"""
LLM recipe parser (step 7).

Takes FetchedPage objects and returns ParsedRecipe objects.
Runs in parallel using ThreadPoolExecutor (spec requirement).

Pre-processing:
  - Each FetchedPage already has `ingredients_excerpt` set by the ingestion pipeline
    (BS4-extracted ingredients section). This is what we send to the LLM — never
    the full page HTML.
  - Sending ~500–1500 chars instead of 10,000+ chars reduces LLM latency significantly.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from bakesquad.llm_client import chat, extract_json
from bakesquad.models import FetchedPage, ParsedRecipe, RecipeIngredient

logger = logging.getLogger(__name__)

# Max workers for parallel parsing — matches MAX_PAGES_PER_RUN
_PARSE_WORKERS = 4

_SYSTEM_PROMPT = (
    "You are a recipe data extractor. Given partial recipe text (title + ingredients section), "
    "extract the structured recipe data and return ONLY a JSON object with these fields:\n\n"
    '- "title": recipe name (string)\n'
    '- "category": one of "cookie", "quick_bread", "cake", "other"\n'
    '- "flour_type": primary flour used — one of: "ap", "almond", "oat", "coconut", "rice", "gf_blend", "other"\n'
    '  Detect from ingredients: almond flour→"almond", oat flour→"oat", coconut flour→"coconut",\n'
    '  rice flour→"rice", any "1:1 gluten-free" or "gluten-free blend"→"gf_blend", default→"ap"\n'
    '- "modifiers": list of applicable tags — choose from:\n'
    '  ["gluten_free", "vegan", "dairy_free", "paleo", "keto", "nut_free"]\n'
    '  Detect from ingredients: no butter/milk/eggs→consider vegan, no wheat flour→gluten_free, etc.\n'
    '  Empty list if none apply.\n'
    '- "ingredients": array of objects, each with:\n'
    '    - "name": ingredient name in lowercase (no quantities, no brand names)\n'
    '    - "quantity": numeric amount (float, 0 if unknown)\n'
    '    - "unit": one of: "cups", "tbsp", "tsp", "g", "oz", "lb", "whole", "sticks"\n'
    '      Use "whole" for countable items (eggs, bananas). '
    '      Convert "1 stick butter" → quantity:1, unit:"sticks"\n'
    '      Convert "1/2 cup" fractions to decimals (0.5)\n'
    '      Convert "3/4 cup" → 0.75, "1/3 cup" → 0.333, "2/3 cup" → 0.667\n'
    '- "yield_description": e.g. "1 loaf", "24 cookies", "one 9-inch cake" (string)\n'
    '- "instruction_count": estimated number of instruction steps (integer, 0 if unknown)\n'
    '- "has_chocolate": true if any chocolate ingredient is present (bool)\n\n'
    "Important:\n"
    "- For 'mashed bananas' or 'ripe bananas': name='mashed banana', unit='whole', qty=count\n"
    "- For 'butter' measured in sticks: unit='sticks'\n"
    "- For 'butter' measured in cups/tbsp: use cups/tbsp\n"
    "- Normalize ingredient names: 'unsalted butter' → 'butter', 'AP flour' → 'all-purpose flour'\n"
    "- Include ALL ingredients, even salt, vanilla, xanthan gum, psyllium husk, etc.\n"
    "Return ONLY the JSON object. No explanation, no markdown fences."
)


def parse_recipe(page: FetchedPage) -> Optional[ParsedRecipe]:
    """Parse one fetched page into a structured recipe. Returns None on failure."""
    # Cap at 1000 chars: BS4 pre-extraction produces a focused ingredients-only text
    # that should fit well within this limit. Keeping input small lets 4 parallel
    # Groq calls stay well under the 6000 TPM rate limit.
    text = (page.ingredients_excerpt or page.raw_text)[:1000]
    if not text.strip():
        logger.warning("No text to parse for %s", page.url)
        return None

    try:
        # max_tokens=700: Groq counts *requested* tokens against TPM limits, not just
        # consumed tokens. A 15-ingredient recipe JSON fits in ~250 output tokens;
        # 700 leaves headroom while keeping 4 parallel calls under the 6000 TPM cap.
        raw = chat(_SYSTEM_PROMPT, text, max_tokens=700, temperature=0)
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError("Expected dict")

        ingredients = []
        for item in data.get("ingredients", []):
            try:
                ing = RecipeIngredient(
                    name=str(item.get("name", "unknown")).lower().strip(),
                    quantity=float(item.get("quantity", 0) or 0),
                    unit=str(item.get("unit", "whole")).lower().strip() or "whole",
                )
                if ing.quantity > 0:
                    ingredients.append(ing)
            except Exception:
                continue

        if not ingredients:
            logger.warning("Parser returned no ingredients for %s", page.url)
            return None

        recipe = ParsedRecipe(
            title=str(data.get("title", page.title or "Unknown Recipe")),
            url=page.url,
            category=data.get("category", "other"),
            flour_type=data.get("flour_type", "ap") or "ap",
            modifiers=data.get("modifiers", []) or [],
            ingredients=ingredients,
            yield_description=str(data.get("yield_description", "")),
            instruction_count=int(data.get("instruction_count", 0) or 0),
            has_chocolate=bool(data.get("has_chocolate", False)),
        )
        logger.info("Parsed %d ingredients from %s", len(ingredients), page.url)
        return recipe

    except Exception as e:
        logger.warning("Parse failed for %s: %s", page.url, e)
        return ParsedRecipe(
            title=page.title or "Unknown Recipe",
            url=page.url,
            category="other",
            ingredients=[],
            parse_error=str(e),
        )


def parse_recipes_parallel(pages: list[FetchedPage]) -> list[ParsedRecipe]:
    """
    Parse all pages in parallel using ThreadPoolExecutor.
    Spec: 'parse all fetched recipes in parallel using the same executor pattern.'
    Drops pages that fail entirely (None return) or have no ingredients.
    """
    results: list[ParsedRecipe] = []
    with ThreadPoolExecutor(max_workers=_PARSE_WORKERS) as executor:
        futures = {executor.submit(parse_recipe, page): page for page in pages}
        for future in as_completed(futures):
            recipe = future.result()
            if recipe is not None and recipe.ingredients:
                results.append(recipe)
    return results
