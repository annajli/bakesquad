"""
Ratio engine — computes category-specific ingredient ratios from normalized recipes.

ZERO LLM calls. All math is deterministic; results are cached in SQLite so that a
previously-analyzed recipe skips parse + normalize + compute entirely on repeat queries.

Reference ranges:
  Quick bread / banana bread: King Arthur Baking + Serious Eats banana bread science
  Cookies: Stella Parks (BraveTart), Kenji Alt-Lopez (Serious Eats)
  Cakes: Rose Levy Beranbaum (Cake Bible), King Arthur
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from bakesquad.memory import cache_get, cache_put
from bakesquad.models import NormalizedIngredient, ParsedRecipe, RatioResult
from bakesquad.normalizer import classify_ingredient, normalize_recipe

logger = logging.getLogger(__name__)


def compute_ratios(recipe: ParsedRecipe) -> RatioResult:
    """
    Main entry point. Checks SQLite cache first; computes on miss.
    Cache hit skips normalize + ratio math entirely (spec requirement).
    """
    cached = cache_get(recipe.url)
    if cached:
        logger.info("Ratio cache HIT: %s", recipe.url)
        result = RatioResult(**{**cached, "from_cache": True})
        return result

    normalized = normalize_recipe(recipe.ingredients)
    result = _compute(recipe.url, recipe.category, normalized, recipe.has_chocolate)
    cache_put(recipe.url, recipe.category, result.model_dump(exclude={"from_cache"}))
    logger.info("Ratio cache MISS (computed): %s", recipe.url)
    return result


def _compute(
    url: str,
    category: str,
    ingredients: list[NormalizedIngredient],
    has_chocolate: bool,
) -> RatioResult:
    # Bucket all ingredients by category
    buckets: dict[str, float] = {}
    for ing in ingredients:
        cat = classify_ingredient(ing.name)
        buckets[cat] = buckets.get(cat, 0.0) + ing.grams

    flour = buckets.get("flour", 0.0)
    fat_butter = buckets.get("fat_butter", 0.0)
    fat_oil = buckets.get("fat_oil", 0.0)
    total_fat = fat_butter + fat_oil
    sugar_white = buckets.get("sugar_white", 0.0)
    sugar_brown = buckets.get("sugar_brown", 0.0)
    total_sugar = sugar_white + sugar_brown
    liquid = buckets.get("liquid", 0.0)
    banana = buckets.get("banana", 0.0)
    egg = buckets.get("egg", 0.0)
    egg_yolk = buckets.get("egg_yolk", 0.0)
    baking_soda = buckets.get("baking_soda", 0.0)
    baking_powder = buckets.get("baking_powder", 0.0)
    total_leavening = baking_soda + baking_powder

    # Determine fat type
    if fat_oil > 0 and fat_butter > 0:
        fat_type: Optional[str] = "mixed"
    elif fat_oil > 0:
        fat_type = "oil"
    elif fat_butter > 0:
        fat_type = "butter"
    else:
        fat_type = "none"

    # Has banana?
    has_banana = banana > 0.0

    base = RatioResult(
        url=url,
        category=category,  # type: ignore[arg-type]
        has_chocolate=has_chocolate,
        has_banana=has_banana,
        flour_grams=flour,
        fat_type=fat_type,  # type: ignore[arg-type]
    )

    if flour < 40.0:
        # Flour < 40g (< 1/3 cup) is almost certainly a parse error — skip ratios.
        # A real single-batch recipe has at least 120g flour.
        logger.warning("Suspiciously low flour (%.0fg) for %s — likely parse error, skipping ratios.", flour, url)
        return base

    if category in ("quick_bread", "cake"):
        # For quick breads, count banana as liquid (it's the primary moisture source)
        effective_liquid = liquid + banana + egg * 0.75 + egg_yolk * 0.5
        base.liquid_to_flour = round(effective_liquid / flour, 3)
        base.fat_to_flour = round(total_fat / flour, 3)
        base.sugar_to_flour = round(total_sugar / flour, 3)
        base.leavening_to_flour = round(total_leavening / flour, 4)

    elif category == "cookie":
        base.butter_to_flour = round(total_fat / flour, 3)
        base.sugar_to_flour = round(total_sugar / flour, 3)
        base.leavening_to_flour = round(total_leavening / flour, 4)
        base.fat_to_flour = round(total_fat / flour, 3)

        if sugar_white > 0:
            base.brown_to_white_sugar = round(sugar_brown / sugar_white, 3)

        base.has_extra_yolks = egg_yolk > (egg * 18)  # extra yolks beyond whole eggs

        if baking_soda > 0 and baking_powder > 0:
            base.leavening_type = "both"
        elif baking_soda > 0:
            base.leavening_type = "soda"
        elif baking_powder > 0:
            base.leavening_type = "powder"
        else:
            base.leavening_type = "none"

    else:
        # "other" category — compute what we can
        effective_liquid = liquid + banana + egg * 0.75
        if effective_liquid > 0:
            base.liquid_to_flour = round(effective_liquid / flour, 3)
        base.fat_to_flour = round(total_fat / flour, 3)
        base.sugar_to_flour = round(total_sugar / flour, 3)
        base.leavening_to_flour = round(total_leavening / flour, 4)

    return base


# ---------------------------------------------------------------------------
# Reference range tables (for display and scoring bounds)
# ---------------------------------------------------------------------------

RATIO_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "quick_bread": {
        "liquid_to_flour":    (0.85, 1.50),   # banana bread can go high
        "fat_to_flour":       (0.28, 0.65),
        "sugar_to_flour":     (0.35, 0.90),
        "leavening_to_flour": (0.008, 0.030),
    },
    "cake": {
        "liquid_to_flour":    (0.80, 1.25),
        "fat_to_flour":       (0.35, 0.80),
        "sugar_to_flour":     (0.65, 1.20),
        "leavening_to_flour": (0.010, 0.040),
    },
    "cookie": {
        "butter_to_flour":    (0.40, 0.75),
        "sugar_to_flour":     (0.55, 1.10),
        "leavening_to_flour": (0.005, 0.025),
        "brown_to_white_sugar": (0.5, 3.0),
    },
}


def ratio_in_range(ratio_name: str, value: Optional[float], category: str) -> bool:
    if value is None:
        return False
    ranges = RATIO_RANGES.get(category, {})
    bounds = ranges.get(ratio_name)
    if bounds is None:
        return True
    lo, hi = bounds
    return lo <= value <= hi
