"""
Ratio engine — computes category-specific ingredient ratios from normalized recipes.

ZERO LLM calls. All math is deterministic; results are cached in SQLite so that a
previously-analyzed recipe skips parse + normalize + compute entirely on repeat queries.

Reference ranges are keyed by "{category}_{flour_type}" (e.g. "cake_almond", "quick_bread_ap").
Ranges for flour types not explicitly listed fall back to the AP flour ranges for that category.

Reference sources:
  AP flour — King Arthur Baking, Serious Eats, BraveTart, Cake Bible
  Almond flour — America's Test Kitchen GF, Elana's Pantry, Detoxinista
  Oat flour — Bob's Red Mill guidelines, King Arthur
  Coconut flour — Bob's Red Mill (coconut flour absorbs ~4× more liquid than AP)
  GF blend — Designed as 1:1 AP substitutes; ranges mirror AP with slightly wider tolerance
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
    result = _compute(
        recipe.url, recipe.category, normalized, recipe.has_chocolate,
        flour_type=recipe.flour_type,
        modifiers=recipe.modifiers,
    )
    cache_put(recipe.url, recipe.category, result.model_dump(exclude={"from_cache"}))
    logger.info("Ratio cache MISS (computed): %s", recipe.url)
    return result


def _compute(
    url: str,
    category: str,
    ingredients: list[NormalizedIngredient],
    has_chocolate: bool,
    flour_type: str = "ap",
    modifiers: Optional[list[str]] = None,
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
    binding_agent = buckets.get("binding_agent", 0.0)

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
        flour_type=flour_type,
        modifiers=modifiers or [],
        has_chocolate=has_chocolate,
        has_banana=has_banana,
        has_binding_agent=binding_agent > 0.0,
        flour_grams=flour,
        fat_type=fat_type,  # type: ignore[arg-type]
    )

    # Flour < 40g (< 1/3 cup) is almost certainly a parse error — skip ratios.
    # Exception: coconut flour recipes use 1/4–1/2 cup (32–64g) per batch.
    min_flour = 20.0 if flour_type == "coconut" else 40.0
    if flour < min_flour:
        logger.warning(
            "Suspiciously low flour (%.0fg) for %s — likely parse error, skipping ratios.",
            flour, url,
        )
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
# Reference range tables — keyed by "{category}_{flour_type}"
#
# Ranges are calibrated per flour type because the same ratio that signals
# a good AP-flour cake (sugar/flour ~0.8) signals an over-sweet almond-flour
# cake (almond flour is denser + naturally sweet, so recipes use more sugar
# per gram of flour but less absolute sugar by volume).
# ---------------------------------------------------------------------------

RATIO_RANGES: dict[str, dict[str, tuple[float, float]]] = {

    # ── AP flour (standard wheat) ─────────────────────────────────────────
    "quick_bread_ap": {
        "liquid_to_flour":    (0.85, 1.50),   # banana bread can go high
        "fat_to_flour":       (0.28, 0.65),
        "sugar_to_flour":     (0.35, 0.90),
        "leavening_to_flour": (0.008, 0.030),
    },
    "cake_ap": {
        "liquid_to_flour":    (0.80, 1.25),
        "fat_to_flour":       (0.35, 0.80),
        "sugar_to_flour":     (0.65, 1.20),
        "leavening_to_flour": (0.010, 0.040),
    },
    "cookie_ap": {
        "fat_to_flour":         (0.40, 0.75),
        "butter_to_flour":      (0.40, 0.75),
        "sugar_to_flour":       (0.55, 1.10),
        "leavening_to_flour":   (0.005, 0.025),
        "brown_to_white_sugar": (0.5, 3.0),
    },

    # ── Almond flour ──────────────────────────────────────────────────────
    # Almond flour is naturally fatty and denser per cup (96g vs 120g).
    # Recipes use much higher fat and sugar ratios relative to flour weight.
    # Liquid ratios are lower because almond flour doesn't absorb as much.
    # Sources: ATK GF Cooking, Elana's Pantry, Detoxinista
    "quick_bread_almond": {
        "liquid_to_flour":    (0.40, 1.20),
        "fat_to_flour":       (0.40, 2.00),
        "sugar_to_flour":     (0.30, 1.50),
        "leavening_to_flour": (0.008, 0.050),
    },
    "cake_almond": {
        "liquid_to_flour":    (0.40, 1.20),
        "fat_to_flour":       (0.50, 2.50),
        "sugar_to_flour":     (0.80, 3.80),   # almond flour cakes are legitimately sweet
        "leavening_to_flour": (0.010, 0.060),
    },
    "cookie_almond": {
        "fat_to_flour":         (0.20, 0.80),
        "butter_to_flour":      (0.20, 0.80),
        "sugar_to_flour":       (0.40, 1.20),
        "leavening_to_flour":   (0.005, 0.040),
        "brown_to_white_sugar": (0.3, 4.0),
    },

    # ── Oat flour ─────────────────────────────────────────────────────────
    # Similar to AP but slightly more absorbent and lighter in texture.
    # Sources: Bob's Red Mill, King Arthur GF guidelines
    "quick_bread_oat": {
        "liquid_to_flour":    (0.80, 1.60),
        "fat_to_flour":       (0.25, 0.70),
        "sugar_to_flour":     (0.30, 0.95),
        "leavening_to_flour": (0.008, 0.035),
    },
    "cake_oat": {
        "liquid_to_flour":    (0.75, 1.30),
        "fat_to_flour":       (0.30, 0.85),
        "sugar_to_flour":     (0.60, 1.30),
        "leavening_to_flour": (0.010, 0.045),
    },
    "cookie_oat": {
        "fat_to_flour":         (0.35, 0.80),
        "butter_to_flour":      (0.35, 0.80),
        "sugar_to_flour":       (0.50, 1.15),
        "leavening_to_flour":   (0.005, 0.030),
        "brown_to_white_sugar": (0.5, 3.5),
    },

    # ── Coconut flour ─────────────────────────────────────────────────────
    # Extremely absorbent — absorbs ~4× more liquid than AP flour per gram.
    # Recipes use far less flour (1/4 to 1/3 the volume) and much more egg/liquid.
    # High fat/flour and liquid/flour ratios are expected and correct.
    # Sources: Bob's Red Mill coconut flour guidelines
    "quick_bread_coconut": {
        "liquid_to_flour":    (2.00, 8.00),   # dramatically higher than AP
        "fat_to_flour":       (0.50, 3.00),
        "sugar_to_flour":     (0.50, 2.50),
        "leavening_to_flour": (0.010, 0.080),
    },
    "cake_coconut": {
        "liquid_to_flour":    (2.00, 8.00),
        "fat_to_flour":       (0.50, 3.00),
        "sugar_to_flour":     (0.80, 3.50),
        "leavening_to_flour": (0.010, 0.080),
    },

    # ── GF blend (1:1 AP substitute) ─────────────────────────────────────
    # Blends like King Arthur Measure-for-Measure or Bob's 1:1 are designed
    # to mimic AP flour behavior. Ranges are AP ranges with slightly wider
    # tolerance since binding agents change the moisture absorption slightly.
    "quick_bread_gf_blend": {
        "liquid_to_flour":    (0.80, 1.60),
        "fat_to_flour":       (0.25, 0.70),
        "sugar_to_flour":     (0.35, 0.95),
        "leavening_to_flour": (0.008, 0.035),
    },
    "cake_gf_blend": {
        "liquid_to_flour":    (0.75, 1.35),
        "fat_to_flour":       (0.30, 0.85),
        "sugar_to_flour":     (0.60, 1.30),
        "leavening_to_flour": (0.010, 0.045),
    },
    "cookie_gf_blend": {
        "fat_to_flour":         (0.38, 0.80),
        "butter_to_flour":      (0.38, 0.80),
        "sugar_to_flour":       (0.50, 1.15),
        "leavening_to_flour":   (0.005, 0.030),
        "brown_to_white_sugar": (0.5, 3.0),
    },
}


def get_ranges(category: str, flour_type: str = "ap") -> dict[str, tuple[float, float]]:
    """
    Return the ratio reference ranges for a (category, flour_type) pair.
    Falls back to AP flour ranges if the specific flour_type isn't listed.
    """
    key = f"{category}_{flour_type}"
    if key in RATIO_RANGES:
        return RATIO_RANGES[key]
    # Fall back to AP for this category
    ap_key = f"{category}_ap"
    return RATIO_RANGES.get(ap_key, {})


def ratio_in_range(
    ratio_name: str,
    value: Optional[float],
    category: str,
    flour_type: str = "ap",
) -> bool:
    if value is None:
        return False
    ranges = get_ranges(category, flour_type)
    bounds = ranges.get(ratio_name)
    if bounds is None:
        return True  # unknown ratio — don't penalize
    lo, hi = bounds
    return lo <= value <= hi
