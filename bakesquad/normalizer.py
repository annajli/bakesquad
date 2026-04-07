"""
Unit normalizer — converts parsed recipe ingredient quantities to grams.

ZERO LLM calls here. If you find yourself reaching for an LLM in this module, stop
and add a row to the lookup tables instead (spec requirement).

Density sources: King Arthur Baking (flour measurements), USDA SR28 (liquids/fats),
Serious Eats (chocolate, nut densities).
"""

from __future__ import annotations

import re
from typing import Optional

from bakesquad.models import NormalizedIngredient, RecipeIngredient

# ---------------------------------------------------------------------------
# Density table: grams per CUP for each ingredient
# (volume → mass conversion; used after converting all units to cups first)
# ---------------------------------------------------------------------------
_GRAMS_PER_CUP: dict[str, float] = {
    # Flours — AP and wheat
    "all-purpose flour": 120.0,
    "all purpose flour": 120.0,
    "ap flour": 120.0,
    "bread flour": 120.0,
    "cake flour": 100.0,
    "whole wheat flour": 120.0,
    "whole-wheat flour": 120.0,
    "flour": 120.0,
    # Alternative / GF flours (King Arthur + manufacturer specs)
    "almond flour": 96.0,
    "almond meal": 96.0,
    "oat flour": 92.0,
    "coconut flour": 128.0,         # extremely absorbent; recipes use far less of it
    "rice flour": 158.0,
    "white rice flour": 158.0,
    "brown rice flour": 163.0,
    "tapioca starch": 120.0,
    "tapioca flour": 120.0,
    "arrowroot starch": 128.0,
    "arrowroot flour": 128.0,
    "arrowroot": 128.0,
    "potato starch": 160.0,
    "cornstarch": 120.0,
    "corn starch": 120.0,
    "gluten-free flour": 120.0,     # generic; approximate
    "gluten free flour": 120.0,
    "gluten-free flour blend": 120.0,
    "gf flour blend": 120.0,
    "1:1 gluten-free flour": 120.0,
    "1-to-1 gluten-free flour": 120.0,
    "gluten-free 1-to-1 baking flour": 120.0,
    # Binding agents (measured in tsp; grams/cup for volume → mass conversion)
    "xanthan gum": 144.0,           # ~3g/tsp × 48 tsp/cup
    "psyllium husk": 192.0,         # ~4g/tsp × 48
    "psyllium husk powder": 192.0,
    "psyllium": 192.0,
    "flax meal": 150.0,             # ~3.1g/tsp × 48
    "ground flaxseed": 150.0,
    "ground flax": 150.0,
    "chia seeds": 160.0,
    "chia seed": 160.0,

    # Sugars
    "granulated sugar": 200.0,
    "white sugar": 200.0,
    "sugar": 200.0,
    "brown sugar": 200.0,         # unpacked; packed ≈ 220g but LLM usually says "packed"
    "light brown sugar": 200.0,
    "dark brown sugar": 200.0,
    "powdered sugar": 120.0,
    "confectioners sugar": 120.0,
    "confectioners' sugar": 120.0,
    "turbinado sugar": 200.0,

    # Fats
    "unsalted butter": 227.0,
    "salted butter": 227.0,
    "butter": 227.0,
    "vegetable oil": 218.0,
    "canola oil": 218.0,
    "olive oil": 216.0,
    "coconut oil": 218.0,
    "neutral oil": 218.0,
    "avocado oil": 218.0,
    "oil": 218.0,
    "shortening": 191.0,
    "lard": 205.0,

    # Liquids
    "whole milk": 240.0,
    "milk": 240.0,
    "skim milk": 245.0,
    "buttermilk": 245.0,
    "heavy cream": 238.0,
    "heavy whipping cream": 238.0,
    "sour cream": 230.0,
    "greek yogurt": 245.0,
    "plain yogurt": 245.0,
    "yogurt": 245.0,
    "water": 240.0,
    "coffee": 240.0,
    "brewed coffee": 240.0,
    "hot water": 240.0,
    "orange juice": 248.0,
    "juice": 248.0,

    # Sweeteners
    "honey": 340.0,
    "maple syrup": 322.0,
    "molasses": 337.0,
    "agave": 328.0,
    "corn syrup": 328.0,

    # Cocoa / chocolate
    "cocoa powder": 85.0,
    "dutch-process cocoa": 85.0,
    "dutch process cocoa": 85.0,
    "natural cocoa": 85.0,
    "unsweetened cocoa": 85.0,
    "chocolate chips": 170.0,
    "semi-sweet chocolate chips": 170.0,
    "semisweet chocolate chips": 170.0,
    "dark chocolate chips": 170.0,
    "milk chocolate chips": 170.0,
    "mini chocolate chips": 168.0,
    "chocolate chunks": 170.0,
    "chopped chocolate": 170.0,

    # Leaveners (measured in tbsp/tsp — cups included for completeness)
    "baking soda": 274.0,
    "baking powder": 192.0,
    "salt": 288.0,
    "kosher salt": 144.0,         # Diamond Crystal is half the density of table salt
    "sea salt": 288.0,

    # Eggs (handled separately by count, but include volume ref)
    "eggs": 240.0,                # 1 cup ≈ 4-5 large eggs

    # Fruit / produce
    "mashed banana": 225.0,
    "mashed bananas": 225.0,
    "banana": 225.0,
    "bananas": 225.0,
    "applesauce": 245.0,
    "pumpkin puree": 245.0,
    "pumpkin": 245.0,

    # Nuts / mix-ins
    "walnuts": 100.0,
    "chopped walnuts": 100.0,
    "pecans": 100.0,
    "chopped pecans": 100.0,
    "almonds": 95.0,
    "rolled oats": 81.0,
    "old-fashioned oats": 81.0,
    "quick oats": 81.0,
    "oats": 81.0,

    # Dairy extras
    "cream cheese": 232.0,
    "ricotta": 246.0,
    "vanilla extract": 208.0,
    "vanilla": 208.0,
}

# ---------------------------------------------------------------------------
# Unit → cup conversion factors
# ---------------------------------------------------------------------------
_UNIT_TO_CUPS: dict[str, float] = {
    # Cups
    "cup": 1.0,
    "cups": 1.0,
    "c": 1.0,
    "c.": 1.0,
    # Tablespoons
    "tablespoon": 1 / 16,
    "tablespoons": 1 / 16,
    "tbsp": 1 / 16,
    "tbs": 1 / 16,
    "tbsps": 1 / 16,
    "t": 1 / 16,
    # Teaspoons
    "teaspoon": 1 / 48,
    "teaspoons": 1 / 48,
    "tsp": 1 / 48,
    "tsps": 1 / 48,
    "t.": 1 / 48,
}

# ---------------------------------------------------------------------------
# Unit → gram direct conversion (mass units need no density lookup)
# ---------------------------------------------------------------------------
_UNIT_TO_GRAMS: dict[str, float] = {
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "gr": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "oz": 28.3495,
    "ounce": 28.3495,
    "ounces": 28.3495,
    "lb": 453.592,
    "lbs": 453.592,
    "pound": 453.592,
    "pounds": 453.592,
}

# ---------------------------------------------------------------------------
# Per-item gram weights for countable ingredients
# ---------------------------------------------------------------------------
_GRAMS_PER_WHOLE: dict[str, float] = {
    "egg": 50.0,            # large egg
    "eggs": 50.0,
    "large egg": 50.0,
    "large eggs": 50.0,
    "medium egg": 44.0,
    "medium eggs": 44.0,
    "egg yolk": 18.0,
    "egg yolks": 18.0,
    "yolk": 18.0,
    "yolks": 18.0,
    "egg white": 32.0,
    "egg whites": 32.0,
    "white": 32.0,
    "banana": 100.0,        # 1 medium banana, mashed ≈ 100g
    "bananas": 100.0,
    "ripe banana": 100.0,
    "ripe bananas": 100.0,
    "stick": 113.5,         # 1 stick butter = 4 oz = 113.4g
    "sticks": 113.5,
    "stick of butter": 113.5,
    "sticks of butter": 113.5,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_ingredient(ing: RecipeIngredient) -> NormalizedIngredient:
    """Convert one ingredient to grams. Marks low_confidence when uncertain."""
    name_lower = ing.name.lower().strip()
    unit_lower = ing.unit.lower().strip()
    qty = ing.quantity

    # 1. Direct mass units — no density lookup needed
    if unit_lower in _UNIT_TO_GRAMS:
        grams = qty * _UNIT_TO_GRAMS[unit_lower]
        return NormalizedIngredient(name=ing.name, grams=grams)

    # 2. "whole" / count units (eggs, bananas, sticks)
    if unit_lower in ("whole", "count", "", "piece", "pieces", "item", "items") or unit_lower in _GRAMS_PER_WHOLE:
        # Check the unit itself first, then the name
        gpw = _GRAMS_PER_WHOLE.get(unit_lower) or _match_density(name_lower, _GRAMS_PER_WHOLE)
        if gpw:
            return NormalizedIngredient(name=ing.name, grams=qty * gpw)
        # Unknown countable ingredient — flag as low confidence
        return NormalizedIngredient(name=ing.name, grams=qty * 50.0, low_confidence=True)

    # 3. Volume units → cups → grams via density table
    cup_factor = _UNIT_TO_CUPS.get(unit_lower)
    if cup_factor is not None:
        cups = qty * cup_factor
        density = _match_density(name_lower, _GRAMS_PER_CUP)
        if density:
            return NormalizedIngredient(name=ing.name, grams=cups * density)
        # Unknown ingredient in a volume unit — flag as low confidence
        return NormalizedIngredient(name=ing.name, grams=cups * 130.0, low_confidence=True)

    # 4. Fallback — unknown unit, flag low confidence
    return NormalizedIngredient(name=ing.name, grams=qty * 30.0, low_confidence=True)


def normalize_recipe(ingredients: list[RecipeIngredient]) -> list[NormalizedIngredient]:
    """Normalize all ingredients in a recipe. No LLM calls."""
    return [normalize_ingredient(ing) for ing in ingredients]


# ---------------------------------------------------------------------------
# Ingredient category helpers (used by ratio engine)
# ---------------------------------------------------------------------------

def classify_ingredient(name: str) -> str:
    """
    Map a normalized ingredient name to a category used by the ratio engine.
    Returns one of: flour, fat_butter, fat_oil, liquid, egg, banana,
    sugar_white, sugar_brown, cocoa, chocolate, baking_soda, baking_powder, other.

    Deterministic lookup — no LLM.
    """
    n = name.lower()

    # Flour
    if "flour" in n:
        return "flour"

    # Baking soda / powder (check before general "baking" match)
    if "baking soda" in n or "bicarbonate" in n:
        return "baking_soda"
    if "baking powder" in n:
        return "baking_powder"

    # Fats — distinguish oil vs butter
    if any(w in n for w in ("butter", "margarine", "shortening", "lard")):
        return "fat_butter"
    if any(w in n for w in ("oil", "crisco")):
        return "fat_oil"

    # Liquids
    if any(w in n for w in ("milk", "buttermilk", "cream", "yogurt", "kefir",
                             "water", "coffee", "juice", "sour cream", "applesauce",
                             "pumpkin")):
        return "liquid"

    # Eggs
    if "yolk" in n:
        return "egg_yolk"
    if "egg" in n:
        return "egg"

    # Banana
    if "banana" in n:
        return "banana"

    # Sugars
    if "brown sugar" in n or "light brown" in n or "dark brown" in n:
        return "sugar_brown"
    if any(w in n for w in ("sugar", "honey", "maple", "molasses", "agave", "syrup")):
        return "sugar_white"

    # Chocolate / cocoa
    if "cocoa" in n:
        return "cocoa"
    if any(w in n for w in ("chocolate chip", "chocolate chunk", "chopped chocolate")):
        return "chocolate"

    # GF binding agents (xanthan gum, psyllium husk, flax/chia eggs)
    if any(w in n for w in ("xanthan", "psyllium", "flax meal", "flaxseed", "ground flax",
                             "chia seed", "chia")):
        return "binding_agent"
    # "flax egg" is a common GF egg substitute; classify with binding agents
    if "flax egg" in n:
        return "binding_agent"

    return "other"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _match_density(name: str, table: dict[str, float]) -> Optional[float]:
    """
    Look up density with fallback substring matching.
    Tries exact match first, then longest-substring match, then word overlap.
    """
    # Exact
    if name in table:
        return table[name]

    # Longest key that is a substring of the ingredient name
    best_key = ""
    best_val: Optional[float] = None
    for key, val in table.items():
        if key in name and len(key) > len(best_key):
            best_key = key
            best_val = val
    if best_val is not None:
        return best_val

    # Name is a substring of a table key (e.g. "flour" → "all-purpose flour")
    for key, val in table.items():
        if name in key:
            return val

    return None
