"""
Scorer (step 10).

Two phases:
  1. Deterministic math — compute per-criterion scores (0–100) from ratio results.
     ZERO LLM calls in this phase.
  2. Batched LLM call — generate "Why this score" explanations + apply scores for
     criteria that ratio data cannot reach (Flavor Complexity, Technique & Layers).

Scoring dispatches per category. Each category has its own named criteria set and
default weights, reflecting the quality axes that actually matter for that baked good.

  cookie        → Chew & Texture, Spread & Structure, Sweetness Balance, Flavor & Technique
  quick_bread   → Moisture & Tenderness, Rise & Dome, Sweetness Balance
  cake          → Moisture & Tenderness, Crumb & Structure, Sweetness Calibration
  yeasted_bread → Hydration, Enrichment Level, Flavor Complexity (LLM-assessed)
  pastry        → Fat & Richness, Structure & Balance, Technique & Layers (LLM-assessed)
  other         → Overall Balance

LLM-assessed criteria (Flavor Complexity, Technique & Layers) start at placeholder 50.0
and are scored 0–100 by the batched add_explanations() call. This keeps the deterministic
phase fast while allowing nuanced differentiation where ratios genuinely can't reach.

Universal add-ons (applied after category criteria):
  GF Binding Agent  — fixed weight 0.20, GF recipes only
  Accessibility     — optional, gated by user_prefs["prefer_accessibility"] > 0
"""

from __future__ import annotations

import logging
from typing import Optional

from bakesquad.llm_client import chat, extract_json
from bakesquad.models import (
    CriterionScore,
    ParsedRecipe,
    QueryPlan,
    RatioResult,
    ScoredRecipe,
)
from bakesquad.ratio_engine import get_ranges

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-category criteria registry
# ---------------------------------------------------------------------------
# Each entry: list of {name, default_weight, llm_assessed}
# llm_assessed=True → placeholder 50.0; add_explanations() fills real score.
# Weights must sum to 1.0 (before universal add-ons rescale the total).

CATEGORY_CRITERIA: dict[str, list[dict]] = {
    "cookie": [
        {"name": "Chew & Texture",     "weight": 0.40, "llm": False},
        {"name": "Spread & Structure", "weight": 0.25, "llm": False},
        {"name": "Sweetness Balance",  "weight": 0.20, "llm": False},
        {"name": "Flavor & Technique", "weight": 0.15, "llm": True},
    ],
    "quick_bread": [
        {"name": "Moisture & Tenderness", "weight": 0.50, "llm": False},
        {"name": "Rise & Dome",           "weight": 0.30, "llm": False},
        {"name": "Sweetness Balance",     "weight": 0.20, "llm": False},
    ],
    "cake": [
        {"name": "Moisture & Tenderness", "weight": 0.45, "llm": False},
        {"name": "Crumb & Structure",     "weight": 0.30, "llm": False},
        {"name": "Sweetness Calibration", "weight": 0.25, "llm": False},
    ],
    "yeasted_bread": [
        {"name": "Hydration",         "weight": 0.30, "llm": False},
        {"name": "Enrichment Level",  "weight": 0.25, "llm": False},
        {"name": "Flavor Complexity", "weight": 0.45, "llm": True},
    ],
    "pastry": [
        {"name": "Fat & Richness",     "weight": 0.35, "llm": False},
        {"name": "Structure & Balance","weight": 0.30, "llm": False},
        {"name": "Technique & Layers", "weight": 0.35, "llm": True},
    ],
    "other": [
        {"name": "Overall Balance", "weight": 1.0, "llm": False},
    ],
}

# Names of criteria that the LLM scores directly (used in add_explanations prompt)
_LLM_CRITERIA_NAMES: set[str] = {
    c["name"]
    for criteria in CATEGORY_CRITERIA.values()
    for c in criteria
    if c["llm"]
}


# ---------------------------------------------------------------------------
# Weight derivation from query
# ---------------------------------------------------------------------------

def derive_weights(plan: QueryPlan, user_prefs: dict) -> dict[str, float]:
    """
    Build criterion weight vector from CATEGORY_CRITERIA defaults, then apply
    query signal boosts and normalize to sum to 1.0.

    User per-category prefs (stored as {criterion_name_snake: weight}) override
    the defaults when present.
    """
    category = plan.category
    spec = CATEGORY_CRITERIA.get(category, CATEGORY_CRITERIA["other"])

    # Start from category defaults
    weights: dict[str, float] = {c["name"]: c["weight"] for c in spec}

    # Apply saved per-category user overrides
    cat_prefs: dict = (user_prefs.get("category_prefs") or {}).get(category, {})
    for name in list(weights.keys()):
        pref_key = name.lower().replace(" & ", "_and_").replace(" ", "_") + "_weight"
        if pref_key in cat_prefs:
            weights[name] = float(cat_prefs[pref_key])

    # Query signal boosts — boost the most relevant criterion(s) for the signal
    query_text = " ".join(plan.hard_constraints + plan.soft_preferences).lower()

    moisture_signals = {"moist", "moisture", "soft", "fresh", "days", "week",
                        "tender", "fudgy", "dense", "juicy"}
    if any(s in query_text for s in moisture_signals):
        _boost(weights, ["Moisture & Tenderness", "Chew & Texture", "Hydration"], 0.15)

    texture_signals = {"crispy", "chewy", "crunchy", "fluffy", "airy", "rise",
                       "lift", "flaky", "crumb", "open", "shatter"}
    if any(s in query_text for s in texture_signals):
        _boost(weights, ["Spread & Structure", "Rise & Dome", "Crumb & Structure",
                         "Technique & Layers"], 0.10)

    flavor_signals = {"flavor", "rich", "buttery", "complex", "depth", "tangy",
                      "ferment", "sourdough", "nutty", "caramel", "brown butter"}
    if any(s in query_text for s in flavor_signals):
        _boost(weights, ["Flavor & Technique", "Flavor Complexity"], 0.10)

    # Normalize
    total = sum(weights.values())
    if total > 0:
        return {k: round(v / total, 3) for k, v in weights.items()}
    return weights


def _boost(weights: dict[str, float], names: list[str], amount: float) -> None:
    """Add `amount` to any of `names` that exist in weights (in place)."""
    for name in names:
        if name in weights:
            weights[name] = min(0.90, weights[name] + amount)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def score_recipe(
    recipe: ParsedRecipe,
    ratios: RatioResult,
    plan: QueryPlan,
    user_prefs: dict,
) -> ScoredRecipe:
    """Compute all criterion scores and composite. No LLM calls."""
    category = ratios.category
    modifiers = ratios.modifiers or []
    flour_type = ratios.flour_type or "ap"

    weights = derive_weights(plan, user_prefs)
    criteria: list[CriterionScore] = []
    violations: list[str] = []

    # --- Dispatch to per-category scoring ---
    if category == "cookie":
        criteria = _score_cookie(recipe, ratios, weights, flour_type)
    elif category == "quick_bread":
        criteria = _score_quick_bread(recipe, ratios, weights, flour_type)
    elif category == "cake":
        criteria = _score_cake(recipe, ratios, weights, flour_type)
    elif category == "yeasted_bread":
        criteria = _score_yeasted_bread(recipe, ratios, weights, flour_type)
    elif category == "pastry":
        criteria = _score_pastry(recipe, ratios, weights, flour_type)
    else:
        criteria = _score_other(recipe, ratios, weights, flour_type)

    # --- Universal add-ons ---

    # GF Binding Agent (fixed weight, GF recipes only)
    if "gluten_free" in modifiers or flour_type in ("almond", "oat", "coconut", "rice", "gf_blend"):
        binding_score = 100.0 if ratios.has_binding_agent else 20.0
        criteria.append(CriterionScore(
            name="GF Binding Agent",
            score=binding_score,
            weight=0.20,
            details="xanthan/psyllium/flax detected" if ratios.has_binding_agent else "no binding agent detected",
        ))

    # Accessibility (optional, LLM-assessed — placeholder until add_explanations fires)
    accessibility_weight = round(float(user_prefs.get("prefer_accessibility", 0.0)) * 0.20, 3)
    if accessibility_weight > 0:
        technique_signals = getattr(recipe, "technique_signals", []) or []
        instruction_count = getattr(recipe, "instruction_count", 0) or 0
        criteria.append(CriterionScore(
            name="Accessibility",
            score=50.0,
            weight=accessibility_weight,
            details=f"instructions={instruction_count}; signals={', '.join(technique_signals) or 'none'}",
        ))

    # --- Hard constraint violations ---
    if "chocolate" in " ".join(plan.hard_constraints + plan.soft_preferences).lower():
        if not ratios.has_chocolate:
            violations.append("No chocolate detected (query requires chocolate)")

    # --- Composite ---
    total_weight = sum(c.weight for c in criteria)
    composite = (
        sum(c.score * c.weight for c in criteria) / total_weight
        if total_weight > 0 else 0.0
    )
    if violations:
        composite = max(0.0, composite - 20.0 * len(violations))
    composite = round(min(100.0, max(0.0, composite)), 1)

    return ScoredRecipe(
        recipe=recipe,
        ratios=ratios,
        criteria=criteria,
        composite_score=composite,
        constraint_violations=violations,
    )


def score_all(
    recipes: list[ParsedRecipe],
    ratios_list: list[RatioResult],
    plan: QueryPlan,
    user_prefs: dict,
) -> list[ScoredRecipe]:
    """Score all recipes, rank by composite score."""
    scored = [score_recipe(r, rt, plan, user_prefs) for r, rt in zip(recipes, ratios_list)]
    scored.sort(key=lambda s: s.composite_score, reverse=True)
    for i, s in enumerate(scored):
        s.rank = i + 1
    return scored


# ---------------------------------------------------------------------------
# Per-category scoring functions
# ---------------------------------------------------------------------------

def _score_cookie(
    recipe: ParsedRecipe,
    ratios: RatioResult,
    weights: dict[str, float],
    flour_type: str,
) -> list[CriterionScore]:
    ranges = get_ranges("cookie", flour_type)
    technique_signals = getattr(recipe, "technique_signals", []) or []

    # --- Chew & Texture ---
    # Primary axes: brown/white sugar ratio (chew), leavening type (crispy vs cakey),
    # fat/flour in range, extra yolks (richness + structure).
    chew = 0.0
    bw = ratios.brown_to_white_sugar
    if bw is not None:
        if bw >= 2.0:    chew += 40
        elif bw >= 1.5:  chew += 32
        elif bw >= 1.0:  chew += 22
        elif bw >= 0.5:  chew += 12
        else:            chew += 5
    else:
        # All brown or all white — infer from sugar_to_flour if available
        sf = ratios.sugar_to_flour
        chew += 20 if (sf and sf > 0.5) else 10

    lt = ratios.leavening_type
    if lt == "both":   chew += 30   # lift + spread = chewy center / crisp edge
    elif lt == "soda": chew += 25   # more spread, browning = caramelized chew
    elif lt == "none": chew += 20   # dense / shortbread-like (intentional)
    elif lt == "powder": chew += 15 # more lift, cakier

    bf = ratios.butter_to_flour or ratios.fat_to_flour
    lo_fat, hi_fat = ranges.get("fat_to_flour", (0.40, 0.75))
    if bf is not None and lo_fat <= bf <= hi_fat:
        chew += 20
    elif bf is not None:
        dist = min(abs(bf - lo_fat), abs(bf - hi_fat))
        chew += max(0.0, 20 * (1 - dist / 0.3))

    if ratios.has_extra_yolks:
        chew += 10

    chew_detail_parts = []
    if bw is not None:
        chew_detail_parts.append(f"brown/white={bw:.2f}")
    if lt:
        chew_detail_parts.append(f"leavening={lt}")
    if bf is not None:
        chew_detail_parts.append(f"fat/flour={bf:.2f}")
    if ratios.has_extra_yolks:
        chew_detail_parts.append("extra yolks")

    # --- Spread & Structure ---
    # Leavening in range (too much → flat/greasy spread; too little → puff with no spread)
    # Fat/flour in range (too high → greasy spread; too low → dry, no spread)
    spread = 0.0
    lf = ratios.leavening_to_flour
    lo_lv, hi_lv = ranges.get("leavening_to_flour", (0.005, 0.025))
    if lf is not None:
        if lo_lv <= lf <= hi_lv:
            spread += 60
        elif lf < lo_lv:
            spread += max(0.0, 60 * (lf / lo_lv))
        else:
            excess = (lf - hi_lv) / hi_lv
            spread += max(0.0, 60 * (1 - excess * 2))

    if bf is not None:
        if lo_fat <= bf <= hi_fat:
            spread += 40
        else:
            dist = min(abs(bf - lo_fat), abs(bf - hi_fat))
            spread += max(0.0, 40 * (1 - dist / 0.3))

    spread_detail_parts = []
    if lf is not None:
        spread_detail_parts.append(f"leavening/flour={lf:.4f}")
    if bf is not None:
        spread_detail_parts.append(f"fat/flour={bf:.2f}")

    # --- Sweetness Balance ---
    sweetness = _score_balance(ratios, "cookie", flour_type)
    sweetness_detail = _balance_detail(ratios)

    # --- Flavor & Technique (LLM placeholder) ---
    flavor_detail = ", ".join(technique_signals) if technique_signals else "awaiting LLM assessment"

    return [
        CriterionScore(name="Chew & Texture",     score=round(min(100.0, chew), 1),   weight=weights.get("Chew & Texture", 0.40),     details=", ".join(chew_detail_parts) or "n/a"),
        CriterionScore(name="Spread & Structure",  score=round(min(100.0, spread), 1), weight=weights.get("Spread & Structure", 0.25), details=", ".join(spread_detail_parts) or "n/a"),
        CriterionScore(name="Sweetness Balance",   score=sweetness,                    weight=weights.get("Sweetness Balance", 0.20),  details=sweetness_detail),
        CriterionScore(name="Flavor & Technique",  score=50.0,                         weight=weights.get("Flavor & Technique", 0.15), details=flavor_detail),
    ]


def _score_quick_bread(
    recipe: ParsedRecipe,  # noqa: ARG001
    ratios: RatioResult,
    weights: dict[str, float],
    flour_type: str,
) -> list[CriterionScore]:
    ranges = get_ranges("quick_bread", flour_type)

    # --- Moisture & Tenderness ---
    # Fat type (oil stays liquid → better multi-day moisture), liquid/flour, banana,
    # and sugar hygroscopicity all contribute.
    moisture = 0.0
    fat_type = ratios.fat_type or "none"
    if fat_type == "oil":     moisture += 30
    elif fat_type == "mixed": moisture += 20
    elif fat_type == "butter": moisture += 10

    lf_val = ratios.liquid_to_flour
    if lf_val is not None:
        lo_lf, hi_lf = ranges.get("liquid_to_flour", (0.85, 1.50))
        mid = (lo_lf + hi_lf) / 2
        half_width = (hi_lf - lo_lf) / 2
        if lo_lf <= lf_val <= hi_lf:
            dist = abs(lf_val - mid) / half_width
            moisture += 35 * (1.0 - 0.5 * dist)
        elif lf_val < lo_lf:
            moisture += max(0.0, 35 * (lf_val / lo_lf) * 0.6)
        else:
            moisture += max(0.0, 35 - (lf_val - hi_lf) * 20)

    sf = ratios.sugar_to_flour
    if sf is not None:
        lo_sf, hi_sf = ranges.get("sugar_to_flour", (0.35, 0.90))
        if lo_sf <= sf <= hi_sf:
            moisture += 15
        elif sf < lo_sf:
            moisture += 8 * (sf / lo_sf)
        else:
            moisture += 11

    if ratios.has_banana:
        moisture += 15

    moisture_detail_parts = []
    if ratios.fat_type:
        moisture_detail_parts.append(f"fat={ratios.fat_type}")
    if lf_val is not None:
        moisture_detail_parts.append(f"liquid/flour={lf_val:.2f}")
    if ratios.has_banana:
        moisture_detail_parts.append("banana present")

    # --- Rise & Dome ---
    # Leavening/flour in range predicts good lift and dome. Over-leavening causes
    # tunneling (large internal voids) — penalized more steeply than under-leavening.
    rise = 0.0
    lv = ratios.leavening_to_flour
    lo_lv, hi_lv = ranges.get("leavening_to_flour", (0.008, 0.030))
    if lv is not None:
        if lo_lv <= lv <= hi_lv:
            rise += 70
        elif lv < lo_lv:
            rise += max(0.0, 70 * (lv / lo_lv))
        else:
            # Over-leavening → tunneling risk; steeper penalty
            excess = (lv - hi_lv) / hi_lv
            rise += max(0.0, 70 * (1 - excess * 2.5))

    fat_ratio = ratios.fat_to_flour
    if fat_ratio is not None:
        lo_fat, hi_fat = ranges.get("fat_to_flour", (0.28, 0.65))
        if lo_fat <= fat_ratio <= hi_fat:
            rise += 30
        else:
            dist = min(abs(fat_ratio - lo_fat), abs(fat_ratio - hi_fat))
            rise += max(0.0, 30 * (1 - dist / 0.3))

    rise_detail_parts = []
    if lv is not None:
        rise_detail_parts.append(f"leavening/flour={lv:.4f}")
        if lv > hi_lv:
            rise_detail_parts.append("tunneling risk")
    if fat_ratio is not None:
        rise_detail_parts.append(f"fat/flour={fat_ratio:.2f}")

    # --- Sweetness Balance ---
    sweetness = _score_balance(ratios, "quick_bread", flour_type)

    return [
        CriterionScore(name="Moisture & Tenderness", score=round(min(100.0, moisture), 1), weight=weights.get("Moisture & Tenderness", 0.50), details=", ".join(moisture_detail_parts) or "n/a"),
        CriterionScore(name="Rise & Dome",            score=round(min(100.0, rise), 1),     weight=weights.get("Rise & Dome", 0.30),           details=", ".join(rise_detail_parts) or "n/a"),
        CriterionScore(name="Sweetness Balance",      score=sweetness,                       weight=weights.get("Sweetness Balance", 0.20),     details=_balance_detail(ratios)),
    ]


def _score_cake(
    recipe: ParsedRecipe,
    ratios: RatioResult,
    weights: dict[str, float],
    flour_type: str,
) -> list[CriterionScore]:
    ranges = get_ranges("cake", flour_type)

    # Custard-style cakes (cheesecake, flourless, clafoutis): flour ratios are
    # meaningless. Use fixed neutral-positive scores; LLM explanations will fill
    # in the texture assessment.
    is_custard = ratios.flour_grams < 40.0 and ratios.liquid_to_flour is None

    # --- Moisture & Tenderness ---
    if is_custard:
        moisture = 70.0
        moisture_detail = "custard-style cake — egg-set structure, flour ratios not applicable"
    else:
        moisture = 0.0
        fat_type = ratios.fat_type or "none"
        if fat_type == "oil":      moisture += 30
        elif fat_type == "mixed":  moisture += 20
        elif fat_type == "butter": moisture += 12

        lf_val = ratios.liquid_to_flour
        if lf_val is not None:
            lo_lf, hi_lf = ranges.get("liquid_to_flour", (0.80, 1.25))
            mid = (lo_lf + hi_lf) / 2
            half_width = (hi_lf - lo_lf) / 2
            if lo_lf <= lf_val <= hi_lf:
                dist = abs(lf_val - mid) / half_width
                moisture += 40 * (1.0 - 0.5 * dist)
            elif lf_val < lo_lf:
                moisture += max(0.0, 40 * (lf_val / lo_lf) * 0.6)
            else:
                moisture += max(0.0, 40 - (lf_val - hi_lf) * 20)

        sf = ratios.sugar_to_flour
        if sf is not None:
            lo_sf, hi_sf = ranges.get("sugar_to_flour", (0.65, 1.20))
            if lo_sf <= sf <= hi_sf:
                moisture += 20
            elif sf > hi_sf:
                moisture += 14
            else:
                moisture += max(0.0, 20 * (sf / lo_sf))

        moisture_detail_parts = []
        if ratios.fat_type:
            moisture_detail_parts.append(f"fat={ratios.fat_type}")
        if lf_val is not None:
            moisture_detail_parts.append(f"liquid/flour={lf_val:.2f}")
        moisture_detail = ", ".join(moisture_detail_parts) or "n/a"

    # --- Crumb & Structure ---
    if is_custard:
        structure = 65.0
        structure_detail = "custard-style cake — no chemical leavening expected"
    else:
        structure = 0.0
        lv = ratios.leavening_to_flour
        lo_lv, hi_lv = ranges.get("leavening_to_flour", (0.010, 0.040))
        if lv is not None:
            if lo_lv <= lv <= hi_lv:
                structure += 60
            elif lv < lo_lv:
                structure += max(0.0, 60 * (lv / lo_lv))
            else:
                excess = (lv - hi_lv) / hi_lv
                structure += max(0.0, 60 * (1 - excess * 2))

        fat_ratio = ratios.fat_to_flour
        lo_fat, hi_fat = ranges.get("fat_to_flour", (0.35, 0.80))
        if fat_ratio is not None:
            if lo_fat <= fat_ratio <= hi_fat:
                structure += 40
            else:
                dist = min(abs(fat_ratio - lo_fat), abs(fat_ratio - hi_fat))
                structure += max(0.0, 40 * (1 - dist / 0.3))

        structure_detail_parts = []
        if lv is not None:
            structure_detail_parts.append(f"leavening/flour={lv:.4f}")
        if fat_ratio is not None:
            structure_detail_parts.append(f"fat/flour={fat_ratio:.2f}")
        structure_detail = ", ".join(structure_detail_parts) or "n/a"

    # --- Sweetness Calibration ---
    sweetness = _score_balance(ratios, "cake", flour_type)

    return [
        CriterionScore(name="Moisture & Tenderness", score=round(min(100.0, moisture), 1),  weight=weights.get("Moisture & Tenderness", 0.45), details=moisture_detail),
        CriterionScore(name="Crumb & Structure",     score=round(min(100.0, structure), 1), weight=weights.get("Crumb & Structure", 0.30),     details=structure_detail),
        CriterionScore(name="Sweetness Calibration", score=sweetness,                        weight=weights.get("Sweetness Calibration", 0.25), details=_balance_detail(ratios)),
    ]


def _score_yeasted_bread(
    recipe: ParsedRecipe,
    ratios: RatioResult,
    weights: dict[str, float],
    flour_type: str,
) -> list[CriterionScore]:
    ranges = get_ranges("yeasted_bread", flour_type)

    # --- Hydration ---
    # 55–85% is the valid range for most styles. Within range = full credit.
    # Lower = potentially dense/stiff; higher = potentially too slack (but
    # high-hydration styles like ciabatta are intentional — not penalized as
    # steeply as under-hydration).
    hydration = 50.0  # neutral default if ratio not available
    hydration_detail = "hydration not measured"
    lf_val = ratios.liquid_to_flour
    if lf_val is not None:
        lo, hi = ranges.get("liquid_to_flour", (0.55, 0.85))
        if lo <= lf_val <= hi:
            hydration = 100.0
        elif lf_val < lo:
            hydration = max(20.0, 100.0 * (lf_val / lo))
        else:
            # High-hydration styles exist — softer penalty above range
            excess = (lf_val - hi) / hi
            hydration = max(40.0, 100.0 * (1 - excess * 0.8))
        hydration_detail = f"liquid/flour={lf_val:.2f} ({lf_val*100:.0f}% hydration)"

    # --- Enrichment Level ---
    # Fat and sugar indicate enrichment style. For yeasted breads, enrichment
    # is a style indicator rather than a quality flaw: lean is correct for
    # sourdough/baguette; rich is correct for brioche/challah. We score based
    # on whether the recipe has measurable enrichment data (not whether it's
    # high or low), which favors recipes with complete ingredient lists.
    enrichment = 50.0  # neutral default
    enrichment_parts: list[str] = []
    fat_ratio = ratios.fat_to_flour
    sugar_ratio = ratios.sugar_to_flour
    data_points = 0

    if fat_ratio is not None:
        data_points += 1
        _, hi_f = ranges.get("fat_to_flour", (0.00, 0.60))
        if fat_ratio <= hi_f:
            enrichment += 20   # in range for some valid style
        else:
            enrichment += 10   # very high fat — unusual but not wrong
        enrichment_parts.append(f"fat/flour={fat_ratio:.2f}")

    if sugar_ratio is not None:
        data_points += 1
        _, hi_s = ranges.get("sugar_to_flour", (0.00, 0.25))
        if sugar_ratio <= hi_s:
            enrichment += 20
        else:
            enrichment += 10
        enrichment_parts.append(f"sugar/flour={sugar_ratio:.2f}")

    if data_points == 0:
        enrichment = 50.0  # can't assess — neutral

    enrichment = round(min(100.0, enrichment), 1)
    enrichment_detail = ", ".join(enrichment_parts) if enrichment_parts else "no enrichment data"

    # --- Flavor Complexity (LLM-assessed placeholder) ---
    # Primary differentiator for yeasted breads. Fermentation depth, crust character,
    # and crumb aroma cannot be derived from ratios. LLM scores 0–100 in add_explanations.
    technique_signals = getattr(recipe, "technique_signals", []) or []
    flavor_detail = ", ".join(technique_signals) if technique_signals else "awaiting LLM assessment"

    return [
        CriterionScore(name="Hydration",         score=round(hydration, 1),   weight=weights.get("Hydration", 0.30),         details=hydration_detail),
        CriterionScore(name="Enrichment Level",  score=enrichment,             weight=weights.get("Enrichment Level", 0.25),  details=enrichment_detail),
        CriterionScore(name="Flavor Complexity", score=50.0,                   weight=weights.get("Flavor Complexity", 0.45), details=flavor_detail),
    ]


def _score_pastry(
    recipe: ParsedRecipe,
    ratios: RatioResult,
    weights: dict[str, float],
    flour_type: str,
) -> list[CriterionScore]:
    ranges = get_ranges("pastry", flour_type)

    # --- Fat & Richness ---
    # High fat/flour is correct for most pastry: tart/pie crust 0.45–0.75,
    # croissant 0.50–1.20 (including separate butter block), choux ~0.5–1.0.
    # Low fat/flour in a pastry is a quality signal worth penalizing.
    fat_ratio = ratios.fat_to_flour
    fat_score = 50.0  # neutral if no data
    fat_detail = "fat not measured"
    if fat_ratio is not None:
        lo, hi = ranges.get("fat_to_flour", (0.40, 1.20))
        if fat_ratio >= lo:
            # In range or above (very high fat = laminated with separate butter block)
            fat_score = min(100.0, 60.0 + (fat_ratio - lo) / (hi - lo) * 40.0)
        else:
            # Below minimum — under-fat pastry
            fat_score = max(10.0, 100.0 * (fat_ratio / lo))
        fat_detail = f"fat/flour={fat_ratio:.2f}"
        if ratios.fat_type:
            fat_detail += f", type={ratios.fat_type}"

    # --- Structure & Balance ---
    # Liquid/flour and sugar/flour in appropriate pastry range.
    # Choux will have very high liquid (detected and not penalized if fat is also high).
    structure = 0.0
    structure_parts: list[str] = []

    lf_val = ratios.liquid_to_flour
    if lf_val is not None:
        lo_lf, hi_lf = ranges.get("liquid_to_flour", (0.15, 3.00))
        if lo_lf <= lf_val <= hi_lf:
            structure += 60
        elif lf_val < lo_lf:
            structure += max(0.0, 60 * (lf_val / lo_lf))
        else:
            structure += 40  # above range — unusual but possible (e.g. very wet choux)
        structure_parts.append(f"liquid/flour={lf_val:.2f}")
    else:
        structure += 30  # no data — neutral partial credit

    sf = ratios.sugar_to_flour
    if sf is not None:
        _, hi_s = ranges.get("sugar_to_flour", (0.00, 0.50))
        if sf <= hi_s:
            structure += 40
        else:
            excess = (sf - hi_s) / hi_s
            structure += max(0.0, 40 * (1 - excess))
        structure_parts.append(f"sugar/flour={sf:.2f}")
    else:
        structure += 20

    structure_detail = ", ".join(structure_parts) if structure_parts else "n/a"

    # --- Technique & Layers (LLM-assessed placeholder) ---
    # Lamination quality, shatter, hollow interior (choux), layer definition —
    # none of these are accessible from ingredient ratios alone.
    technique_signals = getattr(recipe, "technique_signals", []) or []
    technique_detail = ", ".join(technique_signals) if technique_signals else "awaiting LLM assessment"

    return [
        CriterionScore(name="Fat & Richness",      score=round(min(100.0, fat_score), 1),   weight=weights.get("Fat & Richness", 0.35),      details=fat_detail),
        CriterionScore(name="Structure & Balance",  score=round(min(100.0, structure), 1),   weight=weights.get("Structure & Balance", 0.30), details=structure_detail),
        CriterionScore(name="Technique & Layers",   score=50.0,                               weight=weights.get("Technique & Layers", 0.35),  details=technique_detail),
    ]


def _score_other(
    recipe: ParsedRecipe,
    ratios: RatioResult,
    weights: dict[str, float],
    flour_type: str,
) -> list[CriterionScore]:
    balance = _score_balance(ratios, "other", flour_type)
    return [
        CriterionScore(
            name="Overall Balance",
            score=balance,
            weight=weights.get("Overall Balance", 1.0),
            details=_balance_detail(ratios),
        )
    ]


# ---------------------------------------------------------------------------
# Shared scoring helpers
# ---------------------------------------------------------------------------

def _score_balance(ratios: RatioResult, category: str, flour_type: str = "ap") -> float:
    """Sugar/flour in expected range for (category, flour_type). 0–100."""
    sf = ratios.sugar_to_flour
    if sf is None:
        return 50.0
    ranges = get_ranges(category, flour_type)
    lo, hi = ranges.get("sugar_to_flour", (0.35, 1.10))
    if lo <= sf <= hi:
        return 100.0
    elif sf < lo:
        return round(max(0.0, 100.0 * (sf / lo)), 1)
    else:
        excess = (sf - hi) / hi
        return round(max(0.0, 100.0 * (1 - excess)), 1)


def _balance_detail(ratios: RatioResult) -> str:
    parts = []
    if ratios.sugar_to_flour is not None:
        parts.append(f"sugar/flour={ratios.sugar_to_flour:.2f}")
    if ratios.brown_to_white_sugar is not None:
        parts.append(f"brown/white={ratios.brown_to_white_sugar:.2f}")
    return ", ".join(parts) if parts else "n/a"


# ---------------------------------------------------------------------------
# Batched LLM explanation + LLM-criterion scoring
# ---------------------------------------------------------------------------

def add_explanations(scored: list[ScoredRecipe]) -> None:
    """
    Populate .explanation on each ScoredRecipe with one batched LLM call.

    Also scores LLM-assessed criteria (Flavor Complexity, Technique & Layers,
    Flavor & Technique) and applies the technique_note_delta for novel techniques.
    Accessibility scores are filled here when the criterion is present.
    """
    if not scored:
        return

    recipe_summaries = []
    for i, s in enumerate(scored):
        ratio_lines = []
        r = s.ratios
        if r.liquid_to_flour is not None:
            ratio_lines.append(f"liquid/flour={r.liquid_to_flour:.2f}")
        if r.fat_to_flour is not None:
            ratio_lines.append(f"fat/flour={r.fat_to_flour:.2f}")
        if r.sugar_to_flour is not None:
            ratio_lines.append(f"sugar/flour={r.sugar_to_flour:.2f}")
        if r.leavening_to_flour is not None:
            ratio_lines.append(f"leavening/flour={r.leavening_to_flour:.4f}")
        if r.fat_type:
            ratio_lines.append(f"fat_type={r.fat_type}")
        if r.brown_to_white_sugar is not None:
            ratio_lines.append(f"brown/white_sugar={r.brown_to_white_sugar:.2f}")
        if r.has_banana:
            ratio_lines.append("has_banana=yes")
        if r.has_chocolate:
            ratio_lines.append("has_chocolate=yes")

        criteria_lines = [
            f"  {c.name}: {c.score:.0f}/100 (weight={c.weight:.2f})" for c in s.criteria
        ]
        violations_line = (
            f"\n  Violations: {'; '.join(s.constraint_violations)}"
            if s.constraint_violations else ""
        )
        technique_notes = getattr(s.recipe, "technique_notes", "") or ""
        notes_line = f"\n  Novel technique: {technique_notes}" if technique_notes else ""

        # List which criteria are LLM-assessed for this recipe
        llm_criteria = [c for c in s.criteria if c.name in _LLM_CRITERIA_NAMES]
        llm_line = ""
        if llm_criteria:
            names = ", ".join(c.name for c in llm_criteria)
            llm_line = f"\n  LLM-assessed criteria (score these 0–100): {names}"

        recipe_summaries.append(
            f"Recipe {i} — {s.recipe.title} [{s.recipe.url}]\n"
            f"  Category: {s.recipe.category}\n"
            f"  Composite: {s.composite_score:.0f}/100\n"
            + "\n".join(criteria_lines)
            + f"\n  Ratios: {', '.join(ratio_lines) if ratio_lines else 'unknown'}"
            + violations_line
            + notes_line
            + llm_line
        )

    # Build dynamic instruction sections
    has_novel_techniques = any(
        getattr(s.recipe, "technique_notes", "") for s in scored
    )
    has_llm_criteria = any(
        any(c.name in _LLM_CRITERIA_NAMES for c in s.criteria) for s in scored
    )
    has_accessibility = any(
        any(c.name == "Accessibility" for c in s.criteria) for s in scored
    )

    novel_delta_instruction = (
        '\n- "technique_note_delta": float -20.0 to +20.0. How much the "Novel technique" '
        "line (if present) should adjust the Technique Quality or Flavor & Technique score. "
        "0.0 if no novel technique or it is neutral. Omit if no novel technique."
    ) if has_novel_techniques else ""

    llm_criteria_instruction = (
        '\n- "llm_criterion_scores": object mapping criterion name → integer 0–100 score '
        "for each criterion listed under \"LLM-assessed criteria\" in the recipe summary. "
        "Score based on the recipe title, technique notes, ratios, and category context. "
        "Flavor Complexity: 0=bland/simple, 50=average, 100=exceptional depth (fermentation, "
        "brown butter, complex technique). "
        "Technique & Layers: 0=poor lamination/no layers expected, 50=standard, "
        "100=exceptional technique (well-laminated, hollow choux, precise bake). "
        "Flavor & Technique: 0=no technique differentiation, 50=standard, "
        "100=clearly differentiated technique (brown butter, overnight rest, cold ferment). "
        "A stand mixer is an enabler not a barrier. Include for every recipe that has "
        "LLM-assessed criteria."
    ) if has_llm_criteria else ""

    accessibility_instruction = (
        '\n- "accessibility_score": integer 0–100. How easy this recipe is to make at home. '
        "100=simple one-bowl, no special tools. Penalize water baths, steam injection, "
        "lamination schedules. A stand mixer is an enabler (not a barrier). Include for "
        "every recipe."
    ) if has_accessibility else ""

    system = (
        "You are a baking science analyst. For each scored recipe below, write a 2-3 sentence "
        "explanation of WHY it received its score. Translate ratio math into eating-experience "
        "predictions (e.g. 'oil-based fat retains moisture better than butter over multiple days'). "
        "Mention any constraint violations prominently.\n\n"
        'Return a JSON object with key "explanations": a list of objects, each with:\n'
        '- "index" (int)\n'
        '- "text" (string — the 2-3 sentence explanation)'
        + novel_delta_instruction
        + llm_criteria_instruction
        + accessibility_instruction
        + "\nReturn ONLY the JSON object."
    )
    user = "Recipes to explain:\n\n" + "\n\n".join(recipe_summaries)

    try:
        raw = chat(system, user, max_tokens=1200, temperature=0)
        data = extract_json(raw)
        items = data.get("explanations", []) if isinstance(data, dict) else []

        for item in items:
            idx = item.get("index")
            if idx is None or not (0 <= idx < len(scored)):
                continue
            s = scored[idx]
            s.explanation = str(item.get("text", ""))

            # Apply novel-technique delta to the first matching criterion
            delta = item.get("technique_note_delta")
            if delta is not None:
                for target_name in ("Flavor & Technique", "Flavor Complexity",
                                    "Technique & Layers", "Technique Quality"):
                    crit = next((c for c in s.criteria if c.name == target_name), None)
                    if crit is not None:
                        crit.score = round(min(100.0, max(0.0, crit.score + float(delta))), 1)
                        s.technique_note_delta = round(float(delta), 1)
                        break

            # Apply LLM-assessed criterion scores
            llm_scores: dict = item.get("llm_criterion_scores") or {}
            for crit_name, raw_score in llm_scores.items():
                crit = next((c for c in s.criteria if c.name == crit_name), None)
                if crit is not None:
                    crit.score = round(min(100.0, max(0.0, float(raw_score))), 1)

            # Apply accessibility score
            acc_val = item.get("accessibility_score")
            if acc_val is not None:
                acc_crit = next((c for c in s.criteria if c.name == "Accessibility"), None)
                if acc_crit is not None:
                    acc_crit.score = round(min(100.0, max(0.0, float(acc_val))), 1)
                    s.accessibility_score = acc_crit.score

            # Recompute composite once after all criterion updates
            if delta is not None or llm_scores or acc_val is not None:
                total_weight = sum(c.weight for c in s.criteria)
                new_composite = (
                    sum(c.score * c.weight for c in s.criteria) / total_weight
                    if total_weight > 0 else 0.0
                )
                if s.constraint_violations:
                    new_composite = max(0.0, new_composite - 20.0 * len(s.constraint_violations))
                s.composite_score = round(min(100.0, max(0.0, new_composite)), 1)

    except Exception as e:
        logger.warning("Explanation generation failed: %s", e)
        for s in scored:
            s.explanation = "Score explanation unavailable."
