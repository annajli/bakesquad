"""
Scorer (step 10).

Two phases:
  1. Deterministic math — compute per-criterion scores (0–100) from ratio results.
     ZERO LLM calls in this phase.
  2. Batched LLM call — generate all "Why this score" explanations in one prompt.

Scoring is split into:
  - Fixed criteria: always measured; failures penalize regardless of user preferences.
    (leavening_validity, constraint_satisfaction)
  - Variable criteria: weighted by query context + user preference model.
    (moisture_retention, fat_balance, sugar_balance, chew [cookies])

Dynamic weight derivation: the query's soft_preferences and hard_constraints are parsed
to boost relevant weights. "stays moist for days" → moisture_retention weight = 0.85.
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
# Weight derivation from query
# ---------------------------------------------------------------------------

def derive_weights(plan: QueryPlan, user_prefs: dict) -> dict[str, float]:
    """
    Build criterion weight vector from query intent + user preference model.
    Reads soft_preferences and hard_constraints to boost relevant weights.
    """
    # Start from user base weights
    weights = {
        "moisture": user_prefs.get("moisture_base_weight", 0.45),
        "structure": user_prefs.get("structure_base_weight", 0.30),
        "balance": user_prefs.get("balance_base_weight", 0.25),
    }

    # Boost moisture for moisture-related query signals
    moisture_signals = {"moist", "moisture", "soft", "fresh", "days", "week", "tender", "fudgy", "dense"}
    query_text = " ".join(plan.hard_constraints + plan.soft_preferences).lower()
    if any(sig in query_text for sig in moisture_signals):
        weights["moisture"] = min(0.90, weights["moisture"] + 0.40)
        weights["structure"] = max(0.15, weights["structure"] - 0.10)
        weights["balance"] = max(0.10, weights["balance"] - 0.10)

    # Boost structure for texture-related signals
    if any(sig in query_text for sig in {"crispy", "chewy", "crunchy", "fluffy", "airy", "rise"}):
        weights["structure"] = min(0.70, weights["structure"] + 0.20)

    return weights


# ---------------------------------------------------------------------------
# Deterministic scoring math (ZERO LLM calls)
# ---------------------------------------------------------------------------

def score_recipe(
    recipe: ParsedRecipe,
    ratios: RatioResult,
    plan: QueryPlan,
    user_prefs: dict,
) -> ScoredRecipe:
    """Compute all criterion scores and composite score. No LLM."""
    weights = derive_weights(plan, user_prefs)
    category = ratios.category
    flour_type = ratios.flour_type or "ap"
    modifiers = ratios.modifiers or []
    criteria: list[CriterionScore] = []
    violations: list[str] = []

    # --- Moisture retention (variable) ---
    moisture = _score_moisture(ratios, category, flour_type)
    criteria.append(CriterionScore(
        name="Moisture Retention",
        score=moisture,
        weight=weights["moisture"],
        details=_moisture_detail(ratios, category),
    ))

    # --- Structure / leavening (variable, but leavening floor is fixed) ---
    structure = _score_structure(ratios, category, flour_type)
    criteria.append(CriterionScore(
        name="Structure & Leavening",
        score=structure,
        weight=weights["structure"],
        details=_structure_detail(ratios, category),
    ))

    # --- Sugar balance (variable) ---
    balance = _score_balance(ratios, category, flour_type)
    criteria.append(CriterionScore(
        name="Sugar Balance",
        score=balance,
        weight=weights["balance"],
        details=_balance_detail(ratios, category),
    ))

    # --- Binding agent criterion (GF only, fixed weight) ---
    # Gluten-free recipes require a binding agent (xanthan gum, psyllium, flax)
    # to replace the structural role of gluten. Missing one predicts a crumbly result.
    if "gluten_free" in modifiers or flour_type in ("almond", "oat", "coconut", "rice", "gf_blend"):
        binding_score = 100.0 if ratios.has_binding_agent else 20.0
        criteria.append(CriterionScore(
            name="GF Binding Agent",
            score=binding_score,
            weight=0.20,
            details="xanthan/psyllium/flax detected" if ratios.has_binding_agent else "no binding agent detected",
        ))
        if not ratios.has_binding_agent:
            # Note: some almond/oat flour recipes rely on eggs for binding — don't hard-penalize
            # if binding_score < 50 but also not a hard constraint violation.
            pass

    # --- Hard constraint: chocolate (fixed) ---
    if "chocolate" in " ".join(plan.hard_constraints).lower() or "chocolate" in " ".join(plan.soft_preferences).lower():
        if not ratios.has_chocolate:
            violations.append("No chocolate detected (query requires chocolate)")

    # Composite score: weighted sum normalized by total weight
    total_weight = sum(c.weight for c in criteria)
    if total_weight > 0:
        composite = sum(c.score * c.weight for c in criteria) / total_weight
    else:
        composite = 0.0

    # Penalize constraint violations
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
    """Score all recipes, then rank them."""
    scored = []
    for recipe, ratios in zip(recipes, ratios_list):
        s = score_recipe(recipe, ratios, plan, user_prefs)
        scored.append(s)
    scored.sort(key=lambda s: s.composite_score, reverse=True)
    for i, s in enumerate(scored):
        s.rank = i + 1
    return scored


# ---------------------------------------------------------------------------
# Batched LLM explanation call (one call for all recipes)
# Spec: 'generate all "Why this score" explanations in a single batched LLM call'
# ---------------------------------------------------------------------------

def add_explanations(scored: list[ScoredRecipe]) -> None:
    """Populate .explanation on each ScoredRecipe with one batched LLM call."""
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

        violations_line = ""
        if s.constraint_violations:
            violations_line = f"\n  Violations: {'; '.join(s.constraint_violations)}"

        recipe_summaries.append(
            f"Recipe {i} — {s.recipe.title} [{s.recipe.url}]\n"
            f"  Composite: {s.composite_score:.0f}/100\n"
            + "\n".join(criteria_lines)
            + f"\n  Ratios: {', '.join(ratio_lines) if ratio_lines else 'unknown'}"
            + violations_line
        )

    system = (
        "You are a baking science analyst. For each scored recipe below, write a 2-3 sentence "
        "explanation of WHY it received its score. Translate ratio math into eating-experience "
        "predictions (e.g. 'oil-based fat retains moisture better than butter over multiple days'). "
        "Mention any constraint violations prominently.\n\n"
        'Return a JSON object with key "explanations": a list of objects, each with '
        '"index" (int) and "text" (string). Return ONLY the JSON object.'
    )
    user = "Recipes to explain:\n\n" + "\n\n".join(recipe_summaries)

    try:
        raw = chat(system, user, max_tokens=600, temperature=0)
        data = extract_json(raw)
        items = data.get("explanations", []) if isinstance(data, dict) else []
        for item in items:
            idx = item.get("index")
            text = item.get("text", "")
            if idx is not None and 0 <= idx < len(scored):
                scored[idx].explanation = str(text)
    except Exception as e:
        logger.warning("Explanation generation failed: %s", e)
        for s in scored:
            s.explanation = "Score explanation unavailable."


# ---------------------------------------------------------------------------
# Per-criterion scoring functions (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _score_moisture(ratios: RatioResult, category: str, flour_type: str = "ap") -> float:
    """Moisture retention score 0–100. Fat type + liquid/flour + sugar/flour."""
    score = 0.0
    ranges = get_ranges(category, flour_type)

    # Fat type (0–35 pts)
    fat_type = ratios.fat_type or "none"
    if fat_type == "oil":
        score += 35       # Oil stays liquid at room temperature → better multi-day moisture
    elif fat_type == "mixed":
        score += 22
    elif fat_type == "butter":
        score += 12       # Butter solidifies → drier texture on day 2+
    # "none" = 0

    if category in ("quick_bread", "cake"):
        # Liquid-to-flour ratio (0–35 pts) — optimal is the midpoint of the valid range
        lf = ratios.liquid_to_flour
        if lf is not None:
            lo_lf, hi_lf = ranges.get("liquid_to_flour", (0.85, 1.50))
            mid = (lo_lf + hi_lf) / 2
            half_width = (hi_lf - lo_lf) / 2
            if lo_lf <= lf <= hi_lf:
                # Full credit at midpoint, tapering to ~50% at edges
                dist = abs(lf - mid) / half_width
                score += 35 * (1.0 - 0.5 * dist)
            elif lf < lo_lf:
                score += max(0.0, 35 * (lf / lo_lf) * 0.6)
            else:
                score += max(0.0, 35 - (lf - hi_lf) * 20)

        # Sugar hygroscopicity (0–15 pts) — sugar attracts and retains moisture
        sf = ratios.sugar_to_flour
        if sf is not None:
            lo_sf, hi_sf = ranges.get("sugar_to_flour", (0.35, 0.90))
            if lo_sf <= sf <= hi_sf:
                score += 15
            elif sf < lo_sf:
                score += 8 * (sf / lo_sf)
            else:
                score += 11   # More sugar, still ok for moisture

        # Banana bonus (0–15 pts) — banana puree is hygroscopic + moisture-dense
        if ratios.has_banana:
            score += 15

    elif category == "cookie":
        # Cookies: brown sugar (hygroscopic) + butter/flour + fat type already counted
        bw = ratios.brown_to_white_sugar
        if bw is not None:
            if bw >= 1.5:
                score += 30   # High brown sugar = chewy + moist
            elif bw >= 1.0:
                score += 20
            elif bw >= 0.5:
                score += 10
        else:
            # No white sugar (all brown?) = very chewy/moist
            sf = ratios.sugar_to_flour
            if sf and sf > 0.5:
                score += 25

        # Fat/flour balance
        bf = ratios.butter_to_flour or ratios.fat_to_flour
        lo_fat, hi_fat = ranges.get("fat_to_flour", ranges.get("butter_to_flour", (0.40, 0.75)))
        if bf is not None and lo_fat <= bf <= hi_fat:
            score += 20

    return round(min(100.0, score), 1)


def _score_structure(ratios: RatioResult, category: str, flour_type: str = "ap") -> float:
    """Structure/leavening score 0–100. Leavening in range + fat balance."""
    score = 0.0
    ranges = get_ranges(category, flour_type)

    # Leavening ratio (0–60 pts)
    lf = ratios.leavening_to_flour
    lo_lv, hi_lv = ranges.get("leavening_to_flour", (0.008, 0.035))
    if lf is not None:
        if lo_lv <= lf <= hi_lv:
            score += 60
        elif lf < lo_lv:
            # Under-leavened → dense brick; partial credit
            score += max(0, 60 * (lf / lo_lv))
        else:
            # Over-leavened → soapy taste, dry crumb, collapse
            excess_ratio = (lf - hi_lv) / hi_lv
            score += max(0, 60 * (1 - excess_ratio * 2))

    # Fat/flour in range (0–40 pts) — affects crumb tenderness and structure
    fat_ratio = ratios.fat_to_flour or ratios.butter_to_flour
    lo_fat, hi_fat = ranges.get("fat_to_flour", ranges.get("butter_to_flour", (0.28, 0.75)))
    if fat_ratio is not None:
        if lo_fat <= fat_ratio <= hi_fat:
            score += 40
        else:
            dist = min(abs(fat_ratio - lo_fat), abs(fat_ratio - hi_fat))
            score += max(0.0, 40 * (1 - dist / 0.3))

    return round(min(100.0, score), 1)


def _score_balance(ratios: RatioResult, category: str, flour_type: str = "ap") -> float:
    """Sugar balance score 0–100. Sugar/flour in expected range for (category, flour_type)."""
    sf = ratios.sugar_to_flour
    if sf is None:
        return 50.0   # No sugar data — neutral score

    ranges = get_ranges(category, flour_type)
    lo, hi = ranges.get("sugar_to_flour", (0.35, 1.10))

    if lo <= sf <= hi:
        score = 100.0
    elif sf < lo:
        score = max(0.0, 100.0 * (sf / lo))
    else:
        excess = (sf - hi) / hi
        score = max(0.0, 100.0 * (1 - excess))

    return round(score, 1)


def _moisture_detail(ratios: RatioResult, category: str) -> str:
    parts = []
    if ratios.fat_type:
        parts.append(f"fat={ratios.fat_type}")
    if ratios.liquid_to_flour is not None:
        parts.append(f"liquid/flour={ratios.liquid_to_flour:.2f}")
    if ratios.has_banana:
        parts.append("banana present")
    return ", ".join(parts) if parts else "n/a"


def _structure_detail(ratios: RatioResult, category: str) -> str:
    parts = []
    if ratios.leavening_to_flour is not None:
        parts.append(f"leavening/flour={ratios.leavening_to_flour:.4f}")
    fat = ratios.fat_to_flour or ratios.butter_to_flour
    if fat is not None:
        parts.append(f"fat/flour={fat:.2f}")
    return ", ".join(parts) if parts else "n/a"


def _balance_detail(ratios: RatioResult, category: str) -> str:
    parts = []
    if ratios.sugar_to_flour is not None:
        parts.append(f"sugar/flour={ratios.sugar_to_flour:.2f}")
    if ratios.brown_to_white_sugar is not None:
        parts.append(f"brown/white={ratios.brown_to_white_sugar:.2f}")
    return ", ".join(parts) if parts else "n/a"
