"""
Conversational session — holds state between turns and routes follow-up messages.

Three turn types:
  re_filter  — narrow existing results by ingredient constraint (zero LLM calls)
  re_search  — user wants something different; re-run the full pipeline
  factual    — question about results or baking science; answer directly

The classify_turn() LLM call handles all three: it classifies the turn AND generates
the direct_answer for factual turns in the same response, so factual replies cost
exactly one LLM call total (the classify call).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

from bakesquad.llm_client import chat, extract_json
from bakesquad.models import ParsedRecipe, QueryPlan, RatioResult, ScoredRecipe

logger = logging.getLogger(__name__)

TurnType = Literal["re_filter", "re_search", "factual"]


@dataclass
class ConversationSession:
    original_query: str
    messages: list[dict] = field(default_factory=list)  # {"role": "user"/"assistant", "content": str}
    last_plan: Optional[QueryPlan] = None
    last_recipes: list[ParsedRecipe] = field(default_factory=list)
    last_ratios: list[RatioResult] = field(default_factory=list)
    last_scored: list[ScoredRecipe] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def context_summary(self) -> str:
        """Compact summary of the current result set for inclusion in the refine prompt."""
        if not self.last_scored:
            return "No results yet."

        category = self.last_plan.category if self.last_plan else "unknown"
        lines = [f"Current results ({len(self.last_scored)} recipes, category: {category}):"]

        for s in self.last_scored:
            ingr_names = [i.name for i in s.recipe.ingredients[:10]]
            ratios = s.ratios
            fat = ratios.fat_type or "unknown"
            lines.append(
                f"  #{s.rank} [{s.composite_score:.0f}/100] {s.recipe.title}\n"
                f"     fat_type={fat}  has_banana={ratios.has_banana}  has_chocolate={ratios.has_chocolate}\n"
                f"     Ingredients: {', '.join(ingr_names)}"
            )

        if self.last_plan:
            constraints = ", ".join(self.last_plan.hard_constraints) or "none"
            preferences = ", ".join(self.last_plan.soft_preferences) or "none"
            lines.append(f"Active constraints: {constraints}")
            lines.append(f"Active preferences: {preferences}")

        return "\n".join(lines)

    def recent_history(self, n: int = 6) -> str:
        """Last n messages as a formatted string for the refine prompt."""
        recent = self.messages[-n:]
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

    def update_results(
        self,
        plan: QueryPlan,
        recipes: list[ParsedRecipe],
        ratios: list[RatioResult],
        scored: list[ScoredRecipe],
    ) -> None:
        self.last_plan = plan
        self.last_recipes = recipes
        self.last_ratios = ratios
        self.last_scored = scored


# ---------------------------------------------------------------------------
# Turn classification (1 LLM call)
# ---------------------------------------------------------------------------

def classify_turn(session: ConversationSession, user_message: str) -> dict:
    """
    Single LLM call that classifies the follow-up and extracts what needs to change.
    For factual turns, it also generates the direct_answer in the same call — so
    factual replies cost exactly 1 LLM call total.
    """
    system = (
        "You are a baking recipe assistant managing a multi-turn conversation. "
        "The user has already received a ranked list of scored recipes. "
        "They are now sending a follow-up message.\n\n"
        "Classify the follow-up as exactly one of:\n"
        '- "re_filter": user wants to narrow the CURRENT results by adding a constraint '
        '(e.g. "no brown sugar", "only oil-based", "without nuts"). '
        "The existing recipes are re-filtered without a new search.\n"
        '- "re_search": user wants to fundamentally change what they are looking for '
        '(e.g. "find something simpler", "what about a chocolate cake?", '
        '"can you find recipes without chocolate instead?"). A new search runs.\n'
        '- "factual": user is asking a question about the results or baking science '
        '(e.g. "why does oil retain moisture better?", "what does the leavening ratio mean?").\n\n'
        "Return ONLY a JSON object with these fields:\n"
        '- "turn_type": "re_filter" | "re_search" | "factual"\n'
        '- "exclude_ingredients": list of ingredient name strings to exclude (re_filter only, '
        'e.g. ["brown sugar", "walnuts"]). Empty list if not applicable.\n'
        '- "require_fat_type": "oil" | "butter" | null (re_filter only)\n'
        '- "updated_query": string for the new search (re_search only — combine original intent '
        "with the user's new requirement into one natural-language query)\n"
        '- "merged_constraints": list of strings (re_search only — all hard constraints that '
        "should apply to the new search, including ones carried over from before)\n"
        '- "direct_answer": string (factual only — answer the question concisely using the '
        "recipe context and baking science knowledge provided below)\n"
        "Return ONLY the JSON object. No explanation outside of it."
    )

    user = (
        f"Recipe context:\n{session.context_summary()}\n\n"
        f"Conversation so far:\n{session.recent_history(6)}\n\n"
        f"New follow-up message: {user_message}"
    )

    try:
        raw = chat(system, user, max_tokens=600, temperature=0)
        data = extract_json(raw)
        if isinstance(data, dict) and "turn_type" in data:
            return data
    except Exception as e:
        logger.warning("classify_turn failed: %s", e)

    # Fallback: treat as factual with a generic reply
    return {
        "turn_type": "factual",
        "direct_answer": "I had trouble understanding that follow-up. Could you rephrase it?",
    }


# ---------------------------------------------------------------------------
# Re-filter: deterministic ingredient constraint application (zero LLM calls)
# ---------------------------------------------------------------------------

def apply_re_filter(
    session: ConversationSession,
    exclude_ingredients: list[str],
    require_fat_type: Optional[str],
) -> tuple[list[ParsedRecipe], list[RatioResult], list[ScoredRecipe]]:
    """
    Filter the existing result set by ingredient constraints.
    Zero LLM calls — substring matching on normalized ingredient names.

    Returns (recipes, ratios, scored) for recipes that pass all constraints.
    If nothing passes, returns three empty lists (caller should escalate to re_search).
    """
    passing = []

    for recipe, ratio, scored in zip(
        session.last_recipes, session.last_ratios, session.last_scored
    ):
        ok = True

        # Exclude ingredients: "brown sugar" → filter any recipe containing it
        for term in exclude_ingredients:
            term_lower = term.lower().strip()
            if any(term_lower in ing.name.lower() for ing in recipe.ingredients):
                ok = False
                break

        # Fat type requirement: "oil" → only keep oil-based recipes
        if ok and require_fat_type:
            if ratio.fat_type != require_fat_type:
                ok = False

        if ok:
            passing.append((recipe, ratio, scored))

    if not passing:
        return [], [], []

    recipes_out, ratios_out, scored_out = zip(*passing)

    # Re-rank by existing composite score (scores don't change, just re-number ranks)
    combined = sorted(
        zip(recipes_out, ratios_out, scored_out),
        key=lambda x: x[2].composite_score,
        reverse=True,
    )
    recipes_out, ratios_out, scored_out = zip(*combined)
    scored_out = list(scored_out)
    for i, s in enumerate(scored_out):
        s.rank = i + 1

    return list(recipes_out), list(ratios_out), scored_out


# ---------------------------------------------------------------------------
# Helpers for building re-search queries
# ---------------------------------------------------------------------------

def build_re_search_query(session: ConversationSession, refine: dict) -> str:
    """
    Return the query string to use for the re-search.
    Prefers the LLM-generated updated_query; falls back to original_query.
    """
    return refine.get("updated_query") or session.original_query


def build_merged_plan(session: ConversationSession, refine: dict) -> Optional[QueryPlan]:
    """
    Return a QueryPlan with constraints merged from the original plan + refine output.
    Passed to the pipeline so scoring weights carry over correctly.
    Used to pre-seed the new run's query understanding step.
    """
    if not session.last_plan:
        return None

    merged_constraints = refine.get("merged_constraints") or session.last_plan.hard_constraints
    return QueryPlan(
        category=session.last_plan.category,
        hard_constraints=merged_constraints,
        soft_preferences=session.last_plan.soft_preferences,
        queries=session.last_plan.queries,
    )
