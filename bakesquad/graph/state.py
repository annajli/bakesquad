"""
BakeSquadState — the single shared state object passed between all graph nodes.

Design rules:
- Immutable fields (thread_id, user_query) are set once at graph entry.
- List fields that accumulate across turns use Annotated[list, operator.add]
  so LangGraph merges them rather than replacing them.
- Optional fields start as None and are populated by the node responsible.
- user_prefs and liked_recipe_urls are loaded from SQLite at graph entry
  and written back by the memory node after each turn.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore[assignment]

from bakesquad.models import (
    FetchedPage,
    ParsedRecipe,
    QueryPlan,
    RatioResult,
    ScoredRecipe,
    SearchSnippet,
)


class BakeSquadState(TypedDict):
    # ------------------------------------------------------------------ session
    thread_id: str
    # Accumulates the full conversation; each entry is {"role": ..., "content": ...}
    messages: Annotated[list[dict], operator.add]

    # ------------------------------------------------------------------ turn input
    turn_type: Optional[str]          # classify_intent output: new_search | re_filter | re_search | factual
    user_query: str                   # raw text of the current user turn

    # ------------------------------------------------------------------ Step 1 outputs
    query_plan: Optional[QueryPlan]
    category_confidence: float        # 0.0–1.0; below threshold triggers clarify node
    clarify_question: Optional[str]   # set by clarify node before interrupt()
    recency: Optional[str]            # "year" | "month" | None

    # ------------------------------------------------------------------ search outputs (Steps 2–5)
    snippets: list[SearchSnippet]

    # ------------------------------------------------------------------ fetch output (Step 6)
    fetched_pages: list[FetchedPage]
    search_retry_count: int           # incremented on each adaptive retry

    # ------------------------------------------------------------------ parse output (Step 7)
    parsed_recipes: list[ParsedRecipe]
    category_override: Optional[str]  # set by verify node if Step 7 disagrees with Step 1

    # ------------------------------------------------------------------ score output (Steps 8–10)
    ratio_results: list[RatioResult]
    scored_recipes: list[ScoredRecipe]

    # ------------------------------------------------------------------ active filters
    exclude_ingredients: list[str]
    require_fat_type: Optional[str]

    # ------------------------------------------------------------------ factual answer
    factual_answer: Optional[str]

    # ------------------------------------------------------------------ user context
    user_prefs: dict                  # loaded from ~/.bakesquad/user_prefs.json
    liked_recipe_urls: list[str]      # loaded from SQLite liked_recipes table

    # ------------------------------------------------------------------ error accumulator
    # Annotated list so all nodes can append without overwriting each other
    errors: Annotated[list[str], operator.add]
