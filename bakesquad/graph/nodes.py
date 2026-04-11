"""
LangGraph node implementations for BakeSquad (Phase 1 scaffold).

Each function takes BakeSquadState and returns a partial-state dict that
LangGraph merges into the running state.  Nodes call the same underlying
modules that main.py uses — no business logic is duplicated here.

Phase 1 status: all nodes are implemented as thin wrappers around existing
pipeline code.  The clarify node uses langgraph.types.interrupt() so the
graph pauses for user input; all other nodes run to completion.

Nodes:
  classify_intent  — route a follow-up turn (new_search | re_filter | re_search | factual)
  expand_query     — Step 1: QueryPlan via LLM
  clarify          — ask user for more info when category_confidence is low
  search           — Steps 2–5: DuckDuckGo + snippet scoring + adaptive retry
  fetch            — Step 6: parallel page fetch
  parse            — Step 7: parallel LLM recipe parsing
  verify           — Step 7b: category reconciliation (Step 7 majority vote → override Step 1)
  score            — Steps 8–10: ratios + scoring + explanations
  filter_node      — re_filter: apply ingredient/fat-type constraints to existing results
  factual          — answer a factual question using current session context
  memory           — persist liked recipes and update user prefs
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bakesquad.graph.state import BakeSquadState

logger = logging.getLogger(__name__)

# Confidence threshold below which the clarify node is triggered
_CLARIFY_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# classify_intent
# ---------------------------------------------------------------------------

def classify_intent(state: BakeSquadState) -> dict:
    """
    Classify a follow-up user turn as: new_search | re_filter | re_search | factual.

    For the very first turn (no prior query_plan), always returns new_search.
    """
    from bakesquad.session import ConversationSession, classify_turn

    if state.get("query_plan") is None:
        # First turn — no prior context
        return {"turn_type": "new_search"}

    # Build a minimal ConversationSession so classify_turn can inspect history
    session = _state_to_session(state)
    refine = classify_turn(session, state["user_query"])
    turn_type = refine.get("turn_type", "factual")

    update: dict = {"turn_type": turn_type}
    if turn_type == "re_filter":
        update["exclude_ingredients"] = refine.get("exclude_ingredients") or []
        update["require_fat_type"] = refine.get("require_fat_type")
    if turn_type == "factual":
        update["factual_answer"] = refine.get("direct_answer")

    return update


# ---------------------------------------------------------------------------
# expand_query  (Step 1)
# ---------------------------------------------------------------------------

def expand_query(state: BakeSquadState) -> dict:
    """
    Step 1: build QueryPlan from the user query via LLM.

    Also estimates category_confidence from the model's raw response when
    available (future: soft classification).  Currently defaults to 1.0
    because the hard Literal schema forces a committed answer.
    """
    from bakesquad.config import MAX_PAGES_PER_RUN
    from bakesquad.search.ingestion import IngestionPipeline

    pipeline = IngestionPipeline(trusted_sources=[])
    recency = state.get("recency")

    query = state["user_query"]
    hint_plan = state.get("query_plan")

    if hint_plan is not None and state.get("turn_type") == "re_search":
        query = (
            f"[Context: continuing from previous search. "
            f"Category={hint_plan.category}. "
            f"Active constraints: {'; '.join(hint_plan.hard_constraints)}. "
            f"New request: {query}]"
        )

    plan = pipeline._build_query_plan(query, recency)
    return {
        "query_plan": plan,
        "category_confidence": 1.0,   # hard schema — model always commits
        "messages": [{"role": "assistant", "content": f"[Step 1] Category: {plan.category}"}],
    }


# ---------------------------------------------------------------------------
# clarify  (conditional — only reached when confidence < threshold)
# ---------------------------------------------------------------------------

def clarify(state: BakeSquadState) -> dict:
    """
    Pause the graph and ask the user a clarifying question.

    Uses langgraph.types.interrupt() — the graph suspends here and resumes
    when the caller injects the user's answer via graph.invoke() with the
    same thread_id.
    """
    try:
        from langgraph.types import interrupt
    except ImportError:
        # LangGraph not yet installed — fall through without pausing
        logger.warning("langgraph not installed; clarify node will not pause")
        return {}

    question = (
        state.get("clarify_question")
        or "Could you tell me more about what you're looking for? "
           "(e.g. is this a bread, cake, or cookie?)"
    )
    interrupt(question)
    # After resumption, user_query will contain the clarification answer
    return {"clarify_question": None}


# ---------------------------------------------------------------------------
# search  (Steps 2–5)
# ---------------------------------------------------------------------------

def search(state: BakeSquadState) -> dict:
    """Steps 2–5: DuckDuckGo search, snippet scoring, domain cap, adaptive retry."""
    from bakesquad.search.ingestion import IngestionPipeline

    plan = state["query_plan"]
    pipeline = IngestionPipeline(trusted_sources=[])
    recency = state.get("recency")

    candidates = pipeline._search_and_filter(
        state["user_query"], plan.queries, recency
    )

    return {
        "snippets": candidates,
        "messages": [
            {"role": "assistant", "content": f"[Step 2-5] {len(candidates)} candidates found"}
        ],
    }


# ---------------------------------------------------------------------------
# fetch  (Step 6)
# ---------------------------------------------------------------------------

def fetch(state: BakeSquadState) -> dict:
    """Step 6: parallel HTTP fetch for each candidate snippet."""
    from bakesquad.search.ingestion import IngestionPipeline

    pipeline = IngestionPipeline(trusted_sources=[])
    pages = pipeline._fetch_pages(state["snippets"])

    return {
        "fetched_pages": pages,
        "messages": [
            {"role": "assistant", "content": f"[Step 6] {len(pages)} pages fetched"}
        ],
    }


# ---------------------------------------------------------------------------
# parse  (Step 7)
# ---------------------------------------------------------------------------

def parse(state: BakeSquadState) -> dict:
    """Step 7: parallel LLM extraction of structured recipes from fetched pages."""
    from bakesquad.parser import parse_recipes_parallel

    recipes = parse_recipes_parallel(state["fetched_pages"])
    return {
        "parsed_recipes": recipes,
        "messages": [
            {"role": "assistant", "content": f"[Step 7] {len(recipes)} recipes parsed"}
        ],
    }


# ---------------------------------------------------------------------------
# verify  (Step 7b — category reconciliation)
# ---------------------------------------------------------------------------

def verify(state: BakeSquadState) -> dict:
    """
    Step 7b: majority-vote over Step 7 parsed categories.

    If the dominant Step 7 category differs from Step 1, set category_override
    so the score node uses the corrected category for ratio range lookups.
    """
    recipes = state.get("parsed_recipes") or []
    step7_cats = Counter(
        r.category for r in recipes
        if getattr(r, "category", None) and r.category != "other"
    )

    override = None
    if step7_cats:
        dominant = step7_cats.most_common(1)[0][0]
        plan_category = (state.get("query_plan") or {}).get("category") if isinstance(
            (state.get("query_plan") or {}), dict
        ) else getattr(state.get("query_plan"), "category", None)

        if dominant != plan_category:
            logger.warning(
                "verify node: category override %s → %s", plan_category, dominant
            )
            override = dominant

    return {"category_override": override}


# ---------------------------------------------------------------------------
# score  (Steps 8–10)
# ---------------------------------------------------------------------------

def score(state: BakeSquadState) -> dict:
    """Steps 8–10: normalization, ratio engine, scoring, and LLM explanations."""
    from bakesquad.ratio_engine import compute_ratios
    from bakesquad.scorer import add_explanations, score_all

    recipes = state.get("parsed_recipes") or []
    if not recipes:
        return {"scored_recipes": [], "ratio_results": []}

    # Use category_override if the verify node set one
    plan = state["query_plan"]
    if state.get("category_override"):
        plan = plan.model_copy(update={"category": state["category_override"]})

    ratios_list = [compute_ratios(r) for r in recipes]
    user_prefs = state.get("user_prefs") or {}
    scored = score_all(recipes, ratios_list, plan, user_prefs)
    add_explanations(scored)

    top = scored[0].recipe.title if scored else "none"
    return {
        "ratio_results": ratios_list,
        "scored_recipes": scored,
        "messages": [
            {"role": "assistant",
             "content": f"[Step 10] {len(scored)} recipes scored. Top: {top}"}
        ],
    }


# ---------------------------------------------------------------------------
# filter_node  (re_filter — zero LLM calls)
# ---------------------------------------------------------------------------

def filter_node(state: BakeSquadState) -> dict:
    """Apply ingredient/fat-type constraints to the current scored results."""
    from bakesquad.session import apply_re_filter

    session = _state_to_session(state)
    exclude = state.get("exclude_ingredients") or []
    require_fat = state.get("require_fat_type")

    _, _, scored_f = apply_re_filter(session, exclude, require_fat)
    return {
        "scored_recipes": scored_f,
        "messages": [
            {"role": "assistant",
             "content": f"[re_filter] {len(scored_f)} recipes pass filter"}
        ],
    }


# ---------------------------------------------------------------------------
# factual
# ---------------------------------------------------------------------------

def factual(state: BakeSquadState) -> dict:
    """Return a pre-computed factual answer (already set by classify_intent)."""
    answer = state.get("factual_answer") or "I'm not sure about that."
    return {
        "messages": [{"role": "assistant", "content": answer}],
    }


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------

def memory(state: BakeSquadState) -> dict:
    """
    Persist the top-scored recipe to SQLite and update in-memory liked_recipe_urls.

    Currently a stub — full implementation in Phase 2 adds semantic embeddings
    and the expanded liked_recipes schema (user_rating, user_notes, tried_date).
    """
    scored = state.get("scored_recipes") or []
    liked = list(state.get("liked_recipe_urls") or [])

    # Placeholder: auto-like the top result if composite score ≥ 85
    if scored and scored[0].composite_score >= 85:
        url = scored[0].recipe.url
        if url not in liked:
            liked.append(url)
            logger.info("memory node: auto-liked %s", url)

    return {"liked_recipe_urls": liked}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _state_to_session(state: BakeSquadState):
    """Build a minimal ConversationSession from graph state for legacy callers."""
    from bakesquad.session import ConversationSession

    session = ConversationSession(original_query=state.get("user_query", ""))
    plan = state.get("query_plan")
    recipes = state.get("parsed_recipes") or []
    ratios = state.get("ratio_results") or []
    scored = state.get("scored_recipes") or []

    if plan is not None:
        session.update_results(plan, recipes, ratios, scored)

    for msg in state.get("messages") or []:
        if msg.get("role") == "user":
            session.add_user(msg["content"])
        elif msg.get("role") == "assistant":
            session.add_assistant(msg["content"])

    return session
