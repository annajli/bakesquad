"""POST /api/search  — full 11-step pipeline via SSE.
   POST /api/chat    — chat follow-up (re_filter / re_search / factual) via SSE.

Dietary restrictions stored in user_prefs["dietary_restrictions"] are appended
to the query string so the LLM extracts them as hard_constraints in step 1.

Baking-themed buffering messages are emitted before each blocking step so the
frontend can cycle through them while the user waits.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import AsyncGenerator, Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.sessions import get_session, new_session_id, put_session
from api.sse_utils import sse
from bakesquad.config import MAX_PAGES_PER_RUN
from bakesquad.memory import (
    embed_text,
    get_semantic_candidates,
    load_prefs,
    update_user_prefs_from_feedback,
)
from bakesquad.models import QueryPlan
from bakesquad.parser import parse_recipes_parallel
from bakesquad.ratio_engine import compute_ratios
from bakesquad.scorer import add_explanations, score_all
from bakesquad.search.ingestion import IngestionPipeline
from bakesquad.session import (
    ConversationSession,
    apply_re_filter,
    build_merged_plan,
    build_re_search_query,
    classify_turn,
)

router = APIRouter()

# One pun per pipeline step — frontend cycles through these while waiting.
_STEP_MESSAGES = {
    "query_plan": [
        "Preheating the oven...",
        "Reading the recipe card...",
    ],
    "search": [
        "Scouring the recipe shelf...",
        "Checking the pantry for options...",
        "Flipping through the index...",
    ],
    "fetch": [
        "Pulling recipes from the web...",
        "Collecting the good stuff...",
    ],
    "parse": [
        "Reading the ingredient labels...",
        "Decoding the measurements...",
        "Separating the wet from the dry...",
    ],
    "ratio": [
        "Weighing the flour...",
        "Doing the baking math...",
        "Checking the fat-to-flour ratio...",
    ],
    "score": [
        "Taste-testing...",
        "Judging the crumb structure...",
        "Consulting the baking science...",
    ],
}


# ---------------------------------------------------------------------------
# Search request / response shapes
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    recency: Optional[Literal["year", "month"]] = None
    session_id: Optional[str] = None   # provided on re-search continuation


class ChatRequest(BaseModel):
    session_id: str
    message: str
    recency: Optional[Literal["year", "month"]] = None


# ---------------------------------------------------------------------------
# Search pipeline SSE generator
# ---------------------------------------------------------------------------

async def _search_stream(
    query: str,
    recency: Optional[str],
    session_id: str,
    existing_session: Optional[ConversationSession],
) -> AsyncGenerator[str, None]:
    user_prefs = await asyncio.to_thread(load_prefs)

    # Inject dietary restrictions as text so the LLM extracts them as constraints.
    dietary = user_prefs.get("dietary_restrictions") or []
    effective_query = query
    if dietary:
        restrictions = ", ".join(d.replace("_", "-") for d in dietary)
        effective_query = f"{query}. Must be: {restrictions}"

    pipeline = IngestionPipeline(trusted_sources=[])

    # Step 1: Query understanding
    yield sse({"step": "query_plan", "messages": _STEP_MESSAGES["query_plan"]})
    hint_plan = existing_session.last_plan if existing_session else None

    def _build_plan() -> QueryPlan:
        if hint_plan:
            prefix = (
                f"[Continuing search. Category={hint_plan.category}. "
                f"Active constraints: {'; '.join(hint_plan.hard_constraints)}. "
                f"New request: {effective_query}]"
            )
            return pipeline._build_query_plan(prefix, recency)
        return pipeline._build_query_plan(effective_query, recency)

    plan = await asyncio.to_thread(_build_plan)
    yield sse({
        "step": "query_plan_done",
        "data": {
            "category": plan.category,
            "constraints": plan.hard_constraints,
            "preferences": plan.soft_preferences,
        },
    })

    # Steps 2–5: Search + candidate selection
    yield sse({"step": "search", "messages": _STEP_MESSAGES["search"]})
    candidates = await asyncio.to_thread(
        pipeline._search_and_filter, query, plan.queries, recency
    )
    yield sse({"step": "search_done", "data": {"count": len(candidates)}})

    if not candidates:
        yield sse({"step": "error", "message": "No recipes found. Try rephrasing your query."})
        return

    # Step 6: Page fetch
    yield sse({"step": "fetch", "messages": _STEP_MESSAGES["fetch"]})
    pages = await asyncio.to_thread(pipeline._fetch_pages, candidates)
    yield sse({"step": "fetch_done", "data": {"fetched": len(pages), "attempted": min(len(candidates), MAX_PAGES_PER_RUN)}})

    if not pages:
        yield sse({"step": "error", "message": "No recipe pages could be fetched."})
        return

    # Step 7: LLM parse
    yield sse({"step": "parse", "messages": _STEP_MESSAGES["parse"]})
    recipes = await asyncio.to_thread(parse_recipes_parallel, pages)
    yield sse({"step": "parse_done", "data": {"parsed": len(recipes), "failed": len(pages) - len(recipes)}})

    if not recipes:
        yield sse({"step": "error", "message": "Could not extract recipe data from any page."})
        return

    # Step 7b: Category reconciliation — majority vote from parsers overrides step 1.
    step7_cats = Counter(
        r.category for r in recipes
        if getattr(r, "category", None) and r.category != "other"
    )
    if step7_cats:
        dominant = step7_cats.most_common(1)[0][0]
        if dominant != plan.category:
            plan = plan.model_copy(update={"category": dominant})

    # Steps 8–9: Ratios
    yield sse({"step": "ratio", "messages": _STEP_MESSAGES["ratio"]})
    ratios_list = await asyncio.to_thread(lambda: [compute_ratios(r) for r in recipes])
    cache_hits = sum(1 for r in ratios_list if r.from_cache)
    yield sse({"step": "ratio_done", "data": {"cache_hits": cache_hits}})

    # Step 10: Score + explain (uses feedback-inferred weights when toggle is on)
    yield sse({"step": "score", "messages": _STEP_MESSAGES["score"]})
    scoring_prefs = await asyncio.to_thread(update_user_prefs_from_feedback)
    scored = score_all(recipes, ratios_list, plan, scoring_prefs)
    await asyncio.to_thread(add_explanations, scored)

    # Top 3 (or fewer if fewer survived)
    top = scored[:3]

    # Option C: surface similar saved recipes for context
    similar_saved: list[dict] = []
    if scoring_prefs.get("use_feedback_prefs", True):
        q_emb = embed_text(query)
        similar_saved = await asyncio.to_thread(
            get_semantic_candidates, q_emb, 3, plan.category
        )

    # Build / update session for follow-up turns
    session = existing_session or ConversationSession(original_query=query)
    session.update_results(plan, recipes, ratios_list, scored)
    session.add_user(query)
    session.add_assistant(
        f"Found {len(top)} recipes for '{query}'. "
        f"Top: {top[0].recipe.title} ({top[0].composite_score:.0f}/100)."
    )
    put_session(session_id, session)

    yield sse({
        "step": "done",
        "session_id": session_id,
        "results": [s.model_dump() for s in top],
        "similar_saved": similar_saved,
    })


# ---------------------------------------------------------------------------
# Chat follow-up SSE generator
# ---------------------------------------------------------------------------

async def _chat_stream(
    session: ConversationSession,
    session_id: str,
    message: str,
    recency: Optional[str],
) -> AsyncGenerator[str, None]:
    session.add_user(message)

    yield sse({"step": "thinking", "message": "Thinking..."})
    refine = await asyncio.to_thread(classify_turn, session, message)
    turn_type = refine.get("turn_type", "factual")

    if turn_type == "re_filter":
        exclude = [e.strip() for e in (refine.get("exclude_ingredients") or []) if e.strip()]
        require_fat = refine.get("require_fat_type") or None

        _, _, scored_f = apply_re_filter(session, exclude, require_fat)

        if not scored_f:
            msg = "No recipes pass that filter from the current results."
            session.add_assistant(msg)
            yield sse({"step": "done", "type": "re_filter", "results": [], "message": msg})
            return

        user_prefs = await asyncio.to_thread(load_prefs)
        session.update_results(
            session.last_plan,
            [s.recipe for s in scored_f],
            [s.ratios for s in scored_f],
            scored_f,
        )
        session.add_assistant(
            f"Filter applied. {len(scored_f)} recipes pass. "
            f"Top: {scored_f[0].recipe.title}."
        )
        yield sse({
            "step": "done",
            "type": "re_filter",
            "results": [s.model_dump() for s in scored_f[:3]],
        })

    elif turn_type == "re_search":
        new_query = build_re_search_query(session, refine)
        hint_plan = build_merged_plan(session, refine)
        new_session_id_val = new_session_id()

        yield sse({"step": "re_search_start", "query": new_query})
        async for event in _search_stream(new_query, recency, new_session_id_val, session):
            yield event

    else:  # factual
        answer = refine.get("direct_answer") or "I'm not sure — try rephrasing."
        session.add_assistant(answer)
        yield sse({"step": "done", "type": "factual", "answer": answer})


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@router.post("/search")
async def search(req: SearchRequest):
    sid = req.session_id or new_session_id()
    existing = get_session(sid) if req.session_id else None

    return StreamingResponse(
        _search_stream(req.query, req.recency, sid, existing),
        media_type="text/event-stream",
    )


@router.post("/chat")
async def chat(req: ChatRequest):
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Start a new search.")

    return StreamingResponse(
        _chat_stream(session, req.session_id, req.message, req.recency),
        media_type="text/event-stream",
    )
