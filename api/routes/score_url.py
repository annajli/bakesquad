"""POST /api/score-url — score a single recipe URL via SSE.

Pipeline (steps 6–10 only; steps 1–5 are skipped since the URL is given):
  fetch  → parse → ratio → score+explain → done

The recipe category is inferred from the LLM parser (step 7) rather than
requiring the user to select it. A minimal QueryPlan is built from that result
so the scorer has a category and empty constraints.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.sse_utils import sse
from bakesquad.memory import load_prefs, update_user_prefs_from_feedback
from bakesquad.models import FetchedPage, QueryPlan, SearchSnippet
from bakesquad.parser import parse_recipe
from bakesquad.ratio_engine import compute_ratios
from bakesquad.scorer import add_explanations, score_recipe
from bakesquad.search.ingestion import IngestionPipeline

router = APIRouter()

# Baking-themed messages per pipeline step
_MSGS = {
    "fetch":  "Pulling the recipe from the web...",
    "parse":  "Reading the ingredient list...",
    "ratio":  "Weighing the flour and fat...",
    "score":  "Taste-testing the ratios...",
}


class ScoreUrlRequest(BaseModel):
    url: str


def _fetch_url(url: str) -> FetchedPage:
    pipeline = IngestionPipeline(trusted_sources=[])
    snippet = SearchSnippet(url=url, title="", excerpt="")
    return pipeline._fetch_one(snippet)


async def _stream(url: str) -> AsyncGenerator[str, None]:
    # Step 6: Fetch
    yield sse({"step": "fetch", "message": _MSGS["fetch"]})
    try:
        page = await asyncio.to_thread(_fetch_url, url)
    except Exception as exc:
        yield sse({"step": "error", "message": f"Could not fetch that URL: {exc}"})
        return

    if page.fetch_error:
        yield sse({"step": "error", "message": f"Fetch failed: {page.fetch_error}"})
        return

    from bakesquad.config import MIN_PAGE_CONTENT_CHARS
    if len(page.raw_text) < MIN_PAGE_CONTENT_CHARS:
        yield sse({"step": "error", "message": "Page content too short — may not be a recipe."})
        return

    # Step 7: Parse (infers category)
    yield sse({"step": "parse", "message": _MSGS["parse"]})
    recipe = await asyncio.to_thread(parse_recipe, page)

    if recipe is None or not recipe.ingredients:
        yield sse({"step": "error", "message": "No ingredients found. Is this a recipe page?"})
        return

    # Steps 8–9: Ratios
    yield sse({"step": "ratio", "message": _MSGS["ratio"]})
    ratios = await asyncio.to_thread(compute_ratios, recipe)

    # Build a minimal QueryPlan from the inferred category.
    # No hard_constraints / soft_preferences — user didn't express a query.
    plan = QueryPlan(
        category=recipe.category,
        hard_constraints=[],
        soft_preferences=[],
        queries=[],
    )

    # Step 10: Score + explain
    yield sse({"step": "score", "message": _MSGS["score"]})
    user_prefs = await asyncio.to_thread(update_user_prefs_from_feedback)
    scored = score_recipe(recipe, ratios, plan, user_prefs)
    await asyncio.to_thread(add_explanations, [scored])

    yield sse({
        "step": "done",
        "result": scored.model_dump(),
    })


@router.post("/score-url")
async def score_url(req: ScoreUrlRequest):
    return StreamingResponse(_stream(req.url), media_type="text/event-stream")
