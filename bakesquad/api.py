"""
BakeSquad REST API (FastAPI).

Endpoints:
  POST /api/search          — natural-language query → scored recipes
  POST /api/score-url       — paste a URL → score that specific recipe
  GET  /api/saved           — return all liked/stored recipes
  POST /api/feedback        — record liked / disliked / tried / note
  GET  /api/prefs           — get user scoring preferences
  PATCH /api/prefs          — update user scoring preferences
  GET  /api/health          — liveness check

All endpoints are synchronous. The graph invoke() call blocks for the full
pipeline duration (~15–45 s depending on backend). A future streaming endpoint
can wrap graph.astream() for progressive frontend updates.

Usage:
  MODEL_BACKEND=claude uvicorn bakesquad.api:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="BakeSquad API",
    version="0.1.0",
    description="Recipe scoring and retrieval powered by ratio science + LLM analysis.",
)

# Allow any origin during development — tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the graph once at startup — shared across all requests
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        from bakesquad.graph.builder import build_graph
        _graph = build_graph()
    return _graph


def _initial_state(thread_id: str, query: str) -> dict:
    """Minimal valid BakeSquadState for a fresh search turn."""
    from bakesquad.memory import get_liked_urls, load_prefs
    return {
        "thread_id": thread_id,
        "user_query": query,
        "messages": [],
        "turn_type": None,
        "query_plan": None,
        "category_confidence": 1.0,
        "clarify_question": None,
        "recency": None,
        "snippets": [],
        "fetched_pages": [],
        "search_retry_count": 0,
        "parsed_recipes": [],
        "category_override": None,
        "ratio_results": [],
        "scored_recipes": [],
        "exclude_ingredients": [],
        "require_fat_type": None,
        "factual_answer": None,
        "user_prefs": load_prefs(),
        "liked_recipe_urls": get_liked_urls(),
        "errors": [],
    }


def _serialize_scored(scored_recipes: list) -> list[dict]:
    """Convert ScoredRecipe objects to JSON-serializable dicts."""
    result = []
    for s in scored_recipes:
        result.append({
            "rank": s.rank,
            "composite_score": s.composite_score,
            "title": s.recipe.title,
            "url": s.recipe.url,
            "category": s.recipe.category,
            "flour_type": s.recipe.flour_type,
            "modifiers": s.recipe.modifiers,
            "explanation": s.explanation,
            "constraint_violations": s.constraint_violations,
            "technique_note_delta": s.technique_note_delta,
            "accessibility_score": s.accessibility_score,
            "criteria": [
                {
                    "name": c.name,
                    "score": c.score,
                    "weight": c.weight,
                    "details": c.details,
                }
                for c in s.criteria
            ],
            "ratios": {
                "fat_type": s.ratios.fat_type,
                "fat_to_flour": s.ratios.fat_to_flour,
                "sugar_to_flour": s.ratios.sugar_to_flour,
                "liquid_to_flour": s.ratios.liquid_to_flour,
                "leavening_to_flour": s.ratios.leavening_to_flour,
                "brown_to_white_sugar": s.ratios.brown_to_white_sugar,
                "has_banana": s.ratios.has_banana,
                "has_chocolate": s.ratios.has_chocolate,
                "has_binding_agent": s.ratios.has_binding_agent,
                "flour_grams": s.ratios.flour_grams,
            },
        })
    return result


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    thread_id: Optional[str] = None   # provide to continue a conversation
    recency: Optional[str] = None     # "year" | "month" | None


class SearchResponse(BaseModel):
    thread_id: str
    recipes: list[dict]
    query_plan: Optional[dict] = None
    clarify_question: Optional[str] = None
    message: str = ""


class ScoreUrlRequest(BaseModel):
    url: str = Field(..., min_length=10)


class ScoreUrlResponse(BaseModel):
    recipe: Optional[dict] = None
    error: Optional[str] = None


class FeedbackRequest(BaseModel):
    url: str
    action: str = Field(..., pattern="^(liked|disliked|tried|note)$")
    content: str = ""   # note text when action="note"


class PrefsUpdate(BaseModel):
    prefer_accessibility: Optional[float] = Field(None, ge=0.0, le=1.0)
    use_feedback_prefs: Optional[bool] = None
    feedback_min_liked: Optional[int] = Field(None, ge=1)
    preferred_fat: Optional[str] = None    # "oil" | "butter" | None
    sweetness: Optional[str] = None        # "low" | "medium" | "high"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    from bakesquad.llm_client import get_backend, get_model
    return {"status": "ok", "backend": get_backend(), "model": get_model()}


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    Run the full BakeSquad pipeline for a natural-language query.

    Provide thread_id on follow-up turns to continue the same conversation
    (re_filter, re_search, factual). Omit or pass null to start fresh.
    """
    thread_id = req.thread_id or str(uuid.uuid4())
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # On a fresh thread, provide the full initial state.
    # On a continuing thread, only provide the new user query — the graph
    # loads the rest from the SqliteSaver checkpoint.
    if req.thread_id:
        invoke_input = {"user_query": req.query}
    else:
        invoke_input = _initial_state(thread_id, req.query)
        invoke_input["recency"] = req.recency

    try:
        result = graph.invoke(invoke_input, config=config)
    except Exception as exc:
        logger.exception("Graph invocation failed for thread %s", thread_id)
        raise HTTPException(status_code=500, detail=str(exc))

    scored = result.get("scored_recipes") or []
    plan = result.get("query_plan")
    clarify_q = result.get("clarify_question")

    return SearchResponse(
        thread_id=thread_id,
        recipes=_serialize_scored(scored),
        query_plan=plan.model_dump() if plan else None,
        clarify_question=clarify_q,
        message=f"{len(scored)} recipes scored" + (" — clarification needed" if clarify_q else ""),
    )


@app.post("/api/score-url", response_model=ScoreUrlResponse)
def score_url(req: ScoreUrlRequest):
    """
    Score a single recipe from a URL the user has found.

    Skips web search entirely: fetch → parse → ratio → score → store embedding.
    The scored recipe is stored in the corpus so future searches can recall it.
    """
    from bakesquad.memory import (
        embed_text,
        get_liked_urls,
        load_prefs,
        save_embedding,
    )
    from bakesquad.models import SearchSnippet
    from bakesquad.parser import parse_recipes_parallel
    from bakesquad.ratio_engine import compute_ratios
    from bakesquad.scorer import add_explanations, score_all
    from bakesquad.search.ingestion import IngestionPipeline

    url = req.url.strip()

    # Fetch the page
    pipeline = IngestionPipeline()
    snippet = SearchSnippet(url=url, title="", excerpt="")
    page = pipeline._fetch_one(snippet)

    if page.fetch_error:
        return ScoreUrlResponse(error=f"Fetch failed: {page.fetch_error}")
    if not page.ingredients_excerpt or len(page.ingredients_excerpt) < 50:
        return ScoreUrlResponse(error="Could not extract ingredients from this page.")

    # Parse
    recipes = parse_recipes_parallel([page])
    if not recipes or recipes[0].parse_error:
        err = recipes[0].parse_error if recipes else "No recipes parsed"
        return ScoreUrlResponse(error=f"Parse failed: {err}")

    recipe = recipes[0]

    # Build a minimal QueryPlan from parsed recipe fields (no LLM call needed)
    from bakesquad.models import QueryPlan
    plan = QueryPlan(
        category=recipe.category,
        flour_type=recipe.flour_type or "ap",
        modifiers=recipe.modifiers or [],
        hard_constraints=[],
        soft_preferences=[],
        queries=[],
    )

    user_prefs = load_prefs()
    ratios = compute_ratios(recipe)
    scored = score_all([recipe], [ratios], plan, user_prefs)
    add_explanations(scored)

    if not scored:
        return ScoreUrlResponse(error="Scoring produced no results.")

    s = scored[0]

    # Store embedding so this recipe enters the personal corpus
    try:
        ing_names = " ".join(i.name for i in (recipe.ingredients or []))
        text = f"{recipe.title} {recipe.category} {ing_names}"
        save_embedding(url, recipe.title, recipe.category, embed_text(text))
    except Exception as exc:
        logger.warning("score-url: embedding save failed: %s", exc)

    return ScoreUrlResponse(recipe=_serialize_scored([s])[0])


@app.get("/api/saved")
def get_saved():
    """Return all recipes stored in the personal corpus (liked_recipes table)."""
    from bakesquad.memory import get_liked_recipes

    rows = get_liked_recipes()
    recipes = []
    for row in rows:
        try:
            data = json.loads(row["recipe_json"])
            recipes.append({
                "url": row["url"],
                "title": row["title"],
                "rating": row.get("user_rating") or row.get("rating") or 0,
                "notes": row.get("notes") or "",
                "tried_date": row.get("tried_date"),
                "liked_at": row.get("liked_at"),
                "category": (
                    data.get("recipe", {}).get("category")
                    or data.get("ratios", {}).get("category")
                    or "other"
                ),
                "composite_score": data.get("composite_score"),
            })
        except Exception:
            continue
    return {"recipes": recipes, "count": len(recipes)}


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    """
    Record user feedback for a recipe.

    action="liked"    → saves full ScoredRecipe to liked_recipes + records event
    action="disliked" → records event only
    action="tried"    → records event; updates tried_date on liked_recipes row
    action="note"     → records event with note text; updates notes on liked_recipes row
    """
    from bakesquad.memory import (
        add_feedback,
        get_liked_recipes,
        save_liked_recipe,
        update_liked_recipe,
    )
    from datetime import datetime

    add_feedback(req.url, req.action, req.content)

    if req.action == "liked":
        # Find the scored recipe in the corpus (it may already be stored from score-url)
        rows = {r["url"]: r for r in get_liked_recipes()}
        if req.url not in rows:
            # Not yet in liked_recipes — store a minimal placeholder
            save_liked_recipe(req.url, "", {}, rating=1)
        else:
            update_liked_recipe(req.url, user_rating=1)

    elif req.action == "tried":
        update_liked_recipe(req.url, tried_date=datetime.utcnow().date().isoformat())

    elif req.action == "note" and req.content:
        update_liked_recipe(req.url, notes=req.content)

    return {"status": "ok", "url": req.url, "action": req.action}


@app.get("/api/prefs")
def get_prefs():
    """Return the current user scoring preferences."""
    from bakesquad.memory import load_prefs
    return load_prefs()


@app.patch("/api/prefs")
def update_prefs(update: PrefsUpdate):
    """
    Update top-level user scoring preferences.

    Only the fields provided in the request body are changed;
    all others (including category_prefs) are left untouched.
    """
    from bakesquad.memory import load_prefs, save_prefs

    prefs = load_prefs()
    patch = update.model_dump(exclude_none=True)
    prefs.update(patch)
    save_prefs(prefs)
    return prefs
