"""Recipe box CRUD endpoints.

GET    /api/recipe-box         — list saved recipes (sort, category filter, text search)
POST   /api/recipe-box         — save / heart a recipe
PATCH  /api/recipe-box         — update notes on a saved recipe
DELETE /api/recipe-box         — remove a recipe from the box
"""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from bakesquad.memory import (
    delete_liked_recipe,
    get_liked_recipes,
    save_liked_recipe,
    update_liked_recipe,
)

router = APIRouter()

SortOption = Literal["date_desc", "date_asc", "title_asc", "title_desc"]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SaveRecipeRequest(BaseModel):
    url: str
    title: str
    category: str = "other"
    scored_recipe: dict          # model_dump() of ScoredRecipe
    notes: str = ""


class UpdateNotesRequest(BaseModel):
    url: str
    notes: str


class DeleteRequest(BaseModel):
    url: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/recipe-box")
async def list_recipe_box(
    sort: SortOption = "date_desc",
    category: Optional[str] = None,
    search: Optional[str] = None,
):
    rows = get_liked_recipes(sort=sort, category_filter=category, search=search)
    return {"recipes": rows}


@router.post("/recipe-box")
async def save_recipe(req: SaveRecipeRequest):
    save_liked_recipe(
        url=req.url,
        title=req.title,
        category=req.category,
        recipe_dict=req.scored_recipe,
        notes=req.notes,
    )
    return {"saved": True}


@router.patch("/recipe-box")
async def update_notes(req: UpdateNotesRequest):
    rows = get_liked_recipes(search=None)
    urls = {r["url"] for r in rows}
    if req.url not in urls:
        raise HTTPException(status_code=404, detail="Recipe not in box.")
    update_liked_recipe(req.url, notes=req.notes)
    return {"updated": True}


@router.delete("/recipe-box")
async def remove_recipe(req: DeleteRequest):
    delete_liked_recipe(req.url)
    return {"deleted": True}
