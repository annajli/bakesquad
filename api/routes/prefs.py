"""User preferences endpoints.

GET /api/prefs  — load all user prefs (dietary restrictions, memory toggle, etc.)
PUT /api/prefs  — save updated prefs

Dietary restrictions are stored in prefs["dietary_restrictions"] as a list of
strings matching the parser's modifier vocabulary:
  "gluten_free" | "vegan" | "dairy_free" | "nut_free" | "paleo" | "keto"

These are injected server-side into the query pipeline as hard_constraints,
so they must live in SQLite (not just localStorage).

The use_feedback_prefs flag controls whether liked-recipe history biases
scoring weights (Option B memory feature). Toggle off = ignore recipe box
history when deriving weights.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from bakesquad.memory import load_prefs, save_prefs

router = APIRouter()

VALID_DIETARY = frozenset({"gluten_free", "vegan", "dairy_free", "nut_free", "paleo", "keto"})


class PrefsUpdate(BaseModel):
    dietary_restrictions: Optional[List[str]] = None
    use_feedback_prefs: Optional[bool] = None
    prefer_accessibility: Optional[float] = None


@router.get("/prefs")
async def get_prefs():
    prefs = load_prefs()
    return {
        "dietary_restrictions": prefs.get("dietary_restrictions") or [],
        "use_feedback_prefs": prefs.get("use_feedback_prefs", True),
        "prefer_accessibility": prefs.get("prefer_accessibility", 0.0),
    }


@router.put("/prefs")
async def update_prefs(req: PrefsUpdate):
    prefs = load_prefs()

    if req.dietary_restrictions is not None:
        # Validate against known modifier vocabulary
        prefs["dietary_restrictions"] = [
            d for d in req.dietary_restrictions if d in VALID_DIETARY
        ]

    if req.use_feedback_prefs is not None:
        prefs["use_feedback_prefs"] = req.use_feedback_prefs

    if req.prefer_accessibility is not None:
        prefs["prefer_accessibility"] = max(0.0, min(1.0, req.prefer_accessibility))

    save_prefs(prefs)
    return {"saved": True}
