"""
Persistence layer.

Four stores:
  - ratio_cache        (SQLite) — deterministic ratio results keyed by URL.
  - liked_recipes      (SQLite) — ScoredRecipe dicts the user has saved, with rating/notes/tried_date.
  - recipe_embeddings  (SQLite) — float vector representations of liked recipes for semantic retrieval.
  - user_feedback      (SQLite) — event log of liked/disliked/tried/note actions.
  - user_prefs         (JSON)   — global + per-category scoring weight vector.

All stores live under ~/.bakesquad/ so they persist across sessions.

user_prefs structure:
  {
    "moisture_base_weight": 0.45,    # global fallback weights
    "structure_base_weight": 0.30,
    "balance_base_weight": 0.25,
    "preferred_fat": None,           # "oil" | "butter" | None (global)
    "sweetness": "medium",           # "low" | "medium" | "high" (global)
    "use_feedback_prefs": True,      # False = never infer prefs from feedback history
    "feedback_min_liked": 3,         # min liked recipes per category before inferring
    "category_prefs": {              # per-category overrides (populated by inference)
      "cookie": {
        "moisture_base_weight": 0.50,
        "structure_base_weight": 0.28,
        "balance_base_weight": 0.22,
        "preferred_fat": "butter",
        "sweetness": "medium"
      },
      ...
    }
  }
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Storage directory
# ---------------------------------------------------------------------------
_STORE_DIR = Path(os.environ.get("BAKESQUAD_DATA_DIR", Path.home() / ".bakesquad"))
_STORE_DIR.mkdir(parents=True, exist_ok=True)

_DB_PATH = _STORE_DIR / "bakesquad.db"
_PREFS_PATH = _STORE_DIR / "user_prefs.json"

# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables and apply schema migrations idempotently."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ratio_cache (
                url         TEXT PRIMARY KEY,
                category    TEXT NOT NULL,
                ratios_json TEXT NOT NULL,
                computed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS liked_recipes (
                url         TEXT PRIMARY KEY,
                title       TEXT,
                recipe_json TEXT NOT NULL,
                rating      INTEGER DEFAULT 0,
                notes       TEXT DEFAULT '',
                liked_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS recipe_embeddings (
                url         TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                category    TEXT NOT NULL,
                embedding   TEXT NOT NULL,
                embedded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                url         TEXT NOT NULL,
                feedback    TEXT NOT NULL,
                content     TEXT DEFAULT '',
                created_at  TEXT NOT NULL
            );
        """)
        # Additive column migrations for liked_recipes — safe to re-run
        for col_sql in (
            "ALTER TABLE liked_recipes ADD COLUMN user_rating INTEGER DEFAULT 0",
            "ALTER TABLE liked_recipes ADD COLUMN tried_date  TEXT DEFAULT NULL",
        ):
            try:
                conn.execute(col_sql)
            except Exception:
                pass  # column already exists


# ---------------------------------------------------------------------------
# Ratio cache
# ---------------------------------------------------------------------------

def cache_get(url: str) -> Optional[dict]:
    """Return cached ratio dict for a URL, or None on miss."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT ratios_json FROM ratio_cache WHERE url = ?", (url,)
        ).fetchone()
    if row:
        return json.loads(row["ratios_json"])
    return None


def cache_put(url: str, category: str, ratios: dict) -> None:
    """Upsert a ratio result into the cache."""
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO ratio_cache (url, category, ratios_json, computed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                category = excluded.category,
                ratios_json = excluded.ratios_json,
                computed_at = excluded.computed_at
            """,
            (url, category, json.dumps(ratios), datetime.utcnow().isoformat()),
        )


# ---------------------------------------------------------------------------
# Liked recipe store
# ---------------------------------------------------------------------------

def save_liked_recipe(url: str, title: str, recipe_dict: dict, rating: int = 0, notes: str = "") -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO liked_recipes (url, title, recipe_json, rating, notes, liked_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                title = excluded.title,
                recipe_json = excluded.recipe_json,
                rating = excluded.rating,
                notes = excluded.notes,
                liked_at = excluded.liked_at
            """,
            (url, title, json.dumps(recipe_dict), rating, notes, datetime.utcnow().isoformat()),
        )


def get_liked_recipes() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM liked_recipes ORDER BY liked_at DESC").fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# User preference model
# ---------------------------------------------------------------------------

_DEFAULT_PREFS: dict = {
    # Global scoring weights (used when no category-specific override exists)
    "moisture_base_weight": 0.45,
    "structure_base_weight": 0.30,
    "balance_base_weight": 0.25,
    "preferred_fat": None,        # "oil" | "butter" | None
    "sweetness": "medium",        # "low" | "medium" | "high"
    # Feedback inference controls
    "use_feedback_prefs": True,   # set False to disable all feedback-based pref updates
    "feedback_min_liked": 3,      # min liked recipes in a category before inferring prefs
    # Accessibility preference: 0.0 = disabled, 1.0 = full weight (0.20) added to composite
    "prefer_accessibility": 0.0,
    # Per-category weight overrides (populated by update_user_prefs_from_feedback)
    "category_prefs": {},
}


def load_prefs() -> dict:
    if _PREFS_PATH.exists():
        try:
            return {**_DEFAULT_PREFS, **json.loads(_PREFS_PATH.read_text())}
        except Exception:
            pass
    return dict(_DEFAULT_PREFS)


def save_prefs(prefs: dict) -> None:
    _PREFS_PATH.write_text(json.dumps(prefs, indent=2))


# ---------------------------------------------------------------------------
# Liked recipe helpers
# ---------------------------------------------------------------------------

def get_liked_urls() -> list[str]:
    """Return all liked recipe URLs (fast pre-load for memory_node)."""
    with _get_conn() as conn:
        rows = conn.execute("SELECT url FROM liked_recipes").fetchall()
    return [r["url"] for r in rows]


def update_liked_recipe(
    url: str,
    *,
    user_rating: Optional[int] = None,
    notes: Optional[str] = None,
    tried_date: Optional[str] = None,
) -> None:
    """Patch an existing liked_recipes row with user feedback fields."""
    updates: list[str] = []
    params: list = []
    if user_rating is not None:
        updates.append("user_rating = ?")
        params.append(user_rating)
    if notes is not None:
        updates.append("notes = ?")
        params.append(notes)
    if tried_date is not None:
        updates.append("tried_date = ?")
        params.append(tried_date)
    if not updates:
        return
    params.append(url)
    with _get_conn() as conn:
        conn.execute(
            f"UPDATE liked_recipes SET {', '.join(updates)} WHERE url = ?", params
        )


# ---------------------------------------------------------------------------
# User feedback event log
# ---------------------------------------------------------------------------

def add_feedback(url: str, feedback: str, content: str = "") -> None:
    """
    Record a user feedback event.

    feedback: "liked" | "disliked" | "tried" | "note"
    content:  note text when feedback="note", empty otherwise.
    """
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO user_feedback (url, feedback, content, created_at) VALUES (?, ?, ?, ?)",
            (url, feedback, content, datetime.utcnow().isoformat()),
        )


# ---------------------------------------------------------------------------
# Preference inference from feedback history
# ---------------------------------------------------------------------------

def update_user_prefs_from_feedback() -> dict:
    """
    Infer scoring weight adjustments from liked_recipes across all five pref dimensions.

    Respects the use_feedback_prefs flag — returns current prefs unchanged when False.

    Per-category inference (stored in prefs["category_prefs"][category]):
      moisture_base_weight  — normalised mean Moisture Retention score across liked recipes
      structure_base_weight — normalised mean Structure & Leavening score
      balance_base_weight   — normalised mean Sugar Balance score
      preferred_fat         — dominant fat_type if ≥60% of liked recipes share one type
      sweetness             — "low" / "medium" / "high" from median sugar_to_flour ratio

    Requires ≥ prefs["feedback_min_liked"] liked recipes per category before inferring
    (prevents overfitting from a single data point).

    Global prefs are updated from all liked recipes combined (category-agnostic).
    """
    prefs = load_prefs()

    if not prefs.get("use_feedback_prefs", True):
        return prefs

    min_liked: int = prefs.get("feedback_min_liked", 3)
    liked_rows = get_liked_recipes()
    if not liked_rows:
        return prefs

    # ── Parse stored ScoredRecipe dicts ───────────────────────────────────
    # recipe_json stores a ScoredRecipe.model_dump() with keys:
    #   recipe.category, ratios.fat_type, ratios.sugar_to_flour,
    #   criteria[{name, score, weight}]
    _MOISTURE_KEY  = "Moisture Retention"
    _STRUCTURE_KEY = "Structure & Leavening"
    _BALANCE_KEY   = "Sugar Balance"

    # Bucket liked recipes by category
    by_category: dict[str, list[dict]] = {}
    all_parsed: list[dict] = []
    for row in liked_rows:
        try:
            data = json.loads(row["recipe_json"])
        except Exception:
            continue
        category = (
            data.get("recipe", {}).get("category")
            or data.get("ratios", {}).get("category")
            or "other"
        )
        by_category.setdefault(category, []).append(data)
        all_parsed.append(data)

    def _infer_for_group(group: list[dict]) -> dict:
        """Infer all five pref dimensions from a list of parsed ScoredRecipe dicts."""
        n = len(group)
        moisture_scores, structure_scores, balance_scores = [], [], []
        fat_types: list[str] = []
        sugar_ratios: list[float] = []

        for data in group:
            criteria = {c["name"]: c["score"] for c in data.get("criteria", [])}
            ratios = data.get("ratios", {})

            if _MOISTURE_KEY in criteria:
                moisture_scores.append(criteria[_MOISTURE_KEY])
            if _STRUCTURE_KEY in criteria:
                structure_scores.append(criteria[_STRUCTURE_KEY])
            if _BALANCE_KEY in criteria:
                balance_scores.append(criteria[_BALANCE_KEY])

            fat = ratios.get("fat_type")
            if fat in ("oil", "butter", "mixed"):
                fat_types.append(fat)

            sugar = ratios.get("sugar_to_flour")
            if sugar is not None:
                sugar_ratios.append(float(sugar))

        result: dict = {}

        # Weight inference: normalise mean criterion scores to sum to 1.0
        means = {
            "moisture":  sum(moisture_scores)  / len(moisture_scores)  if moisture_scores  else None,
            "structure": sum(structure_scores) / len(structure_scores) if structure_scores else None,
            "balance":   sum(balance_scores)   / len(balance_scores)   if balance_scores   else None,
        }
        valid_means = {k: v for k, v in means.items() if v is not None}
        if len(valid_means) == 3:
            total = sum(valid_means.values())
            if total > 0:
                m_w = round(valid_means["moisture"]  / total, 3)
                s_w = round(valid_means["structure"] / total, 3)
                b_w = round(1.0 - m_w - s_w, 3)   # ensures exact sum of 1
                result["moisture_base_weight"]  = m_w
                result["structure_base_weight"] = s_w
                result["balance_base_weight"]   = b_w

        # Fat preference: dominant type if ≥60% of liked recipes share it
        if fat_types:
            fat_counts: dict[str, int] = {}
            for f in fat_types:
                fat_counts[f] = fat_counts.get(f, 0) + 1
            dominant_fat = max(fat_counts, key=fat_counts.__getitem__)
            result["preferred_fat"] = (
                dominant_fat if fat_counts[dominant_fat] / n >= 0.6 else None
            )

        # Sweetness: median sugar_to_flour ratio
        if sugar_ratios:
            sugar_ratios.sort()
            median = sugar_ratios[len(sugar_ratios) // 2]
            result["sweetness"] = (
                "low" if median < 0.35 else ("high" if median > 0.90 else "medium")
            )

        return result

    # ── Per-category inference ────────────────────────────────────────────
    category_prefs: dict[str, dict] = dict(prefs.get("category_prefs") or {})
    for category, group in by_category.items():
        if len(group) >= min_liked:
            category_prefs[category] = _infer_for_group(group)

    prefs["category_prefs"] = category_prefs

    # ── Global inference from all liked recipes combined ─────────────────
    if len(all_parsed) >= min_liked:
        global_inferred = _infer_for_group(all_parsed)
        for key, value in global_inferred.items():
            if key not in ("moisture_base_weight", "structure_base_weight",
                           "balance_base_weight", "preferred_fat", "sweetness"):
                continue  # never overwrite control fields
            prefs[key] = value

    return prefs


# ---------------------------------------------------------------------------
# Semantic recipe retrieval
# ---------------------------------------------------------------------------

def save_embedding(url: str, title: str, category: str, embedding: list[float]) -> None:
    """Persist a recipe embedding as a JSON float array."""
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO recipe_embeddings (url, title, category, embedding, embedded_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                embedding   = excluded.embedding,
                embedded_at = excluded.embedded_at
            """,
            (url, title, category, json.dumps(embedding), datetime.utcnow().isoformat()),
        )


def get_semantic_candidates(
    query_embedding: list[float],
    top_k: int = 5,
    category_filter: Optional[str] = None,
) -> list[dict]:
    """
    Return the top_k liked recipes most similar to query_embedding (cosine similarity).

    Embeddings are stored as JSON float arrays and compared in Python.
    Upgrade path: swap for sqlite-vec extension when table exceeds ~1000 rows.
    """
    import math

    with _get_conn() as conn:
        sql = "SELECT url, title, category, embedding FROM recipe_embeddings"
        params: list = []
        if category_filter:
            sql += " WHERE category = ?"
            params.append(category_filter)
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return []

    mag_q = math.sqrt(sum(x * x for x in query_embedding))
    if not mag_q:
        return []

    results = []
    for row in rows:
        try:
            emb = json.loads(row["embedding"])
            dot = sum(a * b for a, b in zip(query_embedding, emb))
            mag_e = math.sqrt(sum(x * x for x in emb))
            sim = dot / (mag_q * mag_e) if mag_e else 0.0
            results.append({
                "url": row["url"],
                "title": row["title"],
                "category": row["category"],
                "similarity": sim,
            })
        except Exception:
            continue

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:top_k]
