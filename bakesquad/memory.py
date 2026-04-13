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
    Infer per-category scoring weight overrides from liked_recipes.

    Respects the use_feedback_prefs flag — returns current prefs unchanged when False.

    Per-category inference (stored in prefs["category_prefs"][category]):
      {criterion_name_key: weight}  — normalised mean score per criterion, summing to 1.0.
        Criterion name → key conversion matches derive_weights() in scorer.py:
        "Chew & Texture" → "chew_and_texture_weight"
        "Moisture & Tenderness" → "moisture_and_tenderness_weight"  etc.
      preferred_fat  — dominant fat_type if ≥60% of liked recipes share one type
      sweetness      — "low" / "medium" / "high" from median sugar_to_flour ratio

    Requires ≥ prefs["feedback_min_liked"] liked recipes per category before inferring
    (prevents overfitting from a single data point).

    Global weight inference is intentionally omitted: each category has its own
    criteria set, so cross-category averaging is not meaningful.
    """
    prefs = load_prefs()

    if not prefs.get("use_feedback_prefs", True):
        return prefs

    min_liked: int = prefs.get("feedback_min_liked", 3)
    liked_rows = get_liked_recipes()
    if not liked_rows:
        return prefs

    # Bucket liked recipes by category
    by_category: dict[str, list[dict]] = {}
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

    def _criterion_key(name: str) -> str:
        """Convert criterion display name to user_prefs key (matches derive_weights logic)."""
        return name.lower().replace(" & ", "_and_").replace(" ", "_") + "_weight"

    def _infer_for_group(group: list[dict]) -> dict:
        """
        Infer weight overrides and fat/sweetness preferences from a list of
        ScoredRecipe dicts (stored as model_dump() in recipe_json).
        """
        n = len(group)
        criterion_scores: dict[str, list[float]] = {}
        fat_types: list[str] = []
        sugar_ratios: list[float] = []

        for data in group:
            # Accumulate per-criterion scores across all recipes in the group
            for c in data.get("criteria", []):
                name = c.get("name", "")
                score = c.get("score")
                # Skip universal add-on criteria — they're not per-category weights
                if name in ("GF Binding Agent", "Accessibility") or not name or score is None:
                    continue
                criterion_scores.setdefault(name, []).append(float(score))

            ratios = data.get("ratios", {})
            fat = ratios.get("fat_type")
            if fat in ("oil", "butter", "mixed"):
                fat_types.append(fat)
            sugar = ratios.get("sugar_to_flour")
            if sugar is not None:
                sugar_ratios.append(float(sugar))

        result: dict = {}

        # Weight inference: mean score per criterion, normalised to sum to 1.0
        means = {
            name: sum(scores) / len(scores)
            for name, scores in criterion_scores.items()
            if scores
        }
        total = sum(means.values())
        if means and total > 0:
            names = sorted(means)  # stable ordering
            weights = [round(means[n] / total, 3) for n in names]
            # Ensure exact sum of 1.0 by adjusting the largest weight
            diff = round(1.0 - sum(weights), 3)
            if diff:
                max_idx = weights.index(max(weights))
                weights[max_idx] = round(weights[max_idx] + diff, 3)
            for name, w in zip(names, weights):
                result[_criterion_key(name)] = w

        # Fat preference: dominant type if ≥60% of liked recipes share it
        if fat_types:
            fat_counts: dict[str, int] = {}
            for f in fat_types:
                fat_counts[f] = fat_counts.get(f, 0) + 1
            dominant = max(fat_counts, key=fat_counts.__getitem__)
            result["preferred_fat"] = dominant if fat_counts[dominant] / n >= 0.6 else None

        # Sweetness: median sugar_to_flour ratio
        if sugar_ratios:
            sugar_ratios.sort()
            median = sugar_ratios[len(sugar_ratios) // 2]
            result["sweetness"] = "low" if median < 0.35 else ("high" if median > 0.90 else "medium")

        return result

    # ── Per-category inference ────────────────────────────────────────────
    category_prefs: dict[str, dict] = dict(prefs.get("category_prefs") or {})
    for category, group in by_category.items():
        if len(group) >= min_liked:
            category_prefs[category] = _infer_for_group(group)

    prefs["category_prefs"] = category_prefs
    return prefs


# ---------------------------------------------------------------------------
# Semantic recipe retrieval
# ---------------------------------------------------------------------------

def embed_text(text: str, dims: int = 128) -> list[float]:
    """
    Deterministic hash-based bag-of-words embedding.

    No external dependencies or API calls. Suitable for a small personal corpus
    where approximate semantic overlap between recipe titles and queries is enough.
    Each word is hashed to a bucket; the resulting count vector is L2-normalised.
    """
    import hashlib
    import math

    tokens = text.lower().split()
    vec = [0.0] * dims
    for token in tokens:
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % dims
        vec[bucket] += 1.0
    mag = math.sqrt(sum(x * x for x in vec))
    if mag > 0:
        vec = [x / mag for x in vec]
    return vec


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
