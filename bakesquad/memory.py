"""
Persistence layer.

Three stores:
  - ratio_cache   (SQLite)  — deterministic ratio results keyed by URL; cache hits skip
                              parse + normalize + ratio computation entirely.
  - liked_recipes (SQLite)  — full recipe objects the user has saved with ratings/notes.
  - user_prefs    (JSON)    — preference weight vector, updated over time.

All stores live under ~/.bakesquad/ so they persist across sessions.
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
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ratio_cache (
                url       TEXT PRIMARY KEY,
                category  TEXT NOT NULL,
                ratios_json TEXT NOT NULL,
                computed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS liked_recipes (
                url         TEXT PRIMARY KEY,
                title       TEXT,
                recipe_json TEXT NOT NULL,
                rating      INTEGER,
                notes       TEXT,
                liked_at    TEXT NOT NULL
            );
        """)


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
    "moisture_base_weight": 0.45,
    "structure_base_weight": 0.30,
    "balance_base_weight": 0.25,
    "preferred_fat": None,        # "oil" | "butter" | None
    "sweetness": "medium",        # "low" | "medium" | "high"
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
