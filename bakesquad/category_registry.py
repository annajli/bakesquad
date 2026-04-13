"""
Dynamic category registry — loads categories.yaml once and exposes helpers.

During the transition period (Phases 0–2), the hardcoded Literal in models.py
still governs Pydantic validation.  This module is additive: it provides
synonym lookup and metadata that the LangGraph nodes will use in Phase 2.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

_YAML_PATH = Path(__file__).parent.parent / "categories.yaml"


@functools.lru_cache(maxsize=1)
def load_registry() -> list[dict[str, Any]]:
    """Load and cache the category registry from categories.yaml."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required for the category registry. "
            "Run: pip install pyyaml"
        ) from exc

    with open(_YAML_PATH, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data["categories"]


def get_category_ids() -> list[str]:
    """Return all registered category IDs in definition order."""
    return [c["id"] for c in load_registry()]


def get_category(category_id: str) -> dict[str, Any] | None:
    """Return the full metadata dict for a category ID, or None if not found."""
    for cat in load_registry():
        if cat["id"] == category_id:
            return cat
    return None


def synonym_to_id(term: str) -> str | None:
    """
    Map a surface-form synonym to a category ID.

    Checks both the `id` field and each entry in `synonyms` (case-insensitive).
    Returns None if no match.
    """
    term_lower = term.lower().strip()
    for cat in load_registry():
        if cat["id"] == term_lower:
            return cat["id"]
        for syn in cat.get("synonyms", []):
            if syn.lower() == term_lower:
                return cat["id"]
    return None


def scoring_criteria_for(category_id: str) -> list[str]:
    """Return the ordered list of scoring criterion keys for a category."""
    cat = get_category(category_id)
    if cat is None:
        return ["flavor_balance"]
    return cat.get("scoring_criteria", ["flavor_balance"])


def ratio_ranges_key_for(category_id: str) -> str:
    """Return the ratio_engine lookup key for a category."""
    cat = get_category(category_id)
    if cat is None:
        return "other"
    return cat.get("ratio_ranges_key", category_id)


def build_prompt_category_block() -> str:
    """
    Generate the category description block for Step 1 prompt dynamically from the registry.

    Produces the same format as the hardcoded string in prompts.py so it can
    be dropped in as a replacement once Phase 2 is complete.
    """
    lines: list[str] = [
        '"category": the type of baked good — one of: '
        + ", ".join(get_category_ids()),
    ]
    for cat in load_registry():
        members = cat.get("members", [])
        loaf_note = ""
        if cat.get("loaf_like"):
            loaf_note = ' (loaf-format bakes belong here)'
        if members:
            lines.append(
                f"  {cat['id']} includes: {', '.join(members)}{loaf_note}"
            )
    lines.append("  other: anything that does not fit the above")
    return "\n".join(lines)
