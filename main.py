"""
BakeSquad — full pipeline entry point.

Usage:
    MODEL_BACKEND=groq  python main.py "chocolate chip banana bread that stays moist for days"
    MODEL_BACKEND=claude python main.py "chewy brown butter chocolate chip cookies"
    MODEL_BACKEND=ollama python main.py "moist banana bread"

Definition of done:
    - Completes in <60 s on Claude and Groq backends
    - Returns ≥2 scored, ranked recipes with explanations
    - Per-step elapsed time printed after each step
    - Running same query twice is measurably faster (ratio cache)
    - Switching backends requires only MODEL_BACKEND env var change
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from typing import Literal, Optional

# Force UTF-8 stdout/stderr on Windows so recipe titles with special chars don't crash.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Load .env before any bakesquad imports (sets MODEL_BACKEND, API keys, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ.setdefault("USER_AGENT", "BakeSquad/1.0")

# ---------------------------------------------------------------------------
# Imports (after env is ready)
# ---------------------------------------------------------------------------
from bakesquad.config import STEP_BUDGETS
from bakesquad.llm_client import get_backend, get_model, get_time_budget
from bakesquad.memory import init_db, load_prefs
from bakesquad.parser import parse_recipes_parallel
from bakesquad.ratio_engine import compute_ratios
from bakesquad.scorer import add_explanations, score_all
from bakesquad.search.ingestion import IngestionPipeline

logging.basicConfig(
    level=logging.WARNING,          # Keep INFO/DEBUG off by default for clean output
    format="%(levelname)s  %(name)s  %(message)s",
)

DEFAULT_QUERY = "chocolate chip banana bread that stays moist for days"
BAR_WIDTH = 20   # width of ASCII score bars


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Timer:
    def __init__(self, backend: str):
        self._backend = backend
        self._start = time.monotonic()
        self._step_start = self._start
        self._strict = backend in ("groq", "claude")

    def tick(self, step_name: str, budget_key: Optional[str] = None) -> float:
        now = time.monotonic()
        elapsed = now - self._step_start
        self._step_start = now
        budget = STEP_BUDGETS.get(budget_key or "", 0) if budget_key else 0
        over = elapsed > budget and budget > 0 and self._strict
        status = "!! OVER BUDGET" if over else "ok"
        budget_str = f"  (budget {budget}s)" if budget else ""
        print(f"  {status}  {elapsed:.1f}s{budget_str}")
        return elapsed

    def total(self) -> float:
        return time.monotonic() - self._start


def _bar(score: float) -> str:
    filled = round(score / 100 * BAR_WIDTH)
    return "#" * filled + "." * (BAR_WIDTH - filled)


def _print_header(backend: str, model: str, query: str, recency: Optional[str]) -> None:
    print()
    print("=" * 62)
    print(f"  BakeSquad  |  Backend: {backend} ({model})")
    print("=" * 62)
    print(f"  Query: {query!r}")
    if recency:
        print(f"  Recency:  {recency}")
    print()


def _print_query_understanding(plan) -> None:
    print(f"  Category:    {plan.category}")
    print(f"  Constraints: {', '.join(plan.hard_constraints) or '-'}")
    print(f"  Preferences: {', '.join(plan.soft_preferences) or '-'}")
    print(f"  Queries generated:")
    for q in plan.queries:
        print(f"    - {q}")


def _print_results(scored_recipes, plan) -> None:
    print()
    print("=" * 62)
    print("  RANKED RESULTS")
    print("=" * 62)

    # Query understanding card at top
    print(f"\n  How I understood your query:")
    print(f"    Category: {plan.category}")
    if plan.hard_constraints:
        print(f"    Must have: {', '.join(plan.hard_constraints)}")
    if plan.soft_preferences:
        print(f"    Preferences: {', '.join(plan.soft_preferences)}")

    if not scored_recipes:
        print("\n  No recipes scored. Try a different query.\n")
        return

    print()
    for s in scored_recipes:
        composite_bar = _bar(s.composite_score)
        tag_parts = []
        if s.ratios.fat_type == "oil":
            tag_parts.append("[oil-based]")
        elif s.ratios.fat_type == "butter":
            tag_parts.append("[butter-based]")
        if s.ratios.has_banana:
            tag_parts.append("[banana]")
        if s.ratios.has_chocolate:
            tag_parts.append("[chocolate]")
        if s.ratios.has_extra_yolks:
            tag_parts.append("[extra yolks]")
        tags = "  " + " ".join(tag_parts) if tag_parts else ""

        print(f"  #{s.rank}  {s.recipe.title}")
        print(f"      {s.recipe.url}")
        print(f"      {tags}")
        print(f"      Composite  {composite_bar}  {s.composite_score:.0f}/100")
        print()

        for c in s.criteria:
            bar = _bar(c.score)
            print(f"        {c.name:<24} {bar}  {c.score:.0f}  (w={c.weight:.2f})")

        print()
        print("      Ratios:")
        r = s.ratios
        if r.liquid_to_flour is not None:
            flag = _range_flag(r.liquid_to_flour, r.category, "liquid_to_flour")
            print(f"        liquid/flour:    {r.liquid_to_flour:.3f}  {flag}")
        if r.fat_to_flour is not None:
            flag = _range_flag(r.fat_to_flour, r.category, "fat_to_flour")
            print(f"        fat/flour:       {r.fat_to_flour:.3f}  {flag}")
        if r.sugar_to_flour is not None:
            flag = _range_flag(r.sugar_to_flour, r.category, "sugar_to_flour")
            print(f"        sugar/flour:     {r.sugar_to_flour:.3f}  {flag}")
        if r.leavening_to_flour is not None:
            flag = _range_flag(r.leavening_to_flour, r.category, "leavening_to_flour")
            print(f"        leavening/flour: {r.leavening_to_flour:.4f} {flag}")
        if r.fat_type:
            print(f"        fat source:      {r.fat_type}")
        if r.brown_to_white_sugar is not None:
            flag = _range_flag(r.brown_to_white_sugar, r.category, "brown_to_white_sugar")
            print(f"        brown/white sugar:{r.brown_to_white_sugar:.2f}  {flag}")
        if r.from_cache:
            print(f"        [ratios from cache]")

        if s.constraint_violations:
            print()
            for v in s.constraint_violations:
                print(f"      ** CONSTRAINT RISK: {v}")

        if s.explanation:
            print()
            print("      Why this score:")
            # Wrap explanation at 60 chars
            words = s.explanation.split()
            line = "        "
            for word in words:
                if len(line) + len(word) + 1 > 70:
                    print(line)
                    line = "        " + word
                else:
                    line += (" " if line.strip() else "") + word
            if line.strip():
                print(line)

        print()
        print("  " + "-" * 58)
        print()


def _range_flag(value: float, category: str, ratio_name: str) -> str:
    from bakesquad.ratio_engine import ratio_in_range
    return "ok" if ratio_in_range(ratio_name, value, category) else "!!"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # Parse CLI args
    args = sys.argv[1:]
    recency: Optional[Literal["year", "month"]] = None
    if "--recency" in args:
        idx = args.index("--recency")
        recency = args[idx + 1] if idx + 1 < len(args) else None  # type: ignore[assignment]
        args = args[:idx] + args[idx + 2:]

    query = " ".join(args).strip() or DEFAULT_QUERY
    backend = get_backend()
    model = get_model()
    total_budget = get_time_budget()

    _print_header(backend, model, query, recency)
    init_db()
    user_prefs = load_prefs()
    timer = _Timer(backend)

    # ------------------------------------------------------------------
    # Step 1: Query understanding
    # ------------------------------------------------------------------
    print("[Step 1: Query Understanding]")
    pipeline = IngestionPipeline(trusted_sources=[])
    plan = pipeline._build_query_plan(query, recency)
    _print_query_understanding(plan)
    timer.tick("query_understanding", "query_understanding")

    # ------------------------------------------------------------------
    # Step 2-5: Search, snippet scoring, domain cap, adaptive retry
    # ------------------------------------------------------------------
    print("\n[Step 2-5: Search + Candidate Selection]")
    candidates = pipeline._search_and_filter(query, plan.queries, recency)
    print(f"  {len(candidates)} candidates selected")
    if candidates:
        for c in candidates[:6]:
            print(f"    [{c.relevance_score:.2f}] {c.domain}  {c.title[:60]}")
    timer.tick("search", "search")

    if not candidates:
        print("\n  No candidates found. Try a different query or check your internet connection.")
        return

    # ------------------------------------------------------------------
    # Step 6: Page fetch (parallel)
    # ------------------------------------------------------------------
    print(f"\n[Step 6: Page Fetch]  (parallel, 5 s timeout)")
    from bakesquad.config import MAX_PAGES_PER_RUN
    attempted = min(len(candidates), MAX_PAGES_PER_RUN)
    pages = pipeline._fetch_pages(candidates)
    print(f"  {len(pages)}/{attempted} pages fetched successfully")
    timer.tick("page_fetch", "page_fetch")

    if not pages:
        print("\n  No pages fetched successfully. Aborting.")
        return

    # ------------------------------------------------------------------
    # Step 7: LLM parsing (parallel)
    # ------------------------------------------------------------------
    print(f"\n[Step 7: LLM Parse]  (parallel, {len(pages)} pages)")
    recipes = parse_recipes_parallel(pages)
    print(f"  {len(recipes)} recipes parsed  ({len(pages) - len(recipes)} failed)")
    timer.tick("llm_parsing", "llm_parsing")

    if not recipes:
        print("\n  No recipes parsed successfully. Try a different query.")
        return

    # ------------------------------------------------------------------
    # Steps 8-9: Unit normalization + ratio engine (deterministic, no LLM)
    # ------------------------------------------------------------------
    print(f"\n[Steps 8-9: Normalization + Ratios]")
    ratios_list = []
    cache_hits = 0
    for recipe in recipes:
        r = compute_ratios(recipe)
        ratios_list.append(r)
        if r.from_cache:
            cache_hits += 1
    print(f"  {len(ratios_list)} ratio results  ({cache_hits} from cache)")
    timer.tick("normalization_ratios", "normalization_ratios")

    # ------------------------------------------------------------------
    # Step 10: Scoring (deterministic math) + explanations (batched LLM)
    # ------------------------------------------------------------------
    print(f"\n[Step 10: Scoring + Explanations]")
    scored = score_all(recipes, ratios_list, plan, user_prefs)
    print(f"  {len(scored)} recipes scored")
    print(f"  Generating explanations (1 batched LLM call)...")
    add_explanations(scored)
    timer.tick("scoring_explanations", "scoring_explanations")

    # ------------------------------------------------------------------
    # Step 11: Output
    # ------------------------------------------------------------------
    print(f"\n[Step 11: Output]")
    _print_results(scored, plan)
    timer.tick("output", "output")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = timer.total()
    over = "OVER BUDGET" if total > total_budget else "OK"
    print("=" * 62)
    print(f"  Total: {total:.1f}s  (budget {total_budget}s)  [{over}]")
    print(f"  Backend: {backend} ({model})")
    if cache_hits > 0:
        print(f"  Ratio cache: {cache_hits}/{len(ratios_list)} hits - skipped parse+normalize+ratio")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
