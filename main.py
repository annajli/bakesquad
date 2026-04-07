"""
BakeSquad — pipeline entry point.

Single-run mode (backward compatible):
    MODEL_BACKEND=groq  python main.py "chocolate chip banana bread that stays moist for days"
    MODEL_BACKEND=claude python main.py "chewy brown butter chocolate chip cookies"
    MODEL_BACKEND=ollama python main.py "moist banana bread"
    MODEL_BACKEND=groq  python main.py "sourdough cookies" --recency year

Interactive chat mode (no query argument):
    MODEL_BACKEND=groq python main.py

In chat mode the agent keeps conversation history between turns and supports:
  - re_filter: "no brown sugar", "only oil-based"  — re-ranks existing results instantly
  - re_search: "find me a simpler recipe", "what about a cake?"  — runs a new search
  - factual:   "why does oil retain moisture better?"  — answers using current results
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
from bakesquad.config import MAX_PAGES_PER_RUN, STEP_BUDGETS
from bakesquad.llm_client import get_backend, get_model, get_time_budget
from bakesquad.memory import init_db, load_prefs
from bakesquad.models import ParsedRecipe, QueryPlan, RatioResult, ScoredRecipe
from bakesquad.parser import parse_recipes_parallel
from bakesquad.ratio_engine import compute_ratios, ratio_in_range
from bakesquad.scorer import add_explanations, score_all
from bakesquad.search.ingestion import IngestionPipeline
from bakesquad.session import (
    ConversationSession,
    apply_re_filter,
    build_merged_plan,
    build_re_search_query,
    classify_turn,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)

BAR_WIDTH = 20


# ---------------------------------------------------------------------------
# Display helpers
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


def _range_flag(value: float, category: str, ratio_name: str, flour_type: str = "ap") -> str:
    return "ok" if ratio_in_range(ratio_name, value, category, flour_type) else "!!"


def _print_header(backend: str, model: str, query: str, recency: Optional[str]) -> None:
    print()
    print("=" * 62)
    print(f"  BakeSquad  |  Backend: {backend} ({model})")
    print("=" * 62)
    print(f"  Query: {query!r}")
    if recency:
        print(f"  Recency: {recency}")
    print()


def _print_query_understanding(plan: QueryPlan) -> None:
    print(f"  Category:    {plan.category}")
    print(f"  Constraints: {', '.join(plan.hard_constraints) or '-'}")
    print(f"  Preferences: {', '.join(plan.soft_preferences) or '-'}")
    print("  Queries generated:")
    for q in plan.queries:
        print(f"    - {q}")


def _print_results(scored: list[ScoredRecipe], plan: QueryPlan) -> None:
    print()
    print("=" * 62)
    print("  RANKED RESULTS")
    print("=" * 62)
    print(f"\n  Category: {plan.category}")
    if plan.hard_constraints:
        print(f"  Must have: {', '.join(plan.hard_constraints)}")
    if plan.soft_preferences:
        print(f"  Preferences: {', '.join(plan.soft_preferences)}")

    if not scored:
        print("\n  No recipes to show.\n")
        return

    print()
    for s in scored:
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
        if tags:
            print(f"      {tags.strip()}")
        print(f"      Composite  {_bar(s.composite_score)}  {s.composite_score:.0f}/100")
        print()

        for c in s.criteria:
            print(f"        {c.name:<24} {_bar(c.score)}  {c.score:.0f}  (w={c.weight:.2f})")

        print()
        print("      Ratios:")
        r = s.ratios
        ft = r.flour_type or "ap"
        if r.liquid_to_flour is not None:
            print(f"        liquid/flour:    {r.liquid_to_flour:.3f}  {_range_flag(r.liquid_to_flour, r.category, 'liquid_to_flour', ft)}")
        if r.fat_to_flour is not None:
            print(f"        fat/flour:       {r.fat_to_flour:.3f}  {_range_flag(r.fat_to_flour, r.category, 'fat_to_flour', ft)}")
        if r.sugar_to_flour is not None:
            print(f"        sugar/flour:     {r.sugar_to_flour:.3f}  {_range_flag(r.sugar_to_flour, r.category, 'sugar_to_flour', ft)}")
        if r.leavening_to_flour is not None:
            print(f"        leavening/flour: {r.leavening_to_flour:.4f} {_range_flag(r.leavening_to_flour, r.category, 'leavening_to_flour', ft)}")
        if r.fat_type:
            print(f"        fat source:      {r.fat_type}")
        if r.flour_type and r.flour_type != "ap":
            print(f"        flour type:      {r.flour_type}")
        if r.modifiers:
            print(f"        modifiers:       {', '.join(r.modifiers)}")
        if r.has_binding_agent:
            print(f"        binding agent:   yes")
        if r.brown_to_white_sugar is not None:
            print(f"        brown/white:     {r.brown_to_white_sugar:.2f}  {_range_flag(r.brown_to_white_sugar, r.category, 'brown_to_white_sugar', ft)}")
        if r.from_cache:
            print(f"        [ratios from cache]")

        if s.constraint_violations:
            print()
            for v in s.constraint_violations:
                print(f"      ** CONSTRAINT RISK: {v}")

        if s.explanation:
            print()
            print("      Why this score:")
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


def _print_filter_results(
    scored: list[ScoredRecipe],
    excluded: int,
    exclude_terms: list[str],
    require_fat: Optional[str],
) -> None:
    """Compact view for re_filter results — no ratio detail, just updated ranking."""
    filters = []
    if exclude_terms:
        filters.append(f"excluding: {', '.join(exclude_terms)}")
    if require_fat:
        filters.append(f"fat type: {require_fat} only")
    filter_str = "  |  ".join(filters) if filters else "constraint applied"

    print()
    print(f"  Filter: {filter_str}")
    print(f"  {len(scored)} recipes pass  ({excluded} excluded)")
    print()

    if not scored:
        return

    for s in scored:
        fat_tag = f"[{s.ratios.fat_type}]" if s.ratios.fat_type else ""
        choc_tag = "[chocolate]" if s.ratios.has_chocolate else ""
        tags = " ".join(t for t in [fat_tag, choc_tag] if t)
        print(f"  #{s.rank}  {s.composite_score:.0f}/100  {s.recipe.title}")
        print(f"       {s.recipe.url}")
        if tags:
            print(f"       {tags}")
        print()


# ---------------------------------------------------------------------------
# Core pipeline (extracted so both single-run and re-search can call it)
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str,
    recency: Optional[Literal["year", "month"]] = None,
    session: Optional[ConversationSession] = None,
    user_prefs: Optional[dict] = None,
    hint_plan: Optional[QueryPlan] = None,
    show_timing: bool = True,
) -> Optional[ConversationSession]:
    """
    Run the full 11-step pipeline.

    hint_plan: if provided (re-search from chat), the category and merged constraints
               are passed to the LLM to guide query understanding.
    Returns an updated ConversationSession, or None if the pipeline produced no results.
    """
    backend = get_backend()
    model = get_model()
    total_budget = get_time_budget()

    if user_prefs is None:
        user_prefs = load_prefs()

    timer = _Timer(backend)
    pipeline = IngestionPipeline(trusted_sources=[])

    # Step 1: Query understanding
    print("[Step 1: Query Understanding]")
    if hint_plan:
        # Re-search: seed the query with the category + merged constraints so the
        # LLM doesn't have to re-derive them from scratch.
        hint_prefix = (
            f"[Context: continuing from previous search. "
            f"Category={hint_plan.category}. "
            f"Active constraints: {'; '.join(hint_plan.hard_constraints)}. "
            f"New request: {query}]"
        )
        plan = pipeline._build_query_plan(hint_prefix, recency)
    else:
        plan = pipeline._build_query_plan(query, recency)
    _print_query_understanding(plan)
    timer.tick("query_understanding", "query_understanding")

    # Step 2-5: Search, snippet scoring, domain cap, adaptive retry
    print("\n[Step 2-5: Search + Candidate Selection]")
    candidates = pipeline._search_and_filter(query, plan.queries, recency)
    print(f"  {len(candidates)} candidates selected")
    for c in candidates[:6]:
        print(f"    [{c.relevance_score:.2f}] {c.domain}  {c.title[:58]}")
    timer.tick("search", "search")

    if not candidates:
        print("\n  No candidates found. Try a different query.")
        return None

    # Step 6: Page fetch
    print(f"\n[Step 6: Page Fetch]  (parallel, 5 s timeout)")
    attempted = min(len(candidates), MAX_PAGES_PER_RUN)
    pages = pipeline._fetch_pages(candidates)
    print(f"  {len(pages)}/{attempted} pages fetched successfully")
    timer.tick("page_fetch", "page_fetch")

    if not pages:
        print("\n  No pages fetched. Aborting.")
        return None

    # Step 7: LLM parsing
    print(f"\n[Step 7: LLM Parse]  (parallel, {len(pages)} pages)")
    recipes = parse_recipes_parallel(pages)
    print(f"  {len(recipes)} recipes parsed  ({len(pages) - len(recipes)} failed)")
    timer.tick("llm_parsing", "llm_parsing")

    if not recipes:
        print("\n  No recipes parsed. Try a different query.")
        return None

    # Steps 8-9: Normalization + ratio engine
    print("\n[Steps 8-9: Normalization + Ratios]")
    ratios_list: list[RatioResult] = []
    cache_hits = 0
    for recipe in recipes:
        r = compute_ratios(recipe)
        ratios_list.append(r)
        if r.from_cache:
            cache_hits += 1
    print(f"  {len(ratios_list)} ratio results  ({cache_hits} from cache)")
    timer.tick("normalization_ratios", "normalization_ratios")

    # Step 10: Scoring + explanations
    print("\n[Step 10: Scoring + Explanations]")
    scored = score_all(recipes, ratios_list, plan, user_prefs)
    print(f"  {len(scored)} recipes scored")
    print("  Generating explanations (1 batched LLM call)...")
    add_explanations(scored)
    timer.tick("scoring_explanations", "scoring_explanations")

    # Step 11: Output
    print("\n[Step 11: Output]")
    _print_results(scored, plan)
    timer.tick("output", "output")

    if show_timing:
        total = timer.total()
        over = "OVER BUDGET" if total > total_budget else "OK"
        print("=" * 62)
        print(f"  Total: {total:.1f}s  (budget {total_budget}s)  [{over}]")
        print(f"  Backend: {backend} ({model})")
        if cache_hits > 0:
            print(f"  Ratio cache: {cache_hits}/{len(ratios_list)} hits")
        print("=" * 62)
        print()

    # Build / update session
    if session is None:
        session = ConversationSession(original_query=query)

    session.update_results(plan, recipes, ratios_list, scored)
    session.add_assistant(
        f"Returned {len(scored)} recipes for: {query!r}. "
        f"Top result: {scored[0].recipe.title} ({scored[0].composite_score:.0f}/100)."
    )
    return session


# ---------------------------------------------------------------------------
# Chat turn handlers
# ---------------------------------------------------------------------------

def _handle_re_filter(
    session: ConversationSession,
    refine: dict,
    user_prefs: dict,
) -> None:
    """Apply ingredient/fat-type filter to existing results. Zero LLM calls."""
    exclude = [e.strip() for e in (refine.get("exclude_ingredients") or []) if e.strip()]
    require_fat = refine.get("require_fat_type") or None

    before = len(session.last_scored)
    recipes_f, ratios_f, scored_f = apply_re_filter(session, exclude, require_fat)
    excluded = before - len(scored_f)

    if not scored_f:
        print()
        print("  No recipes pass that filter from the current results.")
        print("  Tip: try a broader constraint, or ask me to search again with the new requirement.")
        session.add_assistant("Re-filter produced 0 results. Suggested a new search.")
        return

    _print_filter_results(scored_f, excluded, exclude, require_fat)
    session.update_results(session.last_plan, recipes_f, ratios_f, scored_f)

    summary = (
        f"Re-filter applied ({', '.join(exclude) or require_fat}). "
        f"{len(scored_f)}/{before} recipes pass. "
        f"Top result: {scored_f[0].recipe.title}."
    )
    session.add_assistant(summary)


def _handle_re_search(
    session: ConversationSession,
    refine: dict,
    recency: Optional[str],
    user_prefs: dict,
) -> None:
    """Re-run the full pipeline with merged constraints from the follow-up."""
    new_query = build_re_search_query(session, refine)
    hint_plan = build_merged_plan(session, refine)

    backend = get_backend()
    model = get_model()
    print()
    print("=" * 62)
    print(f"  BakeSquad  |  Re-search  |  Backend: {backend} ({model})")
    print("=" * 62)
    print(f"  New query: {new_query!r}")
    print()

    updated = run_pipeline(
        query=new_query,
        recency=recency,
        session=session,
        user_prefs=user_prefs,
        hint_plan=hint_plan,
        show_timing=True,
    )
    if updated is None:
        print("  Re-search returned no results.")
        session.add_assistant("Re-search returned no results.")


def _handle_factual(session: ConversationSession, refine: dict) -> None:
    """Print the direct_answer generated by classify_turn. No additional LLM call needed."""
    answer = refine.get("direct_answer") or "I'm not sure about that."
    print()
    print("  " + answer.replace("\n", "\n  "))
    print()
    session.add_assistant(answer)


# ---------------------------------------------------------------------------
# Interactive chat loop
# ---------------------------------------------------------------------------

def chat_loop(recency: Optional[str] = None) -> None:
    backend = get_backend()
    model = get_model()
    user_prefs = load_prefs()

    print()
    print("=" * 62)
    print(f"  BakeSquad  |  Chat Mode  |  Backend: {backend} ({model})")
    print("=" * 62)
    print("  Enter a baking query to get started.")
    print("  Follow up with constraints, questions, or a new request.")
    print("  Type 'quit' to exit.")
    print()

    # Get initial query
    try:
        query = input("  Query: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not query or query.lower() in ("quit", "exit", "q"):
        return

    session = run_pipeline(query, recency=recency, user_prefs=user_prefs, show_timing=True)
    if session is None:
        return
    session.add_user(query)

    # Follow-up loop
    while True:
        print()
        try:
            user_input = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print("\n  Goodbye!\n")
            break

        session.add_user(user_input)

        # Classify the follow-up (1 LLM call)
        print("  Thinking...")
        refine = classify_turn(session, user_input)
        turn_type = refine.get("turn_type", "factual")

        if turn_type == "re_filter":
            _handle_re_filter(session, refine, user_prefs)
        elif turn_type == "re_search":
            _handle_re_search(session, refine, recency, user_prefs)
        else:
            _handle_factual(session, refine)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]
    recency: Optional[Literal["year", "month"]] = None
    if "--recency" in args:
        idx = args.index("--recency")
        recency = args[idx + 1] if idx + 1 < len(args) else None  # type: ignore[assignment]
        args = args[:idx] + args[idx + 2:]

    init_db()

    query = " ".join(args).strip()

    if query:
        # Single-run mode: backward compatible for scripting and testing
        backend = get_backend()
        model = get_model()
        _print_header(backend, model, query, recency)
        user_prefs = load_prefs()
        run_pipeline(query, recency=recency, user_prefs=user_prefs, show_timing=True)
    else:
        # Interactive chat mode
        chat_loop(recency=recency)


if __name__ == "__main__":
    main()
