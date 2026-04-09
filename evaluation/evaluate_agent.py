#!/usr/bin/env python3
"""
BakeSquad Evaluation Framework
================================
Tests and diagnoses the BakeSquad recipe search pipeline across two modes:

  Stage 1 — Classification-only (fast, ~1 LLM call per test case)
    Calls query_plan_prompt + LLM directly without running the full pipeline.
    Instruments chat() to capture every prompt/response pair so failures
    can be traced to prompt wording vs. LLM reasoning.

  Stage 2 — Full pipeline (slow, optional, requires API keys + network)
    Runs the complete 11-step pipeline and captures outputs at each stage.

Usage
-----
  # Stage 1 only (recommended starting point):
  MODEL_BACKEND=groq python evaluation/evaluate_agent.py

  # With debug output (prints each LLM prompt + raw response):
  MODEL_BACKEND=groq python evaluation/evaluate_agent.py --debug

  # Run both stages (needs real search + LLM):
  MODEL_BACKEND=groq python evaluation/evaluate_agent.py --full-pipeline

  # Subset of tests by group:
  MODEL_BACKEND=groq python evaluation/evaluate_agent.py --groups known_issue edge_case

  # Custom dataset:
  python evaluation/evaluate_agent.py --dataset path/to/my_dataset.json

Outputs (written to evaluation/results/):
  eval_results_<timestamp>.json   — per-case structured results
  eval_report_<timestamp>.md      — human-readable analysis report
  confusion_matrix_<timestamp>.txt — ASCII confusion matrix

Requirements
------------
  pip install python-dotenv   (already used by main.py)
  All other deps from the main project (openai, anthropic, etc.)
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import sys
import time

# Force UTF-8 stdout/stderr on Windows (same fix as main.py)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Path setup: allow running from repo root or from evaluation/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# LLM instrumentation — wrap chat() BEFORE importing any bakesquad code
# ---------------------------------------------------------------------------

_llm_trace: list[dict] = []          # appended to by the wrapper
_instrumentation_active = False


def _install_chat_instrumentation() -> None:
    """
    Monkey-patch bakesquad.llm_client.chat() so every call is logged.

    Each entry in _llm_trace records:
      step_label   — injected by the caller (see _labeled_chat)
      system       — full system prompt sent to the LLM
      user         — user message sent to the LLM
      raw_response — raw text returned by the LLM (before JSON parsing)
      elapsed_s    — wall-clock time for this call
      error        — exception message if the call failed, else null
    """
    global _instrumentation_active
    if _instrumentation_active:
        return

    import bakesquad.llm_client as llm_mod  # noqa: F401

    _original_chat = llm_mod.chat

    def _instrumented_chat(
        system: str,
        user: str,
        *,
        temperature: float = 0,
        max_tokens: int = 2048,
        _step_label: str = "unknown",
    ) -> str:
        t0 = time.monotonic()
        error_msg = None
        raw = ""
        try:
            raw = _original_chat(system, user, temperature=temperature, max_tokens=max_tokens)
            return raw
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            _llm_trace.append({
                "step_label": _step_label,
                "system": system,
                "user": user,
                "raw_response": raw,
                "elapsed_s": round(time.monotonic() - t0, 3),
                "error": error_msg,
            })

    llm_mod.chat = _instrumented_chat  # type: ignore[attr-defined]
    _instrumentation_active = True


def _labeled_chat(system: str, user: str, step_label: str, **kwargs: Any) -> str:
    """Call the (instrumented) chat() with a step label injected."""
    import bakesquad.llm_client as llm_mod
    return llm_mod.chat(system, user, _step_label=step_label, **kwargs)  # type: ignore[call-arg]


def _pop_trace() -> list[dict]:
    """Return and clear the current trace buffer."""
    entries = list(_llm_trace)
    _llm_trace.clear()
    return entries


# ---------------------------------------------------------------------------
# Imports (after instrumentation setup)
# ---------------------------------------------------------------------------
_install_chat_instrumentation()

import bakesquad.llm_client as llm_mod
from bakesquad.llm_client import extract_json
from bakesquad.search.prompts import query_plan_prompt


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["test_cases"]


# ---------------------------------------------------------------------------
# Stage 1: Classification testing
# ---------------------------------------------------------------------------

def run_classification_test(
    tc: dict,
    debug: bool = False,
) -> dict:
    """
    Run a single test case through Step 1 (query understanding / classification).

    Returns a result dict with:
      id, query, expected_category, predicted_category, correct,
      ambiguous, acceptable_categories, group,
      query_plan (full parsed plan),
      llm_trace (list of instrumented LLM calls),
      error (if something failed),
      elapsed_s
    """
    _pop_trace()  # clear any leftover trace entries

    t0 = time.monotonic()
    error_msg = None
    predicted_category = None
    query_plan_dict = None

    try:
        system, user = query_plan_prompt(tc["query"], recency=None, trusted_sources=[])

        if debug:
            print(f"\n{'─'*60}")
            print(f"[{tc['id']}] {tc['query']!r}")
            print(f"  SYSTEM PROMPT (first 300 chars):\n    {system[:300]}...")
            print(f"  USER: {user!r}")

        raw = _labeled_chat(system, user, step_label="step1_query_plan")
        parsed = extract_json(raw)

        if debug:
            print(f"  RAW RESPONSE: {raw[:400]}")
            print(f"  PARSED: {parsed}")

        query_plan_dict = parsed if isinstance(parsed, dict) else {}
        predicted_category = query_plan_dict.get("category", "PARSE_ERROR")

    except Exception as exc:
        error_msg = str(exc)
        predicted_category = "ERROR"
        if debug:
            print(f"  ERROR: {exc}")

    elapsed = round(time.monotonic() - t0, 3)
    trace = _pop_trace()

    # Correctness: exact match, OR within acceptable_categories for ambiguous
    is_ambiguous = tc.get("ambiguous", False)
    acceptable = tc.get("acceptable_categories", [tc["expected_category"]])
    correct_exact = (predicted_category == tc["expected_category"])
    correct_acceptable = (predicted_category in acceptable)

    return {
        "id": tc["id"],
        "query": tc["query"],
        "group": tc["group"],
        "expected_category": tc["expected_category"],
        "predicted_category": predicted_category,
        "correct_exact": correct_exact,
        "correct_acceptable": correct_acceptable,
        "ambiguous": is_ambiguous,
        "acceptable_categories": acceptable,
        "query_plan": query_plan_dict,
        "llm_trace": trace,
        "error": error_msg,
        "elapsed_s": elapsed,
        "notes": tc.get("notes", ""),
    }


def run_stage1(
    test_cases: list[dict],
    debug: bool = False,
    groups: Optional[list[str]] = None,
) -> list[dict]:
    """Run all (or filtered) test cases through Stage 1 classification."""
    if groups:
        test_cases = [tc for tc in test_cases if tc["group"] in groups]

    results = []
    total = len(test_cases)
    print(f"\n{'='*60}")
    print(f"  Stage 1: Classification Testing  ({total} cases)")
    print(f"{'='*60}")

    for i, tc in enumerate(test_cases, 1):
        status_prefix = f"[{i:02d}/{total}]"
        result = run_classification_test(tc, debug=debug)

        mark = "✓" if result["correct_exact"] else ("~" if result["correct_acceptable"] else "✗")
        ambig_tag = " (ambiguous)" if result["ambiguous"] else ""
        print(
            f"  {status_prefix} {mark}  {tc['id']:<5}  "
            f"{tc['query'][:40]:<40}  "
            f"expected={result['expected_category']:<12}  "
            f"got={result['predicted_category']:<12}"
            f"{ambig_tag}"
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Stage 2: Full pipeline testing
# ---------------------------------------------------------------------------

def run_full_pipeline_test(tc: dict, debug: bool = False) -> dict:
    """
    Run a single test case through the complete 11-step BakeSquad pipeline.

    Instruments every LLM call (via the already-installed wrapper).
    Captures: query plan, snippets fetched, pages fetched, parsed recipes,
              ratio results, and scored recipes.

    NOTE: This requires real API keys and network access. It will be slow
    (30–60 s per query with Groq/Claude, up to 3 min with Ollama).
    """
    _pop_trace()
    t0 = time.monotonic()
    error_msg = None
    stage_outputs: dict[str, Any] = {}

    try:
        # Import lazily to avoid DB init at module load
        from bakesquad.memory import init_db, load_prefs
        from bakesquad.models import QueryPlan, ScoredRecipe
        from bakesquad.search.ingestion import IngestionPipeline
        from bakesquad.parser import parse_recipes_parallel
        from bakesquad.ratio_engine import compute_ratios
        from bakesquad.scorer import score_all, add_explanations

        init_db()
        user_prefs = load_prefs()
        pipeline = IngestionPipeline(trusted_sources=[])

        # Step 1 — Query plan
        sys_prompt, usr_prompt = query_plan_prompt(tc["query"], recency=None, trusted_sources=[])
        raw_plan = _labeled_chat(sys_prompt, usr_prompt, step_label="step1_query_plan")
        plan_dict = extract_json(raw_plan)
        plan = QueryPlan(**plan_dict)
        stage_outputs["query_plan"] = plan.model_dump()

        # Steps 2–6 — Search + fetch
        pages = pipeline.run(plan)
        stage_outputs["pages_fetched"] = [
            {"url": p.url, "title": p.title, "fetch_error": p.fetch_error}
            for p in pages
        ]

        if not pages:
            stage_outputs["result"] = "no_pages"
            return _build_pipeline_result(tc, stage_outputs, _pop_trace(), error_msg, t0)

        # Step 7 — Parse
        parsed = parse_recipes_parallel(pages, plan)
        stage_outputs["parsed_recipes"] = [
            {
                "title": r.title,
                "url": r.url,
                "category": r.category,
                "flour_type": r.flour_type,
                "modifiers": r.modifiers,
                "has_chocolate": r.has_chocolate,
                "parse_error": r.parse_error,
            }
            for r in parsed
        ]

        if not parsed:
            stage_outputs["result"] = "no_parsed_recipes"
            return _build_pipeline_result(tc, stage_outputs, _pop_trace(), error_msg, t0)

        # Steps 8–9 — Normalize + ratios
        ratios = [compute_ratios(r) for r in parsed]
        stage_outputs["ratios"] = [r.model_dump() for r in ratios]

        # Step 10 — Score
        scored = score_all(parsed, ratios, plan, user_prefs)
        scored = add_explanations(scored)
        stage_outputs["scored_recipes"] = [
            {
                "rank": s.rank,
                "title": s.recipe.title,
                "url": s.recipe.url,
                "composite_score": s.composite_score,
                "constraint_violations": s.constraint_violations,
                "criteria": [c.model_dump() for c in s.criteria],
            }
            for s in scored
        ]
        stage_outputs["result"] = "ok"

    except Exception as exc:
        error_msg = str(exc)
        stage_outputs["result"] = "error"
        if debug:
            import traceback
            traceback.print_exc()

    return _build_pipeline_result(tc, stage_outputs, _pop_trace(), error_msg, t0)


def _build_pipeline_result(
    tc: dict,
    stage_outputs: dict,
    trace: list[dict],
    error_msg: Optional[str],
    t0: float,
) -> dict:
    plan = stage_outputs.get("query_plan", {})
    predicted_category = plan.get("category", "ERROR")
    is_ambiguous = tc.get("ambiguous", False)
    acceptable = tc.get("acceptable_categories", [tc["expected_category"]])
    return {
        "id": tc["id"],
        "query": tc["query"],
        "group": tc["group"],
        "expected_category": tc["expected_category"],
        "predicted_category": predicted_category,
        "correct_exact": predicted_category == tc["expected_category"],
        "correct_acceptable": predicted_category in acceptable,
        "ambiguous": is_ambiguous,
        "acceptable_categories": acceptable,
        "stage_outputs": stage_outputs,
        "llm_trace": trace,
        "error": error_msg,
        "elapsed_s": round(time.monotonic() - t0, 3),
        "notes": tc.get("notes", ""),
    }


def run_stage2(
    test_cases: list[dict],
    debug: bool = False,
    groups: Optional[list[str]] = None,
    max_cases: int = 10,
) -> list[dict]:
    """Run full pipeline on a (limited) subset of test cases."""
    if groups:
        test_cases = [tc for tc in test_cases if tc["group"] in groups]
    test_cases = test_cases[:max_cases]

    results = []
    total = len(test_cases)
    print(f"\n{'='*60}")
    print(f"  Stage 2: Full Pipeline Testing  ({total} cases, may be slow)")
    print(f"{'='*60}")

    for i, tc in enumerate(test_cases, 1):
        print(f"  [{i:02d}/{total}] Running full pipeline for: {tc['query']!r} ...")
        result = run_full_pipeline_test(tc, debug=debug)
        mark = "✓" if result["correct_exact"] else ("~" if result["correct_acceptable"] else "✗")
        print(
            f"           {mark}  "
            f"expected={result['expected_category']:<12}  "
            f"got={result['predicted_category']:<12}  "
            f"({result['elapsed_s']:.1f}s)"
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    """
    Compute classification accuracy and error patterns.

    Splits non-ambiguous cases from ambiguous cases.
    Returns a metrics dict suitable for JSON serialisation.
    """
    non_ambiguous = [r for r in results if not r["ambiguous"]]
    ambiguous = [r for r in results if r["ambiguous"]]

    total_na = len(non_ambiguous)
    correct_exact = sum(1 for r in non_ambiguous if r["correct_exact"])
    errors = [r for r in non_ambiguous if not r["correct_exact"] and not r.get("error")]

    # Per-group accuracy
    groups: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in non_ambiguous:
        g = r["group"]
        groups[g]["total"] += 1
        if r["correct_exact"]:
            groups[g]["correct"] += 1

    group_accuracy = {
        g: {
            "total": v["total"],
            "correct": v["correct"],
            "accuracy": round(v["correct"] / v["total"], 3) if v["total"] else 0.0,
        }
        for g, v in sorted(groups.items())
    }

    # Confusion matrix (predicted → expected counts)
    categories = ["cookie", "quick_bread", "cake", "other"]
    confusion: dict[str, dict[str, int]] = {
        pred: {exp: 0 for exp in categories} for pred in categories + ["ERROR", "PARSE_ERROR"]
    }
    for r in non_ambiguous:
        pred = r["predicted_category"]
        exp = r["expected_category"]
        if pred not in confusion:
            confusion[pred] = {exp2: 0 for exp2 in categories}
        if exp in confusion[pred]:
            confusion[pred][exp] += 1
        else:
            confusion[pred][exp] = 1

    # Error pattern analysis
    error_patterns = _analyse_errors(errors)

    # Ambiguous breakdown
    ambig_stats = {
        "total": len(ambiguous),
        "within_acceptable": sum(1 for r in ambiguous if r["correct_acceptable"]),
        "outside_acceptable": sum(1 for r in ambiguous if not r["correct_acceptable"]),
    }
    if ambiguous:
        ambig_stats["rate_within_acceptable"] = round(
            ambig_stats["within_acceptable"] / ambig_stats["total"], 3
        )

    return {
        "total_non_ambiguous": total_na,
        "correct_exact": correct_exact,
        "accuracy": round(correct_exact / total_na, 3) if total_na else 0.0,
        "error_count": len(errors),
        "error_rate": round(len(errors) / total_na, 3) if total_na else 0.0,
        "group_accuracy": group_accuracy,
        "confusion_matrix": confusion,
        "error_patterns": error_patterns,
        "ambiguous": ambig_stats,
    }


def _analyse_errors(errors: list[dict]) -> list[dict]:
    """
    Cluster misclassified cases into named failure patterns.

    Patterns detected:
      - quick_bread_as_other: quick_bread expected, 'other' predicted
      - quick_bread_as_cake:  quick_bread expected, 'cake' predicted
      - loaf_framing:         query contains 'loaf' but not 'bread'/'muffin'
      - bread_keyword_escape: query contains 'bread' but classified wrong
      - modifier_interference: dietary/flour modifier changed classification
    """
    patterns: list[dict] = []

    qb_as_other = [
        r for r in errors
        if r["expected_category"] == "quick_bread" and r["predicted_category"] == "other"
    ]
    if qb_as_other:
        patterns.append({
            "name": "quick_bread_classified_as_other",
            "count": len(qb_as_other),
            "cases": [r["id"] for r in qb_as_other],
            "queries": [r["query"] for r in qb_as_other],
            "description": (
                "The LLM output 'other' for queries the schema defines as quick_bread. "
                "Common with banana bread and zucchini bread — items whose name "
                "contains a non-category word ('bread') that the model may generalise "
                "beyond the quick_bread bucket."
            ),
        })

    qb_as_cake = [
        r for r in errors
        if r["expected_category"] == "quick_bread" and r["predicted_category"] == "cake"
    ]
    if qb_as_cake:
        patterns.append({
            "name": "quick_bread_classified_as_cake",
            "count": len(qb_as_cake),
            "cases": [r["id"] for r in qb_as_cake],
            "queries": [r["query"] for r in qb_as_cake],
            "description": (
                "Moist quick breads (banana bread, pumpkin bread) are sometimes "
                "mistaken for cakes due to similar texture/ingredient vocabulary."
            ),
        })

    loaf_framing = [
        r for r in errors
        if "loaf" in r["query"].lower()
        and "bread" not in r["query"].lower()
        and "muffin" not in r["query"].lower()
    ]
    if loaf_framing:
        patterns.append({
            "name": "loaf_framing_not_recognised",
            "count": len(loaf_framing),
            "cases": [r["id"] for r in loaf_framing],
            "queries": [r["query"] for r in loaf_framing],
            "description": (
                "The query used 'loaf' as the product type instead of 'bread' or 'muffin'. "
                "The schema examples don't use 'loaf'; the model may not map it to quick_bread."
            ),
        })

    modifier_interference = [
        r for r in errors
        if r["expected_category"] == "quick_bread"
        and any(
            m in r["query"].lower()
            for m in ["gluten-free", "almond flour", "paleo", "keto", "vegan"]
        )
    ]
    if modifier_interference:
        patterns.append({
            "name": "modifier_overrides_category",
            "count": len(modifier_interference),
            "cases": [r["id"] for r in modifier_interference],
            "queries": [r["query"] for r in modifier_interference],
            "description": (
                "Dietary or flour modifiers (gluten-free, almond flour, paleo) caused "
                "the model to deviate from the correct category — possibly because "
                "the modifier dominated the prompt reasoning."
            ),
        })

    other_errors = [
        r for r in errors
        if r not in qb_as_other
        and r not in qb_as_cake
        and r not in loaf_framing
        and r not in modifier_interference
    ]
    if other_errors:
        patterns.append({
            "name": "uncategorised_errors",
            "count": len(other_errors),
            "cases": [r["id"] for r in other_errors],
            "queries": [r["query"] for r in other_errors],
            "description": "Misclassifications not matching any named pattern.",
        })

    return patterns


# ---------------------------------------------------------------------------
# Pipeline diagnostics
# ---------------------------------------------------------------------------

def diagnose_pipeline_issues(results: list[dict]) -> dict:
    """
    Analyse WHERE in the pipeline errors originate and whether they propagate.

    For Stage 1 results this is limited to Step 1 (query plan).
    For Stage 2 results this covers Steps 1–10.
    """
    diagnostics: dict[str, Any] = {}

    # Step 1 propagation: if query plan gets category wrong, all downstream
    # steps use the wrong reference ranges — no correction is possible.
    step1_errors = [
        r for r in results
        if not r["ambiguous"]
        and not r["correct_exact"]
        and (r.get("query_plan") or {}).get("category") not in (None, "ERROR", "PARSE_ERROR")
    ]

    diagnostics["step1_classification_errors"] = len(step1_errors)
    diagnostics["step1_propagation_note"] = (
        "Step 1 errors propagate irreversibly. Once the query plan sets the wrong "
        "category, Steps 9–10 compute ratios against the wrong reference ranges, "
        "and Step 10 scoring uses the wrong criterion weights. The pipeline has no "
        "mechanism to revisit the category after Step 1."
    )

    # Check if any Stage 2 results show category disagreement between Step 1 and Step 7
    step1_vs_step7_disagreements = []
    for r in results:
        so = r.get("stage_outputs", {})
        plan_cat = so.get("query_plan", {}).get("category")
        parsed = so.get("parsed_recipes", [])
        for p in parsed:
            if p.get("category") and p["category"] != plan_cat:
                step1_vs_step7_disagreements.append({
                    "id": r["id"],
                    "query": r["query"],
                    "step1_category": plan_cat,
                    "step7_category": p["category"],
                    "recipe_title": p.get("title", ""),
                })

    diagnostics["step1_vs_step7_disagreements"] = step1_vs_step7_disagreements
    if step1_vs_step7_disagreements:
        diagnostics["step1_vs_step7_note"] = (
            "Step 7 (parser) inferred a different category than Step 1 for some recipes. "
            "The pipeline uses the Step 1 category for ratio range selection — "
            "the Step 7 re-classification is recorded but does not affect scoring."
        )

    # LLM trace analysis: flag calls where the raw response needed fallback parsing
    json_parse_struggles = []
    for r in results:
        for entry in r.get("llm_trace", []):
            raw = entry.get("raw_response", "")
            if raw and ("```" in raw or raw.strip().startswith("I ") or raw.strip().startswith("The ")):
                json_parse_struggles.append({
                    "id": r["id"],
                    "step": entry["step_label"],
                    "response_preview": raw[:200],
                })
    diagnostics["json_parse_struggles"] = json_parse_struggles

    return diagnostics


# ---------------------------------------------------------------------------
# Confusion matrix printer
# ---------------------------------------------------------------------------

def render_confusion_matrix(confusion: dict[str, dict[str, int]]) -> str:
    cats = ["cookie", "quick_bread", "cake", "other"]
    col_w = 12
    lines = []
    header = f"{'Predicted →':>14}" + "".join(f"{c:>{col_w}}" for c in cats)
    lines.append(header)
    lines.append("  Expected ↓" + "─" * (col_w * len(cats) + 4))
    for exp in cats:
        row = f"  {exp:<14}"
        for pred in cats:
            cell = confusion.get(pred, {}).get(exp, 0)
            row += f"{cell:>{col_w}}"
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    stage1_results: list[dict],
    stage2_results: list[dict],
    metrics: dict,
    diagnostics: dict,
    backend: str,
    model: str,
    timestamp: str,
) -> str:
    """Generate a markdown evaluation report."""
    total = metrics["total_non_ambiguous"]
    acc = metrics["accuracy"]
    errors = metrics["error_count"]
    ambig = metrics["ambiguous"]
    ga = metrics["group_accuracy"]

    # Known-issue group stats
    ki = ga.get("known_issue", {})
    ki_acc = ki.get("accuracy", "N/A")

    lines = [
        f"# BakeSquad Evaluation Report",
        f"",
        f"**Generated:** {timestamp}  ",
        f"**Backend:** {backend} (`{model}`)  ",
        f"**Stage 1 cases:** {len(stage1_results)}  ",
        f"**Stage 2 cases:** {len(stage2_results)}  ",
        f"",
        f"---",
        f"",
        f"## 1. Summary of Findings",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Non-ambiguous cases | {total} |",
        f"| Exact accuracy | **{acc:.1%}** |",
        f"| Misclassifications | {errors} |",
        f"| Ambiguous cases | {ambig['total']} |",
        f"| Ambiguous within acceptable | {ambig.get('within_acceptable', 'N/A')} |",
        f"",
        f"### Known-Issue Group (TC25–TC29)",
        f"",
        f"| Group | Cases | Correct | Accuracy |",
        f"|-------|-------|---------|----------|",
    ]

    for g, v in ga.items():
        lines.append(f"| {g} | {v['total']} | {v['correct']} | {v['accuracy']:.0%} |")

    lines += [
        f"",
        f"---",
        f"",
        f"## 2. Quantitative Results",
        f"",
        f"### Confusion Matrix (non-ambiguous cases)",
        f"",
        f"```",
        render_confusion_matrix(metrics["confusion_matrix"]),
        f"```",
        f"",
        f"Rows = ground truth (expected), Columns = model prediction.",
        f"Diagonal entries are correct classifications.",
        f"",
        f"---",
        f"",
        f"## 3. Key Failure Modes",
        f"",
    ]

    patterns = metrics["error_patterns"]
    if not patterns:
        lines.append("_No error patterns detected — all non-ambiguous cases classified correctly._")
    else:
        for p in patterns:
            lines += [
                f"### {p['name'].replace('_', ' ').title()} ({p['count']} cases)",
                f"",
                f"**Affected queries:**",
            ]
            for q in p["queries"]:
                lines.append(f"- `{q}`")
            lines += [f"", f"**Analysis:** {p['description']}", f""]

    lines += [
        f"---",
        f"",
        f"## 4. Root Cause Analysis",
        f"",
        f"### A. Prompt Design Assessment",
        f"",
        f"The `query_plan_prompt` in `bakesquad/search/prompts.py` **already includes** the",
        f"correct examples:",
        f"",
        f"```",
        f'Input: "banana bread"',
        f'→ {{"category":"quick_bread", ...}}',
        f"",
        f'Input: "zucchini bread with chocolate chips"',
        f'→ {{"category":"quick_bread", ...}}',
        f"```",
        f"",
        f"The schema definition also explicitly states:",
        f"> `quick_bread includes: banana bread, zucchini bread, pumpkin bread, muffins, scones, cornbread`",
        f"",
        f"**Conclusion:** The prompt is *correctly designed*. If failures occur on these",
        f"specific queries, the root cause is **not** prompt omission — it is either:",
        f"",
        f"1. **LLM reasoning inconsistency** — the model outputs correctly in most runs",
        f"   but fails on certain runs, especially with weaker models (Groq llama-3.1-8b).",
        f"2. **Model-specific knowledge gaps** — some models associate 'banana bread' with",
        f"   artisan/yeast bread rather than quick bread, overriding the in-context examples.",
        f"3. **Temperature/sampling variance** — at temperature=0 this should be deterministic,",
        f"   but quantised local models may not behave deterministically.",
        f"",
        f"### B. Classification Schema Gaps",
        f"",
        f"The schema has **four mutually exclusive, non-overlapping categories** with no",
        f"mechanism for partial or multi-label classification.",
        f"",
        f"Gap items identified:",
        f"- `loaf` is not listed as a quick_bread synonym in the examples",
        f"- `bar cookies` (brownies, lemon bars) rely on the model knowing they are cookies",
        f"- No catch-all fallback within a category (e.g., 'if unsure between quick_bread/cake → quick_bread')",
        f"",
        f"### C. Pipeline Rigidity",
        f"",
        f"The pipeline is a **fixed 11-step DAG with no feedback loops:**",
        f"",
        f"```",
        f"Step 1 → category (FIXED) → propagates to Steps 9, 10",
        f"Step 7 → may infer a different category from ingredients",
        f"        but this is IGNORED for ratio range selection",
        f"```",
        f"",
        f"**Impact of a Step 1 misclassification:**",
        f"- Step 9 computes ratios against **wrong reference ranges** (e.g., cookie ranges for a quick bread)",
        f"- Step 10 uses **wrong scoring criteria** (e.g., brown/white sugar ratio for a loaf)",
        f"- No step corrects or revisits the category decision",
        f"- The final score is **meaningless** for misclassified recipes",
        f"",
    ]

    if diagnostics.get("step1_vs_step7_disagreements"):
        lines += [
            f"**Step 1 vs Step 7 disagreements detected:**",
            f"",
        ]
        for d in diagnostics["step1_vs_step7_disagreements"]:
            lines.append(
                f"- `{d['query']}`: Step 1 → `{d['step1_category']}`, "
                f"Step 7 → `{d['step7_category']}` (recipe: {d['recipe_title']!r})"
            )
        lines.append("")

    lines += [
        f"---",
        f"",
        f"## 5. Recommended Fixes",
        f"",
        f"### Fix 1: Prompt Hardening (Low Effort, Immediate)",
        f"",
        f"Add explicit `loaf` synonyms to the `query_plan_prompt` schema description:",
        f"",
        f"```diff",
        f"- '  quick_bread includes: banana bread, zucchini bread, pumpkin bread, muffins, scones, cornbread\\n'",
        f"+ '  quick_bread includes: banana bread, zucchini bread, pumpkin bread, muffins, scones, cornbread, '",
        f"+     'and ANY baked loaf that does not require yeast (banana nut loaf, zucchini loaf, etc.)\\n'",
        f"```",
        f"",
        f"Add a conflict-resolution tie-breaker example:",
        f"",
        f"```",
        f'Input: "banana nut loaf"',
        f'→ {{"category":"quick_bread", ...}}   # loaf ≠ yeast bread',
        f"```",
        f"",
        f"### Fix 2: Step 7 Category Reconciliation (Medium Effort)",
        f"",
        f"After Step 7 (parsing), compare the parser-inferred category against the",
        f"Step 1 query plan category. If they disagree for a majority of fetched",
        f"recipes, override the plan category before Steps 9–10.",
        f"",
        f"```python",
        f"# In run_pipeline(), after parse_recipes_parallel():",
        f"step7_categories = Counter(r.category for r in parsed if not r.parse_error)",
        f"dominant_step7 = step7_categories.most_common(1)[0][0] if step7_categories else None",
        f"if dominant_step7 and dominant_step7 != plan.category:",
        f"    logging.warning(",
        f"        'Category override: Step1=%s → Step7=%s (recipes=%s)',",
        f"        plan.category, dominant_step7, dict(step7_categories)",
        f"    )",
        f"    plan = plan.model_copy(update={{'category': dominant_step7}})",
        f"```",
        f"",
        f"### Fix 3: Soft Classification (High Effort, Best Long-Term)",
        f"",
        f"Replace the hard `Literal['cookie','quick_bread','cake','other']` with a",
        f"scored confidence dict, then select the category only when confidence is",
        f"above a threshold:",
        f"",
        f"```python",
        f"# New QueryPlan field:",
        f'category_scores: dict[str, float]  # e.g. {{"quick_bread": 0.9, "other": 0.1}}',
        f"category: str  # argmax of category_scores",
        f"",
        f"# Prompt change: ask the LLM for confidence scores instead of hard label.",
        f"# If max confidence < 0.6, log a warning and use the Step 7 re-classification.",
        f"```",
        f"",
        f"### Fix 4: Add Regression Tests (Maintenance)",
        f"",
        f"Add `pytest` tests that run the classification-only path (Stage 1) for the",
        f"five known-issue queries and assert `quick_bread`:",
        f"",
        f"```python",
        f"# tests/test_classification.py",
        f"@pytest.mark.parametrize('query', [",
        f"    'banana bread', 'zucchini bread', 'chocolate chip banana bread',",
        f"    'moist zucchini bread with walnuts', 'gluten-free banana bread',",
        f"])",
        f"def test_quick_bread_classification(query):",
        f"    system, user = query_plan_prompt(query, recency=None, trusted_sources=[])",
        f"    raw = chat(system, user)",
        f"    plan = extract_json(raw)",
        f"    assert plan['category'] == 'quick_bread', f'Got {{plan[\"category\"]}} for {{query!r}}'",
        f"```",
        f"",
        f"---",
        f"",
        f"## 6. Architecture Evaluation",
        f"",
        f"### Current Architecture: Fixed Sequential DAG",
        f"",
        f"```",
        f"Query → [Step 1: classify] → [Steps 2-6: search+fetch] → [Step 7: parse]",
        f"      → [Steps 8-9: ratios] → [Step 10: score] → Output",
        f"         ↑ category locked here; no recovery downstream",
        f"```",
        f"",
        f"**Strengths:**",
        f"- Deterministic, fast, predictable latency",
        f"- Minimal LLM calls (3–4 total per run)",
        f"- Easily debuggable (each step's output is a typed Pydantic model)",
        f"",
        f"**Weaknesses:**",
        f"- Single point of failure at Step 1",
        f"- No correction loop if Step 1 is wrong",
        f"- Step 7 re-classification data is unused",
        f"",
        f"### Alternative 1: Step 7 Feedback Loop",
        f"",
        f"```",
        f"Query → [Step 1] → [Steps 2-6] → [Step 7] ─────────────────────┐",
        f"                                      ↓                         |",
        f"                              category disagrees?               |",
        f"                              yes → override plan.category      |",
        f"                              no  → continue as-is             |",
        f"                                      ↓                         |",
        f"                            [Steps 8-9] → [Step 10] ← ──────────┘",
        f"```",
        f"",
        f"Cost: ~0 extra LLM calls. Recommended as Fix 2 above.",
        f"",
        f"### Alternative 2: Soft Classification",
        f"",
        f"Ask the LLM to return confidence scores for all categories.",
        f"Fall back to Step 7 when Step 1 confidence is low.",
        f"Cost: minor prompt change, same number of LLM calls.",
        f"",
        f"### Alternative 3: Single-Pass Reasoning (ReAct-style)",
        f"",
        f"Replace the entire pipeline with a ReAct loop that searches, reads pages,",
        f"and computes ratios as tool calls. More flexible but:",
        f"- Adds ~3–5× the LLM calls",
        f"- Unpredictable latency (would blow the 60 s budget)",
        f"- Harder to debug",
        f"- Not recommended given the tight budget constraints of this project",
        f"",
        f"**Verdict:** Fix 2 (category reconciliation) gives the best improvement-to-effort",
        f"ratio. Fix 1 (prompt hardening) is a free win that should be done immediately.",
        f"",
        f"---",
        f"",
        f"## 7. Instrumented LLM Traces",
        f"",
        f"All LLM prompts and raw responses are stored per-result in",
        f"`eval_results_<timestamp>.json` under the `llm_trace` key.",
        f"",
        f"Each trace entry contains:",
        f"- `step_label` — which pipeline step made the call",
        f"- `system` — full system prompt",
        f"- `user` — user message",
        f"- `raw_response` — unmodified LLM output (before JSON extraction)",
        f"- `elapsed_s` — latency for this call",
        f"- `error` — exception message if the call failed",
        f"",
        f"Use `--debug` flag to print traces to stdout during evaluation.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_results(
    results: list[dict],
    metrics: dict,
    diagnostics: dict,
    output_dir: Path,
    timestamp: str,
    stage: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "metadata": {
            "timestamp": timestamp,
            "stage": stage,
            "backend": os.environ.get("MODEL_BACKEND", "ollama"),
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
        "results": results,
    }
    path = output_dir / f"eval_results_{stage}_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def save_report(report: str, output_dir: Path, timestamp: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"eval_report_{timestamp}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return path


def save_confusion_matrix(confusion: dict, output_dir: Path, timestamp: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"confusion_matrix_{timestamp}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_confusion_matrix(confusion))
    return path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BakeSquad Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_HERE / "eval_dataset.json",
        help="Path to evaluation dataset JSON (default: evaluation/eval_dataset.json)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print each LLM prompt + raw response to stdout",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Also run Stage 2 (full 11-step pipeline) on a subset of cases",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        metavar="GROUP",
        help=(
            "Only run cases from these groups. "
            "Available: clear_cookie, clear_cake, clear_quick_bread, "
            "known_issue, edge_case, clear_other, ambiguous"
        ),
    )
    parser.add_argument(
        "--stage2-max",
        type=int,
        default=10,
        metavar="N",
        help="Max cases for Stage 2 full-pipeline run (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_HERE / "results",
        help="Directory to write output files (default: evaluation/results/)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Backend info
    backend = os.environ.get("MODEL_BACKEND", "ollama")
    try:
        model = llm_mod.get_model()
    except Exception:
        model = "unknown"

    print(f"\nBakeSquad Evaluation Framework")
    print(f"  Backend : {backend} ({model})")
    print(f"  Dataset : {args.dataset}")
    print(f"  Output  : {args.output_dir}")
    print(f"  Debug   : {args.debug}")
    print(f"  Timestamp: {timestamp}")

    # Load dataset
    test_cases = load_dataset(args.dataset)
    print(f"\n  Loaded {len(test_cases)} test cases from {args.dataset.name}")

    # Stage 1
    s1_results = run_stage1(test_cases, debug=args.debug, groups=args.groups)
    s1_metrics = compute_metrics(s1_results)
    s1_diagnostics = diagnose_pipeline_issues(s1_results)

    print(f"\n  Stage 1 Accuracy: {s1_metrics['accuracy']:.1%} "
          f"({s1_metrics['correct_exact']}/{s1_metrics['total_non_ambiguous']} non-ambiguous)")

    ki = s1_metrics["group_accuracy"].get("known_issue", {})
    if ki:
        print(f"  Known-Issue Group: {ki['accuracy']:.0%} ({ki['correct']}/{ki['total']})")

    # Stage 2 (optional)
    s2_results: list[dict] = []
    s2_metrics: dict = {}
    s2_diagnostics: dict = {}
    if args.full_pipeline:
        stage2_cases = [tc for tc in test_cases if not tc.get("ambiguous")]
        s2_results = run_stage2(
            stage2_cases,
            debug=args.debug,
            groups=args.groups,
            max_cases=args.stage2_max,
        )
        s2_metrics = compute_metrics(s2_results)
        s2_diagnostics = diagnose_pipeline_issues(s2_results)
        print(f"\n  Stage 2 Accuracy: {s2_metrics['accuracy']:.1%} "
              f"({s2_metrics['correct_exact']}/{s2_metrics['total_non_ambiguous']})")

    # Save outputs
    s1_path = save_results(s1_results, s1_metrics, s1_diagnostics, args.output_dir, timestamp, "s1")
    print(f"\n  Stage 1 results saved: {s1_path}")

    if s2_results:
        s2_path = save_results(
            s2_results, s2_metrics, s2_diagnostics, args.output_dir, timestamp, "s2"
        )
        print(f"  Stage 2 results saved: {s2_path}")

    report = generate_report(
        s1_results,
        s2_results,
        s1_metrics,
        s1_diagnostics,
        backend,
        model,
        timestamp,
    )
    report_path = save_report(report, args.output_dir, timestamp)
    print(f"  Report saved        : {report_path}")

    cm_path = save_confusion_matrix(s1_metrics["confusion_matrix"], args.output_dir, timestamp)
    print(f"  Confusion matrix    : {cm_path}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE")
    print(f"  Stage 1 accuracy: {s1_metrics['accuracy']:.1%}")
    if ki:
        ki_result = "PASS" if ki["accuracy"] == 1.0 else "FAIL"
        print(f"  Known-issue cases: {ki_result} ({ki['accuracy']:.0%})")

    patterns = s1_metrics.get("error_patterns", [])
    if patterns:
        print(f"\n  Failure patterns detected:")
        for p in patterns:
            print(f"    - {p['name']}: {p['count']} case(s)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
