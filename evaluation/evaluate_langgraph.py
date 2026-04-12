"""
BakeSquad LangGraph Evaluation Framework

Compares the fixed-DAG architecture against the LangGraph architecture across
four evaluation stages:

  Stage 1A  Classification accuracy          (fast, no network)
  Stage 1B  Preference/constraint extraction recall   (fast, no network)
  Stage 1C  Query specificity analysis        (derived from 1A+1B, no network)
  Stage 1D  Turn-classification routing       (fast, no network)
  Stage 2   Scoring transparency              (--full-pipeline, network + API)
  Stage 3   DAG vs LangGraph comparison       (--compare-langgraph, requires langgraph)

Usage:
  # Stage 1 only (fast, ~2 min on Groq):
  MODEL_BACKEND=groq python evaluation/evaluate_langgraph.py

  # Stage 1 + 2 scoring transparency (5 pipeline runs):
  MODEL_BACKEND=groq python evaluation/evaluate_langgraph.py --full-pipeline

  # All stages including LangGraph comparison:
  MODEL_BACKEND=groq python evaluation/evaluate_langgraph.py --full-pipeline --compare-langgraph

  # Debug: print every LLM prompt + raw response:
  MODEL_BACKEND=groq python evaluation/evaluate_langgraph.py --debug

  # Filter to specific groups:
  MODEL_BACKEND=groq python evaluation/evaluate_langgraph.py --groups preference_rich multi_turn
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ.setdefault("USER_AGENT", "BakeSquad-Eval/2.0")

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_EVAL_DIR   = Path(__file__).parent
_DATASET_V2 = _EVAL_DIR / "eval_dataset_v2.json"
_DATASET_V1 = _EVAL_DIR / "eval_dataset.json"


# ---------------------------------------------------------------------------
# LLM instrumentation (mirrors evaluate_agent.py)
# ---------------------------------------------------------------------------

_llm_trace: list[dict] = []

def _install_chat_instrumentation() -> None:
    import bakesquad.llm_client as llm_mod
    _orig = llm_mod.chat

    def _instrumented(system, user, *, temperature=0, max_tokens=2048, _step_label="unknown"):
        t0 = time.monotonic()
        raw, err = "", None
        try:
            raw = _orig(system, user, temperature=temperature, max_tokens=max_tokens)
            return raw
        except Exception as exc:
            err = str(exc)
            raise
        finally:
            _llm_trace.append({
                "step_label":   _step_label,
                "system":       system[:800],
                "user":         user[:400],
                "raw_response": raw[:600],
                "elapsed_s":    round(time.monotonic() - t0, 3),
                "error":        err,
            })

    llm_mod.chat = _instrumented


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]


def _filter_cases(cases: list[dict], groups: Optional[list[str]]) -> list[dict]:
    if not groups:
        return cases
    return [c for c in cases if c.get("group") in groups]


# ---------------------------------------------------------------------------
# Stage 1A  —  Classification accuracy
# ---------------------------------------------------------------------------

def _run_classification(case: dict, debug: bool) -> dict:
    """Run Step 1 QueryPlan extraction for one test case."""
    from bakesquad.llm_client import chat, extract_json
    from bakesquad.search.prompts import query_plan_prompt

    query = case["query"]
    system, user = query_plan_prompt(query, None, [])
    t0 = time.monotonic()
    raw, error, predicted = "", None, None

    try:
        raw = chat(system, user, temperature=0, max_tokens=512)
        data = extract_json(raw)
        predicted = data.get("category", "PARSE_ERROR") if isinstance(data, dict) else "PARSE_ERROR"
        plan_data = data if isinstance(data, dict) else {}
    except Exception as exc:
        error = str(exc)
        predicted = "ERROR"
        plan_data = {}

    elapsed = round(time.monotonic() - t0, 2)
    expected = case.get("expected_category")
    correct = (predicted == expected) if expected else None

    if debug:
        mark = "OK" if correct else "FAIL"
        print(f"  [{case['id']}] {mark}  got={predicted!r}  expected={expected!r}  ({elapsed}s)")

    return {
        "id":               case["id"],
        "query":            query,
        "group":            case.get("group"),
        "expected_category": expected,
        "predicted_category": predicted,
        "correct":          correct,
        "plan_data":        plan_data,
        "elapsed_s":        elapsed,
        "error":            error,
        "llm_trace":        list(_llm_trace[-1:]),
    }


def run_stage1a(cases: list[dict], debug: bool) -> list[dict]:
    """Stage 1A: classification accuracy across all cases that have expected_category."""
    scoreable = [c for c in cases if c.get("expected_category") and "turns" not in c]
    print(f"\n[Stage 1A] Classification accuracy — {len(scoreable)} cases")
    results = []
    for i, case in enumerate(scoreable, 1):
        print(f"  {i}/{len(scoreable)}  {case['id']}  {case['query'][:55]}", end="  ")
        r = _run_classification(case, debug)
        mark = "OK" if r["correct"] else "FAIL" if r["correct"] is False else "—"
        print(mark)
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Stage 1B  —  Preference / constraint extraction recall
# ---------------------------------------------------------------------------

def _check_keywords(extracted_list: list[str], expected_keywords: list[str]) -> dict:
    """
    Check how many expected_keywords appear as substrings in any extracted string.
    Returns: {found: list, missed: list, recall: float}
    """
    if not expected_keywords:
        return {"found": [], "missed": [], "recall": 1.0}
    haystack = " ".join(extracted_list).lower()
    found = [kw for kw in expected_keywords if kw.lower() in haystack]
    missed = [kw for kw in expected_keywords if kw.lower() not in haystack]
    return {
        "found":  found,
        "missed": missed,
        "recall": round(len(found) / len(expected_keywords), 3),
    }


def run_stage1b(stage1a_results: list[dict], cases_by_id: dict, debug: bool) -> list[dict]:
    """
    Stage 1B: preference/constraint extraction recall.
    Operates on cases that have expected_constraint_keywords or expected_preference_keywords.
    Uses the plan_data already extracted in Stage 1A — no extra LLM calls.
    """
    pref_cases = [
        c for c in cases_by_id.values()
        if c.get("expected_constraint_keywords") is not None
        or c.get("expected_preference_keywords") is not None
        or c.get("expected_modifiers") is not None
        or c.get("expected_flour_type") is not None
    ]
    if not pref_cases:
        print("\n[Stage 1B] No preference_rich cases in selected groups — skipped")
        return []

    print(f"\n[Stage 1B] Preference/constraint extraction recall — {len(pref_cases)} cases")

    # Build a lookup from id → plan_data from Stage 1A results
    plan_by_id = {r["id"]: r.get("plan_data", {}) for r in stage1a_results}

    results = []
    for case in pref_cases:
        case_id = case["id"]
        plan = plan_by_id.get(case_id, {})
        if not plan:
            # Case wasn't in Stage 1A (different group filter); run it now
            r1a = _run_classification(case, debug=False)
            plan = r1a.get("plan_data", {})

        hard_constraints  = plan.get("hard_constraints", []) or []
        soft_preferences  = plan.get("soft_preferences",  []) or []
        extracted_mods    = plan.get("modifiers",          []) or []
        extracted_flour   = plan.get("flour_type", "ap")   or "ap"

        constraint_check = _check_keywords(
            hard_constraints, case.get("expected_constraint_keywords") or []
        )
        preference_check = _check_keywords(
            soft_preferences, case.get("expected_preference_keywords") or []
        )

        expected_mods = case.get("expected_modifiers") or []
        mod_match = all(m in extracted_mods for m in expected_mods)
        mod_missing = [m for m in expected_mods if m not in extracted_mods]

        expected_flour = case.get("expected_flour_type")
        flour_correct = (extracted_flour == expected_flour) if expected_flour else None

        result = {
            "id":                   case_id,
            "query":                case["query"],
            "group":                case.get("group"),
            "constraint_recall":    constraint_check["recall"],
            "constraint_found":     constraint_check["found"],
            "constraint_missed":    constraint_check["missed"],
            "preference_recall":    preference_check["recall"],
            "preference_found":     preference_check["found"],
            "preference_missed":    preference_check["missed"],
            "modifier_match":       mod_match,
            "modifier_missing":     mod_missing,
            "flour_type_correct":   flour_correct,
            "extracted_flour_type": extracted_flour,
            "expected_flour_type":  expected_flour,
        }

        if debug:
            cr = constraint_check["recall"]
            pr = preference_check["recall"]
            print(f"  [{case_id}] constraint_recall={cr:.0%}  pref_recall={pr:.0%}  flour={'OK' if flour_correct else 'FAIL' if flour_correct is False else '—'}")

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Stage 1C  —  Query specificity analysis
# ---------------------------------------------------------------------------

def run_stage1c(stage1a_results: list[dict], stage1b_results: list[dict]) -> dict:
    """
    Stage 1C: compare extraction completeness across short / medium / long queries.
    Derived from Stage 1A plan_data — no extra calls needed.
    """
    print("\n[Stage 1C] Query specificity analysis")

    groups_of_interest = ["short_query", "medium_query", "long_query"]
    by_group: dict[str, list] = {g: [] for g in groups_of_interest}

    # Collect plan_data for each specificity group
    for r in stage1a_results:
        g = r.get("group")
        if g in by_group:
            by_group[g].append(r)

    # Constraint/preference recall from Stage 1B, keyed by id
    recall_by_id = {r["id"]: r for r in stage1b_results}

    analysis: dict[str, dict] = {}
    for group in groups_of_interest:
        cases = by_group[group]
        if not cases:
            continue

        n_constraints_list, n_prefs_list = [], []
        c_recalls, p_recalls = [], []

        for r in cases:
            plan = r.get("plan_data", {})
            n_constraints_list.append(len(plan.get("hard_constraints") or []))
            n_prefs_list.append(len(plan.get("soft_preferences") or []))
            if r["id"] in recall_by_id:
                rb = recall_by_id[r["id"]]
                c_recalls.append(rb["constraint_recall"])
                p_recalls.append(rb["preference_recall"])

        analysis[group] = {
            "n":                    len(cases),
            "avg_constraints":      round(sum(n_constraints_list) / len(cases), 2),
            "avg_preferences":      round(sum(n_prefs_list) / len(cases), 2),
            "avg_constraint_recall": round(sum(c_recalls) / len(c_recalls), 3) if c_recalls else None,
            "avg_preference_recall": round(sum(p_recalls) / len(p_recalls), 3) if p_recalls else None,
        }

        a = analysis[group]
        print(f"  {group:<15}  n={a['n']}  avg_constraints={a['avg_constraints']:.1f}  "
              f"avg_preferences={a['avg_preferences']:.1f}  "
              f"constraint_recall={a['avg_constraint_recall'] or 'n/a'}  "
              f"preference_recall={a['avg_preference_recall'] or 'n/a'}")

    return analysis


# ---------------------------------------------------------------------------
# Stage 1D  —  Turn classification (classify_intent routing)
# ---------------------------------------------------------------------------

def _make_mock_session(query: str, category: str = "cookie"):
    """Build a minimal ConversationSession for turn-classification testing."""
    from bakesquad.models import (
        CriterionScore, ParsedRecipe, QueryPlan,
        RatioResult, RecipeIngredient, ScoredRecipe,
    )
    from bakesquad.session import ConversationSession

    plan = QueryPlan(
        category=category,
        hard_constraints=[],
        soft_preferences=[],
        queries=[query, query + " recipe"],
    )
    recipes, ratios_list, scored_list = [], [], []
    for i in range(3):
        recipe = ParsedRecipe(
            title=f"Mock Recipe {i + 1}",
            url=f"https://example.com/recipe-{i + 1}",
            category=category,
            ingredients=[
                RecipeIngredient(name="all-purpose flour", quantity=2.0, unit="cups"),
                RecipeIngredient(name="butter", quantity=0.5, unit="cups"),
                RecipeIngredient(name="sugar", quantity=1.0, unit="cups"),
            ],
        )
        ratio = RatioResult(
            url=recipe.url,
            category=category,
            fat_type="butter",
            fat_to_flour=0.55,
            sugar_to_flour=0.75,
            leavening_to_flour=0.015,
        )
        score = ScoredRecipe(
            recipe=recipe,
            ratios=ratio,
            criteria=[
                CriterionScore(name="Moisture Retention",    score=70.0, weight=0.45),
                CriterionScore(name="Structure & Leavening", score=65.0, weight=0.30),
                CriterionScore(name="Sugar Balance",         score=80.0, weight=0.25),
            ],
            composite_score=72.0 - i * 3,
            rank=i + 1,
        )
        recipes.append(recipe)
        ratios_list.append(ratio)
        scored_list.append(score)

    session = ConversationSession(original_query=query)
    session.update_results(plan, recipes, ratios_list, scored_list)
    return session


def run_stage1d(cases: list[dict], debug: bool) -> list[dict]:
    """Stage 1D: test classify_intent turn routing for multi-turn sequences."""
    from bakesquad.session import classify_turn

    mt_cases = [c for c in cases if c.get("group") == "multi_turn" and "turns" in c]
    if not mt_cases:
        print("\n[Stage 1D] No multi_turn cases in selected groups — skipped")
        return []

    print(f"\n[Stage 1D] Turn classification routing — {len(mt_cases)} sequences")
    results = []

    for case in mt_cases:
        seq_result = {
            "id":          case["id"],
            "description": case.get("description", ""),
            "turns":       [],
            "all_correct": True,
        }
        turns = case["turns"]
        # First turn is always new_search (no LLM call needed — just the ground truth check)
        first_turn = turns[0]
        seq_result["turns"].append({
            "content":             first_turn["content"],
            "expected_turn_type":  first_turn.get("expected_turn_type"),
            "predicted_turn_type": "new_search",
            "correct":             first_turn.get("expected_turn_type") == "new_search",
            "note":                "first turn always new_search",
        })

        # Build a mock session seeded with the first turn's topic
        session = _make_mock_session(first_turn["content"])
        session.add_user(first_turn["content"])
        session.add_assistant(f"Returned 3 recipes for: {first_turn['content']!r}.")

        for turn in turns[1:]:
            user_msg = turn["content"]
            expected_type = turn.get("expected_turn_type")
            expected_extract = turn.get("expected_extract", {})

            t0 = time.monotonic()
            try:
                refine = classify_turn(session, user_msg)
                predicted = refine.get("turn_type", "ERROR")
                elapsed = round(time.monotonic() - t0, 2)
            except Exception as exc:
                predicted = "ERROR"
                refine = {}
                elapsed = round(time.monotonic() - t0, 2)

            correct = (predicted == expected_type) if expected_type else None

            # Check extracted fields for re_filter turns
            extract_checks = {}
            if expected_extract and predicted == "re_filter":
                for key, expected_val in expected_extract.items():
                    actual_val = refine.get(key)
                    if isinstance(expected_val, list):
                        # Check all expected items appear somewhere in the extracted list
                        actual_lower = [str(v).lower() for v in (actual_val or [])]
                        match = all(
                            any(ev.lower() in av for av in actual_lower)
                            for ev in expected_val
                        )
                    else:
                        match = str(actual_val).lower() == str(expected_val).lower()
                    extract_checks[key] = {"expected": expected_val, "actual": actual_val, "match": match}

            if not correct:
                seq_result["all_correct"] = False

            if debug:
                mark = "OK" if correct else "FAIL"
                print(f"  [{case['id']}] turn={user_msg[:40]!r}  {mark}  predicted={predicted!r}  expected={expected_type!r}")

            seq_result["turns"].append({
                "content":             user_msg,
                "expected_turn_type":  expected_type,
                "predicted_turn_type": predicted,
                "correct":             correct,
                "extract_checks":      extract_checks,
                "elapsed_s":           elapsed,
            })

            # Advance session for next turn
            session.add_user(user_msg)
            session.add_assistant(f"Handled turn: {predicted}.")

        mark = "OK" if seq_result["all_correct"] else "FAIL"
        print(f"  {case['id']}  {case.get('description', '')}  [{mark}]")
        results.append(seq_result)

    return results


# ---------------------------------------------------------------------------
# Stage 2  —  Scoring transparency (full pipeline)
# ---------------------------------------------------------------------------

def _build_scoring_trace(
    scored_recipe,
    category_used: str,
    category_source: str,
    user_prefs: dict,
) -> dict:
    """Build a detailed scoring trace dict from a ScoredRecipe."""
    from bakesquad.scorer import TECHNIQUE_SYNERGIES

    # Determine weight source
    cat_prefs = (user_prefs.get("category_prefs") or {}).get(category_used, {})
    if cat_prefs and any(k in cat_prefs for k in ("moisture_base_weight", "structure_base_weight")):
        weight_source = "category_prefs"
    elif any(
        user_prefs.get(k) != default
        for k, default in [
            ("moisture_base_weight", 0.45),
            ("structure_base_weight", 0.30),
            ("balance_base_weight", 0.25),
        ]
    ):
        weight_source = "global_prefs"
    else:
        weight_source = "defaults"

    # Build per-criterion detail
    technique_signals = list(getattr(scored_recipe.recipe, "technique_signals", []) or [])
    technique_notes   = str(getattr(scored_recipe.recipe, "technique_notes", "") or "")
    note_delta        = scored_recipe.technique_note_delta

    # Identify which synergies fired
    signal_set = set(technique_signals)
    synergies_triggered = []
    for required, bonus in TECHNIQUE_SYNERGIES.get(category_used, []):
        if required.issubset(signal_set):
            synergies_triggered.append({
                "signals": sorted(required),
                "bonus":   bonus,
            })

    criteria_detail = []
    for c in scored_recipe.criteria:
        entry: dict[str, Any] = {
            "name":    c.name,
            "score":   c.score,
            "weight":  c.weight,
            "details": c.details,
        }
        if c.name == "Technique Quality":
            base = round(c.score - (note_delta or 0.0), 1) if note_delta else c.score
            entry.update({
                "technique_signals":    technique_signals,
                "technique_notes":      technique_notes,
                "technique_base_score": base,
                "technique_note_delta": note_delta,
                "synergies_triggered":  synergies_triggered,
            })
        criteria_detail.append(entry)

    return {
        "url":                  scored_recipe.recipe.url,
        "title":                scored_recipe.recipe.title,
        "rank":                 scored_recipe.rank,
        "composite_score":      scored_recipe.composite_score,
        "category_used":        category_used,
        "category_source":      category_source,
        "weight_source":        weight_source,
        "criteria":             criteria_detail,
        "constraint_violations": scored_recipe.constraint_violations,
        "explanation":          scored_recipe.explanation,
    }


def run_stage2(cases: list[dict], max_cases: int, debug: bool) -> list[dict]:
    """Stage 2: full pipeline runs with complete scoring transparency capture."""
    from collections import Counter

    from bakesquad.memory import init_db, load_prefs
    from bakesquad.parser import parse_recipes_parallel
    from bakesquad.ratio_engine import compute_ratios
    from bakesquad.scorer import add_explanations, score_all
    from bakesquad.search.ingestion import IngestionPipeline

    init_db()
    user_prefs = load_prefs()

    # Run on scoring_transparency group first, then fill from other groups
    st_cases = [c for c in cases if c.get("group") == "scoring_transparency" and "turns" not in c]
    other_cases = [c for c in cases if c.get("group") != "scoring_transparency" and "turns" not in c and c.get("expected_category")]
    run_cases = (st_cases + other_cases)[:max_cases]

    if not run_cases:
        print("\n[Stage 2] No eligible cases — skipped")
        return []

    print(f"\n[Stage 2] Scoring transparency — {len(run_cases)} pipeline runs")
    results = []

    for i, case in enumerate(run_cases, 1):
        query = case["query"]
        print(f"\n  [{i}/{len(run_cases)}] {case['id']}  {query[:60]}")
        pipeline = IngestionPipeline(trusted_sources=[])
        run_result: dict[str, Any] = {
            "id":    case["id"],
            "query": query,
            "group": case.get("group"),
            "expected_technique_signals": case.get("expected_technique_signals"),
            "error": None,
        }

        try:
            # Step 1
            plan = pipeline._build_query_plan(query, None)
            print(f"    Step 1  category={plan.category}")

            # Steps 2–5
            candidates = pipeline._search_and_filter(query, plan.queries, None)
            print(f"    Step 2-5  {len(candidates)} candidates")
            if not candidates:
                run_result["error"] = "no candidates"
                results.append(run_result)
                continue

            # Step 6
            pages = pipeline._fetch_pages(candidates)
            print(f"    Step 6  {len(pages)} pages fetched")
            if not pages:
                run_result["error"] = "no pages fetched"
                results.append(run_result)
                continue

            # Step 7
            recipes = parse_recipes_parallel(pages)
            print(f"    Step 7  {len(recipes)} recipes parsed")
            if not recipes:
                run_result["error"] = "no recipes parsed"
                results.append(run_result)
                continue

            # Step 7b: category reconciliation
            from collections import Counter as _Counter
            step7_cats = _Counter(
                r.category for r in recipes
                if getattr(r, "category", None) and r.category != "other" and not r.parse_error
            )
            category_override = None
            category_source = "step1"
            if step7_cats:
                dominant, count = step7_cats.most_common(1)[0]
                if dominant != plan.category and count >= len(recipes) / 2:
                    category_override = dominant
                    category_source = "step7_override"
                    print(f"    Step 7b  category override: {plan.category} → {dominant}")

            effective_plan = (
                plan.model_copy(update={"category": category_override})
                if category_override else plan
            )
            category_used = effective_plan.category

            # Steps 8–9
            ratios_list = [compute_ratios(r) for r in recipes]

            # Step 10
            scored = score_all(recipes, ratios_list, effective_plan, user_prefs)
            add_explanations(scored)
            print(f"    Step 10  {len(scored)} recipes scored — top: {scored[0].recipe.title[:50]!r} ({scored[0].composite_score:.0f}/100)")

            # Build scoring traces
            scoring_traces = [
                _build_scoring_trace(s, category_used, category_source, user_prefs)
                for s in scored
            ]

            # Technique signal coverage analysis
            all_signals: list[str] = []
            all_notes:   list[str] = []
            delta_events: list[dict] = []
            for s in scored:
                sigs = list(getattr(s.recipe, "technique_signals", []) or [])
                notes = str(getattr(s.recipe, "technique_notes", "") or "")
                all_signals.extend(sigs)
                if notes:
                    all_notes.append(notes)
                if s.technique_note_delta is not None:
                    delta_events.append({
                        "title":  s.recipe.title,
                        "notes":  notes,
                        "delta":  s.technique_note_delta,
                    })

            # Compare expected vs extracted technique signals (scoring_transparency cases)
            expected_sigs = case.get("expected_technique_signals") or []
            signal_match: Optional[dict] = None
            if expected_sigs:
                found = [s for s in expected_sigs if s in all_signals]
                signal_match = {
                    "expected": expected_sigs,
                    "found":    found,
                    "missed":   [s for s in expected_sigs if s not in all_signals],
                    "recall":   round(len(found) / len(expected_sigs), 3),
                }

            run_result.update({
                "plan": {
                    "category":        plan.category,
                    "flour_type":      plan.flour_type,
                    "modifiers":       plan.modifiers,
                    "hard_constraints": plan.hard_constraints,
                    "soft_preferences": plan.soft_preferences,
                },
                "category_override":        category_override,
                "category_source":          category_source,
                "recipes_scored":           len(scored),
                "top_recipe":               scored[0].recipe.title if scored else None,
                "top_composite_score":      scored[0].composite_score if scored else None,
                "scoring_traces":           scoring_traces,
                "technique_signals_found":  sorted(set(all_signals)),
                "technique_notes_found":    all_notes,
                "technique_delta_events":   delta_events,
                "signal_match":             signal_match,
            })

        except Exception as exc:
            import traceback
            run_result["error"] = str(exc)
            if debug:
                traceback.print_exc()
            print(f"    ERROR: {exc}")

        results.append(run_result)

    return results


# ---------------------------------------------------------------------------
# Stage 3  —  DAG vs LangGraph comparison
# ---------------------------------------------------------------------------

def run_stage3(cases: list[dict], max_cases: int, debug: bool) -> list[dict]:
    """
    Stage 3: run the same queries through both the fixed DAG and the LangGraph graph,
    then compare outputs (category, top recipe, composite score, timing).
    """
    try:
        from bakesquad.graph.builder import build_graph
    except ImportError as e:
        print(f"\n[Stage 3] Skipped — langgraph not installed: {e}")
        return []

    from bakesquad.memory import init_db, load_prefs
    from bakesquad.parser import parse_recipes_parallel
    from bakesquad.ratio_engine import compute_ratios
    from bakesquad.scorer import add_explanations, score_all
    from bakesquad.search.ingestion import IngestionPipeline

    init_db()
    user_prefs = load_prefs()

    run_cases = [c for c in cases if c.get("expected_category") and "turns" not in c][:max_cases]
    if not run_cases:
        print("\n[Stage 3] No eligible cases — skipped")
        return []

    print(f"\n[Stage 3] DAG vs LangGraph comparison — {len(run_cases)} cases")

    try:
        graph = build_graph(checkpointing=False)
        print("  LangGraph graph compiled OK")
    except Exception as e:
        print(f"  LangGraph graph build failed: {e}")
        return []

    results = []
    for i, case in enumerate(run_cases, 1):
        query = case["query"]
        print(f"\n  [{i}/{len(run_cases)}] {case['id']}  {query[:55]}")
        cmp: dict[str, Any] = {"id": case["id"], "query": query, "dag": {}, "langgraph": {}, "comparison": {}}

        # ── Fixed DAG run ──────────────────────────────────────────────────
        try:
            t0 = time.monotonic()
            pipeline = IngestionPipeline(trusted_sources=[])
            plan = pipeline._build_query_plan(query, None)
            candidates = pipeline._search_and_filter(query, plan.queries, None)
            pages = pipeline._fetch_pages(candidates) if candidates else []
            recipes = parse_recipes_parallel(pages) if pages else []

            # Step 7b reconciliation
            from collections import Counter as _Counter
            step7_cats = _Counter(
                r.category for r in recipes
                if getattr(r, "category", None) and r.category != "other" and not r.parse_error
            )
            if step7_cats:
                dominant, count = step7_cats.most_common(1)[0]
                if dominant != plan.category and count >= len(recipes) / 2:
                    plan = plan.model_copy(update={"category": dominant})

            ratios_list = [compute_ratios(r) for r in recipes] if recipes else []
            scored = score_all(recipes, ratios_list, plan, user_prefs) if recipes else []
            if scored:
                add_explanations(scored)

            cmp["dag"] = {
                "category":         plan.category,
                "top_recipe_url":   scored[0].recipe.url if scored else None,
                "top_recipe_title": scored[0].recipe.title if scored else None,
                "top_score":        scored[0].composite_score if scored else None,
                "recipes_found":    len(scored),
                "elapsed_s":        round(time.monotonic() - t0, 1),
            }
            print(f"    DAG:       category={plan.category}  top={scored[0].composite_score:.0f}/100  ({cmp['dag']['elapsed_s']}s)")
        except Exception as exc:
            cmp["dag"]["error"] = str(exc)
            print(f"    DAG ERROR: {exc}")

        # ── LangGraph run ──────────────────────────────────────────────────
        try:
            import uuid
            t0 = time.monotonic()
            thread_id = str(uuid.uuid4())
            initial_state = {
                "thread_id":          thread_id,
                "user_query":         query,
                "messages":           [{"role": "user", "content": query}],
                "turn_type":          None,
                "query_plan":         None,
                "category_confidence": 1.0,
                "clarify_question":   None,
                "recency":            None,
                "snippets":           [],
                "fetched_pages":      [],
                "search_retry_count": 0,
                "parsed_recipes":     [],
                "category_override":  None,
                "ratio_results":      [],
                "scored_recipes":     [],
                "exclude_ingredients": [],
                "require_fat_type":   None,
                "factual_answer":     None,
                "user_prefs":         user_prefs,
                "liked_recipe_urls":  [],
                "errors":             [],
            }
            config = {"configurable": {"thread_id": thread_id}}
            final_state = graph.invoke(initial_state, config=config)

            lg_scored = final_state.get("scored_recipes") or []
            lg_plan   = final_state.get("query_plan")
            lg_override = final_state.get("category_override")
            lg_category = lg_override or (lg_plan.category if lg_plan else None)

            cmp["langgraph"] = {
                "category":           lg_category,
                "category_override":  lg_override,
                "top_recipe_url":     lg_scored[0].recipe.url if lg_scored else None,
                "top_recipe_title":   lg_scored[0].recipe.title if lg_scored else None,
                "top_score":          lg_scored[0].composite_score if lg_scored else None,
                "recipes_found":      len(lg_scored),
                "elapsed_s":          round(time.monotonic() - t0, 1),
            }
            print(f"    LangGraph: category={lg_category}  top={lg_scored[0].composite_score:.0f}/100  ({cmp['langgraph']['elapsed_s']}s)")
        except Exception as exc:
            cmp["langgraph"]["error"] = str(exc)
            if debug:
                import traceback; traceback.print_exc()
            print(f"    LangGraph ERROR: {exc}")

        # ── Comparison ─────────────────────────────────────────────────────
        dag = cmp["dag"]
        lg  = cmp["langgraph"]
        if not dag.get("error") and not lg.get("error"):
            cat_match   = dag.get("category") == lg.get("category")
            url_match   = dag.get("top_recipe_url") == lg.get("top_recipe_url")
            score_delta = (
                round(abs((lg.get("top_score") or 0) - (dag.get("top_score") or 0)), 1)
                if dag.get("top_score") is not None and lg.get("top_score") is not None
                else None
            )
            timing_delta = (
                round((lg.get("elapsed_s") or 0) - (dag.get("elapsed_s") or 0), 1)
                if dag.get("elapsed_s") is not None and lg.get("elapsed_s") is not None
                else None
            )
            cmp["comparison"] = {
                "category_match":   cat_match,
                "top_url_match":    url_match,
                "score_delta":      score_delta,
                "timing_delta_s":   timing_delta,
            }
            print(f"    Comparison: cat_match={'YES' if cat_match else 'NO'}  url_match={'YES' if url_match else 'NO'}  score_delta={score_delta}  timing_delta={timing_delta}s")

        results.append(cmp)

    return results


# ---------------------------------------------------------------------------
# Metrics + report generation
# ---------------------------------------------------------------------------

def compute_stage1a_metrics(results: list[dict]) -> dict:
    valid    = [r for r in results if r["correct"] is not None]
    correct  = [r for r in valid if r["correct"]]
    by_group: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in valid:
        g = r.get("group", "unknown")
        by_group[g]["total"] += 1
        if r["correct"]:
            by_group[g]["correct"] += 1
    return {
        "total":        len(valid),
        "correct":      len(correct),
        "accuracy":     round(len(correct) / len(valid), 3) if valid else 0,
        "by_group":     {g: {"total": v["total"], "correct": v["correct"],
                              "accuracy": round(v["correct"] / v["total"], 3)}
                         for g, v in by_group.items()},
    }


def compute_stage1b_metrics(results: list[dict]) -> dict:
    if not results:
        return {}
    c_recalls = [r["constraint_recall"] for r in results if r.get("expected_constraint_keywords") is not None or r.get("constraint_found") is not None]
    p_recalls = [r["preference_recall"]  for r in results if r.get("expected_preference_keywords") is not None or r.get("preference_found") is not None]
    # Note: We get all results, filter only those that actually had expected values
    all_flour_checks  = [r["flour_type_correct"] for r in results if r.get("flour_type_correct") is not None]
    return {
        "n":                        len(results),
        "avg_constraint_recall":    round(sum(c_recalls) / len(c_recalls), 3) if c_recalls else None,
        "avg_preference_recall":    round(sum(p_recalls) / len(p_recalls), 3) if p_recalls else None,
        "flour_type_accuracy":      round(sum(all_flour_checks) / len(all_flour_checks), 3) if all_flour_checks else None,
    }


def compute_stage2_metrics(results: list[dict]) -> dict:
    ok = [r for r in results if not r.get("error")]
    all_signals = []
    note_events = 0
    reconciliation_events = 0
    for r in ok:
        all_signals.extend(r.get("technique_signals_found") or [])
        note_events += len(r.get("technique_delta_events") or [])
        if r.get("category_override"):
            reconciliation_events += 1
    from collections import Counter
    signal_counts = Counter(all_signals)
    # Technique signal recall (scoring_transparency cases only)
    st = [r for r in ok if r.get("signal_match")]
    avg_signal_recall = (
        round(sum(r["signal_match"]["recall"] for r in st) / len(st), 3) if st else None
    )
    return {
        "runs_ok":                  len(ok),
        "runs_error":               len(results) - len(ok),
        "top_technique_signals":    signal_counts.most_common(10),
        "technique_delta_events":   note_events,
        "category_reconciliations": reconciliation_events,
        "avg_signal_recall":        avg_signal_recall,
    }


def compute_stage3_metrics(results: list[dict]) -> dict:
    valid = [r for r in results if not r.get("dag", {}).get("error") and not r.get("langgraph", {}).get("error")]
    if not valid:
        return {}
    cat_matches   = sum(1 for r in valid if r.get("comparison", {}).get("category_match"))
    url_matches   = sum(1 for r in valid if r.get("comparison", {}).get("top_url_match"))
    score_deltas  = [r["comparison"]["score_delta"] for r in valid if r.get("comparison", {}).get("score_delta") is not None]
    timing_deltas = [r["comparison"]["timing_delta_s"] for r in valid if r.get("comparison", {}).get("timing_delta_s") is not None]
    return {
        "n":                    len(valid),
        "category_match_rate":  round(cat_matches / len(valid), 3),
        "top_url_match_rate":   round(url_matches / len(valid), 3),
        "avg_score_delta":      round(sum(score_deltas) / len(score_deltas), 2) if score_deltas else None,
        "avg_timing_delta_s":   round(sum(timing_deltas) / len(timing_deltas), 1) if timing_deltas else None,
    }


def generate_report(
    s1a_results, s1b_results, s1c_analysis, s1d_results,
    s2_results, s3_results,
    s1a_metrics, s1b_metrics, s2_metrics, s3_metrics,
    output_dir: Path,
    timestamp: str,
) -> Path:
    lines = [
        f"# BakeSquad LangGraph Evaluation Report",
        f"**Generated:** {timestamp}",
        f"**Model:** `{os.environ.get('MODEL_BACKEND','?')}` / `{_get_model_name()}`",
        "",
        "---",
        "",
        "## Stage 1A — Classification Accuracy",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total cases | {s1a_metrics.get('total', 0)} |",
        f"| Correct | {s1a_metrics.get('correct', 0)} |",
        f"| Accuracy | {s1a_metrics.get('accuracy', 0):.1%} |",
        "",
        "**By group:**",
        "",
        "| Group | Total | Correct | Accuracy |",
        "|-------|-------|---------|----------|",
    ]
    for g, gm in sorted((s1a_metrics.get("by_group") or {}).items()):
        lines.append(f"| {g} | {gm['total']} | {gm['correct']} | {gm['accuracy']:.1%} |")

    lines += [
        "",
        "---",
        "",
        "## Stage 1B — Preference/Constraint Extraction Recall",
        "",
    ]
    if s1b_metrics:
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Cases tested | {s1b_metrics.get('n', 0)} |",
            f"| Avg constraint recall | {s1b_metrics.get('avg_constraint_recall') or 'n/a'} |",
            f"| Avg preference recall | {s1b_metrics.get('avg_preference_recall') or 'n/a'} |",
            f"| Flour type accuracy   | {s1b_metrics.get('flour_type_accuracy') or 'n/a'} |",
            "",
            "**Per-case results:**",
            "",
            "| ID | Query | Constraint recall | Pref recall | Flour OK |",
            "|----|-------|-------------------|-------------|----------|",
        ]
        for r in s1b_results:
            cr = f"{r['constraint_recall']:.0%}"
            pr = f"{r['preference_recall']:.0%}"
            fl = ("YES" if r["flour_type_correct"] else "NO") if r["flour_type_correct"] is not None else "—"
            q  = r["query"][:45]
            lines.append(f"| {r['id']} | {q} | {cr} | {pr} | {fl} |")
    else:
        lines.append("_No preference-rich cases in selected groups._")

    lines += [
        "",
        "---",
        "",
        "## Stage 1C — Query Specificity Analysis",
        "",
        "Hypothesis: longer queries → more constraints/preferences extracted.",
        "",
        "| Group | n | Avg constraints | Avg preferences | Constraint recall | Pref recall |",
        "|-------|---|-----------------|-----------------|-------------------|-------------|",
    ]
    for g in ["short_query", "medium_query", "long_query"]:
        if g in (s1c_analysis or {}):
            a = s1c_analysis[g]
            cr = f"{a['avg_constraint_recall']:.1%}" if a["avg_constraint_recall"] is not None else "n/a"
            pr = f"{a['avg_preference_recall']:.1%}" if a["avg_preference_recall"] is not None else "n/a"
            lines.append(f"| {g} | {a['n']} | {a['avg_constraints']} | {a['avg_preferences']} | {cr} | {pr} |")

    lines += [
        "",
        "---",
        "",
        "## Stage 1D — Turn Classification Routing",
        "",
    ]
    if s1d_results:
        correct_seqs = sum(1 for r in s1d_results if r.get("all_correct"))
        lines.append(f"{correct_seqs}/{len(s1d_results)} sequences fully correct.")
        lines.append("")
        for seq in s1d_results:
            mark = "OK" if seq["all_correct"] else "FAIL"
            lines.append(f"**{seq['id']}** [{mark}] {seq.get('description', '')}")
            for turn in seq.get("turns", []):
                c = turn.get("content", "")[:50]
                pred = turn.get("predicted_turn_type", "?")
                exp  = turn.get("expected_turn_type", "?")
                tm   = "OK" if turn.get("correct") else "FAIL" if turn.get("correct") is False else "—"
                lines.append(f"  - {tm}  `{pred}` (expected `{exp}`)  — {c!r}")
            lines.append("")
    else:
        lines.append("_No multi-turn cases in selected groups._")

    lines += [
        "---",
        "",
        "## Stage 2 — Scoring Transparency",
        "",
    ]
    if s2_results:
        m = s2_metrics
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Pipeline runs OK | {m.get('runs_ok', 0)} |",
            f"| Runs with errors | {m.get('runs_error', 0)} |",
            f"| Category reconciliation events | {m.get('category_reconciliations', 0)} |",
            f"| Novel technique delta events | {m.get('technique_delta_events', 0)} |",
            f"| Avg technique signal recall | {m.get('avg_signal_recall') or 'n/a'} |",
            "",
            "**Most frequent technique signals extracted:**",
            "",
        ]
        for sig, cnt in (m.get("top_technique_signals") or []):
            lines.append(f"- `{sig}`: {cnt}")
        lines.append("")
        lines.append("**Scoring traces:**")
        lines.append("")
        for r in s2_results:
            if r.get("error"):
                lines.append(f"### {r['id']} — ERROR: {r['error']}")
                continue
            lines.append(f"### {r['id']} — {r['query'][:60]}")
            lines.append(f"- Category: `{r.get('category_source','?')}` → `{r.get('plan',{}).get('category','?')}`")
            if r.get("category_override"):
                lines.append(f"- **Step 7b override applied**: `{r['category_override']}`")
            lines.append(f"- Top recipe: {r.get('top_recipe','?')} ({r.get('top_composite_score','?')}/100)")
            sigs = r.get("technique_signals_found") or []
            lines.append(f"- Technique signals: {', '.join(sigs) if sigs else 'none'}")
            for note in r.get("technique_notes_found") or []:
                lines.append(f"- Novel technique note: _{note}_")
            for ev in r.get("technique_delta_events") or []:
                lines.append(f"- Technique delta applied: `{ev['delta']:+}` for {ev['title']!r}")
            if r.get("signal_match"):
                sm = r["signal_match"]
                lines.append(f"- Expected signals: {sm['expected']}  found: {sm['found']}  missed: {sm['missed']}  recall: {sm['recall']:.0%}")
            lines.append("")
            for trace in r.get("scoring_traces") or []:
                lines.append(f"  **#{trace['rank']} {trace['title'][:50]}** — composite: {trace['composite_score']}/100  (weights from: {trace['weight_source']})")
                for c in trace.get("criteria") or []:
                    tech_note = ""
                    if c["name"] == "Technique Quality":
                        sigs_str = ", ".join(c.get("technique_signals") or []) or "none"
                        delta_str = f"  delta={c.get('technique_note_delta'):+}" if c.get("technique_note_delta") is not None else ""
                        synergy_str = f"  synergies={['+'.join(s['signals']) for s in (c.get('synergies_triggered') or [])]}" if c.get("synergies_triggered") else ""
                        tech_note = f" | signals: {sigs_str}{delta_str}{synergy_str}"
                        if c.get("technique_notes"):
                            tech_note += f" | note: {c['technique_notes'][:60]}"
                    lines.append(f"    - {c['name']}: {c['score']:.0f}/100 (w={c['weight']:.2f}){tech_note}")
                lines.append("")
    else:
        lines.append("_Run with `--full-pipeline` to enable Stage 2._")

    lines += [
        "---",
        "",
        "## Stage 3 — DAG vs LangGraph Comparison",
        "",
    ]
    if s3_results:
        m = s3_metrics
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Cases compared | {m.get('n', 0)} |",
            f"| Category agreement | {m.get('category_match_rate', 0):.1%} |",
            f"| Top recipe URL match | {m.get('top_url_match_rate', 0):.1%} |",
            f"| Avg composite score delta | {m.get('avg_score_delta') or 'n/a'} |",
            f"| Avg timing delta (LG - DAG) | {m.get('avg_timing_delta_s') or 'n/a'}s |",
            "",
            "| ID | Category DAG | Category LG | Cat match | Score DAG | Score LG | Score delta |",
            "|----|--------------|-------------|-----------|-----------|----------|-------------|",
        ]
        for r in s3_results:
            d, l, cmp = r.get("dag", {}), r.get("langgraph", {}), r.get("comparison", {})
            if d.get("error") or l.get("error"):
                lines.append(f"| {r['id']} | ERROR | ERROR | — | — | — | — |")
            else:
                cat_m = "YES" if cmp.get("category_match") else "NO"
                lines.append(
                    f"| {r['id']} | {d.get('category','?')} | {l.get('category','?')} | "
                    f"{cat_m} | {d.get('top_score','?')} | {l.get('top_score','?')} | "
                    f"{cmp.get('score_delta','?')} |"
                )
    else:
        lines.append("_Run with `--compare-langgraph` to enable Stage 3._")

    lines += ["", "---", "", "*Generated by evaluate_langgraph.py*"]

    report_path = output_dir / f"eval_report_lg_{timestamp}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _get_model_name() -> str:
    try:
        from bakesquad.llm_client import get_model
        return get_model()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BakeSquad LangGraph evaluation")
    parser.add_argument("--full-pipeline",      action="store_true", help="Enable Stage 2 (network + API)")
    parser.add_argument("--compare-langgraph",  action="store_true", help="Enable Stage 3 (requires langgraph)")
    parser.add_argument("--stage2-max",         type=int, default=5,  metavar="N")
    parser.add_argument("--stage3-max",         type=int, default=5,  metavar="N")
    parser.add_argument("--groups",             nargs="+",            metavar="GROUP")
    parser.add_argument("--debug",              action="store_true")
    parser.add_argument("--output-dir",         default=str(_EVAL_DIR / "results"), metavar="DIR")
    parser.add_argument("--dataset",            default=str(_DATASET_V2), metavar="PATH")
    parser.add_argument("--dataset-v1",         default=str(_DATASET_V1), metavar="PATH",
                        help="v1 dataset appended for classification regression")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    _install_chat_instrumentation()

    # Load datasets
    cases_v2 = _load_dataset(Path(args.dataset))
    cases_v1: list[dict] = []
    if Path(args.dataset_v1).exists():
        cases_v1 = _load_dataset(Path(args.dataset_v1))
        print(f"Loaded {len(cases_v1)} cases from v1 dataset (classification regression)")

    # Combined for Stage 1A regression; v2 only for preference/turn stages
    all_cases = cases_v2 + cases_v1
    filtered  = _filter_cases(all_cases, args.groups)
    filtered_v2 = _filter_cases(cases_v2, args.groups)

    print(f"BakeSquad LangGraph Evaluation — {len(filtered)} cases ({len(filtered_v2)} from v2)")
    print(f"Backend: {os.environ.get('MODEL_BACKEND','?')} / {_get_model_name()}")
    print(f"Output:  {output_dir}")

    cases_by_id = {c["id"]: c for c in filtered_v2}

    # Stage 1A: classification accuracy (all cases with expected_category + turns not present)
    s1a_results = run_stage1a(filtered, args.debug)

    # Stage 1B: preference capture (v2 cases only)
    s1b_results = run_stage1b(s1a_results, cases_by_id, args.debug)

    # Stage 1C: specificity analysis (derived, no extra LLM calls)
    s1c_analysis = run_stage1c(s1a_results, s1b_results)

    # Stage 1D: turn classification (v2 multi_turn cases)
    s1d_results = run_stage1d(filtered_v2, args.debug)

    # Stage 2: scoring transparency
    s2_results: list[dict] = []
    if args.full_pipeline:
        s2_results = run_stage2(filtered_v2, args.stage2_max, args.debug)

    # Stage 3: DAG vs LangGraph
    s3_results: list[dict] = []
    if args.compare_langgraph:
        s3_results = run_stage3(filtered, args.stage3_max, args.debug)

    # Compute metrics
    s1a_metrics = compute_stage1a_metrics(s1a_results)
    s1b_metrics = compute_stage1b_metrics(s1b_results)
    s2_metrics  = compute_stage2_metrics(s2_results)
    s3_metrics  = compute_stage3_metrics(s3_results)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Stage 1A  accuracy: {s1a_metrics.get('accuracy',0):.1%}  ({s1a_metrics.get('correct',0)}/{s1a_metrics.get('total',0)})")
    if s1b_metrics:
        print(f"  Stage 1B  constraint recall: {s1b_metrics.get('avg_constraint_recall') or 'n/a'}  pref recall: {s1b_metrics.get('avg_preference_recall') or 'n/a'}")
    if s2_results:
        print(f"  Stage 2   {s2_metrics.get('runs_ok',0)} runs OK  reconciliations: {s2_metrics.get('category_reconciliations',0)}  novel deltas: {s2_metrics.get('technique_delta_events',0)}")
    if s3_results:
        print(f"  Stage 3   category agreement: {s3_metrics.get('category_match_rate',0):.1%}  url match: {s3_metrics.get('top_url_match_rate',0):.1%}")
    print(f"{'='*60}")

    # Save full results JSON
    all_results = {
        "timestamp":    timestamp,
        "backend":      os.environ.get("MODEL_BACKEND", "?"),
        "model":        _get_model_name(),
        "stage1a":      s1a_results,
        "stage1b":      s1b_results,
        "stage1c":      s1c_analysis,
        "stage1d":      s1d_results,
        "stage2":       s2_results,
        "stage3":       s3_results,
        "metrics": {
            "stage1a": s1a_metrics,
            "stage1b": s1b_metrics,
            "stage2":  s2_metrics,
            "stage3":  s3_metrics,
        },
    }
    results_path = output_dir / f"eval_results_lg_{timestamp}.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults: {results_path}")

    # Generate markdown report
    report_path = generate_report(
        s1a_results, s1b_results, s1c_analysis, s1d_results,
        s2_results, s3_results,
        s1a_metrics, s1b_metrics, s2_metrics, s3_metrics,
        output_dir, timestamp,
    )
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    main()
