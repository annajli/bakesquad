# BakeSquad — LangGraph Migration Context

> **Purpose:** Working reference for migrating BakeSquad from its current fixed
> sequential pipeline to a LangGraph-based agent with a web frontend, persistent
> user memory, dynamic category support, and instruction-aware scoring.
>
> Keep this file updated as decisions are made. All file:line references point to
> the state of the codebase at migration start.

---

## Table of Contents

1. [Why migrate to LangGraph](#1-why-migrate-to-langgraph)
2. [Current architecture — accurate map](#2-current-architecture--accurate-map)
3. [Target architecture](#3-target-architecture)
4. [Graph state schema](#4-graph-state-schema)
5. [Node definitions](#5-node-definitions)
6. [Edge routing logic](#6-edge-routing-logic)
7. [Dynamic category registry](#7-dynamic-category-registry)
8. [Instruction-aware scoring](#8-instruction-aware-scoring)
9. [Expanded memory layer](#9-expanded-memory-layer)
10. [API layer — FastAPI + SSE](#10-api-layer--fastapi--sse)
11. [Frontend architecture](#11-frontend-architecture)
12. [Migration roadmap](#12-migration-roadmap)
13. [New dependencies](#13-new-dependencies)
14. [What does NOT change](#14-what-does-not-change)

---

## 1. Why Migrate to LangGraph

### Problems with the current fixed DAG (`main.py`)

| Problem | Root cause | User impact |
|---------|-----------|-------------|
| Step 1 classification errors propagate silently | Category locked at Step 1; no feedback loop | Wrong ratio ranges → meaningless scores |
| Hard `Literal["cookie","quick_bread","cake","other"]` | Category hardcoded across 4 files | Adding yeasted bread, pastry, etc. requires code surgery |
| `ConversationSession` lives only in RAM | No serialisation, no thread IDs | Chat history lost on restart; no multi-user support |
| No streaming | Pipeline is synchronous; all output printed at end | Frontend impossible without full rewrite |
| `qwen3.5` drops responses into `<think>` blocks | `_strip_think_tags` deletes full output on ~30% of calls | Silent failures that look like empty results |
| Scoring ignores recipe instructions | `scorer.py` only reads `RatioResult` fields | Technique signals (mixing method, temp, rest time) not used |
| Single-file 570-line `main.py` entry point | No separation of graph logic from I/O | Hard to test, hard to serve via API |

### What LangGraph gives us

- **Conditional edges** — route by classification confidence; loop back once if Step 7 disagrees with Step 1
- **Interrupt nodes** — pause graph when confidence is low and ask user a clarifying question; resume on reply
- **`SqliteSaver` checkpointing** — every graph state is persisted at each node; conversation resumes across restarts
- **Thread IDs** — multiple users / sessions run isolated graphs sharing one SQLite store
- **Built-in streaming** — `graph.stream()` yields partial state after each node; plug directly into SSE
- **Tool nodes** — search, fetch, parse, score become callable tools; the LLM can invoke subsets adaptively
- **Open category routing** — new category = new conditional branch; no changes to existing nodes

---

## 2. Current Architecture — Accurate Map

### File inventory

| File | Lines | Role |
|------|-------|------|
| `main.py` | 570 | Entry point + all display logic + `run_pipeline()` + `chat_loop()` |
| `bakesquad/search/ingestion.py` | 484 | Steps 1–6: query plan, search, scoring, domain cap, fetch |
| `bakesquad/search/prompts.py` | 121 | All LLM prompt builders (plain strings, no LangChain) |
| `bakesquad/parser.py` | ~200 | Step 7: parallel LLM recipe parsing (ThreadPoolExecutor) |
| `bakesquad/normalizer.py` | ~300 | Step 8: unit→grams lookup table (zero LLM) |
| `bakesquad/ratio_engine.py` | ~250 | Step 9: deterministic ratio math + SQLite cache |
| `bakesquad/scorer.py` | 400 | Step 10: deterministic scoring math + batched LLM explanations |
| `bakesquad/session.py` | 230 | Multi-turn state + `classify_turn()` + `apply_re_filter()` |
| `bakesquad/memory.py` | 145 | SQLite (ratio_cache, liked_recipes) + user_prefs JSON |
| `bakesquad/llm_client.py` | 222 | Multi-backend LLM wrapper (ollama/groq/claude) |
| `bakesquad/models.py` | 142 | Pydantic models for all pipeline stages |
| `bakesquad/config.py` | ~50 | Pipeline constants (budgets, caps, cutoffs) |

### The 11-step pipeline (what `run_pipeline()` in `main.py:272` does)

```
User query (str)
   │
   ▼ LLM call #1 — query_plan_prompt (ingestion.py:148)
[Step 1] QueryPlan: category + flour_type + modifiers + hard_constraints + soft_preferences + queries[2]
   │
   ▼ DuckDuckGo × 2 queries, sequential (ingestion.py:195)
[Step 2] list[SearchSnippet] (deduped by normalised URL)
   │
   ▼ LLM call #2 — snippet_relevance_prompt, batched (ingestion.py:275)
[Step 3] relevance_score per snippet; heuristic pre-filter (roundups, paywall, URL pattern)
   │
   ▼ deterministic (ingestion.py:300)
[Step 4] domain cap (max 2 per domain); trusted sources win tiebreaks
   │
   ▼ if <3 candidates: LLM call #3 — relaxed_queries_prompt (ingestion.py:342)
[Step 5] adaptive retry with broader queries
   │
   ▼ parallel requests+BS4, ThreadPoolExecutor 4 workers (ingestion.py:357)
[Step 6] list[FetchedPage] with pre-extracted ingredients_excerpt (≤2000 chars sent to LLM)
   │
   ▼ parallel LLM calls #4..N — _SYSTEM_PROMPT in parser.py (one per page)
[Step 7] list[ParsedRecipe]: title, category, flour_type, modifiers, ingredients, has_chocolate
         NOTE: ParsedRecipe.category is computed here but NEVER used to override QueryPlan.category
   │
   ▼ zero LLM — lookup tables (normalizer.py)
[Step 8] list[NormalizedIngredient]: all quantities in grams
   │
   ▼ zero LLM — ratio math; SQLite cache by URL (ratio_engine.py)
[Step 9] list[RatioResult]: liquid/fat/sugar/leavening ratios vs RATIO_RANGES
   │
   ▼ zero LLM scoring math + LLM call #(N+1) batched explanations (scorer.py)
[Step 10] list[ScoredRecipe]: composite_score, criteria[3–4], constraint_violations, explanation
   │
   ▼ print to stdout (main.py:125)
[Step 11] Ranked output
```

### Known architectural gaps (confirmed by evaluation runs)

1. **Step 7 re-classification is discarded.** `ParsedRecipe.category` is inferred from
   ingredients (correctly) but `compute_ratios()` in `main.py:~350` reads `plan.category`
   (Step 1), not `recipe.category` (Step 7). A wrong Step 1 category propagates invisibly.

2. **Instruction text is not scored.** `scorer.py` reads only `RatioResult` fields.
   Technique signals (mixing method, bake temperature, chilling time) are in
   `FetchedPage.raw_text` but never extracted or used.

3. **`qwen3.5` Ollama backend drops ~30% of responses.** `_strip_think_tags` in
   `llm_client.py:163` deletes `<think>…</think>` blocks, but `qwen3.5` sometimes
   puts its entire output inside `<think>`. Fix: after stripping, if result is empty,
   fall back to extracting JSON from *within* the largest `<think>` block.

4. **Category is `Literal[...]`.** Adding a new category requires edits to:
   `models.py:44,74,99`, `ratio_engine.py` (RATIO_RANGES keys), `scorer.py` (all
   `if category == ...` branches), `search/prompts.py` (schema description + examples).

5. **No streaming.** All output goes through `print()`. The pipeline must complete
   before any result is visible.

---

## 3. Target Architecture

### Graph overview

```
                         ┌─────────────────────────────┐
                         │      BakeSquad Agent Graph  │
                         └───────────┬─────────────────┘
                                     │
                              [START / entry]
                                     │
                         ┌───────────▼─────────────┐
                         │    classify_intent       │  (1 LLM call)
                         │  new_search | refine     │  routes on turn_type
                         │  factual | clarify       │
                         └──┬─────────┬──────┬──────┘
                            │         │      │
               new_search   │   refine│      │factual
                            ▼         ▼      ▼
              ┌─────────────────┐  ┌──────┐  ┌───────────┐
              │  expand_query   │  │filter│  │  factual  │
              │ (dynamic cats)  │  │_node │  │  _node    │
              └────────┬────────┘  └──────┘  └───────────┘
                       │
              ┌────────▼────────┐
              │   search_node   │  DuckDuckGo × 2 + snippet scoring
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │   fetch_node    │  parallel requests+BS4
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │   parse_node    │  parallel LLM parsing
              └────────┬────────┘
                       │
              ┌────────▼────────┐   category mismatch?
              │  verify_node    │──────────────────────► expand_query (once)
              └────────┬────────┘   ok: continue
                       │
              ┌────────▼────────┐
              │   score_node    │  ratio + instruction scoring
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  memory_node    │  write cache + update prefs
              └────────┬────────┘
                       │
                    [END / stream]
```

### Conversation loop (multi-turn)

The graph runs once per turn. `SqliteSaver` checkpoints state after each node.
On the next user message the graph is resumed from the last checkpoint for that
`thread_id`, exactly like a long-running conversation.

```
User sends message
      │
      ▼
API server loads checkpoint for thread_id
      │
      ▼
Graph resumes at classify_intent
      │  (has access to last_scored, last_plan, message_history from checkpoint)
      ▼
Routes to: new_search | refine | factual | clarify
      │
      ▼
Streams partial state back via SSE
      │
      ▼
Checkpoint saved
```

---

## 4. Graph State Schema

Replace `ConversationSession` (session.py:29) with a LangGraph `TypedDict` state.
Every node reads from and writes to this object; LangGraph diffs it at each step.

```python
# bakesquad/graph/state.py

from __future__ import annotations
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
import operator

from bakesquad.models import (
    FetchedPage, ParsedRecipe, QueryPlan,
    RatioResult, SearchSnippet, ScoredRecipe,
)


class BakeSquadState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────
    thread_id: str
    messages: Annotated[list[dict], operator.add]  # append-only message history
    turn_type: Optional[str]          # "new_search" | "refine" | "factual" | "clarify"
    user_query: str                   # current turn's raw input

    # ── Query understanding ───────────────────────────────────────
    query_plan: Optional[QueryPlan]
    category_confidence: float        # 0.0–1.0; triggers clarify node if < threshold
    clarify_question: Optional[str]   # set by clarify_node if confidence low
    recency: Optional[str]            # "year" | "month" | None

    # ── Search & fetch ────────────────────────────────────────────
    snippets: list[SearchSnippet]
    fetched_pages: list[FetchedPage]
    search_retry_count: int           # caps at 1 to prevent infinite retry

    # ── Parse & verify ────────────────────────────────────────────
    parsed_recipes: list[ParsedRecipe]
    category_override: Optional[str]  # set by verify_node if Step7 disagrees Step1

    # ── Score ─────────────────────────────────────────────────────
    scored_recipes: list[ScoredRecipe]

    # ── Refine / filter ───────────────────────────────────────────
    exclude_ingredients: list[str]
    require_fat_type: Optional[str]

    # ── Factual ───────────────────────────────────────────────────
    factual_answer: Optional[str]

    # ── User memory ───────────────────────────────────────────────
    user_prefs: dict                  # loaded from DB at session start
    liked_recipe_urls: list[str]      # preloaded for fast filter

    # ── Error tracking ────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]  # append-only error log
```

**Why `Annotated[list, operator.add]`?** LangGraph uses reducers to merge state
updates from concurrent nodes. `operator.add` means each node appends to the list
rather than replacing it — safe for `messages` and `errors` which accumulate across turns.

---

## 5. Node Definitions

Each node is a plain Python function `(state: BakeSquadState) -> dict` that returns
a partial state update. LangGraph merges the update into the full state.

### 5.1 `classify_intent_node`

**Replaces:** `session.classify_turn()` in `session.py:91`  
**LLM calls:** 1 (for non-trivial follow-ups; 0 for first turn which is always `new_search`)

```python
# bakesquad/graph/nodes/classify_intent.py

def classify_intent_node(state: BakeSquadState) -> dict:
    """
    Route incoming message to: new_search | refine | factual | clarify.

    For the first message in a thread (no query_plan yet) → always new_search.
    For follow-ups → 1 LLM call using session context.
    """
    if state["query_plan"] is None:
        return {
            "turn_type": "new_search",
            "user_query": state["messages"][-1]["content"],
        }

    # Use existing classify_turn logic (single LLM call)
    # Pass message_history + last_scored context
    ...
```

### 5.2 `expand_query_node`

**Replaces:** `IngestionPipeline._build_query_plan()` in `ingestion.py:145`  
**LLM calls:** 1 (query_plan_prompt)  
**New behaviour:** reads category registry (§7) dynamically; stores `category_confidence`

```python
def expand_query_node(state: BakeSquadState) -> dict:
    """
    Step 1: Build QueryPlan from user query.

    Key additions vs. current code:
    - Reads CATEGORY_REGISTRY (YAML) to build the prompt's category list dynamically.
    - Asks LLM for category_scores dict (confidence per category) in addition to
      the hard category label.
    - Stores category_confidence = max(category_scores.values()).
    - If category_confidence < CLARIFY_THRESHOLD (0.55), sets turn_type="clarify"
      so the graph pauses and asks the user a clarifying question.
    """
    ...
    return {
        "query_plan": plan,
        "category_confidence": confidence,
        "turn_type": "new_search" if confidence >= CLARIFY_THRESHOLD else "clarify",
    }
```

### 5.3 `clarify_node`

**New node — no equivalent in current code**  
**LLM calls:** 1 (generates the clarifying question)

```python
def clarify_node(state: BakeSquadState) -> dict:
    """
    When classification confidence is below threshold, generate a single
    clarifying question and interrupt the graph.

    Example: "banana bread cake" → confidence 0.45 →
      'Did you mean a quick bread (like banana bread) or a banana-flavoured cake?'

    LangGraph interrupt() pauses execution here until the user replies.
    The next turn's classify_intent sees the clarification and re-routes to new_search.
    """
    from langgraph.types import interrupt
    question = _generate_clarify_question(state)
    interrupt(question)  # graph pauses; resumes when user sends next message
    return {"clarify_question": question}
```

### 5.4 `search_node`

**Replaces:** `IngestionPipeline._search_and_filter()` in `ingestion.py:322`  
**LLM calls:** 1 (snippet relevance, batched) + optional 1 (relaxed queries)

```python
def search_node(state: BakeSquadState) -> dict:
    """
    Steps 2–5: DuckDuckGo search + snippet scoring + domain cap + adaptive retry.
    Uses state["query_plan"].queries (exactly 2).
    Preserves the existing sequential-search behaviour (ddgs deadlock fix, ingestion.py:195).
    """
    ...
    return {"snippets": passing_snippets, "search_retry_count": retries}
```

### 5.5 `fetch_node`

**Replaces:** `IngestionPipeline._fetch_pages()` in `ingestion.py:357`  
**LLM calls:** 0

```python
def fetch_node(state: BakeSquadState) -> dict:
    """
    Step 6: Parallel page fetch (ThreadPoolExecutor, 4 workers).
    Same logic as today; no LLM calls.
    Extracts ingredients_excerpt per page (ingestion.py:420).
    """
    ...
    return {"fetched_pages": pages}
```

### 5.6 `parse_node`

**Replaces:** `parse_recipes_parallel()` in `parser.py`  
**LLM calls:** N (one per fetched page, parallel)  
**New behaviour:** also extracts `technique_signals` (see §8)

```python
def parse_node(state: BakeSquadState) -> dict:
    """
    Step 7: Parallel LLM parsing of fetched pages.
    Parser _SYSTEM_PROMPT extended to also extract technique_signals list.

    technique_signals examples:
      ["fold_method", "brown_butter", "chill_24h", "bake_325f", "batter_rest_30m"]
    """
    ...
    return {"parsed_recipes": parsed}
```

### 5.7 `verify_node`

**New node — implements the Step 7 feedback loop identified in evaluation**  
**LLM calls:** 0

```python
def verify_node(state: BakeSquadState) -> dict:
    """
    Compare Step 1 category (query_plan.category) against Step 7 categories
    (ParsedRecipe.category for each recipe).

    If a majority of parsed recipes disagree with the query plan category,
    set category_override to the majority Step 7 category.
    The score_node reads category_override (if set) before query_plan.category.

    Also triggers a graph loop: if category_override is set AND search_retry_count == 0,
    route back to expand_query with updated category to re-run search.
    Capped at 1 retry to prevent infinite loops.
    """
    from collections import Counter
    step7_cats = Counter(
        r.category for r in state["parsed_recipes"]
        if not r.parse_error and r.category != "other"
    )
    if not step7_cats:
        return {}
    dominant, count = step7_cats.most_common(1)[0]
    plan_cat = state["query_plan"].category
    if dominant != plan_cat and count >= len(state["parsed_recipes"]) / 2:
        return {"category_override": dominant}
    return {}
```

### 5.8 `score_node`

**Replaces:** `score_all()` + `add_explanations()` in `scorer.py`  
**LLM calls:** 1 (batched explanation generation, unchanged)  
**New behaviour:** adds `technique_score` criterion (see §8); reads `category_override`

```python
def score_node(state: BakeSquadState) -> dict:
    """
    Step 10: Deterministic scoring + batched LLM explanations.
    Uses category_override if set, else query_plan.category.
    Adds technique_score as a 4th scoring criterion.
    """
    effective_category = state.get("category_override") or state["query_plan"].category
    # update plan.category for ratio range lookup
    plan = state["query_plan"].model_copy(update={"category": effective_category})
    ...
    return {"scored_recipes": scored}
```

### 5.9 `filter_node`

**Replaces:** `apply_re_filter()` in `session.py:149`  
**LLM calls:** 0

```python
def filter_node(state: BakeSquadState) -> dict:
    """
    Refine turn: filter existing scored_recipes by ingredient exclusions
    and fat type requirement. Zero LLM calls.
    Falls back to new_search if nothing passes.
    """
    ...
    return {"scored_recipes": filtered} if filtered else {"turn_type": "new_search"}
```

### 5.10 `factual_node`

**Replaces:** the `"factual"` branch of `classify_turn()` in `session.py:91`  
**LLM calls:** 1 (answer generation)

```python
def factual_node(state: BakeSquadState) -> dict:
    """
    Answer a baking science question using the current result context.
    classify_intent already generated direct_answer; this node formats and stores it.
    """
    ...
    return {"factual_answer": answer}
```

### 5.11 `memory_node`

**Replaces / extends:** `cache_put()` calls scattered in `main.py` + `memory.py`  
**LLM calls:** 0

```python
def memory_node(state: BakeSquadState) -> dict:
    """
    Write-through memory operations:
    1. Upsert ratio_cache for each scored recipe URL (existing behaviour).
    2. Update user_prefs if feedback signals are present in state.
    3. If user rated/liked a recipe this turn, insert into liked_recipes.
    4. Optionally update recipe_embeddings table for semantic search (§9).
    """
    ...
    return {"user_prefs": updated_prefs}
```

---

## 6. Edge Routing Logic

```python
# bakesquad/graph/graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from bakesquad.graph.state import BakeSquadState
from bakesquad.graph import nodes

def build_graph(db_path: str) -> CompiledGraph:
    g = StateGraph(BakeSquadState)

    # Register nodes
    g.add_node("classify_intent", nodes.classify_intent_node)
    g.add_node("expand_query",    nodes.expand_query_node)
    g.add_node("clarify",         nodes.clarify_node)
    g.add_node("search",          nodes.search_node)
    g.add_node("fetch",           nodes.fetch_node)
    g.add_node("parse",           nodes.parse_node)
    g.add_node("verify",          nodes.verify_node)
    g.add_node("score",           nodes.score_node)
    g.add_node("filter",          nodes.filter_node)
    g.add_node("factual",         nodes.factual_node)
    g.add_node("memory",          nodes.memory_node)

    # Entry
    g.set_entry_point("classify_intent")

    # classify_intent → branch by turn_type
    g.add_conditional_edges(
        "classify_intent",
        lambda s: s["turn_type"],
        {
            "new_search": "expand_query",
            "refine":     "filter",
            "factual":    "factual",
        },
    )

    # expand_query → branch on confidence
    g.add_conditional_edges(
        "expand_query",
        lambda s: "clarify" if s["turn_type"] == "clarify" else "search",
        {"clarify": "clarify", "search": "search"},
    )

    # clarify → END (waits for user; next turn re-enters at classify_intent)
    g.add_edge("clarify", END)

    # Linear search → fetch → parse → verify
    g.add_edge("search", "fetch")
    g.add_edge("fetch",  "parse")
    g.add_edge("parse",  "verify")

    # verify → branch: re-expand if category mismatch and first attempt
    g.add_conditional_edges(
        "verify",
        lambda s: (
            "expand_query"
            if s.get("category_override") and s["search_retry_count"] == 0
            else "score"
        ),
        {"expand_query": "expand_query", "score": "score"},
    )

    # score → memory → END
    g.add_edge("score",   "memory")
    g.add_edge("memory",  END)

    # filter → memory (re-filtered results still write cache)
    g.add_edge("filter",  "memory")

    # factual → END (no memory write needed)
    g.add_edge("factual", END)

    # Compile with SQLite checkpoint persistence
    checkpointer = SqliteSaver.from_conn_string(db_path)
    return g.compile(checkpointer=checkpointer, interrupt_before=["clarify"])
```

---

## 7. Dynamic Category Registry

Replace `Literal["cookie", "quick_bread", "cake", "other"]` with a YAML registry
loaded at startup. New categories require only a new YAML entry.

```yaml
# bakesquad/categories.yaml

categories:

  cookie:
    aliases: [cookies, bars, brownies, shortbread, biscotti, rugelach]
    ratio_axes: [fat_to_flour, sugar_to_flour, leavening_to_flour, brown_to_white_sugar]
    scoring_criteria: [moisture_retention, structure_leavening, sugar_balance, chew]
    base_weights: {moisture: 0.40, structure: 0.35, balance: 0.25}

  quick_bread:
    aliases:
      - muffins, scones, loaf, cornbread
      - banana bread, banana nut bread, banana loaf
      - zucchini bread, zucchini loaf, pumpkin bread
      - quick loaf  # catches "a moist loaf for gift giving"
    ratio_axes: [liquid_to_flour, fat_to_flour, sugar_to_flour, leavening_to_flour]
    scoring_criteria: [moisture_retention, structure_leavening, sugar_balance]
    base_weights: {moisture: 0.45, structure: 0.30, balance: 0.25}

  cake:
    aliases: [layer cake, bundt, cupcakes, cheesecake, coffee cake, lava cake]
    ratio_axes: [liquid_to_flour, fat_to_flour, sugar_to_flour, leavening_to_flour]
    scoring_criteria: [moisture_retention, structure_leavening, sugar_balance]
    base_weights: {moisture: 0.45, structure: 0.30, balance: 0.25}

  # ── New categories (add without code changes) ──────────────────────────

  yeasted_bread:
    aliases: [sourdough, focaccia, brioche, baguette, sandwich bread, dinner rolls]
    ratio_axes: [hydration_to_flour, yeast_to_flour, salt_to_flour]
    scoring_criteria: [crust_development, crumb_structure, fermentation_quality]
    base_weights: {crust: 0.35, crumb: 0.35, fermentation: 0.30}
    ratio_ranges:
      hydration_to_flour: [0.60, 0.85]
      yeast_to_flour:     [0.005, 0.025]
      salt_to_flour:      [0.015, 0.022]

  pastry:
    aliases: [croissants, danish, puff pastry, tart, pie crust, phyllo]
    ratio_axes: [fat_to_flour, water_to_flour]
    scoring_criteria: [lamination, flakiness, butter_quality]
    base_weights: {lamination: 0.45, flakiness: 0.35, butter_quality: 0.20}

  other:
    aliases: []   # catch-all; minimal scoring
    ratio_axes: []
    scoring_criteria: []
    base_weights: {}
```

### Registry loader

```python
# bakesquad/category_registry.py

import yaml
from pathlib import Path
from functools import lru_cache

REGISTRY_PATH = Path(__file__).parent / "categories.yaml"

@lru_cache(maxsize=1)
def load_registry() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text())

def category_names() -> list[str]:
    return list(load_registry()["categories"].keys())

def all_aliases() -> dict[str, str]:
    """Map every alias → canonical category name."""
    result = {}
    for cat, meta in load_registry()["categories"].items():
        for alias in meta.get("aliases", []):
            result[alias.lower()] = cat
    return result

def ratio_axes(category: str) -> list[str]:
    return load_registry()["categories"].get(category, {}).get("ratio_axes", [])

def ratio_ranges(category: str) -> dict:
    return load_registry()["categories"].get(category, {}).get("ratio_ranges", {})
```

### Dynamic prompt injection

`query_plan_prompt()` in `search/prompts.py` is rewritten to call `load_registry()`
and inject the live category list and aliases, replacing the hardcoded description.

```python
def query_plan_prompt(query, recency, trusted_sources):
    from bakesquad.category_registry import load_registry
    registry = load_registry()

    cat_lines = []
    for cat, meta in registry["categories"].items():
        aliases = ", ".join(meta.get("aliases", [])[:6])  # first 6 for brevity
        cat_lines.append(f'  {cat}: includes {aliases}')
    category_block = "\n".join(cat_lines)

    system = (
        f'- "category": one of: {", ".join(registry["categories"].keys())}\n'
        f'{category_block}\n'
        ...
    )
    return system, query
```

---

## 8. Instruction-Aware Scoring

### What to extract (additions to `parser.py` `_SYSTEM_PROMPT`)

The existing system prompt already extracts ingredients, category, modifiers, and
`has_chocolate`. Add a `technique_signals` list:

```
Also extract "technique_signals": a list of technique keywords present in the
instructions. Include only items from this vocabulary:

  Mixing:       fold_method, cream_method, melt_method, one_bowl
  Temperature:  bake_low (≤325°F), bake_standard (326–375°F), bake_high (>375°F)
  Resting:      chill_dough, batter_rest, overnight_rest
  Fat prep:     brown_butter, clarified_butter, room_temp_butter
  Leavening:    double_leavening (both baking powder and soda used)
  Special:      water_bath, steam_bake, dutch_oven, parchment_lined

Return an empty list if none apply.
```

### New `technique_score` criterion in `scorer.py`

```python
# Add to bakesquad/scorer.py

TECHNIQUE_RULES: dict[str, dict[str, float]] = {
    "quick_bread": {
        "fold_method":      +20,   # correct muffin method → tender crumb
        "bake_low":         +10,   # gentle heat → moist interior
        "bake_high":        -15,   # overbakes edges before centre sets
        "one_bowl":         +5,    # less gluten development from minimal mixing
        "brown_butter":     +10,   # added flavour complexity
        "batter_rest":      +8,    # better hydration
    },
    "cookie": {
        "brown_butter":     +15,   # Maillard flavour
        "chill_dough":      +15,   # better flavour development, less spread
        "overnight_rest":   +10,
        "cream_method":     +8,    # proper fat aeration
        "double_leavening": +5,
    },
    "cake": {
        "cream_method":     +15,   # proper fat aeration → fine crumb
        "water_bath":       +20,   # moisture during baking (cheesecakes)
        "room_temp_butter": +10,   # creams properly
        "bake_low":         +10,
        "bake_high":        -10,
    },
}

def _score_technique(signals: list[str], category: str) -> float:
    score = 50.0  # neutral baseline (no instruction data)
    if not signals:
        return score
    rules = TECHNIQUE_RULES.get(category, {})
    for signal in signals:
        score += rules.get(signal, 0)
    return round(min(100.0, max(0.0, score)), 1)
```

### Add to `score_recipe()` in `scorer.py`

```python
# Inside score_recipe(), after the existing three criteria:
technique_signals = getattr(recipe, "technique_signals", [])
if technique_signals or True:   # always include; shows 50 when no data
    tech_score = _score_technique(technique_signals, category)
    criteria.append(CriterionScore(
        name="Technique Quality",
        score=tech_score,
        weight=0.15,   # fixed weight; doesn't shift with user prefs
        details=", ".join(technique_signals) if technique_signals else "no technique data",
    ))
```

### Add `technique_signals` to `ParsedRecipe` in `models.py`

```python
class ParsedRecipe(BaseModel):
    ...
    technique_signals: list[str] = Field(default_factory=list)
```

---

## 9. Expanded Memory Layer

### Schema additions to `memory.py`

```sql
-- Run once on migration; existing tables unchanged
ALTER TABLE liked_recipes ADD COLUMN user_notes TEXT DEFAULT '';
ALTER TABLE liked_recipes ADD COLUMN user_rating INTEGER DEFAULT 0;
ALTER TABLE liked_recipes ADD COLUMN tried_date  TEXT DEFAULT NULL;

CREATE TABLE IF NOT EXISTS recipe_embeddings (
    url         TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    category    TEXT NOT NULL,
    embedding   BLOB NOT NULL,    -- float32[] via sqlite-vec or stored as JSON
    embedded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT NOT NULL,
    feedback    TEXT NOT NULL,    -- "liked" | "disliked" | "tried" | "note"
    content     TEXT DEFAULT '',  -- note text if feedback="note"
    created_at  TEXT NOT NULL
);
```

### `memory.py` additions

```python
# New functions to add to bakesquad/memory.py

def update_liked_recipe(url: str, rating: int = None, notes: str = None, tried_date: str = None):
    """Patch an existing liked recipe with user feedback fields."""
    ...

def add_feedback(url: str, feedback: str, content: str = ""):
    """Record a user feedback event (liked/disliked/tried/note)."""
    ...

def get_semantic_candidates(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Nearest-neighbour search over recipe_embeddings.
    Returns top_k liked recipes most similar to the query embedding.
    Used to pre-populate results from memory before running a new web search.

    Implementation options (in order of simplicity):
      Option A: store embeddings as JSON float arrays; cosine-similarity in Python (no dep)
      Option B: sqlite-vec extension (C extension, fast, pip install sqlite-vec)
      Option C: ChromaDB (separate process, richer API, overkill for MVP)
    Start with Option A; migrate to B when the liked_recipes table grows.
    """
    ...

def update_user_prefs_from_feedback(feedback_batch: list[dict]) -> dict:
    """
    Infer pref weight adjustments from recent feedback events.
    E.g. consistently liking oil-based recipes → preferred_fat = "oil".
    Returns updated prefs dict (caller persists with save_prefs()).
    """
    ...
```

### Fast retrieval in `memory_node`

When the user's query closely matches a liked recipe (cosine similarity > 0.85),
`memory_node` can inject that recipe directly into `state["scored_recipes"]` at
rank 0 before running the full search — cutting the latency to ~0 for repeat queries.

```python
# In memory_node, during new_search turns:
if state.get("query_embedding"):
    cache_hits = get_semantic_candidates(state["query_embedding"], top_k=3)
    for hit in cache_hits:
        if hit["similarity"] > MEMORY_HIT_THRESHOLD:   # 0.85
            inject_cached_recipe(state, hit)            # add to scored_recipes
```

---

## 10. API Layer — FastAPI + SSE

### Endpoints

```python
# bakesquad/api/main.py   (replaces CLI main.py)

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command
import json

app = FastAPI()
graph = build_graph(db_path=str(_DB_PATH))   # LangGraph compiled graph

@app.post("/query")
async def start_query(body: QueryRequest):
    """
    Start a new search or continue a conversation thread.
    Returns thread_id; client polls /stream/{thread_id} for results.
    """
    thread_id = body.thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    # Kick off graph run in background
    asyncio.create_task(_run_graph(body.query, config))
    return {"thread_id": thread_id}

@app.get("/stream/{thread_id}")
async def stream_results(thread_id: str):
    """
    Server-Sent Events stream of graph node outputs for a thread.
    Client receives partial results as each node completes.
    """
    async def event_generator():
        config = {"configurable": {"thread_id": thread_id}}
        async for chunk in graph.astream(None, config=config, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                yield f"data: {json.dumps({'node': node_name, 'data': _serialise(node_output)})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/feedback")
async def submit_feedback(body: FeedbackRequest):
    """Save user rating, notes, or liked/tried status for a recipe."""
    save_liked_recipe(body.url, body.title, body.recipe_dict,
                      rating=body.rating, notes=body.notes)
    return {"ok": True}

@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    """Return full message history for a conversation thread."""
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    return {"messages": state.values.get("messages", [])}

@app.get("/liked")
async def get_liked():
    return {"recipes": get_liked_recipes()}
```

### SSE message types

| `node` value | Frontend action |
|-------------|-----------------|
| `expand_query` | Show category badge + constraint chips |
| `search` | Show "Searching…" with snippet count |
| `fetch` | Show "Fetching N pages…" |
| `parse` | Show recipe title previews as they arrive |
| `score` | Render score cards (streamed, not all-at-once) |
| `clarify` | Show clarifying question in chat bubble; block input |
| `factual` | Insert assistant message in chat |
| `[DONE]` | Hide loading states |

---

## 11. Frontend Architecture

### Recommended stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Framework | **Next.js 14 (App Router)** | SSE + React Server Components; easy to deploy on Vercel |
| Styling | **Tailwind CSS** | Fast iteration; no CSS files |
| Chat UI | Custom `<ChatPanel>` | Need mixed message types (chat bubbles + recipe cards) |
| Score cards | Custom `<RecipeCard>` | Score bar, ratio details, criteria breakdown |
| State | **Zustand** | Lightweight; thread_id + messages + results |
| SSE client | `EventSource` API | Browser-native; no extra dep |

**Alternative for fast MVP:** [Gradio](https://gradio.app/) with a custom `gr.Blocks`
layout. Python-only, ~100 lines to get chat + card display. Switch to Next.js when
you need more control over the UI.

### Component tree

```
<App>
  <Sidebar>
    <ThreadList />          // past conversations by thread_id
    <LikedRecipes />        // saved recipes with notes
  </Sidebar>

  <Main>
    <QueryBar />            // text input + submit + recency toggle
    <ChatHistory>
      <UserMessage />
      <AssistantMessage />  // factual answers
      <ClarifyPrompt />     // pause-and-ask bubble
      <RecipeResultSet>     // rendered after score node streams
        <RecipeCard>
          <ScoreBar />
          <CriteriaBreakdown />
          <RatioDetail />
          <TechniqueSignals />   // NEW: shows technique_signals
          <FeedbackButtons>      // Like / Tried / Add note
        </RecipeCard>
      </RecipeResultSet>
    </ChatHistory>
  </Main>
</App>
```

### Key frontend behaviours

1. **Streaming render**: open `EventSource("/stream/{thread_id}")` on submit.
   Each `score` event appends a `<RecipeCard>`. User sees results arrive one by one,
   not all at once after 30s.

2. **Clarify interrupt**: when `node === "clarify"`, disable the query bar and render
   the question as an assistant bubble. User's next message resumes the graph.

3. **Feedback loop**: "Like", "Tried it", and "Add note" buttons on each card call
   `POST /feedback`. The `memory_node` picks these up on the next search and updates
   both the SQLite store and user_prefs weights.

4. **Liked recipe fast lane**: when the backend returns a recipe with
   `source: "memory"` in its data, render a "From your saved recipes" badge.

---

## 12. Migration Roadmap

### Phase 0 — Immediate fixes (no architecture change)

| Task | File | Effort |
|------|------|--------|
| Fix Ollama think-tag fallback | `llm_client.py:163` | 30 min |
| Add `technique_signals` to parser prompt | `parser.py:_SYSTEM_PROMPT` | 2 h |
| Add `technique_score` criterion | `scorer.py` | 2 h |
| Add Step 7 → Step 1 category reconciliation | `main.py` after `parse_recipes_parallel()` | 1 h |

### Phase 1 — Persistence and API layer

| Task | New files | Effort |
|------|-----------|--------|
| Schema migration SQL | `bakesquad/memory.py` | 1 h |
| FastAPI server with `/query` + `/stream` + `/feedback` | `bakesquad/api/main.py` | 2 d |
| SSE streaming wrapper around existing `run_pipeline()` | `bakesquad/api/stream.py` | 1 d |

The Phase 1 API wraps the **existing** `run_pipeline()` without touching the pipeline
internals. This gets you a working API endpoint + SSE stream before any LangGraph work.

### Phase 2 — LangGraph migration

| Task | New files | Effort |
|------|-----------|--------|
| `BakeSquadState` TypedDict | `bakesquad/graph/state.py` | 2 h |
| Extract each step into a node function | `bakesquad/graph/nodes/*.py` | 2 d |
| Wire graph with `StateGraph` | `bakesquad/graph/graph.py` | 1 d |
| `SqliteSaver` integration + thread_id | `bakesquad/graph/graph.py` | 2 h |
| `verify_node` (Step 7 feedback loop) | `bakesquad/graph/nodes/verify.py` | 3 h |
| `clarify_node` + `interrupt()` | `bakesquad/graph/nodes/clarify.py` | 3 h |
| Update API to call `graph.astream()` | `bakesquad/api/main.py` | 4 h |

### Phase 3 — Dynamic categories + frontend

| Task | New files | Effort |
|------|-----------|--------|
| `categories.yaml` registry | `bakesquad/categories.yaml` | 2 h |
| Registry loader + dynamic prompt injection | `bakesquad/category_registry.py` | 3 h |
| Remove `Literal[...]` from `models.py` | `bakesquad/models.py` | 1 h |
| Next.js frontend scaffold | `frontend/` | 3–5 d |
| `<RecipeCard>` + score bar components | `frontend/components/` | 2 d |
| SSE client + chat panel | `frontend/components/ChatHistory.tsx` | 1 d |
| Feedback buttons → `/feedback` API | `frontend/components/RecipeCard.tsx` | 4 h |

### Phase 4 — Semantic memory

| Task | Effort |
|------|--------|
| Embedding generation on liked recipe save | 2 h |
| Cosine similarity search in `memory_node` | 3 h |
| "From your saved recipes" UI badge | 2 h |
| `user_feedback` preference learning | 4 h |

---

## 13. New Dependencies

```toml
# Add to requirements.txt / pyproject.toml

# LangGraph
langgraph>=0.2              # graph + SqliteSaver
langchain-core>=0.3         # message types + tool interface

# API layer
fastapi>=0.115
uvicorn[standard]>=0.30
python-multipart>=0.0.9     # for form data in feedback endpoint

# Category registry
pyyaml>=6.0

# Frontend (separate package.json in frontend/)
# next@14, react@18, tailwindcss@3, zustand@4

# Optional: semantic memory
# sqlite-vec>=0.1            # fast vector search in SQLite
# openai>=1.0                # if using OpenAI embeddings
# sentence-transformers>=2.0 # if using local embeddings (MiniLM-L6)
```

---

## 14. What Does NOT Change

The following components are well-designed and require no changes during migration.
LangGraph nodes wrap them as-is.

| Component | Why it stays | LangGraph role |
|-----------|-------------|----------------|
| `llm_client.py` | Multi-backend wrapper is clean; just fix the Ollama think-tag bug | All nodes call it directly |
| `normalizer.py` | Zero-LLM lookup tables; deterministic | Called from within `score_node` |
| `ratio_engine.py` | Deterministic math + SQLite cache; correct | Called from within `score_node`; RATIO_RANGES extended for new categories |
| `search/prompts.py` | Prompt builders are modular | `expand_query_node` calls `query_plan_prompt()`; made dynamic per §7 |
| `bakesquad.db` schema | Tables already exist; just add columns per §9 | `memory_node` reads/writes via `memory.py` |
| DuckDuckGo sequential search | ddgs deadlock fix (ingestion.py:195) must be preserved | `search_node` keeps sequential behaviour |
| `config.py` constants | Budgets and caps are correct | Imported directly in relevant nodes |
