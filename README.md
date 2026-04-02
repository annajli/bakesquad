# BakeSquad

A recipe search and scoring agent that cuts through review-inflation bias on popular recipe sites. Instead of ranking by social proof (star ratings, review counts), BakeSquad evaluates recipes based on **ingredient ratios** and **baking science principles** — then personalizes ranking using your preference profile.

> Final project for an AI agents workshop.

---

## The problem

AllRecipes, Food Network, and similar sites rank recipes by PageRank-style popularity. A recipe posted in 2010 with 4,000 reviews consistently outranks a technically superior recipe from 2023. BakeSquad evaluates the recipe itself — its ingredient ratios and structure — not its audience size.

---

## How it works

BakeSquad runs an 11-step fixed pipeline from natural language query to ranked output:

```
Query
  │
  ▼
[1] Query Understanding      — 1 LLM call: extract category, constraints, scoring weights
  │
  ▼
[2] DuckDuckGo Search        — 2 query variants (broad + specific), deduplicated
  │
  ▼
[3] Snippet Pre-check        — 1 batched LLM call scores all snippets at once
  │
  ▼
[4] Domain Cap               — max 2 results per domain, relevance-ranked
  │
  ▼
[5] Adaptive Retry           — if < 3 candidates, retry with relaxed queries
  │
  ▼
[6] Parallel Page Fetch      — requests + BeautifulSoup, 5 s timeout, max 4 pages
  │
  ▼
[7] LLM Parse (parallel)     — BS4 pre-extracts ingredients section; LLM structures it
  │
  ▼
[8] Unit Normalization       — lookup table converts all quantities to grams (no LLM)
  │
  ▼
[9] Ratio Engine             — deterministic math; results cached in SQLite by URL
  │
  ▼
[10] Scoring + Explanations  — deterministic scores; 1 batched LLM call for "why"
  │
  ▼
[11] Ranked Output           — score cards with ratio breakdowns and explanations
```

**Key performance properties:**
- Every LLM call that can be batched *is* batched — never one call per item
- Page fetching is parallel (`ThreadPoolExecutor`)
- Unit normalization, ratio math, and scoring contain **zero LLM calls**
- Ratio results are cached in SQLite — repeat queries skip parse + normalize + ratio entirely

---

## MVP scope

Supports three baked-good categories:

| Category | Example query |
|---|---|
| `quick_bread` | `"chocolate chip banana bread that stays moist for days"` |
| `cookie` | `"chewy brown butter chocolate chip cookies"` |
| `cake` | `"moist chocolate birthday cake with cream cheese frosting"` |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy `.env.example` (or create `.env`) with the keys for whichever backends you plan to use:

```env
GROQ_API_KEY=your_groq_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

Groq keys are free at [console.groq.com](https://console.groq.com). Ollama requires no key.

---

## Running the agent

```bash
MODEL_BACKEND=groq  python main.py "chocolate chip banana bread that stays moist for days"
MODEL_BACKEND=claude python main.py "chewy brown butter chocolate chip cookies"
MODEL_BACKEND=ollama python main.py "moist banana bread"
```

**With recency filter:**
```bash
MODEL_BACKEND=groq python main.py "sourdough discard cookies" --recency year
MODEL_BACKEND=groq python main.py "viral banana bread" --recency month
```

Switching backends requires only changing `MODEL_BACKEND` — no code changes.

---

## LLM backends

| `MODEL_BACKEND` | Model | Time budget | Notes |
|---|---|---|---|
| `groq` | `llama-3.1-8b-instant` | 60 s | Free tier; fast; use for dev |
| `claude` | `claude-sonnet-4-20250514` | 60 s | Best explanation quality; use for demos |
| `ollama` | `qwen3:8b` (local) | 3 min | No cost; requires Ollama running locally |

All LLM calls route through `bakesquad/llm_client.py` — a thin wrapper that handles backend selection, rate-limit retries, and think-tag stripping (for qwen3).

---

## Example output

```
==============================================================
  BakeSquad  |  Backend: groq (llama-3.1-8b-instant)
==============================================================
  Query: 'chocolate chip banana bread that stays moist for days'

[Step 1: Query Understanding]
  Category:    quick_bread
  Constraints: contains chocolate, contains banana, stays moist for days
  Queries generated:
    - moist chocolate chip banana bread recipe
    - best ever chocolate chip banana bread recipe with buttermilk and brown sugar
  ok  3.3s  (budget 5s)

[Step 2-5: Search + Candidate Selection]
  11 candidates selected
  ok  4.1s  (budget 8s)

[Step 6: Page Fetch]  (parallel, 5 s timeout)
  4/4 pages fetched successfully
  ok  1.3s  (budget 10s)

...

  #1  Perfect Chocolate Chip Banana Bread
      https://cafedelites.com/banana-bread/
        [butter-based] [banana] [chocolate]
      Composite  ###############.....  73/100

        Moisture Retention       ############........  61  (w=0.85)
        Structure & Leavening    ####################  100  (w=0.20)
        Sugar Balance            ####################  100  (w=0.15)

      Ratios:
        liquid/flour:    1.786  !!
        fat/flour:       0.540  ok
        sugar/flour:     0.952  ok
        leavening/flour: 0.0272 ok
        fat source:      butter

      Why this score:
        Recipe scored 73/100 with well-balanced leavening and sugar.
        Butter-based fat is good for initial moisture but loses ground
        to oil-based recipes on day 2+. Liquid/flour ratio is slightly
        above the optimal range, which may soften the crumb further.

==============================================================
  Total: 29.2s  (budget 60s)  [OK]
  Backend: groq (llama-3.1-8b-instant)
  Ratio cache: 2/4 hits - skipped parse+normalize+ratio
==============================================================
```

---

## Scoring model

### Ratio reference ranges

Ratios are computed from normalized gram weights (King Arthur Baking + USDA density tables).

**Quick bread / banana bread:**

| Ratio | Optimal range | What it predicts |
|---|---|---|
| liquid / flour | 0.85 – 1.50 | Crumb moisture; banana counts as liquid |
| fat / flour | 0.28 – 0.65 | Tenderness and shelf life |
| sugar / flour | 0.35 – 0.90 | Sweetness balance; sugar is hygroscopic |
| leavening / flour | 0.008 – 0.030 | Rise, crumb openness |
| fat source | oil > mixed > butter | Multi-day moisture retention |

**Cookie:**

| Ratio | Optimal range | What it predicts |
|---|---|---|
| butter / flour | 0.40 – 0.75 | Spread and richness |
| sugar / flour | 0.55 – 1.10 | Sweetness and texture |
| brown / white sugar | 0.5 – 3.0 | Chew (high brown) vs crispness (high white) |
| leavening / flour | 0.005 – 0.025 | Lift and spread control |

### Criteria types

| Type | Description |
|---|---|
| **Fixed** | Always scored; failures penalize regardless of user profile (leavening floor, constraint satisfaction) |
| **Variable** | Weighted by query context + user preference model (moisture retention, sugar balance) |

### Dynamic weight derivation

Weights shift based on what you ask for:

- `"stays moist for days"` → moisture retention weight boosted to ~0.85
- `"crispy edges"` → structure weight boosted
- `"not too sweet"` → sugar balance weight boosted

---

## Persistence

All persistent data lives in `~/.bakesquad/`:

| Store | Format | Purpose |
|---|---|---|
| `bakesquad.db` → `ratio_cache` | SQLite | Ratio results keyed by URL; cache hits skip parse + normalize + ratio |
| `bakesquad.db` → `liked_recipes` | SQLite | Recipes you've saved with ratings and notes |
| `user_prefs.json` | JSON | Base criterion weights (updated over time) |

On a **cache hit**, the pipeline skips steps 7–9 entirely and jumps straight to scoring. Observed speedup: ~29 s first run → ~19 s with 3/4 cache hits.

---

## Project structure

```
bakesquad/
├── main.py                    # Entry point — full pipeline with per-step timing
├── requirements.txt
├── CONTEXT.md                 # Full system specification
└── bakesquad/
    ├── config.py              # Pipeline constants and step time budgets
    ├── models.py              # Pydantic data models (SearchSnippet → ScoredRecipe)
    ├── llm_client.py          # Multi-backend LLM wrapper (ollama / groq / claude)
    ├── memory.py              # SQLite + JSON persistence layer
    ├── normalizer.py          # Unit → grams conversion (lookup table, no LLM)
    ├── ratio_engine.py        # Deterministic ratio math + SQLite cache
    ├── parser.py              # Parallel LLM recipe parsing
    ├── scorer.py              # Scoring math + batched LLM explanations
    └── search/
        ├── ingestion.py       # Steps 1–6: query plan → search → fetch
        └── prompts.py         # Plain-string prompt builders
```

---

## Design decisions and tradeoffs

### No LangChain

CONTEXT.md specifies LangChain. We bypass it and call the OpenAI-compatible / Anthropic APIs directly. LangChain template + chain overhead added ~1–2 s per LLM call, which was incompatible with the 60 s total budget. All deviations are commented in the code at the point of deviation.

### Fixed pipeline, not ReAct

The orchestrator is a deterministic 11-step pipeline, not a ReAct loop. For a task with well-defined sub-steps, a fixed pipeline is strictly better: predictable latency, easy to budget, and impossible for the LLM to get stuck in a reasoning loop. ReAct would buy nothing here and cost budget control.

### Sequential DuckDuckGo search

The spec calls for parallel search queries. DDGS v9 uses `primp` (a Rust HTTP client) which deadlocks when called from multiple Python threads on Windows. With only 2 queries, sequential execution takes ~2–3 s — well within the 8 s search budget — so parallelism provides no meaningful benefit.

### Domain weighting

No developer-defined domain tier is applied. All sources compete on snippet relevance score alone. If two snippets score equally, a user-configured trusted source wins the tiebreak. This avoids recreating authority bias under a different name.

---

## Known limitations

- Unit normalization is imperfect for vague measurements (`"a handful of chocolate chips"`) — these are flagged as low-confidence
- Ingredient ratio ranges encode opinionated food science sources (King Arthur, Serious Eats, Stella Parks) — alternatives may disagree
- Technique signals (resting dough, browning butter) are not scorable from ingredient lists; recipe specificity (instruction count) is used as a proxy
- Paywalled Substack posts return a login wall — detected at snippet pre-check and skipped
- No ground truth for validation without a controlled blind tasting study

---

## Validation plan

Pick 4–5 recipes spanning the agent's score range for a single category. Bake under controlled conditions. Blind taste test with 3+ people. Check whether rank order correlates with agent scores. Even a small sample provides real empirical results — a weak correlation is a finding, not a failure.
