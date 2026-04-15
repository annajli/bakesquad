# BakeSquad

A recipe scoring agent that cuts through review-inflation bias. Instead of ranking by social proof (star ratings, review counts), BakeSquad evaluates recipes using **ingredient ratios**, **baking science principles**, and **LLM-assessed technique quality** — then personalizes ranking using a preference profile built from your history.

---

## The problem

AllRecipes, Food Network, and similar sites rank by PageRank-style popularity. A recipe posted in 2010 with 4,000 reviews consistently outranks a technically superior recipe from 2023. BakeSquad evaluates the recipe itself — its ratios and structure — not its audience size.

You can also paste in a URL you found anywhere and score it directly, building a personal corpus of vetted recipes that improves future recommendations.

---

## How it works

BakeSquad is a LangGraph-orchestrated agent with a multi-step pipeline from natural language query (or URL) to ranked output.

```
User input: query or URL
       │
       ▼
[classify_intent]  — turn type: new_search | re_filter | re_search | factual
       │
       ▼
[expand_query]     — Step 1: QueryPlan via LLM (category, constraints, query variants)
       │         └─ confidence < 0.55 → [clarify] node asks user a question
       ▼
[search]           — Steps 2–5: DuckDuckGo, snippet scoring, domain cap, adaptive retry
       │
       ▼
[fetch]            — Step 6: JSON-LD extraction (primary) → BS4 fallback; parallel
       │
       ▼
[parse]            — Step 7: parallel LLM recipe parsing into structured ingredients
       │
       ▼
[verify]           — Step 7b: majority-vote category reconciliation
       │
       ▼
[score]            — Steps 8–10: unit normalization → ratio engine → per-category
       │             scoring math → LLM explanations + LLM-assessed criteria
       │             + corpus recall (semantically similar previously-scored recipes)
       ▼
[memory]           — cache ratios, save embeddings, update preference model
       │
       ▼
Ranked output
```

### Conversation turns

After an initial search, follow-up messages are classified and routed:

| Turn type | Example | Behaviour |
|---|---|---|
| `re_filter` | "only show me oil-based ones" | Zero LLM — filter existing results |
| `re_search` | "actually I want chocolate banana bread" | Full new pipeline with prior context |
| `factual` | "why does oil keep bread moist longer?" | Single LLM answer, no search |

---

## Scoring model

Scoring is **per-category** — each baked-good type has its own named criteria reflecting the quality axes that actually matter for that style.

### Category criteria

| Category | Criteria | Notes |
|---|---|---|
| `cookie` | Chew & Texture, Spread & Structure, Sweetness Balance, Flavor & Technique† | Brown/white sugar split predicts chew; leavening type predicts spread |
| `quick_bread` | Moisture & Tenderness, Rise & Dome, Sweetness Balance | Over-leavening flagged as tunneling risk |
| `cake` | Moisture & Tenderness, Crumb & Structure, Sweetness Calibration | Custard cakes (cheesecake, flourless) bypass flour-ratio scoring |
| `yeasted_bread` | Hydration, Enrichment Level, Flavor Complexity† | Baker's % hydration; lean vs enriched style |
| `pastry` | Fat & Richness, Structure & Balance, Technique & Layers† | Fat/flour is dominant signal; wide range covers shortcrust → croissant |
| `brownie` | Fudge Factor, Chocolate Intensity†, Sweetness & Crust | High fat/flour and no leavening are correct signals, not defects; Chocolate Intensity covers cocoa type and bloom technique; Blondies use the same criteria with Chocolate Intensity scoring butterscotch/brown butter depth instead |
| `other` | Overall Balance | Sugar/flour as general proxy |

† LLM-assessed: starts at placeholder 50/100; scored 0–100 in the batched explanation call.

Universal add-ons (applied after category criteria):

| Criterion | When active | Weight |
|---|---|---|
| GF Binding Agent | Gluten-free modifiers or non-AP flour | 0.20 fixed |
| Accessibility | `prefer_accessibility > 0` in user prefs | 0–0.20 scaled |

### Dynamic weight derivation

Weights shift based on query signals:

- `"stays moist for days"` → Moisture & Tenderness boosted
- `"crispy edges"`, `"open crumb"` → structure criterion boosted
- `"brown butter"`, `"fermented"` → flavor criterion boosted

Weights also learn from your liked recipe history — per-category inferences stored in `user_prefs.json`.

### Ratio reference ranges

All ratios are computed from normalized gram weights (King Arthur Baking + USDA density tables). Ranges are calibrated per flour type (AP, almond, oat, coconut, GF blend).

| Category | Key ratios |
|---|---|
| Quick bread | liquid/flour 0.85–1.50, fat/flour 0.28–0.65, leavening/flour 0.008–0.030 |
| Cake | liquid/flour 0.80–1.25, fat/flour 0.35–0.80, sugar/flour 0.65–1.20 |
| Cookie | fat/flour 0.40–0.75, brown/white sugar 0.5–3.0, leavening/flour 0.005–0.025 |
| Yeasted bread | liquid/flour 0.55–0.85 (hydration), fat/flour 0.00–0.60 |
| Pastry | fat/flour 0.40–1.20, liquid/flour 0.15–3.00 |
| Brownie | fat/flour 1.00–2.50, sugar/flour 2.00–5.00, leavening/flour 0.000–0.015 |

---

## Supported categories

| Category | Includes |
|---|---|
| `cookie` | Drop cookies, bar cookies, shortbread |
| `quick_bread` | Banana bread, zucchini bread, muffins, scones, cornbread |
| `cake` | Layer cakes, bundt cakes, cheesecakes, cupcakes |
| `yeasted_bread` | Sourdough, focaccia, baguette, dinner rolls, brioche, challah |
| `pastry` | Croissants, danish, choux (éclairs, profiteroles), tarts, pies |
| `brownie` | Fudgy brownies, cakey brownies, blondies, marble brownies |
| `other` | Anything uncategorizable |

---

## Persistence

All data lives in `~/.bakesquad/`:

| Store | Format | Purpose |
|---|---|---|
| `bakesquad.db → ratio_cache` | SQLite | Ratio results keyed by URL; cache hits skip parse + normalize entirely |
| `bakesquad.db → liked_recipes` | SQLite | Recipes saved with ratings, notes, tried date |
| `bakesquad.db → recipe_embeddings` | SQLite | Hash-based embeddings for semantic corpus recall |
| `bakesquad.db → user_feedback` | SQLite | Event log: liked / disliked / tried / note |
| `user_prefs.json` | JSON | Scoring weights + per-category inferences from feedback history |
| `graph.db` | SQLite | LangGraph conversation checkpoints (multi-turn state) |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file:

```env
ANTHROPIC_API_KEY=your_anthropic_key_here
GROQ_API_KEY=your_groq_key_here        # optional
```

### 3. Run the API server

```bash
MODEL_BACKEND=claude uvicorn bakesquad.api:app --reload --port 8000
```

### 4. Run the CLI (optional)

```bash
MODEL_BACKEND=claude python main.py "chewy brown butter chocolate chip cookies"
MODEL_BACKEND=claude python main.py "sourdough with open crumb" --recency year
```

---

## LLM backends

| `MODEL_BACKEND` | Model | Notes |
|---|---|---|
| `claude` | `claude-sonnet-4-20250514` | Best quality; recommended |
| `groq` | `llama-3.1-8b-instant` | Free tier; fast; good for development |
| `ollama` | `qwen3:8b` (local) | No cost; requires Ollama running locally |

All calls route through `bakesquad/llm_client.py` — a thin wrapper handling backend selection, rate-limit retries, and think-tag stripping.

---

## API

The FastAPI server runs at `http://localhost:8000`. Interactive docs at `/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/search` | Natural-language query → scored recipes. Pass `thread_id` to continue a conversation. |
| `POST` | `/api/score-url` | Paste a URL → score that recipe and store it in your corpus |
| `GET` | `/api/saved` | All recipes in your personal corpus |
| `POST` | `/api/feedback` | Record liked / disliked / tried / note for a recipe URL |
| `GET` | `/api/prefs` | Current scoring preferences |
| `PATCH` | `/api/prefs` | Update preferences (accessibility weight, fat preference, etc.) |
| `GET` | `/api/health` | Liveness check + backend info |

`POST /api/search` returns both `recipes` (fully scored) and `candidates` (all URLs that passed snippet scoring, including those that failed to fetch or parse) — so you can visit high-relevance recipes that couldn't be scored automatically.

---

## Project structure

```
bakesquad/
├── main.py                    # CLI entry point
├── requirements.txt
├── categories.yaml            # Category registry: criteria, ratio keys, synonyms
└── bakesquad/
    ├── api.py                 # FastAPI REST layer
    ├── config.py              # Pipeline constants and time budgets
    ├── models.py              # Pydantic models (SearchSnippet → ScoredRecipe)
    ├── llm_client.py          # Multi-backend LLM wrapper
    ├── memory.py              # SQLite + JSON persistence; preference inference; embeddings
    ├── normalizer.py          # Unit → grams conversion (lookup table, no LLM)
    ├── ratio_engine.py        # Deterministic ratio math + SQLite cache
    ├── parser.py              # Parallel LLM recipe parsing
    ├── scorer.py              # Per-category scoring + batched LLM explanations
    ├── session.py             # Turn classification helpers
    ├── category_registry.py   # categories.yaml loader
    ├── graph/
    │   ├── builder.py         # LangGraph StateGraph assembly + SqliteSaver checkpointing
    │   ├── nodes.py           # Node implementations (classify_intent, search, score, memory…)
    │   └── state.py           # BakeSquadState TypedDict
    └── search/
        ├── ingestion.py       # Steps 1–6: query plan → DDG search → fetch (JSON-LD + BS4)
        └── prompts.py         # Plain-string prompt builders
```

---

## Design decisions

### LangGraph over a fixed pipeline

The original implementation was a deterministic 11-step pipeline. LangGraph adds multi-turn conversation routing (re_filter, re_search, factual), clarification interrupts when category confidence is low, and SqliteSaver checkpointing for persistent conversation state across API requests — none of which are possible with a fixed pipeline.

### JSON-LD extraction before BS4

Most major recipe sites and WordPress recipe plugins embed `schema.org/Recipe` structured data. JSON-LD extraction is tried first — it's machine-readable, survives layout changes, and dramatically improves parse success rates. BS4 heuristic extraction is the fallback for sites that don't implement structured data.

### Per-category scoring over universal criteria

A single rubric (moisture/structure/balance) applied across all categories loses the signal that matters per style: chew and spread for cookies, hydration for yeasted breads, lamination for pastry. Each category now has its own named criteria; LLM-assessed criteria handle axes (flavor complexity, technique quality) that ingredient ratios can't reach.

### No LangChain

LangChain template and chain overhead added ~1–2 s per LLM call, incompatible with the 60 s time budget. All LLM calls route through a thin `llm_client.py` wrapper instead.

### Brownie as a first-class category

Brownies and blondies have a ratio signature that is structurally incompatible with both `cookie` and `cake`: fat/flour 1.0–2.5, sugar/flour 2.0–5.0, and intentionally absent leavening. Both values read as severe out-of-range violations under cake or cookie scoring, producing scores in the 16–26/100 range for well-regarded recipes. The fix is a dedicated `brownie` category with its own ratio ranges, scoring criteria, and parser disambiguation notes.

The ratio cache stores computed ratio numbers keyed by URL. On a cache hit, the stored `category` field is ignored in favour of the live parser's output — so adding new categories never requires manual cache invalidation.

### Sequential DuckDuckGo search

DDGS v9 uses `primp` (a Rust HTTP client) that deadlocks when called from multiple Python threads on Windows. With 2 queries, sequential execution takes ~2–3 s — well within budget.

---

## Known limitations

- Unit normalization is imperfect for vague measurements (`"a handful of chocolate chips"`) — flagged as low-confidence
- Technique signal recall is low (~8%) — web recipes describe technique in prose, not vocabulary terms. Differentiation comes from LLM-assessed criteria and `technique_notes` delta
- JavaScript-rendered sites (Serious Eats, NYT Cooking) return empty content to requests — neither JSON-LD nor BS4 can help
- Paywalled content (Substack, Patreon) is detected at snippet pre-check and skipped; URLs are still surfaced in the `candidates` response field
- No ground truth for scoring validation without a controlled blind tasting study
