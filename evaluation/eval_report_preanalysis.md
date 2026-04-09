# BakeSquad Evaluation Report — Pre-Analysis
### (Code-review based; no LLM run required)

**Prepared:** 2026-04-09  
**Scope:** Static analysis of `bakesquad/` codebase + design of evaluation framework  
**Purpose:** Diagnose the known banana-bread / zucchini-bread misclassification before
running a live evaluation. Serves as the baseline interpretation of what the evaluation
script should confirm or refute.

---

## 1. Summary of Findings

| Finding | Severity | Stage |
|---------|----------|-------|
| Banana bread / zucchini bread reported as `other` | High | Step 1 |
| Prompt already contains correct examples — not a schema gap | — | Step 1 |
| Step 7 re-classification is computed but never fed back | Medium | Step 7 |
| No feedback loop — Step 1 errors propagate to all downstream steps | High | Architecture |
| `loaf` framing not listed as a quick_bread synonym | Low | Step 1 |
| No existing automated tests | High | Testing |

**Overall:** The pipeline is architecturally sound but brittle at the single Step 1
classification decision. The known misclassification is most likely a **model-dependent
reasoning failure**, not a prompt design flaw — because the prompt already has the right
examples. The fix is a combination of (a) prompt hardening for edge framing and
(b) using Step 7's re-classification as a correction signal.

---

## 2. Known Issue: Banana Bread / Zucchini Bread → `other`

### What the prompt says

`bakesquad/search/prompts.py:query_plan_prompt()` (lines 39–83) contains:

```
"category": the type of baked good — one of: cookie, quick_bread, cake, other
  quick_bread includes: banana bread, zucchini bread, pumpkin bread, muffins, scones, cornbread
```

And includes **two explicit few-shot examples** of the exact failing queries:

```
Input: "banana bread"
→ {"category":"quick_bread", ...}

Input: "zucchini bread with chocolate chips"
→ {"category":"quick_bread", ...}
```

### Hypothesis matrix

| Root Cause | Likely? | Evidence |
|------------|---------|---------|
| Missing from prompt schema | **No** | Schema line explicitly lists both |
| Missing from few-shot examples | **No** | Both have their own examples |
| LLM reasoning failure (weaker models) | **Yes** | Likely — Groq llama-3.1-8b may override examples |
| Temperature variance / non-determinism | **Possible** | Local/quantised models aren't always deterministic at T=0 |
| `other` as the default fallback | **Possible** | If the model is uncertain, it may pick the safe bucket |
| `bread` keyword triggering yeast-bread association | **Yes** | Models trained broadly may associate "bread" with sourdough/baguette |

### Why "bread" may override few-shot examples

LLMs with strong priors about the word "bread" (as a yeast-leavened product) may
discount the explicit in-context examples when the surface-level keyword strongly
activates a different concept. This is a known *prior–context tension* failure mode,
more common in small/quantised models (Groq llama-3.1-8b, Qwen3-8b) than in larger
frontier models (Claude Sonnet).

**Prediction for evaluation run:**
- With `MODEL_BACKEND=claude` → TC25, TC26 will pass (Claude respects few-shot well)
- With `MODEL_BACKEND=groq` → TC25, TC26 may fail (llama-3.1-8b has weaker instruction following)
- With `MODEL_BACKEND=ollama` (qwen3:8b) → Inconsistent, model-run dependent

---

## 3. Pipeline Architecture Analysis

### The 11-step fixed DAG

```
User query
    │
    ▼
[Step 1] query_plan_prompt + LLM
    │  Sets: category, flour_type, modifiers, queries
    │  ← SINGLE POINT OF FAILURE
    │  ← category is LOCKED here and never revised
    ▼
[Steps 2–5] DuckDuckGo search + snippet scoring
    │  Uses: plan.queries
    ▼
[Step 6] Page fetch (parallel)
    ▼
[Step 7] LLM parse (parallel)
    │  Also classifies: category, flour_type, modifiers per recipe
    │  ← This classification is COMPUTED but IGNORED for scoring
    ▼
[Steps 8–9] Normalize + compute ratios
    │  Uses: plan.category to select RATIO_RANGES reference bucket
    │  ← Wrong category → wrong reference ranges → meaningless ratio scores
    ▼
[Step 10] Score + explain
    │  Uses: plan.category for criterion selection
    │  ← Wrong category → wrong scoring criteria
    ▼
Output
```

### Impact of Step 1 misclassification

If `banana bread` is classified as `other` at Step 1:

1. **Step 9**: Ratios computed but no reference range exists for `other` — the ratio
   engine falls back to an empty/generic range, so all ratios score "out of range"
   regardless of actual values.

2. **Step 10**: Scoring uses `other` criteria — no `moisture_retention` or
   `leavening_validity` criteria that are specific to quick breads. The composite
   score is essentially random noise.

3. **No correction**: The pipeline finishes without error — it silently produces
   a bad result. The user sees ranked recipes with meaningless scores.

### Step 7 re-classification is wasted

`bakesquad/parser.py` (`_SYSTEM_PROMPT`) instructs the LLM parser to infer `category`
from ingredients (e.g., `mashed banana` → `quick_bread`). This is stored in
`ParsedRecipe.category` but is never compared to `QueryPlan.category` downstream.

In `main.py:run_pipeline()`, after Step 7, `parsed_recipes` are passed directly to
`compute_ratios()` which reads `plan.category` (from Step 1), not `recipe.category`.
The Step 7 re-classification is purely informational.

This is the most actionable architectural fix: a 5-line reconciliation step between
Steps 7 and 9.

---

## 4. Classification Schema Evaluation

### Category coverage

| Category | Schema members | Ambiguous edge cases |
|----------|---------------|----------------------|
| `cookie` | drop cookies, bar cookies, brownies, shortbread | Brookies (brownie+cookie)? Cookie cake? |
| `quick_bread` | banana bread, zucchini bread, pumpkin bread, muffins, scones, cornbread | Any "loaf" item, Dutch baby |
| `cake` | layer cakes, bundt cakes, cheesecakes, cupcakes | Coffee cake (also breakfast-y), lava cake |
| `other` | anything else | Most bread, pastry, yeasted items |

**Gap: `loaf` framing** — The schema examples show `bread` and `muffin` but not `loaf`.
A query like `"banana nut loaf"` gives the model no explicit signal. The correct output
is `quick_bread`, but models may classify it as `other` because "loaf" isn't in the
schema definition.

**Not a gap: brownies / lemon bars** — These are listed as `bar cookies` in the schema,
and most modern LLMs correctly classify them.

### Schema design tradeoffs

The hard `Literal["cookie", "quick_bread", "cake", "other"]` constraint:

**Pros:**
- Enforces valid Pydantic model instantiation
- Drives deterministic downstream logic (ratio ranges, scoring criteria)

**Cons:**
- No confidence signal — the model must commit to one category even when uncertain
- No multi-label support (e.g., a recipe that spans cake and quick_bread)
- `other` acts as a trash bucket with no scoring benefit

---

## 5. Comparison of Architecture Alternatives

### Option A: Current (Fixed DAG)
- **Latency**: 30–60 s (API), 60–180 s (Ollama)
- **LLM calls**: 3–4 total
- **Failure mode**: Single wrong Step 1 → silent bad output
- **Recovery**: None

### Option B: Step 7 Feedback (Recommended Near-Term Fix)
- **Latency**: +0 s (no extra LLM call)
- **LLM calls**: Same
- **Failure mode**: Reduced — majority-vote from parsed recipes overrides Step 1 if they disagree
- **Recovery**: Partial (helps when Step 1 is wrong but Step 7 is right)
- **Implementation**: ~10 lines in `main.py`

```python
# After parse_recipes_parallel(), before compute_ratios():
from collections import Counter
step7_cats = Counter(
    r.category for r in parsed if not r.parse_error and r.category != "other"
)
if step7_cats:
    dominant = step7_cats.most_common(1)[0][0]
    if dominant != plan.category:
        logging.warning("Category override: %s → %s", plan.category, dominant)
        plan = plan.model_copy(update={"category": dominant})
```

### Option C: Soft Classification
- **Latency**: +0 s (same LLM call, richer output)
- **LLM calls**: Same
- **Implementation**: Prompt change + downstream threshold logic
- **Benefit**: Explicit uncertainty signal; low-confidence queries can trigger
  a fallback or a second-pass confirmation

Prompt addition:
```
Also return "category_scores": an object mapping each category to a confidence
float (0.0–1.0) that sums to 1.0. Example:
{"cookie":0.05, "quick_bread":0.92, "cake":0.02, "other":0.01}
```

### Option D: ReAct / Tool-Augmented Agent
- **Latency**: 3–5× current (many LLM calls for tool use)
- **LLM calls**: Unpredictable
- **Not recommended** given the 60 s budget constraint

---

## 6. Recommended Action Plan

### Immediate (before next demo)

1. **Prompt hardening** — add `loaf` synonym and a tie-breaker example to
   `query_plan_prompt`. Estimated time: 10 minutes.

2. **Regression test** — add 5 pytest assertions for the known-issue queries.
   Estimated time: 15 minutes.

### Short-term (next sprint)

3. **Step 7 feedback loop** — reconcile Step 1 vs Step 7 category before ratios.
   Estimated time: 1 hour.

4. **Soft classification** — add `category_scores` to `QueryPlan` and use it in
   the Step 7 reconciliation logic. Estimated time: 2–3 hours.

### Long-term (if expanding the product)

5. **Expand `quick_bread` schema** — add explicit entries for scone variants,
   loaf cakes, and savory quick breads.

6. **Ambiguous query handling** — if `max(category_scores) < 0.6`, ask the user
   a clarifying question before searching.

---

## 7. How to Run the Live Evaluation

### Prerequisites

```bash
# Install deps (already done if main.py works):
pip install python-dotenv

# Set your backend:
export MODEL_BACKEND=groq    # or claude, ollama
export GROQ_API_KEY=...      # if using groq
export ANTHROPIC_API_KEY=... # if using claude
```

### Stage 1 only (recommended starting point, ~2 min on Groq)

```bash
cd /path/to/bakesquad
MODEL_BACKEND=groq python evaluation/evaluate_agent.py
```

### With debug output (prints every LLM prompt + raw response)

```bash
MODEL_BACKEND=groq python evaluation/evaluate_agent.py --debug
```

### Only the known-issue cases

```bash
MODEL_BACKEND=groq python evaluation/evaluate_agent.py --groups known_issue edge_case
```

### Full pipeline (needs network + API, ~5–10 min)

```bash
MODEL_BACKEND=groq python evaluation/evaluate_agent.py --full-pipeline --stage2-max 15
```

### All three backends (compare model behaviour)

```bash
MODEL_BACKEND=groq    python evaluation/evaluate_agent.py --output-dir evaluation/results/groq
MODEL_BACKEND=claude  python evaluation/evaluate_agent.py --output-dir evaluation/results/claude
MODEL_BACKEND=ollama  python evaluation/evaluate_agent.py --output-dir evaluation/results/ollama
```

### Output files

All outputs go to `evaluation/results/` (or `--output-dir`):

| File | Contents |
|------|----------|
| `eval_results_s1_<timestamp>.json` | Per-case results incl. full LLM traces |
| `eval_report_<timestamp>.md` | Human-readable analysis report |
| `confusion_matrix_<timestamp>.txt` | ASCII confusion matrix |

---

## 8. What to Look for in Results

### Confirming the known issue

Check `eval_results_s1_*.json` for TC25–TC29. If `predicted_category == "other"`,
look at `llm_trace[0].raw_response` to see whether:

- The model ignored the few-shot examples entirely (reasoning gap)
- The model acknowledged "banana bread" as quick_bread but changed its mind (conflict)
- The model gave a correct answer but JSON extraction failed (parsing issue)

### Diagnosing prompt vs reasoning

If TC25 (`"banana bread"`) fails but TC27 (`"chocolate chip banana bread"`) passes:
→ Adding more tokens around the key phrase helps → **prompt sparsity** issue

If TC25 fails with Claude but passes with Groq:
→ Model-specific, not prompt-specific → **model capability** issue

If TC25 and TC30 (`"banana nut loaf"`) both fail:
→ The word "banana" alone is insufficient → **keyword gap** in schema

If only TC30 fails:
→ The `loaf` framing is the issue → **ontology gap** (easily fixed with prompt change)
