# Baking Agent — Project Context

## Overview

A recipe search and scoring agent that cuts through review-inflation bias on popular recipe sites. Instead of returning the most popular recipes by PageRank, the agent evaluates recipes based on ingredient ratios, baking science principles, and personalized user preferences. It takes a natural language query, searches the web for candidate recipes, parses and analyzes them, and returns a ranked list with per-criterion score explanations.

**Course context:** Final project for an AI agents workshop. Deliverables are a working system, a presentation, and a written report covering problem statement, system design, and results.

**Core problem:** Review-aggregator sites (AllRecipes, Food Network, etc.) rank recipes by social proof, not quality. A recipe posted in 2010 with 4,000 reviews outranks a technically superior recipe from 2023. This agent evaluates the recipe itself — its ingredient ratios and structure — not its popularity.

---

## Tech Stack

- **Agent framework:** LangChain
- **Search tool:** DuckDuckGo (`DuckDuckGoSearchResults`)
- **Local model (candidate):** [qwen3](https://ollama.com/library/qwen3-next) via Ollama
- **Page fetching:** `requests` + `BeautifulSoup`, or LangChain `WebBaseLoader`
- **Language:** Python

---

## System Architecture

### Layers

```
User Interface
  ├── Query input (natural language, URL, or raw text paste)
  ├── Search settings (recency dial, user-configured trusted sources)
  └── User profile (liked recipes, preference weights)

Orchestrator (LLM brain)
  ├── Search & Ingestion subsystem
  └── Memory subsystem (persistent across sessions)
  └── Processing Pipeline (shared)

Output: Ranked results with score explanations
```

### Orchestrator responsibilities

The orchestrator is the LLM reasoning layer. It is not a simple function router — it plans and sequences tool calls based on the query and user context. It reads from memory before doing anything, decides how many search queries to fire, evaluates whether the candidate pool is sufficient, and synthesizes natural language explanations of scores.

---

## Tools

### 1. Search & Ingestion

Runs as a multi-step loop — not a single tool call.

**Step 1 — Query diversification**
From the user's natural language input, generate N query variants:
- Broad variant: `"banana bread chocolate recipe"`
- Specific variant: `"chocolate chip banana bread moist"`
- Recency variant: filtered to past year/month (mapped from user's recency dial)
- Source-targeted variants: only if the user has configured trusted sources — append `site:` queries for those domains. No developer-defined source list is applied by default.

**Step 2 — Multi-query search**
Run all N queries via DuckDuckGo. Collect snippets (title + excerpt + URL). Deduplicate across queries.

> Note: DuckDuckGo returns snippets, not full page content. A separate fetch step is required.

**Step 3 — Snippet relevance pre-check**
Before fetching any full pages, run a fast LLM pass scoring each snippet against the parsed query intent. This avoids wasting fetch and parse cycles on irrelevant results. This is also where per-domain selection happens: if AllRecipes appears 5 times, only the highest-scoring snippet advances.

Substack handling: paywalled posts typically have thin or missing excerpts — detect and skip these at this stage.

**Step 4 — Domain cap + selection**
Enforce a per-domain cap (suggested: 1–2 recipes per domain). Output: a ranked, deduplicated list of URLs diverse across sources.

**Step 5 — Adaptive fetch decision**
If fewer than 5 relevant URLs survive steps 3–4, loop back to step 2 with expanded/relaxed queries. If ≥ 5 survive, continue to full fetch.

**Step 6 — Full page fetch**
Fetch complete page content for each selected URL using `WebBaseLoader` or `requests` + `BeautifulSoup`.

### 2. LLM Parser

Takes raw page HTML/text, outputs a structured recipe object:
```json
{
  "title": "...",
  "url": "...",
  "ingredients": [
    { "name": "all-purpose flour", "quantity": 200, "unit": "g" },
    ...
  ],
  "yield": "1 loaf",
  "instructions": ["..."]
}
```

Using the LLM to parse (rather than CSS selectors) makes the system robust to layout variation across sites and handles Substack, personal blogs, and other non-standard formats automatically.

### 3. Unit Normalizer

Converts all ingredient quantities to grams using established density tables (King Arthur, USDA). Flags low-confidence conversions (e.g., "a handful of chocolate chips") explicitly in output rather than silently guessing — downstream ratio scores for flagged ingredients carry reduced confidence.

### 4. Ratio Engine

Computes category-specific ingredient ratios from the normalized recipe. Results are **cached per recipe** (keyed on URL or ingredient hash) — ratios are deterministic math and do not change between users or queries.

**Quick bread / cake ratios:**
- Liquid-to-flour ratio (moisture predictor)
- Fat-to-flour ratio
- Sugar-to-flour ratio
- Fat type: oil vs butter (oil predicts better multi-day moisture retention)
- Leavening ratio (baking soda / baking powder to flour)

**Cookie-specific ratios:**
- Brown-to-white sugar ratio (brown sugar → chew + moisture; white sugar → spread + crispness)
- Butter-to-flour ratio
- Egg yolk vs whole egg ratio
- Leavening: baking soda vs baking powder balance

**Outlier handling:** Ratio values outside consensus ranges are flagged as either likely errors or deliberate stylistic choices. The LLM reasoning step distinguishes between these — a very high hydration in a banana bread may be intentional (fudgy, dense texture), not a mistake.

### 5. Scorer

Takes ratio engine output + user preference model → weighted score per criterion → composite score.

**Key design principle:** Criteria are split into two types:

| Type | Description | Examples |
|---|---|---|
| Fixed | Always measured; failures filter or penalize regardless of user | Recipe specificity, ratio plausibility, leavening floor, hard constraints from query |
| Variable | Weighted by user preference model + query context | Sweetness balance, moisture retention, chew, fat source preference, texture profile |

Fixed criteria define the **candidate pool** (recipes failing them are filtered before personalization runs). Variable criteria define **ranking within that pool**.

Scoring weights are generated dynamically from the query — "stays moist for days" boosts moisture retention weight to ~0.9 for that search, even if the user's base profile doesn't set it that high.

**Hard constraints vs preferences:** A hard constraint from the query (e.g., "chocolate not overpowering") is treated differently from a preference dial. Constraint violations are flagged visibly in the output ("Constraint risk") even if the recipe otherwise scores well.

---

## Memory System

Persistent across sessions. Three components:

### Preference model
Structured weights derived from: (a) liked recipe loading with user notes, (b) explicit preference elicitation conversation at onboarding, (c) accumulated post-bake feedback. Informs both query understanding (boosts relevant weights even when not explicitly stated in query) and scoring.

### Liked recipe store
Full structured recipe objects for recipes the user has explicitly saved. Used for similarity-based recommendations and as seeding data for the preference model.

### Ratio cache
Stores ratio engine output keyed by recipe URL (or ingredient hash). Since ratios are deterministic, a previously-analyzed recipe skips parse + ratio steps and goes straight to scoring. Builds up a library of pre-analyzed recipes over time.

### Feedback log
Time-stamped record of: recipe recommended → whether user baked it → post-bake rating + notes. Raw history that feeds preference model updates. Even a simple thumbs-up/down per result provides a useful signal over multiple sessions.

---

## Search & Ingestion Design Decisions

### Countering PageRank bias
- **Query diversification** surfaces different result sets from the same search tool
- **Per-domain cap** prevents any single domain from flooding the candidate pool
- **Recency filter** (user-selectable) surfaces newer recipes that haven't accumulated reviews
- **Snippet pre-check** selects by relevance to the query, not by position in search results
- **Ratio engine as equalizer** — once a recipe is in the pool, its score is based entirely on ingredient math, not source. A technically excellent recipe from an unknown blog can outscore a mediocre one from a famous site.

### Domain weighting — design decision and tradeoff

Hardcoding a preferred domain list (e.g. always favoring Serious Eats over AllRecipes) substitutes *editorial credibility* for *review volume* as the trust signal. This is arguably better than raw popularity, but it still recreates a form of authority bias — it disadvantages newer food writers, niche Substack newsletters, and personal blogs from skilled home bakers who haven't built large audiences. That's a softer version of the same problem the agent sets out to solve.

**Chosen approach: domain weighting is a tiebreaker only, and is user-configured.**

- No developer-defined domain tier is applied by default
- The snippet relevance pre-check does all candidate selection work, scored against the query — not the source
- If two snippets score equally on relevance, a user-configured trusted source wins the tiebreak
- Users can add their own trusted sources (specific Substack writers, blogs they follow, etc.) — this shifts curation from the developer to the person who actually knows what they like
- The per-domain cap (1–2 results per domain) prevents flooding without privileging any particular source

### Substack specifics
Public Substack posts are indexable and fetchable. Paywalled posts return a login wall — detected at snippet pre-check stage (thin/missing excerpt) and skipped. Worth including as a domain for food newsletter writers.

### Adaptive fetching trigger
Two conditions trigger a second fetch round:
1. Fewer than 3–5 relevant candidates survive snippet pre-check
2. No candidates satisfy the hard constraints from the query (e.g., none actually contain chocolate at a plausible ratio)

---

## Output Format

Each result card includes:
- Recipe title, source domain, URL
- Composite score (0–100)
- Per-criterion score bars with numeric values
- Expandable "Why this score" — natural language explanation translating ratio math into eating-experience predictions
- Tags (key technique signals: browned butter, oil-based, rested dough, etc.)
- Constraint risk badge when a hard query constraint is likely violated
- "Review inflated" badge when a high-review-count recipe scores poorly

Top of results: "How I understood your query" card showing parsed category, extracted constraints, scoring weights used, and which weights were boosted from profile memory.

---

## Query Understanding

The orchestrator's first step on receiving a natural language query:

1. **Extract category** — what type of baked good (quick bread, cookie, cake, etc.)
2. **Extract hard constraints** — things that must be true ("has chocolate", "stays moist for days")
3. **Extract preferences** — soft requirements that inform weighting ("not overpowering")
4. **Merge with profile** — boost weights from user's preference history even if not stated
5. **Generate search queries** — diversified variants for the ingestion loop
6. **Determine criteria weights** — dynamic weight vector for this specific query

---

## Limitations (for report)

- Unit normalization is imperfect for volume measurements; low-confidence conversions are flagged
- Technique matters as much as ratios but is hard to score from ingredient lists alone — recipe specificity (precision of instructions) is used as a proxy
- Substack paywalled content is not accessible
- Ratio canonical ranges are contested; the reference table encodes opinionated food science sources
- No ground truth for validation without a controlled blind tasting study
- Candidate pool construction is still an open problem — any source-weighting heuristic risks recreating authority bias under a different name; the chosen approach (relevance-first, user-configured tiebreaker) mitigates but does not fully resolve this

---

## Validation Plan

Pick 4–5 recipes spanning a range of agent scores for a single category. Bake under controlled conditions. Blind taste test with 3+ people. Check whether rank order correlates with agent scores. Even a small sample provides real empirical results for the report — if correlation is weak, that is a finding, not a failure.

---

## Open Questions / TODOs

- [ ] Decide on two focus categories for MVP (suggested: cookies + quick breads/banana bread)
- [ ] Build ratio reference table for each category (canonical ranges from Serious Eats, King Arthur, Stella Parks)
- [ ] Define preference elicitation conversation flow (concrete questions → mapped to weights)
- [ ] Determine ratio cache storage format (SQLite, JSON file, or in-memory dict for MVP)
- [ ] Test qwen3 parsing quality against a Claude/GPT-4 class model on messy recipe HTML
- [ ] Design user-configurable trusted sources interface (how users add/remove their own preferred domains)
- [ ] Implement feedback loop (even a simple thumbs up/down per result)
- [ ] Define ingredient substitution handling (from brainstorm doc: if no matching recipe found, generate a new one based on top-ranked recipe + substitution rules)