# Per-domain result cap — flat across all domains.
# No developer-defined domain tiers: all sources compete on snippet relevance score alone.
# Users can configure their own trusted sources; those are used as a tiebreaker only.
DOMAIN_CAP = 2

# Pipeline thresholds
# DEVIATION from CONTEXT.md: MIN_CANDIDATE_THRESHOLD reduced from 5 to 3.
# Reason: we fetch at most 4 pages per run, so requiring 5 candidates is unreachable
# without triggering the adaptive retry every single time.
MIN_CANDIDATE_THRESHOLD = 3
RELEVANCE_SCORE_CUTOFF = 0.4
SUBSTACK_PAYWALL_CUTOFF = 80      # excerpt shorter than this chars on substack.com → skip
SNIPPET_SCORE_CHUNK_SIZE = 40     # max snippets per LLM scoring call (batched, never 1/snippet)
MIN_PAGE_CONTENT_CHARS = 300      # fetched pages shorter than this are dropped as thin/gated

# Search
# DEVIATION from CONTEXT.md: QUERIES_PER_RUN capped at 2 (broad + specific) per the explicit
# budget requirement: "cap at 2 search queries unless the user explicitly requests recency
# filtering." CONTEXT.md describes up to 5 variants; we generate exactly 2 for speed.
QUERIES_PER_RUN = 2
MAX_RESULTS_PER_QUERY = 8
MAX_SEARCH_WORKERS = 2            # parallel DDG threads — 2 because we only fire 2 queries

# Fetch
# DEVIATION from CONTEXT.md: timeout reduced from 15s to 5s per the explicit spec requirement:
# "Enforce a 5-second timeout per request; drop non-responsive pages rather than waiting."
MAX_FETCH_WORKERS = 4
FETCH_TIMEOUT_SECONDS = 5
MAX_PAGES_PER_RUN = 4             # hard cap — fetch at most 4 pages per run

# LLM
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
LLM_TEMPERATURE = 0

# Step time budgets in seconds (Claude and Groq backends — Ollama is relaxed to 180s total)
STEP_BUDGETS = {
    "query_understanding": 5,
    "search": 8,
    "snippet_precheck": 6,
    "page_fetch": 10,
    "llm_parsing": 15,
    "normalization_ratios": 2,
    "scoring_explanations": 8,
    "output": 1,
}
