# Per-domain result cap — flat across all domains.
# No developer-defined domain tiers: all sources compete on snippet relevance score alone.
# Users can configure their own trusted sources; those are used as a tiebreaker only.
DOMAIN_CAP = 2

# Pipeline thresholds
MIN_CANDIDATE_THRESHOLD = 5       # below this, trigger a retry with relaxed queries
RELEVANCE_SCORE_CUTOFF = 0.4      # snippets below this score are dropped at step 3
SUBSTACK_PAYWALL_CUTOFF = 80      # excerpt shorter than this chars on substack.com → skip
SNIPPET_SCORE_CHUNK_SIZE = 40     # max snippets per LLM scoring call
MIN_PAGE_CONTENT_CHARS = 500      # fetched pages shorter than this are dropped as thin/gated

# Search
QUERIES_PER_RUN = 5               # number of diversified query variants to generate
MAX_RESULTS_PER_QUERY = 8         # DuckDuckGo results per query
MAX_SEARCH_WORKERS = 3            # parallel DDG threads (keep low to avoid rate limits)

# Fetch
MAX_FETCH_WORKERS = 4             # parallel page fetches
FETCH_TIMEOUT_SECONDS = 15

# LLM
DEFAULT_OLLAMA_MODEL = "qwen3"
LLM_TEMPERATURE = 0
