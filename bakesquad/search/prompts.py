from langchain_core.prompts import ChatPromptTemplate

# Combined step-1 prompt: query understanding + diversification in one LLM call.
# Returns a JSON object with category, hard_constraints, soft_preferences, and queries.
#
# {trusted_sources_instruction} is populated at runtime:
#   - If the user has configured trusted sources → instructs the LLM to add site:-targeted variants
#   - If not → empty string; no developer-defined source list is injected
# {recency_instruction} is populated at runtime based on the user's recency dial setting.
QUERY_PLAN = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a baking recipe search assistant. Given a user's natural language query, "
            "analyze it and return a JSON object with exactly these keys:\n\n"
            "- \"category\": the type of baked good — one of: cookie, quick_bread, cake, bread, other\n"
            "- \"hard_constraints\": list of strings — things that MUST be true about the recipe "
            "(e.g. 'contains chocolate', 'stays moist for days', 'gluten-free'). Empty list if none.\n"
            "- \"soft_preferences\": list of strings — texture/flavor preferences that inform ranking "
            "(e.g. 'chewy', 'not too sweet', 'crispy edges'). Empty list if none.\n"
            "- \"queries\": list of {n} web search strings to find candidate recipes. Include:\n"
            "  - 1 broad variant (general ingredient/category terms)\n"
            "  - 1 specific/descriptive variant (texture, technique, key ingredients)\n"
            "{recency_instruction}"
            "{trusted_sources_instruction}"
            "  - Fill remaining slots with additional variants covering different angles\n\n"
            "Return only the JSON object, no other text."
        ),
    ),
    ("human", "{query}"),
])

# Injected into QUERY_PLAN at runtime based on recency dial
RECENCY_YEAR = "  - 1 recency variant scoped to the past year (append '2024 OR 2025' to the query)\n"
RECENCY_MONTH = "  - 1 recency variant scoped to the past month (append the current month and year)\n"
RECENCY_OFF = "  - 1 recency variant that appends the current year to surface newer results\n"

# Injected into QUERY_PLAN at runtime based on user-configured trusted sources
TRUSTED_SOURCES_ON = (
    "  - {n_site} site:-targeted variant(s) using site: for the following user-configured "
    "trusted domains: {trusted_domains}\n"
)
TRUSTED_SOURCES_OFF = ""  # no site: queries when user has not configured trusted sources

SNIPPET_RELEVANCE = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You evaluate recipe search result snippets for relevance to a user query. "
            "Given a numbered list of snippets, return a JSON object with key \"scores\": "
            "a list of objects, one per snippet, each with fields: "
            "\"index\" (int), \"score\" (float 0.0–1.0), \"reason\" (short string). "
            "Scoring rules:\n"
            "- 1.0 = clearly a recipe that matches the query\n"
            "- 0.7–0.9 = likely a recipe, partial match (e.g. right technique, wrong flavor)\n"
            "- 0.4–0.6 = uncertain — might be a recipe but hard to tell from snippet\n"
            "- 0.0–0.3 = not a recipe, listicle/roundup, irrelevant, or gated content\n"
            "- Score 0.0 for substack.com snippets with very short or missing excerpt text "
            "(suspected paywall)\n"
            "- Score 0.0 for '10 best X recipes' roundups that link out to other recipes "
            "rather than providing one\n"
            "- Score 0.0 for social media group/community posts (e.g. facebook.com/groups/)\n"
            "Return only the JSON object, no other text."
        ),
    ),
    ("human", "Query: {query}\n\nSnippets:\n{snippet_list}"),
])

RELAXED_QUERIES = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "The previous recipe search returned too few relevant results. "
            "Generate 3–4 broader, more permissive query variants for the same recipe intent. "
            "Remove all site: restrictions. Use more general ingredient or category terms. "
            'Return a JSON object with key "queries" containing a list of strings. '
            "Return only the JSON object, no other text."
        ),
    ),
    ("human", "Original query: {query}\n\nPrevious queries tried:\n{previous_queries}"),
])
