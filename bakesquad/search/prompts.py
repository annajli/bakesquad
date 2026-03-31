from langchain_core.prompts import ChatPromptTemplate

# {trusted_sources_instruction} is populated at runtime:
#   - If the user has configured trusted sources → a line instructing the LLM to include
#     site:-targeted variants for those domains.
#   - If not → empty string. No developer-defined source list is injected.
QUERY_DIVERSIFICATION = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You generate diverse web search query variants for recipe search. "
            "Given a user's natural language recipe query, return {n} search strings as a JSON object "
            'with a single key "queries" containing a list of strings. '
            "Include:\n"
            "- 1 broad variant (general ingredient/category terms)\n"
            "- 1 specific/descriptive variant (texture, technique, key ingredients)\n"
            "- 1 recency variant (append the current year to surface newer results)\n"
            "{trusted_sources_instruction}"
            "Return only the JSON object, no other text."
        ),
    ),
    ("human", "{query}"),
])

# trusted_sources_instruction values:
TRUSTED_SOURCES_ON = (
    "- {n_site} site:-targeted variants using site: for the following user-configured trusted domains: "
    "{trusted_domains}\n"
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
            "- 1.0 = clearly a recipe matching the query\n"
            "- 0.5–0.9 = likely a recipe, partial match\n"
            "- 0.0–0.4 = not a recipe, listicle/roundup, irrelevant, or paywalled\n"
            "- Score 0.0 for substack.com snippets with very short or missing excerpt text "
            "(suspected paywall — login wall returned instead of content)\n"
            "- Score 0.0 for '10 best X recipes' roundups that link out to other recipes\n"
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
