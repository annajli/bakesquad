"""
Plain-string prompt builders for the search pipeline.

DEVIATION from CONTEXT.md: CONTEXT.md specifies LangChain as the framework and uses
ChatPromptTemplate. Replaced with plain (system, user) string functions so all LLM
calls route through llm_client.chat() — no LangChain dependency anywhere in this module.
Reason: LangChain template/chain overhead was ~1-2 s per call, incompatible with the
60 s total budget.
"""

from __future__ import annotations


def query_plan_prompt(
    query: str,
    recency: str | None,
    trusted_sources: list[str],
) -> tuple[str, str]:
    """
    Step 1: produce (system, user) for query understanding + search query diversification.
    Returns exactly 2 search queries (broad + specific) per the cap-at-2 requirement.
    """
    recency_line = ""
    if recency == "year":
        recency_line = '  - Make query[1] a recency variant scoped to the past year (append "2024 OR 2025")\n'
    elif recency == "month":
        recency_line = "  - Make query[1] a recency variant scoped to the past month\n"
    else:
        recency_line = '  - Make query[1] more specific/descriptive (texture, technique, key ingredients)\n'

    trusted_line = ""
    if trusted_sources:
        domains = ", ".join(trusted_sources[:2])
        trusted_line = f"  - User has these trusted domains (mention them in queries if relevant): {domains}\n"

    system = (
        "You are a baking recipe search assistant. Given a user's natural language query, "
        "analyze it and return a JSON object with exactly these keys:\n\n"
        '- "category": the type of baked good — one of: cookie, quick_bread, cake, other\n'
        '  quick_bread includes: banana bread, zucchini bread, pumpkin bread, muffins, scones, cornbread\n'
        '  cookie includes: drop cookies, bar cookies, brownies, shortbread\n'
        '  cake includes: layer cakes, bundt cakes, cheesecakes, cupcakes\n'
        '  other: anything that does not fit the above\n'
        '- "flour_type": the primary flour the recipe likely uses — one of:\n'
        '    "ap" (all-purpose, default), "almond", "oat", "coconut", "rice", "gf_blend" (1:1 GF blend), "other"\n'
        '  Use "ap" unless the query explicitly mentions a non-AP flour or is gluten-free.\n'
        '  If gluten-free and no specific flour mentioned, use "gf_blend".\n'
        '- "modifiers": list of applicable dietary/technique tags from:\n'
        '    ["gluten_free", "vegan", "dairy_free", "paleo", "keto", "nut_free"]\n'
        '  Empty list if none apply.\n'
        '- "hard_constraints": list of strings — things that MUST be true about the recipe '
        '(e.g. "contains chocolate", "stays moist for days"). Empty list if none.\n'
        '- "soft_preferences": list of strings — texture/flavor preferences that inform ranking. '
        "Empty list if none.\n"
        '- "queries": list of EXACTLY 2 web search strings to find candidate recipes:\n'
        "  - queries[0]: broad variant using general ingredient/category terms\n"
        f"{recency_line}"
        f"{trusted_line}"
        "\nExamples:\n\n"
        'Input: "banana bread"\n'
        '{"category":"quick_bread","flour_type":"ap","modifiers":[],'
        '"hard_constraints":[],"soft_preferences":[],'
        '"queries":["banana bread recipe","moist classic banana bread recipe"]}\n\n'
        'Input: "zucchini bread with chocolate chips"\n'
        '{"category":"quick_bread","flour_type":"ap","modifiers":[],'
        '"hard_constraints":["contains chocolate"],"soft_preferences":[],'
        '"queries":["zucchini bread chocolate chip recipe","moist chocolate chip zucchini bread recipe"]}\n\n'
        'Input: "gluten-free almond flour banana bread"\n'
        '{"category":"quick_bread","flour_type":"almond","modifiers":["gluten_free"],'
        '"hard_constraints":[],"soft_preferences":[],'
        '"queries":["almond flour banana bread recipe","gluten free almond flour banana bread moist"]}\n\n'
        'Input: "chewy brown butter chocolate chip cookies"\n'
        '{"category":"cookie","flour_type":"ap","modifiers":[],'
        '"hard_constraints":["contains chocolate"],"soft_preferences":["chewy texture","brown butter flavor"],'
        '"queries":["brown butter chocolate chip cookie recipe","chewy brown butter chocolate chip cookies recipe"]}\n\n'
        'Input: "moist red velvet cake"\n'
        '{"category":"cake","flour_type":"ap","modifiers":[],'
        '"hard_constraints":["red velvet flavor","moist texture"],"soft_preferences":[],'
        '"queries":["red velvet cake recipe","moist classic red velvet cake recipe from scratch"]}\n\n'
        "Return only the JSON object, no other text."
    )
    user = query
    return system, user


def snippet_relevance_prompt(query: str, snippet_list: str) -> tuple[str, str]:
    """
    Step 3: batch relevance scoring — all snippets in one prompt, one JSON array back.
    NEVER called once per snippet; always batched.
    """
    system = (
        "You evaluate recipe search result snippets for relevance to a user query. "
        'Given a numbered list of snippets, return a JSON object with key "scores": '
        "a list of objects, one per snippet, each with: "
        '"index" (int), "score" (float 0.0–1.0), "reason" (short string). '
        "Scoring rules:\n"
        "- 1.0 = clearly a single recipe matching the query\n"
        "- 0.7–0.9 = likely a single recipe, partial match\n"
        "- 0.4–0.6 = uncertain — might be a recipe but hard to tell\n"
        "- 0.0–0.3 = not a recipe, roundup/listicle, irrelevant, or gated content\n"
        "- 0.0 for substack.com snippets with very short/missing excerpts (paywall)\n"
        "- 0.0 for '10 best X' roundups that link out to multiple other recipes\n"
        "- 0.0 for social media group/community posts\n"
        "Return only the JSON object, no other text."
    )
    user = f"Query: {query}\n\nSnippets:\n{snippet_list}"
    return system, user


def relaxed_queries_prompt(base_query: str, previous_queries: str) -> tuple[str, str]:
    """Step 5: generate broader queries when candidate pool is too small."""
    system = (
        "The previous recipe search returned too few relevant results. "
        "Generate 2 broader, more permissive query variants for the same recipe intent. "
        "Remove all site: restrictions. Use more general ingredient or category terms. "
        'Return a JSON object with key "queries" containing a list of 2 strings. '
        "Return only the JSON object, no other text."
    )
    user = f"Original query: {base_query}\n\nPrevious queries tried:\n{previous_queries}"
    return system, user
