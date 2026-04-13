"""
Plain-string prompt builders for the search pipeline.

DEVIATION from CONTEXT.md: CONTEXT.md specifies LangChain as the framework and uses
ChatPromptTemplate. Replaced with plain (system, user) string functions so all LLM
calls route through llm_client.chat() — no LangChain dependency anywhere in this module.
Reason: LangChain template/chain overhead was ~1-2 s per call, incompatible with the
60 s total budget.
"""

from __future__ import annotations


def _build_category_block() -> str:
    """
    Build the category description block from the registry.

    Falls back to the hardcoded description if pyyaml is not installed or the
    registry file is missing — keeps the CLI working without the Phase 2 deps.

    The Literal in models.py now matches the full registry (cookie, quick_bread,
    cake, yeasted_bread, pastry, other).
    """
    try:
        from bakesquad.category_registry import load_registry
        categories = load_registry()
        cat_ids = [cat["id"] for cat in categories]
        lines = [
            '- "category": the type of baked good — must be EXACTLY one of: '
            + ", ".join(cat_ids)
        ]
        for cat in categories:
            cat_id = cat["id"]
            members = cat.get("members", [])
            if cat_id == "other":
                lines.append("  other: truly uncategorizable baked goods only")
                continue
            member_str = ", ".join(members) if members else cat_id
            line = f'  {cat_id} includes: {member_str}'
            if cat.get("loaf_like") and cat_id == "quick_bread":
                line += (
                    '\n  NOTE: any baked "loaf" that is NOT a yeast bread is quick_bread'
                    ' (e.g. "banana nut loaf", "zucchini chocolate loaf", "pumpkin loaf" → quick_bread)'
                )
            lines.append(line)
        return "\n".join(lines) + "\n"
    except Exception:
        return (
            '- "category": the type of baked good — must be EXACTLY one of:'
            ' cookie, quick_bread, cake, yeasted_bread, pastry, other\n'
            '  cookie includes: drop cookies, bar cookies, brownies, shortbread\n'
            '  quick_bread includes: banana bread, zucchini bread, pumpkin bread, muffins, scones, cornbread\n'
            '  NOTE: any baked "loaf" that is NOT a yeast bread is quick_bread'
            ' (e.g. "banana nut loaf", "zucchini chocolate loaf", "pumpkin loaf" → quick_bread)\n'
            '  cake includes: layer cakes, bundt cakes, cheesecakes, cupcakes\n'
            '  yeasted_bread includes: sourdough, focaccia, baguette, dinner rolls, pizza dough,'
            ' brioche, challah, sandwich loaves\n'
            '  pastry includes: croissants, danish pastries, choux (eclairs, profiteroles), tarts, pies\n'
            '  other: truly uncategorizable baked goods only\n'
        )


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
        + _build_category_block()
        + '- "flour_type": the primary flour the recipe likely uses — one of:\n'
        '    "ap" (all-purpose, default), "almond", "oat", "coconut", "rice", "gf_blend" (1:1 GF blend), "other"\n'
        '  Use "ap" unless the query explicitly mentions a non-AP flour or is gluten-free.\n'
        '  If gluten-free and no specific flour mentioned, use "gf_blend".\n'
        '- "modifiers": list of applicable dietary/technique tags from:\n'
        '    ["gluten_free", "vegan", "dairy_free", "paleo", "keto", "nut_free"]\n'
        '  Empty list if none apply.\n'
        '- "hard_constraints": list of strings — things that MUST be true about the recipe.\n'
        '  RULE 1: Any explicit exclusion ("no X", "without X", "X-free", "X allergy") MUST go in\n'
        '  hard_constraints. E.g. "no nuts" → hard_constraints: ["no nuts"].\n'
        '  RULE 2: Any named key ingredient inclusion ("chocolate chip cookies", "blueberry muffins",\n'
        '  "banana bread", "with walnuts") MUST also go in hard_constraints as "contains X".\n'
        '  E.g. "chocolate chip cookies" → hard_constraints: ["contains chocolate chips"].\n'
        '  E.g. "blueberry muffins" → hard_constraints: ["contains blueberries"].\n'
        '- "soft_preferences": list of strings — texture, flavor, or technique preferences that\n'
        '  inform ranking but are not hard requirements.\n'
        '  RULE: Temporal preservation ("stays moist for N days", "keeps soft", "lasts a week")\n'
        '  belongs here, e.g. "stays moist for 3–4 days".\n'
        '- "queries": list of EXACTLY 2 web search strings to find candidate recipes:\n'
        "  - queries[0]: broad variant using general ingredient/category terms\n"
        f"{recency_line}"
        f"{trusted_line}"
        '- "confidence": float 0.0–1.0, your confidence in the category classification.\n'
        '  Use 1.0 for clear cases. Use 0.5–0.8 when the query is ambiguous between categories.\n'
        '- "clarification_question": if confidence < 0.7, a short question to ask the user to\n'
        '  resolve the ambiguity (e.g. "Are you looking for a yeast bread or a quick loaf?").\n'
        '  Use an empty string "" when confidence >= 0.7.\n'
        "\nExamples:\n\n"
        'Input: "banana bread"\n'
        '{"category":"quick_bread","flour_type":"ap","modifiers":[],'
        '"hard_constraints":["contains banana"],"soft_preferences":[],'
        '"queries":["banana bread recipe","moist classic banana bread recipe"],'
        '"confidence":1.0,"clarification_question":""}\n\n'
        'Input: "lemon bars"\n'
        '{"category":"cookie","flour_type":"ap","modifiers":[],'
        '"hard_constraints":["contains lemon"],"soft_preferences":[],'
        '"queries":["lemon bar recipe","classic lemon bars shortbread crust recipe"],'
        '"confidence":1.0,"clarification_question":""}\n\n'
        'Input: "chocolate chip cookies without any nuts"\n'
        '{"category":"cookie","flour_type":"ap","modifiers":[],'
        '"hard_constraints":["contains chocolate chips","no nuts"],"soft_preferences":[],'
        '"queries":["chocolate chip cookie recipe no nuts","nut-free chocolate chip cookies"],'
        '"confidence":1.0,"clarification_question":""}\n\n'
        'Input: "moist banana bread that stays soft for at least 5 days"\n'
        '{"category":"quick_bread","flour_type":"ap","modifiers":[],'
        '"hard_constraints":["banana"],"soft_preferences":["moist","stays soft for 5 days"],'
        '"queries":["moist banana bread recipe","banana bread stays soft days recipe"],'
        '"confidence":1.0,"clarification_question":""}\n\n'
        'Input: "banana nut loaf"\n'
        '{"category":"quick_bread","flour_type":"ap","modifiers":[],'
        '"hard_constraints":["contains banana","contains nuts"],"soft_preferences":[],'
        '"queries":["banana nut loaf recipe","moist banana walnut quick bread loaf recipe"],'
        '"confidence":1.0,"clarification_question":""}\n\n'
        'Input: "sourdough loaf"\n'
        '{"category":"yeasted_bread","flour_type":"ap","modifiers":[],'
        '"hard_constraints":[],"soft_preferences":[],'
        '"queries":["sourdough loaf recipe","classic sourdough bread recipe"],'
        '"confidence":1.0,"clarification_question":""}\n\n'
        'Input: "classic sourdough with open crumb"\n'
        '{"category":"yeasted_bread","flour_type":"ap","modifiers":[],'
        '"hard_constraints":[],"soft_preferences":["open crumb structure"],'
        '"queries":["sourdough bread recipe open crumb","classic sourdough loaf recipe"],'
        '"confidence":1.0,"clarification_question":""}\n\n'
        'Input: "homemade croissants from scratch"\n'
        '{"category":"pastry","flour_type":"ap","modifiers":[],'
        '"hard_constraints":[],"soft_preferences":["flaky layers","buttery"],'
        '"queries":["homemade croissant recipe from scratch","flaky butter croissants recipe"],'
        '"confidence":1.0,"clarification_question":""}\n\n'
        'Input: "chewy brown butter chocolate chip cookies"\n'
        '{"category":"cookie","flour_type":"ap","modifiers":[],'
        '"hard_constraints":["contains chocolate"],"soft_preferences":["chewy texture","brown butter flavor"],'
        '"queries":["brown butter chocolate chip cookie recipe","chewy brown butter chocolate chip cookies recipe"],'
        '"confidence":1.0,"clarification_question":""}\n\n'
        'Input: "a rich loaf for gifting"\n'
        '{"category":"quick_bread","flour_type":"ap","modifiers":[],'
        '"hard_constraints":[],"soft_preferences":["rich flavor"],'
        '"queries":["gift loaf recipe","moist quick bread loaf recipe gift"],'
        '"confidence":0.6,"clarification_question":"Are you thinking of a quick loaf (like banana or zucchini bread) or a yeasted loaf (like brioche or a sandwich bread)?"}\n\n'
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
