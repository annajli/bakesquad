"""
Search + ingestion pipeline (steps 1–6).

DEVIATION from CONTEXT.md: CONTEXT.md specifies LangChain as the agent framework.
Replaced with direct llm_client.chat() calls and plain-string prompts (search/prompts.py).
Reason: removing LangChain cuts ~1-2 s per LLM call, critical for the 60 s budget.

DEVIATION from CONTEXT.md: WebBaseLoader removed. Fetch uses requests+BeautifulSoup only.
Reason: WebBaseLoader requires langchain_community which we're removing; requests+BS4
is simpler, faster to import, and gives us more control over extraction.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from bakesquad.config import (
    DOMAIN_CAP,
    FETCH_TIMEOUT_SECONDS,
    MAX_FETCH_WORKERS,
    MAX_PAGES_PER_RUN,
    MAX_RESULTS_PER_QUERY,
    MIN_CANDIDATE_THRESHOLD,
    MIN_PAGE_CONTENT_CHARS,
    QUERIES_PER_RUN,
    RELEVANCE_SCORE_CUTOFF,
    SNIPPET_SCORE_CHUNK_SIZE,
    SUBSTACK_PAYWALL_CUTOFF,
)
from bakesquad.llm_client import chat, extract_json
from bakesquad.models import FetchedPage, QueryPlan, SearchSnippet
from bakesquad.search.prompts import (
    query_plan_prompt,
    relaxed_queries_prompt,
    snippet_relevance_prompt,
)

try:
    from ddgs import DDGS
except ImportError as e:
    raise ImportError("Install ddgs: pip install ddgs") from e

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# URL path segments that reliably indicate non-recipe pages.
_SKIP_URL_SEGMENTS = frozenset({
    "/groups/",
    "/r/",
    "/tag/",
    "/category/",
    "/author/",
    "/about/",
    "/contact/",
})

# Title patterns indicating a roundup rather than a single recipe.
_ROUNDUP_STARTERS = (
    "best ", "top ", "10 ", "15 ", "20 ", "25 ", "30 ",
    "the best ", "the top ",
)

# Common recipe plugin ingredient container class names for BS4 extraction.
_INGREDIENT_CLASSES = [
    "wprm-recipe-ingredients-container",
    "wprm-recipe-ingredient",
    "tasty-recipe-ingredients",
    "recipe-ingredients",
    "ingredients",
    "ingredient-list",
    "recipe__ingredient",
    "structured-ingredients__list",
    "ingredient",
]


class IngestionPipeline:
    """
    6-step search and ingestion pipeline:

      1. Query plan       — single LLM call: category, constraints, 2 search queries
      2. Parallel search  — DuckDuckGo queries run concurrently, deduplicated by URL
      3. Snippet scoring  — heuristic pre-filter, then one batched LLM relevance call
      4. Domain cap       — flat cap per domain; user trusted sources win tiebreaks
      5. Adaptive retry   — if < MIN_CANDIDATE_THRESHOLD survive, retry with relaxed queries
      6. Page fetch       — requests+BS4, parallelised; thin pages dropped post-fetch
    """

    def __init__(
        self,
        *,
        trusted_sources: Optional[list[str]] = None,
        num_query_variants: int = QUERIES_PER_RUN,
        min_candidates: int = MIN_CANDIDATE_THRESHOLD,
        domain_cap: int = DOMAIN_CAP,
        relevance_cutoff: float = RELEVANCE_SCORE_CUTOFF,
        max_fetch_workers: int = MAX_FETCH_WORKERS,
    ):
        self.trusted_sources: set[str] = set(trusted_sources or [])
        self.num_query_variants = num_query_variants
        self.min_candidates = min_candidates
        self.domain_cap = domain_cap
        self.relevance_cutoff = relevance_cutoff
        self.max_fetch_workers = max_fetch_workers
        self.last_query_plan: Optional[QueryPlan] = None

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def run(
        self,
        query: str,
        recency: Literal["year", "month", None] = None,
    ) -> list[FetchedPage]:
        """Execute the full pipeline. Returns fetched pages ready for the LLM parser."""
        plan = self._build_query_plan(query, recency)
        candidates = self._search_and_filter(query, plan.queries, recency)
        return self._fetch_pages(candidates)

    def run_search_only(
        self,
        query: str,
        recency: Literal["year", "month", None] = None,
    ) -> list[SearchSnippet]:
        """Steps 1–4 only — no fetch. For inspection/testing."""
        plan = self._build_query_plan(query, recency)
        return self._search_and_filter(query, plan.queries, recency)

    # -------------------------------------------------------------------------
    # Step 1: Query plan (category + constraints + queries in one LLM call)
    # -------------------------------------------------------------------------

    def _build_query_plan(
        self, query: str, recency: Literal["year", "month", None]
    ) -> QueryPlan:
        system, user = query_plan_prompt(
            query=query,
            recency=recency,
            trusted_sources=sorted(self.trusted_sources),
        )
        raw = chat(system, user, max_tokens=512)
        try:
            data = extract_json(raw)
            if isinstance(data, list):
                raise ValueError("Expected dict, got list")
            queries = data.get("queries", []) or [query]
            # Enforce 2-query cap
            queries = queries[:2]
            if len(queries) < 1:
                queries = [query]
            plan = QueryPlan(
                category=data.get("category", "other"),
                flour_type=data.get("flour_type", "ap") or "ap",
                modifiers=data.get("modifiers", []) or [],
                hard_constraints=data.get("hard_constraints", []),
                soft_preferences=data.get("soft_preferences", []),
                queries=queries,
            )
        except Exception as exc:
            logger.warning(
                "Query plan parse failed (%s). Raw LLM response: %r",
                exc, raw[:400],
            )
            plan = QueryPlan(
                category="other",
                flour_type="ap",
                modifiers=[],
                hard_constraints=[],
                soft_preferences=[],
                queries=[query],
            )
        self.last_query_plan = plan
        logger.info(
            "Step 1: category=%s | constraints=%s | %d queries",
            plan.category, plan.hard_constraints, len(plan.queries),
        )
        return plan

    # -------------------------------------------------------------------------
    # Step 2: Parallel multi-query search
    # -------------------------------------------------------------------------

    def _search_all(self, queries: list[str]) -> list[SearchSnippet]:
        """
        DEVIATION from CONTEXT.md: searches run sequentially, not in parallel threads.
        Root cause: ddgs v9 uses primp (Rust HTTP client) which deadlocks when invoked
        from multiple Python threads simultaneously on Windows. Sequential execution with
        2 queries takes ~2-3 s total, well within the 8 s search budget, so parallelism
        provides no meaningful benefit here.
        """
        seen: dict[str, SearchSnippet] = {}

        for q in queries:
            try:
                with DDGS() as ddgs:
                    for r in ddgs.text(q, max_results=MAX_RESULTS_PER_QUERY, timeout=8):
                        if not r.get("href"):
                            continue
                        url = r["href"]
                        key = _normalise_url(url)
                        if key not in seen:
                            seen[key] = SearchSnippet(
                                url=url,
                                title=r.get("title", ""),
                                excerpt=r.get("body", ""),
                                query_source=q,
                            )
            except Exception as e:
                logger.warning("DDG search failed for %r: %s", q, e)

        snippets = list(seen.values())
        logger.info("Step 2: %d unique snippets from %d queries", len(snippets), len(queries))
        return snippets

    # -------------------------------------------------------------------------
    # Step 3: Snippet relevance pre-check (batched — never one call per snippet)
    # -------------------------------------------------------------------------

    def _score_snippets(self, snippets: list[SearchSnippet], query: str) -> list[SearchSnippet]:
        if not snippets:
            return []

        # Heuristic pre-filter — no LLM needed for obvious cases
        for s in snippets:
            if s.relevance_score is not None:
                continue
            reason = _heuristic_skip(s)
            if reason:
                s.relevance_score = 0.0
                s.skip_reason = reason

        # Substack paywall heuristic
        for s in snippets:
            if s.relevance_score is None and "substack.com" in s.domain:
                if len(s.excerpt) < SUBSTACK_PAYWALL_CUTOFF:
                    s.relevance_score = 0.0
                    s.skip_reason = "paywall"

        # Batched LLM scoring — all undecided snippets in one prompt
        to_score = [s for s in snippets if s.relevance_score is None]
        for i in range(0, len(to_score), SNIPPET_SCORE_CHUNK_SIZE):
            self._score_chunk(to_score[i : i + SNIPPET_SCORE_CHUNK_SIZE], query)

        passing = []
        for s in snippets:
            if s.relevance_score is None:
                s.relevance_score = 0.5
            if s.relevance_score < self.relevance_cutoff:
                if not s.skip_reason:
                    s.skip_reason = "low_score"
            else:
                passing.append(s)

        heuristic_dropped = sum(
            1 for s in snippets if s.skip_reason in {"non_recipe_url", "likely_roundup", "paywall"}
        )
        logger.info(
            "Step 3: %d/%d passed (%d heuristic-dropped)",
            len(passing), len(snippets), heuristic_dropped,
        )
        return passing

    def _score_chunk(self, snippets: list[SearchSnippet], query: str) -> None:
        snippet_list = "\n".join(
            f"{i}. [{s.domain}] {s.title} — {s.excerpt[:200]}"
            for i, s in enumerate(snippets)
        )
        system, user = snippet_relevance_prompt(query, snippet_list)
        try:
            raw = chat(system, user, max_tokens=1024)
            data = extract_json(raw)
            items = data.get("scores", []) if isinstance(data, dict) else []
            for item in items:
                idx = item.get("index")
                score = item.get("score")
                if idx is not None and score is not None and 0 <= idx < len(snippets):
                    snippets[idx].relevance_score = float(score)
        except Exception:
            logger.warning("Snippet scoring failed; defaulting chunk to 0.5.")
            for s in snippets:
                if s.relevance_score is None:
                    s.relevance_score = 0.5

    # -------------------------------------------------------------------------
    # Step 4: Domain cap
    # -------------------------------------------------------------------------

    def _apply_domain_cap(self, snippets: list[SearchSnippet]) -> list[SearchSnippet]:
        snippets = sorted(
            snippets,
            key=lambda s: (s.relevance_score or 0.0, s.domain in self.trusted_sources),
            reverse=True,
        )
        domain_counts: dict[str, int] = {}
        result = []
        for s in snippets:
            count = domain_counts.get(s.domain, 0)
            if count < self.domain_cap:
                result.append(s)
                domain_counts[s.domain] = count + 1
            else:
                s.skip_reason = "domain_cap"
        logger.info("Step 4: %d snippets after domain cap", len(result))
        return result

    # -------------------------------------------------------------------------
    # Steps 2–5: Search loop with one adaptive retry
    # -------------------------------------------------------------------------

    def _search_and_filter(
        self,
        query: str,
        queries: list[str],
        recency: Literal["year", "month", None] = None,
    ) -> list[SearchSnippet]:
        candidates: list[SearchSnippet] = []
        for attempt in range(2):
            if attempt == 1:
                logger.info("Step 5: only %d candidates — retrying with relaxed queries.", len(candidates))
                queries = self._relax_queries(queries, query)

            raw = self._search_all(queries)
            scored = self._score_snippets(raw, query)
            candidates = self._apply_domain_cap(scored)

            if len(candidates) >= self.min_candidates:
                break
        return candidates

    def _relax_queries(self, original_queries: list[str], base_query: str) -> list[str]:
        system, user = relaxed_queries_prompt(base_query, "\n".join(original_queries))
        try:
            raw = chat(system, user, max_tokens=256)
            data = extract_json(raw)
            queries = data.get("queries", []) if isinstance(data, dict) else []
            return queries[:2] or [base_query]
        except Exception:
            logger.warning("Relaxed query generation failed; falling back to base query.")
            return [base_query]

    # -------------------------------------------------------------------------
    # Step 6: Full page fetch — parallel, 5 s timeout, thin-page drop
    # -------------------------------------------------------------------------

    def _fetch_pages(self, snippets: list[SearchSnippet]) -> list[FetchedPage]:
        # Cap to MAX_PAGES_PER_RUN candidates before fetching
        snippets = snippets[:MAX_PAGES_PER_RUN]
        pages: list[FetchedPage] = []
        with ThreadPoolExecutor(max_workers=self.max_fetch_workers) as executor:
            futures = {executor.submit(self._fetch_one, s): s for s in snippets}
            for future in as_completed(futures):
                page = future.result()
                if page.fetch_error:
                    continue
                if len(page.raw_text) < MIN_PAGE_CONTENT_CHARS:
                    logger.debug("Dropped thin page (%d chars): %s", len(page.raw_text), page.url)
                    continue
                pages.append(page)
        logger.info("Step 6: %d/%d pages fetched successfully", len(pages), len(snippets))
        return pages

    def _fetch_one(self, snippet: SearchSnippet) -> FetchedPage:
        try:
            resp = requests.get(
                snippet.url,
                headers={"User-Agent": _USER_AGENT},
                timeout=FETCH_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            title = snippet.title or (soup.title.string.strip() if soup.title else "")

            # Strategy 1: JSON-LD structured data (schema.org/Recipe)
            jsonld_excerpt = _extract_jsonld_ingredients(soup, title)
            if jsonld_excerpt:
                # Still build raw_text for the thin-page guard — use full stripped text
                raw_text = soup.get_text(separator="\n", strip=True)
                logger.debug("Fetched %s via JSON-LD (%d chars)", snippet.url, len(raw_text))
                return FetchedPage(
                    url=snippet.url,
                    title=title,
                    raw_text=raw_text,
                    ingredients_excerpt=jsonld_excerpt,
                    fetch_method="requests_bs4",
                    snippet=snippet,
                )

            # Strategy 2: BS4 fallback — remove noise then heuristic ingredient extraction
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                tag.decompose()
            raw_text = soup.get_text(separator="\n", strip=True)
            ingredients_excerpt = _extract_ingredients_section(soup, title)

            logger.debug("Fetched %s via BS4 (%d chars)", snippet.url, len(raw_text))
            return FetchedPage(
                url=snippet.url,
                title=title,
                raw_text=raw_text,
                ingredients_excerpt=ingredients_excerpt,
                fetch_method="requests_bs4",
                snippet=snippet,
            )
        except Exception as e:
            logger.warning("Fetch failed for %s: %s", snippet.url, e)
            return FetchedPage(
                url=snippet.url,
                title=snippet.title,
                raw_text="",
                ingredients_excerpt="",
                fetch_method="requests_bs4",
                fetch_error=str(e),
                snippet=snippet,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_jsonld_ingredients(soup: BeautifulSoup, title: str) -> str:
    """
    Extract ingredient list from schema.org/Recipe JSON-LD blocks.

    Most major recipe sites and WordPress recipe plugins (WP Recipe Maker,
    Tasty Recipes, Create) embed structured data in <script type="application/ld+json">
    tags. This is machine-readable and survives layout changes.

    Returns a formatted ingredients excerpt string, or "" if no Recipe JSON-LD found.
    """
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue

        # Handle both bare objects and @graph arrays
        candidates: list[dict] = []
        if isinstance(data, dict):
            graph = data.get("@graph", [])
            if graph:
                candidates.extend(g for g in graph if isinstance(g, dict))
            else:
                candidates.append(data)
        elif isinstance(data, list):
            candidates.extend(item for item in data if isinstance(item, dict))

        for node in candidates:
            # Normalise @type — may be a string or list
            node_type = node.get("@type", "")
            if isinstance(node_type, list):
                node_type = " ".join(node_type)
            if "Recipe" not in node_type:
                continue

            ingredients = node.get("recipeIngredient", [])
            if not ingredients or not isinstance(ingredients, list):
                continue

            # Use JSON-LD title if better than snippet title
            jsonld_title = node.get("name", title) or title
            ingredient_text = "\n".join(str(i) for i in ingredients if i)

            if len(ingredient_text) < 20:
                continue  # suspiciously short — probably malformed

            # Include yield and description for context the parser can use
            extra_parts = []
            yield_val = node.get("recipeYield")
            if yield_val:
                yield_str = yield_val if isinstance(yield_val, str) else ", ".join(str(v) for v in yield_val)
                extra_parts.append(f"Yield: {yield_str}")
            description = node.get("description", "")
            if description:
                extra_parts.append(f"Description: {str(description)[:300]}")

            header = "\n".join(extra_parts)
            excerpt = (
                f"Title: {jsonld_title}\n\n"
                + (f"{header}\n\n" if header else "")
                + f"Ingredients:\n{ingredient_text}"
            )
            logger.debug(
                "JSON-LD extraction succeeded for %s (%d ingredients)",
                jsonld_title, len(ingredients),
            )
            return excerpt[:3000]

    return ""


def _extract_ingredients_section(soup: BeautifulSoup, title: str) -> str:
    """
    Pre-extract just the ingredients list from a parsed page.
    Spec: 'Before sending HTML to the LLM, use BeautifulSoup to pre-extract only
    the ingredients section and title.'
    Sends ~300–800 chars to the LLM instead of 10,000+ chars of full page text.
    """
    # Strategy 1: common recipe plugin class names
    for cls in _INGREDIENT_CLASSES:
        el = soup.find(class_=lambda c: c is not None and cls in (" ".join(c) if isinstance(c, list) else c).lower())
        if el:
            text = el.get_text(separator="\n", strip=True)
            if len(text) > 50:
                return f"Title: {title}\n\nIngredients:\n{text[:2000]}"

    # Strategy 2: find heading containing "ingredient" then grab the next list
    for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
        if "ingredient" in heading.get_text().lower():
            parts = []
            for sibling in heading.next_siblings:
                tag_name = getattr(sibling, "name", None)
                if tag_name in ("ul", "ol", "div"):
                    parts.append(sibling.get_text(separator="\n", strip=True))
                    break
                elif tag_name in ("h1", "h2", "h3"):
                    break
            if parts:
                return f"Title: {title}\n\nIngredients:\n" + "\n".join(parts[:1])[:2000]

    # Fallback: regex scan of lines for quantity patterns in stripped text
    all_text = soup.get_text(separator="\n", strip=True)
    lines = all_text.split("\n")
    ingr_start = -1
    for i, line in enumerate(lines):
        low = line.lower().strip()
        if low in ("ingredients", "ingredients:") or (
            "ingredient" in low and len(low) < 40
        ):
            ingr_start = i
            break
    if ingr_start >= 0:
        section = "\n".join(lines[ingr_start : ingr_start + 60])
        return f"Title: {title}\n\n{section[:2000]}"

    # Last resort: first 2500 chars of full text
    return f"Title: {title}\n\n{all_text[:2500]}"


def _heuristic_skip(s: SearchSnippet) -> str | None:
    """Fast pre-filter before LLM scoring. Returns skip reason or None."""
    if any(seg in s.url for seg in _SKIP_URL_SEGMENTS):
        return "non_recipe_url"
    title_lower = s.title.lower()
    if "recipe" in title_lower and any(title_lower.startswith(w) for w in _ROUNDUP_STARTERS):
        return "likely_roundup"
    return None


def _normalise_url(url: str) -> str:
    """Lowercase scheme+host, strip trailing slash + query params — for deduplication."""
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme.lower()}://{host}{path}"
