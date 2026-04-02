from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama

from bakesquad.config import (
    DEFAULT_OLLAMA_MODEL,
    DOMAIN_CAP,
    FETCH_TIMEOUT_SECONDS,
    LLM_TEMPERATURE,
    MAX_FETCH_WORKERS,
    MAX_RESULTS_PER_QUERY,
    MAX_SEARCH_WORKERS,
    MIN_CANDIDATE_THRESHOLD,
    MIN_PAGE_CONTENT_CHARS,
    QUERIES_PER_RUN,
    RELEVANCE_SCORE_CUTOFF,
    SNIPPET_SCORE_CHUNK_SIZE,
    SUBSTACK_PAYWALL_CUTOFF,
)
from bakesquad.models import FetchedPage, QueryPlan, SearchSnippet
from bakesquad.search.prompts import (
    QUERY_PLAN,
    RECENCY_MONTH,
    RECENCY_OFF,
    RECENCY_YEAR,
    RELAXED_QUERIES,
    SNIPPET_RELEVANCE,
    TRUSTED_SOURCES_OFF,
    TRUSTED_SOURCES_ON,
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

# URL path segments that reliably indicate non-recipe pages regardless of domain.
# Kept narrow to avoid false positives — recipe blogs use /category/ for browsing pages,
# not for the recipes themselves; social media group/community paths return gated content.
_SKIP_URL_SEGMENTS = frozenset({
    "/groups/",       # Facebook groups, Reddit communities — gated or discussion, not recipes
    "/r/",            # Reddit subreddit paths
    "/tag/",          # Blog tag archive pages
    "/category/",     # Blog category archive pages
    "/author/",       # Author bio pages
    "/about/",        # About pages
    "/contact/",      # Contact pages
})

# Title patterns that indicate a roundup/listicle rather than a single recipe.
# Only triggered when the title also contains "recipe" to avoid false positives
# (e.g. "Best Banana Bread" without "recipes" could still be a single recipe page).
_ROUNDUP_STARTERS = (
    "best ", "top ", "10 ", "15 ", "20 ", "25 ", "30 ",
    "the best ", "the top ",
)


class IngestionPipeline:
    """
    6-step search and ingestion pipeline:

      1. Query plan       — single LLM call extracts category, constraints, preferences, queries
      2. Parallel search  — DuckDuckGo queries run concurrently, deduplicated by URL
      3. Snippet scoring  — heuristic pre-filter, then batched LLM relevance scoring
      4. Domain cap       — flat cap per domain; user trusted sources win tiebreaks
      5. Adaptive retry   — if < MIN_CANDIDATE_THRESHOLD survive, retry with relaxed queries
      6. Page fetch       — WebBaseLoader with requests+BS4 fallback, parallelised;
                            thin pages (< MIN_PAGE_CONTENT_CHARS) dropped post-fetch
    """

    def __init__(
        self,
        *,
        llm=None,
        trusted_sources: Optional[list[str]] = None,
        num_query_variants: int = QUERIES_PER_RUN,
        min_candidates: int = MIN_CANDIDATE_THRESHOLD,
        domain_cap: int = DOMAIN_CAP,
        relevance_cutoff: float = RELEVANCE_SCORE_CUTOFF,
        max_fetch_workers: int = MAX_FETCH_WORKERS,
    ):
        self.llm = llm or ChatOllama(model=DEFAULT_OLLAMA_MODEL, temperature=LLM_TEMPERATURE)
        # User-configured trusted sources — tiebreaker in domain cap sort and
        # triggers site:-targeted query variants. No developer-defined list applied.
        self.trusted_sources: set[str] = set(trusted_sources or [])
        self.num_query_variants = num_query_variants
        self.min_candidates = min_candidates
        self.domain_cap = domain_cap
        self.relevance_cutoff = relevance_cutoff
        self.max_fetch_workers = max_fetch_workers
        self.last_query_plan: Optional[QueryPlan] = None  # set after each run

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
        candidates = self._search_and_filter(query, plan.queries)
        return self._fetch_pages(candidates)

    def run_search_only(
        self,
        query: str,
        recency: Literal["year", "month", None] = None,
    ) -> list[SearchSnippet]:
        """Steps 1–4 only — no fetch. Useful for inspecting candidate selection."""
        plan = self._build_query_plan(query, recency)
        return self._search_and_filter(query, plan.queries)

    # -------------------------------------------------------------------------
    # Step 1: Query plan (category + constraints + queries in one LLM call)
    # -------------------------------------------------------------------------

    def _build_query_plan(
        self, query: str, recency: Literal["year", "month", None]
    ) -> QueryPlan:
        recency_instruction = {
            "year": RECENCY_YEAR,
            "month": RECENCY_MONTH,
        }.get(recency, RECENCY_OFF)

        if self.trusted_sources:
            n_site = min(len(self.trusted_sources), 2)
            trusted_instruction = TRUSTED_SOURCES_ON.format(
                n_site=n_site,
                trusted_domains=", ".join(sorted(self.trusted_sources)),
            )
        else:
            trusted_instruction = TRUSTED_SOURCES_OFF

        chain = QUERY_PLAN | self.llm
        result = chain.invoke({
            "query": query,
            "n": self.num_query_variants,
            "recency_instruction": recency_instruction,
            "trusted_sources_instruction": trusted_instruction,
        })
        text = _clean_llm_text(result)

        try:
            data = json.loads(_extract_json(text))
            plan = QueryPlan(
                category=data.get("category", "other"),
                hard_constraints=data.get("hard_constraints", []),
                soft_preferences=data.get("soft_preferences", []),
                queries=data.get("queries", []) or [query],
            )
        except (json.JSONDecodeError, ValueError):
            logger.warning("Query plan JSON parse failed; falling back to original query.")
            plan = QueryPlan(
                category="other",
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
        seen: dict[str, SearchSnippet] = {}
        lock = threading.Lock()

        def _search_one(q: str) -> list[tuple[str, dict]]:
            rows = []
            try:
                with DDGS() as ddgs:
                    for r in ddgs.text(q, max_results=MAX_RESULTS_PER_QUERY):
                        if r.get("href"):
                            rows.append((q, r))
            except Exception as e:
                logger.warning("DDG search failed for '%s': %s", q, e)
            return rows

        with ThreadPoolExecutor(max_workers=MAX_SEARCH_WORKERS) as executor:
            for rows in executor.map(_search_one, queries):
                for q, r in rows:
                    url = r["href"]
                    key = _normalise_url(url)
                    with lock:
                        if key not in seen:
                            seen[key] = SearchSnippet(
                                url=url,
                                title=r.get("title", ""),
                                excerpt=r.get("body", ""),
                                query_source=q,
                            )

        snippets = list(seen.values())
        logger.info("Step 2: %d unique snippets from %d queries", len(snippets), len(queries))
        return snippets

    # -------------------------------------------------------------------------
    # Step 3: Snippet relevance pre-check
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

        # Batch LLM scoring for everything not yet decided
        to_score = [s for s in snippets if s.relevance_score is None]
        for i in range(0, len(to_score), SNIPPET_SCORE_CHUNK_SIZE):
            self._score_chunk(to_score[i : i + SNIPPET_SCORE_CHUNK_SIZE], query)

        passing = []
        for s in snippets:
            if s.relevance_score is None:
                s.relevance_score = 0.5  # safe default; shouldn't reach here
            if s.relevance_score < self.relevance_cutoff:
                if not s.skip_reason:
                    s.skip_reason = "low_score"
            else:
                passing.append(s)

        heuristic_dropped = sum(1 for s in snippets if s.skip_reason in {"non_recipe_url", "likely_roundup", "paywall"})
        logger.info(
            "Step 3: %d/%d passed (%d dropped by heuristic pre-filter)",
            len(passing), len(snippets), heuristic_dropped,
        )
        return passing

    def _score_chunk(self, snippets: list[SearchSnippet], query: str) -> None:
        snippet_list = "\n".join(
            f"{i}. [{s.domain}] {s.title} — {s.excerpt[:200]}"
            for i, s in enumerate(snippets)
        )
        chain = SNIPPET_RELEVANCE | self.llm
        result = chain.invoke({"query": query, "snippet_list": snippet_list})
        text = _clean_llm_text(result)
        try:
            items = json.loads(_extract_json(text)).get("scores", [])
            for item in items:
                idx = item.get("index")
                score = item.get("score")
                if idx is not None and score is not None and 0 <= idx < len(snippets):
                    snippets[idx].relevance_score = float(score)
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Snippet scoring JSON parse failed; defaulting chunk to 0.5.")
            for s in snippets:
                if s.relevance_score is None:
                    s.relevance_score = 0.5

    # -------------------------------------------------------------------------
    # Step 4: Domain cap
    # -------------------------------------------------------------------------

    def _apply_domain_cap(self, snippets: list[SearchSnippet]) -> list[SearchSnippet]:
        # Primary sort: relevance score. Tiebreak: user-configured trusted sources rank higher.
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

    def _search_and_filter(self, query: str, queries: list[str]) -> list[SearchSnippet]:
        candidates: list[SearchSnippet] = []
        for attempt in range(2):
            if attempt == 1:
                logger.info("Step 5: Only %d candidates — retrying with relaxed queries.", len(candidates))
                queries = self._relax_queries(queries, query)

            raw = self._search_all(queries)
            scored = self._score_snippets(raw, query)
            candidates = self._apply_domain_cap(scored)

            if len(candidates) >= self.min_candidates:
                break

        return candidates

    def _relax_queries(self, original_queries: list[str], base_query: str) -> list[str]:
        chain = RELAXED_QUERIES | self.llm
        result = chain.invoke({
            "query": base_query,
            "previous_queries": "\n".join(original_queries),
        })
        text = _clean_llm_text(result)
        try:
            return json.loads(_extract_json(text)).get("queries", []) or [base_query]
        except (json.JSONDecodeError, ValueError):
            logger.warning("Relaxed query generation failed; falling back to base query.")
            return [base_query]

    # -------------------------------------------------------------------------
    # Step 6: Full page fetch (with thin-page drop)
    # -------------------------------------------------------------------------

    def _fetch_pages(self, snippets: list[SearchSnippet]) -> list[FetchedPage]:
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
        logger.info("Step 6: %d pages fetched", len(pages))
        return pages

    def _fetch_one(self, snippet: SearchSnippet) -> FetchedPage:
        # Try WebBaseLoader first
        try:
            loader = WebBaseLoader(snippet.url)
            docs = loader.load()
            raw_text = "\n".join(d.page_content for d in docs).strip()
            if raw_text:
                logger.debug("WebBaseLoader OK: %s (%d chars)", snippet.url, len(raw_text))
                return FetchedPage(
                    url=snippet.url,
                    title=snippet.title,
                    raw_text=raw_text,
                    fetch_method="webbaseloader",
                    snippet=snippet,
                )
        except Exception as e:
            logger.debug("WebBaseLoader failed for %s: %s", snippet.url, e)

        # Fallback: requests + BeautifulSoup
        try:
            resp = requests.get(
                snippet.url,
                headers={"User-Agent": _USER_AGENT},
                timeout=FETCH_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            raw_text = soup.get_text(separator="\n", strip=True)
            title = snippet.title or (soup.title.string.strip() if soup.title else "")
            logger.debug("requests+BS4 OK: %s (%d chars)", snippet.url, len(raw_text))
            return FetchedPage(
                url=snippet.url,
                title=title,
                raw_text=raw_text,
                fetch_method="requests_bs4",
                snippet=snippet,
            )
        except Exception as e:
            logger.warning("All fetch methods failed for %s: %s", snippet.url, e)
            return FetchedPage(
                url=snippet.url,
                title=snippet.title,
                raw_text="",
                fetch_method="requests_bs4",
                fetch_error=str(e),
                snippet=snippet,
            )


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _heuristic_skip(s: SearchSnippet) -> str | None:
    """
    Fast pre-filter before LLM scoring. Returns a skip reason string, or None to proceed.
    Kept intentionally narrow — only patterns that reliably indicate non-recipe content.
    """
    # URL path segments that indicate archive/community/meta pages
    if any(seg in s.url for seg in _SKIP_URL_SEGMENTS):
        return "non_recipe_url"

    # Roundup/listicle detection: numbered starter + "recipe" anywhere in title
    title_lower = s.title.lower()
    if "recipe" in title_lower and any(title_lower.startswith(w) for w in _ROUNDUP_STARTERS):
        return "likely_roundup"

    return None


def _normalise_url(url: str) -> str:
    """Lowercase scheme+host, strip path trailing slash and query params — for deduplication."""
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme.lower()}://{host}{path}"


def _extract_json(text: str) -> str:
    """Pull the outermost {...} block out of a string."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in LLM response.")
    return text[start:end]


def _clean_llm_text(result) -> str:
    """Extract text content from an LLM result and strip qwen3 <think> blocks."""
    text = result.content if hasattr(result, "content") else str(result)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
