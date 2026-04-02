from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
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
    MIN_CANDIDATE_THRESHOLD,
    QUERIES_PER_RUN,
    RELEVANCE_SCORE_CUTOFF,
    SUBSTACK_PAYWALL_CUTOFF,
)
from bakesquad.models import FetchedPage, SearchSnippet
from bakesquad.search.prompts import (
    QUERY_DIVERSIFICATION,
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


class IngestionPipeline:
    """
    6-step search and ingestion pipeline:

      1. Query diversification  — LLM generates N query variants
      2. Multi-query search     — DuckDuckGo, deduplicated by URL
      3. Snippet relevance      — LLM scores each snippet 0–1; Substack paywall heuristic
      4. Domain cap             — flat cap of DOMAIN_CAP results per domain
      5. Adaptive retry         — if < MIN_CANDIDATE_THRESHOLD survive, retry with relaxed queries
      6. Full page fetch        — WebBaseLoader with requests+BS4 fallback, parallelised
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
        # User-configured trusted sources — used as tiebreaker in domain cap sort, and
        # to generate site:-targeted query variants. No developer-defined list is applied.
        self.trusted_sources: set[str] = set(trusted_sources or [])
        self.num_query_variants = num_query_variants
        self.min_candidates = min_candidates
        self.domain_cap = domain_cap
        self.relevance_cutoff = relevance_cutoff
        self.max_fetch_workers = max_fetch_workers

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def run(self, query: str) -> list[FetchedPage]:
        """Execute the full pipeline. Returns fetched pages ready for the LLM parser."""
        queries = self._diversify_queries(query)
        candidates = self._search_and_filter(query, queries)
        return self._fetch_pages(candidates)

    def run_search_only(self, query: str) -> list[SearchSnippet]:
        """Steps 1–4 only — no fetch. Useful for inspecting candidate selection."""
        queries = self._diversify_queries(query)
        return self._search_and_filter(query, queries)

    # -------------------------------------------------------------------------
    # Step 1: Query diversification
    # -------------------------------------------------------------------------

    def _diversify_queries(self, query: str) -> list[str]:
        if self.trusted_sources:
            n_site = min(len(self.trusted_sources), 2)
            trusted_instruction = TRUSTED_SOURCES_ON.format(
                n_site=n_site,
                trusted_domains=", ".join(sorted(self.trusted_sources)),
            )
        else:
            trusted_instruction = TRUSTED_SOURCES_OFF

        chain = QUERY_DIVERSIFICATION | self.llm
        result = chain.invoke({
            "query": query,
            "n": self.num_query_variants,
            "trusted_sources_instruction": trusted_instruction,
        })
        text = _clean_llm_text(result)
        try:
            queries = json.loads(_extract_json(text)).get("queries", [])
        except (json.JSONDecodeError, ValueError):
            logger.warning("Query diversification JSON parse failed; using original query.")
            queries = []

        queries = queries or [query]
        logger.info("Step 1: %d query variants generated", len(queries))
        return queries

    # -------------------------------------------------------------------------
    # Step 2: Multi-query search
    # -------------------------------------------------------------------------

    def _search_all(self, queries: list[str]) -> list[SearchSnippet]:
        seen: dict[str, SearchSnippet] = {}  # normalised URL → first-seen snippet
        with DDGS() as ddgs:
            for q in queries:
                try:
                    results = ddgs.text(q, max_results=MAX_RESULTS_PER_QUERY)
                    for r in results:
                        url = r.get("href", "")
                        if not url:
                            continue
                        key = _normalise_url(url)
                        if key not in seen:
                            seen[key] = SearchSnippet(
                                url=url,
                                title=r.get("title", ""),
                                excerpt=r.get("body", ""),
                                query_source=q,
                            )
                except Exception as e:
                    logger.warning("DDG search failed for query '%s': %s", q, e)

        snippets = list(seen.values())
        logger.info("Step 2: %d unique snippets from %d queries", len(snippets), len(queries))
        return snippets

    # -------------------------------------------------------------------------
    # Step 3: Snippet relevance pre-check
    # -------------------------------------------------------------------------

    def _score_snippets(self, snippets: list[SearchSnippet], query: str) -> list[SearchSnippet]:
        if not snippets:
            return []

        # Substack paywall heuristic — no LLM needed
        for s in snippets:
            if "substack.com" in s.domain and len(s.excerpt) < SUBSTACK_PAYWALL_CUTOFF:
                s.relevance_score = 0.0
                s.skip_reason = "paywall"

        to_score = [s for s in snippets if s.relevance_score is None]
        chunk_size = 15
        for i in range(0, len(to_score), chunk_size):
            self._score_chunk(to_score[i : i + chunk_size], query)

        passing = []
        for s in snippets:
            if s.relevance_score is None:
                s.relevance_score = 0.5  # shouldn't happen, but safe default
            if s.relevance_score < self.relevance_cutoff:
                if not s.skip_reason:
                    s.skip_reason = "low_score"
            else:
                passing.append(s)

        logger.info("Step 3: %d/%d snippets passed relevance pre-check", len(passing), len(snippets))
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
    # Step 6: Full page fetch
    # -------------------------------------------------------------------------

    def _fetch_pages(self, snippets: list[SearchSnippet]) -> list[FetchedPage]:
        pages: list[FetchedPage] = []
        with ThreadPoolExecutor(max_workers=self.max_fetch_workers) as executor:
            futures = {executor.submit(self._fetch_one, s): s for s in snippets}
            for future in as_completed(futures):
                pages.append(future.result())
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

def _normalise_url(url: str) -> str:
    """Lowercase scheme+host, strip trailing slash — used for deduplication."""
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
