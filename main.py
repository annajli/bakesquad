"""
BakeSquad — demo entry point.

Usage:
    python main.py
    python main.py "chewy brown butter chocolate chip cookies"
"""

import logging
import sys

from langchain_ollama import ChatOllama

from bakesquad.config import DEFAULT_OLLAMA_MODEL, LLM_TEMPERATURE
from bakesquad.search.ingestion import IngestionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)

DEFAULT_QUERY = "moist chocolate chip banana bread stays fresh for days"


def main() -> None:
    query = " ".join(sys.argv[1:]) or DEFAULT_QUERY
    print(f"\nQuery: {query!r}\n{'─' * 60}")

    llm = ChatOllama(model=DEFAULT_OLLAMA_MODEL, temperature=LLM_TEMPERATURE)

    # Pass trusted_sources to enable site:-targeted query variants.
    # Empty list = no source-targeted queries (default behaviour per spec).
    pipeline = IngestionPipeline(
        llm=llm,
        trusted_sources=[],  # e.g. ["seriouseats.com", "smittenkitchen.com"]
    )

    # Run steps 1–4 first so we can inspect candidate selection before fetching
    print("Running search + candidate selection (steps 1–4)…")
    candidates = pipeline.run_search_only(query)

    print(f"\n{'─' * 60}")
    print(f"Candidates ({len(candidates)} selected):")
    for i, s in enumerate(candidates, 1):
        print(f"  {i:>2}. [{s.relevance_score:.2f}] {s.domain}")
        print(f"       {s.title}")
        print(f"       {s.url}")

    if not candidates:
        print("\nNo candidates found. Try a different query.")
        return

    # Fetch full pages (step 6)
    print(f"\n{'─' * 60}")
    print("Fetching full pages (step 6)…")
    pages = pipeline._fetch_pages(candidates)

    print(f"\n{'─' * 60}")
    print(f"Fetched pages ({len(pages)}):")
    for page in pages:
        status = f"ERROR: {page.fetch_error}" if page.fetch_error else f"{len(page.raw_text):,} chars via {page.fetch_method}"
        print(f"  • {page.title or '(no title)'}")
        print(f"    {page.url}")
        print(f"    {status}")

    skipped = sum(1 for c in candidates if c.skip_reason)
    print(f"\nDone. {len(pages)} pages ready for parsing. ({skipped} snippets skipped earlier.)")


if __name__ == "__main__":
    main()
