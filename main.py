"""
BakeSquad — demo entry point.

Usage:
    python main.py
    python main.py "chewy brown butter chocolate chip cookies"
    python main.py "moist banana bread" --recency year
    python main.py "sourdough discard cookies" --recency month
"""

import logging
import os
import sys

# Must be set before langchain_community.document_loaders is imported
os.environ.setdefault("USER_AGENT", "BakeSquad/1.0")

from langchain_ollama import ChatOllama

from bakesquad.config import DEFAULT_OLLAMA_MODEL, LLM_TEMPERATURE
from bakesquad.search.ingestion import IngestionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)

DEFAULT_QUERY = "moist chocolate chip banana bread stays fresh for days"


def main() -> None:
    args = sys.argv[1:]
    recency = None
    if "--recency" in args:
        idx = args.index("--recency")
        recency = args[idx + 1] if idx + 1 < len(args) else None
        args = args[:idx] + args[idx + 2:]

    query = " ".join(args) or DEFAULT_QUERY
    print(f"\nQuery: {query!r}  (recency={recency or 'none'})")
    print("─" * 60)

    llm = ChatOllama(model=DEFAULT_OLLAMA_MODEL, temperature=LLM_TEMPERATURE)

    pipeline = IngestionPipeline(
        llm=llm,
        trusted_sources=[],  # e.g. ["seriouseats.com", "smittenkitchen.com"]
    )

    print("Running search + candidate selection (steps 1–4)…")
    candidates = pipeline.run_search_only(query, recency=recency)

    # Show what the LLM understood from the query
    plan = pipeline.last_query_plan
    if plan:
        print(f"\n{'─' * 60}")
        print("Query understanding:")
        print(f"  Category:          {plan.category}")
        print(f"  Hard constraints:  {plan.hard_constraints or '—'}")
        print(f"  Soft preferences:  {plan.soft_preferences or '—'}")
        print(f"  Search queries:    {len(plan.queries)} variants")
        for q in plan.queries:
            print(f"    • {q}")

    print(f"\n{'─' * 60}")
    print(f"Candidates ({len(candidates)} selected):")
    for i, s in enumerate(candidates, 1):
        print(f"  {i:>2}. [{s.relevance_score:.2f}] {s.domain}")
        print(f"       {s.title}")
        print(f"       {s.url}")

    if not candidates:
        print("\nNo candidates found. Try a different query.")
        return

    print(f"\n{'─' * 60}")
    print("Fetching full pages (step 6)…")
    pages = pipeline._fetch_pages(candidates)

    print(f"\n{'─' * 60}")
    print(f"Fetched pages ({len(pages)}):")
    for page in pages:
        print(f"  • {page.title or '(no title)'}")
        print(f"    {page.url}")
        print(f"    {len(page.raw_text):,} chars via {page.fetch_method}")

    thin_dropped = len(candidates) - len(pages)
    print(f"\nDone. {len(pages)} pages ready for parsing.", end="")
    if thin_dropped:
        print(f" ({thin_dropped} dropped as thin/gated content.)", end="")
    print()


if __name__ == "__main__":
    main()
