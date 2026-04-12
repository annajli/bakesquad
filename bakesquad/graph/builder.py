"""
Graph builder — assembles the LangGraph StateGraph for BakeSquad.

Usage:
    from bakesquad.graph.builder import build_graph
    graph = build_graph()
    result = graph.invoke({"thread_id": "...", "user_query": "banana bread", ...})

The graph is built lazily (call build_graph() once; reuse the compiled object).
SqliteSaver checkpointing is enabled by default, writing to ~/.bakesquad/graph.db.

Phase 1 status:
  - Full node wiring is implemented.
  - clarify node is wired with interrupt_before so the graph pauses for user input.
  - Streaming via graph.astream() is ready but requires an async caller (Phase 3 FastAPI).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CHECKPOINTS_PATH = Path.home() / ".bakesquad" / "graph.db"


def _should_clarify(state: dict) -> str:
    """Route to clarify node when category confidence is below threshold."""
    from bakesquad.graph.nodes import _CLARIFY_THRESHOLD
    if state.get("category_confidence", 1.0) < _CLARIFY_THRESHOLD:
        return "clarify"
    return "search"


def _route_turn_type(state: dict) -> str:
    """After classify_intent (or filter_node escalation), route to the appropriate sub-pipeline."""
    turn = state.get("turn_type", "new_search")
    if turn in ("new_search", "re_search"):
        return "expand_query"
    if turn == "re_filter":
        return "filter_node"
    return "factual"


def _has_results(state: dict) -> str:
    """After search, check whether candidates exist before fetching."""
    return "fetch" if state.get("snippets") else "END"


def _has_pages(state: dict) -> str:
    """After fetch, check whether pages were retrieved before parsing."""
    return "parse" if state.get("fetched_pages") else "END"


def _has_recipes(state: dict) -> str:
    """After parse, check whether any recipes were parsed before scoring."""
    return "verify" if state.get("parsed_recipes") else "END"


def build_graph(checkpointing: bool = True):
    """
    Compile and return the BakeSquad LangGraph.

    Args:
        checkpointing: If True, attaches SqliteSaver so conversation state
                       persists across invocations by thread_id.  Set False
                       for unit tests that don't need persistence.

    Returns:
        A compiled LangGraph runnable.
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise ImportError(
            "langgraph is required. Run: pip install langgraph"
        ) from exc

    from bakesquad.graph.nodes import (
        classify_intent,
        clarify,
        expand_query,
        factual,
        fetch,
        filter_node,
        memory,
        parse,
        score,
        search,
        verify,
    )
    from bakesquad.graph.state import BakeSquadState

    builder = StateGraph(BakeSquadState)

    # ------------------------------------------------------------------ nodes
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("expand_query", expand_query)
    builder.add_node("clarify", clarify)
    builder.add_node("search", search)
    builder.add_node("fetch", fetch)
    builder.add_node("parse", parse)
    builder.add_node("verify", verify)
    builder.add_node("score", score)
    builder.add_node("filter_node", filter_node)
    builder.add_node("factual", factual)
    builder.add_node("memory", memory)

    # ------------------------------------------------------------------ entry
    builder.set_entry_point("classify_intent")

    # ------------------------------------------------------------------ edges
    # Turn routing: classify_intent → expand_query | filter_node | factual
    builder.add_conditional_edges("classify_intent", _route_turn_type)

    # Step 1 → optional clarify → search
    builder.add_conditional_edges("expand_query", _should_clarify)
    builder.add_edge("clarify", "expand_query")   # resume after user answers

    # Search → fetch → parse → verify → score → memory
    builder.add_conditional_edges("search", _has_results)
    builder.add_conditional_edges("fetch", _has_pages)
    builder.add_conditional_edges("parse", _has_recipes)
    builder.add_edge("verify", "score")
    builder.add_edge("score", "memory")
    builder.add_edge("memory", END)

    # filter_node → memory normally; if filter found nothing it sets turn_type=re_search
    # and routes back through classify_intent → expand_query for a full new search.
    builder.add_conditional_edges(
        "filter_node",
        lambda s: "expand_query" if s.get("turn_type") == "re_search" else "memory",
    )

    # Factual answer → done
    builder.add_edge("factual", END)

    # ------------------------------------------------------------------ checkpointing
    checkpointer = None
    if checkpointing:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            _CHECKPOINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            checkpointer = SqliteSaver.from_conn_string(str(_CHECKPOINTS_PATH))
            logger.info("Graph checkpointing enabled: %s", _CHECKPOINTS_PATH)
        except ImportError:
            logger.warning(
                "langgraph[sqlite] not available; running without checkpointing. "
                "Install with: pip install 'langgraph[sqlite]'"
            )

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["clarify"],  # pause before clarify so caller can inject user answer
    )
