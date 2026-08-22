"""Shared runtime wiring between the UI and the compiled pipeline.

The corpus, the retriever and the graph are built once per Streamlit
session through ``st.cache_resource``; a request is invoked here and its
outcome recorded in session state. Both pages read the same objects, so
they live in one module rather than being rebuilt per page.

Presentation and integration only (AGENTS.md section 3): nothing here
re-implements routing, thresholds, or validation.
"""

from __future__ import annotations

import time
from datetime import datetime

import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from src.graph import build_graph
from src.ingestion.loader import load_documents
from src.retrievers.local_tfidf import LocalTfidfRetriever
from src.schemas import (
    Document,
    PipelineState,
)

# Page objects for the current script run so each rail can link to the other
# view. Streamlit rebuilds them on every rerun, so this is a cache of the
# current run only, never cross-session state.
_PAGE_REFS: dict[str, object] = {}


@st.cache_resource
def _corpus() -> list[Document]:
    """Load the validated corpus once per Streamlit process."""
    return load_documents()


@st.cache_resource
def _graph_and_retriever() -> tuple[
    CompiledStateGraph, LocalTfidfRetriever, datetime
]:
    """Build the TF-IDF index and compile the graph once per process.

    Streamlit reruns the script on every interaction; caching keeps the
    expensive index build and graph compilation out of that loop. The
    build timestamp is real and feeds the Ops "last index build" stat.
    """
    retriever = LocalTfidfRetriever(_corpus())
    built_at = datetime.now().astimezone()
    return (
        build_graph(retriever=retriever, documents=_corpus()),
        retriever,
        built_at,
    )


def _invoke_graph(query: str) -> tuple[PipelineState, float]:
    """Run one query through the real pipeline, measuring wall latency."""
    graph, _, _ = _graph_and_retriever()
    started = time.perf_counter()
    state: PipelineState = graph.invoke({"query": query})
    return state, time.perf_counter() - started


def _record_request(query: str, state: PipelineState, latency: float) -> None:
    """Append one real request outcome to the session history.

    The request id is session-scoped presentation bookkeeping (it indexes
    this history list); every other stored field comes from the graph or
    is measured here. History is telemetry for the audit view, never
    conversation memory for the pipeline.
    """
    history = st.session_state.setdefault("history", [])
    history.append(
        {
            "request_id": f"Q-{len(history) + 1:03d}",
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "query": query,
            "state": state,
            "latency_seconds": round(latency, 3),
        }
    )


def _session_id() -> str:
    """Return this browser session's identifier, creating it on first use.

    Presentation bookkeeping only: it labels the session in the chat top bar
    so a user can quote it to support, and it shares the clock with the
    request ids the console lists for the same session.
    """
    session_id = st.session_state.get("session_id")
    if session_id is None:
        started = datetime.now().astimezone()
        session_id = f"RAG-{started.strftime('%Y%m%d-%H%M')}"
        st.session_state["session_id"] = session_id
    return session_id
