"""Shared runtime wiring between the UI and the compiled pipeline.

The corpus, the retriever and the graph are built once per Streamlit
session through ``st.cache_resource``; a request is invoked here and its
outcome recorded in session state. Both pages read the same objects, so
they live in one module rather than being rebuilt per page.

Presentation and integration only (AGENTS.md section 3): nothing here
re-implements routing, thresholds, or validation.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime

import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from src.fallback import ReasonCode
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
    """Run one query through the real pipeline, measuring wall latency.

    Every node inside the graph degrades to a logged fallback on its own,
    so a failure that escapes ``invoke`` is a fault in the runtime around
    them -- and on a Streamlit page an escaping exception is rendered as
    a traceback that may carry a filesystem path, a prompt fragment or a
    provider payload (AGENTS.md section 4, invariant 10). This seam
    absorbs it into the service-state the presentation layer already
    knows how to draw. Only the exception type reaches stderr, and no
    telemetry is written here: logging is the graph's job, and the UI
    layer holds no business logic (AGENTS.md section 3).
    """
    graph, _, _ = _graph_and_retriever()
    started = time.perf_counter()
    try:
        state: PipelineState = graph.invoke({"query": query})
    except Exception as exc:
        print(
            f"ui.runtime: request degraded after {type(exc).__name__}",
            file=sys.stderr,
        )
        state = _unhandled_failure_state(query)
    return state, time.perf_counter() - started


def _unhandled_failure_state(query: str) -> PipelineState:
    """Build the state of a request the pipeline could not finish.

    Args:
        query: The employee's question, kept so the console can show
            which request degraded.

    Returns:
        A fallback state carrying the stage-failure reason code. The
        telemetry flag is False because nothing was written: the graph
        never reached its own logging node, and claiming a recorded
        question the sink never saw is the dishonesty Finding 7 removed.
    """
    return {
        "query": query,
        "route": "fallback",
        "fallback_reason": ReasonCode.EVIDENCE_FAILURE.value,
        "telemetry_logged": False,
    }


def _record_request(query: str, state: PipelineState, latency: float) -> None:
    """Append one real request outcome to the active session's history.

    The request id is session-scoped presentation bookkeeping (it indexes
    that session's history list); every other stored field comes from the
    graph or is measured here. History is telemetry for the audit view,
    never conversation memory for the pipeline: no earlier turn is passed
    back into ``invoke``.
    """
    session = _active_session()
    history = session["history"]
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
    # Rebinding keeps the flat key and the record pointing at the same
    # list even if something replaced one of them between runs.
    _bind_active(session)


def _next_session_id(started: datetime) -> str:
    """Return an id for a new session that no existing one already holds.

    The minute-resolution stamp is what the top bar shows and what the
    user quotes to support, so it is kept; two sessions opened inside one
    minute would collide on it, and a collision here is not cosmetic --
    the id keys the rail buttons.
    """
    base = f"RAG-{started.strftime('%Y%m%d-%H%M')}"
    taken = {
        session["session_id"]
        for session in st.session_state.get("sessions", [])
    }
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _new_session_record() -> dict:
    """Build one empty session: its id, its start clock, its history."""
    started = datetime.now().astimezone()
    return {
        "session_id": _next_session_id(started),
        "started_at": started.isoformat(timespec="seconds"),
        "history": [],
    }


def _bind_active(session: dict) -> None:
    """Point the flat session-state keys at one session.

    ``history`` and ``session_id`` stay as top-level keys because the
    console reads them directly; binding aliases them onto the active
    session's own list rather than copying it, so a request recorded here
    is the same object the console renders.
    """
    st.session_state["history"] = session["history"]
    st.session_state["session_id"] = session["session_id"]
    st.session_state["active_session"] = session["session_id"]


def _sessions() -> list[dict]:
    """Return every session of this browser session, oldest first.

    The list lives in Streamlit's session state and therefore for as long
    as the browser tab does. It is deliberately not persisted: the JSONL
    sink is the only thing that keeps employee questions across sessions,
    and it stays behind ``ENABLE_OPS_VIEW`` (AGENTS.md section 7).
    """
    sessions = st.session_state.get("sessions")
    if not sessions:
        sessions = [_new_session_record()]
        st.session_state["sessions"] = sessions
        _bind_active(sessions[0])
    return sessions


def _active_session() -> dict:
    """Return the session the user is currently reading."""
    sessions = _sessions()
    active = st.session_state.get("active_session")
    for session in sessions:
        if session["session_id"] == active:
            return session
    # The pointer named a session that no longer exists; the newest one is
    # the only defensible fallback, and rebinding keeps the flat keys true.
    _bind_active(sessions[-1])
    return sessions[-1]


def _start_new_session() -> None:
    """Open a new session, keeping the previous one in the rail.

    An empty current session is reused rather than stacked: pressing the
    button twice should not leave a row of sessions that never held a
    question.
    """
    current = _active_session()
    if not current["history"]:
        _bind_active(current)
        return
    session = _new_session_record()
    _sessions().append(session)
    _bind_active(session)


def _switch_session(session_id: str) -> None:
    """Make one archived session the active one again."""
    for session in _sessions():
        if session["session_id"] == session_id:
            _bind_active(session)
            return


def _session_id() -> str:
    """Return the active session's identifier, creating it on first use.

    Presentation bookkeeping only: it labels the session in the chat top
    bar so a user can quote it to support, and it shares the clock with
    the request ids the console lists for the same session.
    """
    return _active_session()["session_id"]
