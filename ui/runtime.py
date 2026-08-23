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


def _invoke_graph(query: str) -> tuple[PipelineState, float, dict[str, float]]:
    """Run one query through the real pipeline, timing each node.

    Every node inside the graph degrades to a logged fallback on its own,
    so a failure that escapes the run is a fault in the runtime around
    them -- and on a Streamlit page an escaping exception is rendered as
    a traceback that may carry a filesystem path, a prompt fragment or a
    provider payload (AGENTS.md section 4, invariant 10). This seam
    absorbs it into the service-state the presentation layer already
    knows how to draw. Only the exception type reaches stderr, and no
    telemetry is written here: logging is the graph's job, and the UI
    layer holds no business logic (AGENTS.md section 3).

    The graph is streamed rather than invoked so the wall clock can be
    read between node updates. This is still one run of one request --
    streaming is how that run reports its steps -- and the final state is
    the last full-value chunk, which is what ``invoke`` would return. The
    alternative would be a per-node timing field inside
    ``PipelineState``, which changes the pipeline's contract for a number
    only the console reads.

    Returns:
        The final state, the total wall latency, and seconds per node
        keyed by the graph's own node names. A node that never ran is
        absent rather than zero, and a degraded run returns whatever was
        measured before the fault.
    """
    graph, _, _ = _graph_and_retriever()
    started = time.perf_counter()
    mark = started
    timings: dict[str, float] = {}
    state: PipelineState = {"query": query}
    try:
        for mode, chunk in graph.stream(
            {"query": query}, stream_mode=["updates", "values"]
        ):
            now = time.perf_counter()
            if mode == "updates":
                # A node can report more than once on a loop; the times
                # add up rather than the last one replacing the first.
                for node in chunk:
                    timings[node] = timings.get(node, 0.0) + (now - mark)
            else:
                state = chunk
            mark = now
    except Exception as exc:
        print(
            f"ui.runtime: request degraded after {type(exc).__name__}",
            file=sys.stderr,
        )
        state = _unhandled_failure_state(query)
    return state, time.perf_counter() - started, timings


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


def _record_request(
    query: str,
    state: PipelineState,
    latency: float,
    node_seconds: dict[str, float] | None = None,
) -> None:
    """Append one real request outcome to the active session's history.

    The request id is session-scoped presentation bookkeeping (it indexes
    that session's history list); every other stored field comes from the
    graph or is measured here. History is telemetry for the audit view,
    never conversation memory for the pipeline: no earlier turn is passed
    back into the graph.

    Args:
        query: The question as it was submitted.
        state: Final pipeline state for this request.
        latency: Total wall seconds measured around the run.
        node_seconds: Seconds per node from ``_invoke_graph``. A caller
            with no measurement passes nothing rather than zeros, because
            a zero would read as a node that ran instantly.
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
            # Measured around each node by the streaming run, so the
            # trace can say where a slow request spent its time instead
            # of only how long it took in total.
            "node_seconds": {
                node: round(seconds, 4)
                for node, seconds in (node_seconds or {}).items()
            },
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


def _all_requests() -> list[dict]:
    """Return every request of this browser tab, oldest first.

    The console is a view over the whole tab rather than over whichever
    session the assistant happens to be showing: an operator reading the
    route split or the fallback triage wants what the app did, not what
    one transcript did. Each row is a shallow copy carrying the id of the
    session it came from, because ``Q-001`` exists once per session and
    the console keys its selection on the pair.

    Returns:
        Session records with an added ``session_id``, ordered by their
        timestamp. Records themselves are not mutated.
    """
    rows: list[dict] = []
    for session in _sessions():
        for record in session["history"]:
            rows.append({**record, "session_id": session["session_id"]})
    # The sessions list is already in creation order and each history is
    # append-only, so this only matters if two sessions ever interleave;
    # sorting on the stamp keeps that case honest rather than assuming.
    rows.sort(key=lambda row: row["timestamp"])
    return rows


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
