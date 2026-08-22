"""Command-line interface for the enterprise support pipeline.

Presentation only (AGENTS.md section 3): this module parses arguments,
invokes the compiled graph, and renders ``PipelineState``. All pipeline
behaviour lives in ``src``.

Credentials are not required to start: the guardrail and low-score
routes never reach an LLM, so they must stay demonstrable without a key
(remediation plan Finding 8). ``--check`` reports configuration state for
an operator who wants that answer before running a query.

Usage:
    python main.py "How do I claim a taxi fare after overtime?"
    python main.py --interactive
    python main.py --check
"""

from __future__ import annotations

import sys

from langgraph.graph.state import CompiledStateGraph

from src import config
from src.fallback import response_text_for_state
from src.graph import build_graph, load_screened_corpus
from src.schemas import PipelineState

BANNER = "=" * 68
DIVIDER = "-" * 68

# The score label must present the number as a similarity heuristic,
# never as an answer-correctness probability (AGENTS.md section 4).
SCORE_LABEL = "Retrieval similarity score (heuristic)"

# Reported by --check. The credential is described by presence only; its
# value never reaches the terminal (AGENTS.md section 7).
STARTUP_ERROR_TEXT = (
    "ERROR: the knowledge base could not be loaded ({name}). "
    "Check CORPUS_DIR and the document frontmatter."
)
# Reported by --check so an operator can see the ingestion screen ran,
# rather than having to trust that it exists. The quarantined text itself
# is never printed: it is exactly the string that must not be repeated
# anywhere a reader might take it for an instruction.
CORPUS_SCREEN_CLEAN_TEXT = (
    "Ingestion screen: {count} documents loaded, no instruction-shaped "
    "lines quarantined"
)
CORPUS_SCREEN_QUARANTINED_TEXT = (
    "Ingestion screen: {source_id} had {lines} instruction-shaped line(s) "
    "quarantined before indexing"
)
KEY_CONFIGURED_TEXT = "OPENAI_API_KEY: configured"
KEY_MISSING_TEXT = (
    "OPENAI_API_KEY: missing. Guardrail refusals and low-score fallbacks "
    "still work; questions that need the answer service will report that "
    "it is unavailable.\n"
    "Copy .env.example to .env and add your OpenAI API key to enable them."
)


def _response_text(state: PipelineState) -> str:
    """Pick the user-facing text for the finished request's route."""
    fixed_text = response_text_for_state(
        state.get("route"),
        state.get("guardrail_reason"),
        state.get("fallback_reason"),
        telemetry_logged=state.get("telemetry_logged", True),
    )
    if fixed_text is not None:
        return fixed_text
    return state.get("answer", "")


def _render_sources(state: PipelineState) -> None:
    """List the validated citations with their document titles."""
    citations = state.get("valid_citations", [])
    if not citations:
        return
    titles = {
        document.source_id: document.title
        for document in state.get("answer_evidence", [])
    }
    print("\nSources:")
    for source_id in citations:
        title = titles.get(source_id, "")
        print(f"  {source_id} - {title}" if title else f"  {source_id}")


def _render_diagnostics(state: PipelineState) -> None:
    """Show route and scores without presenting them as probabilities."""
    print(f"\nRoute: {state.get('route', 'unknown')}")
    raw_score = state.get("raw_retrieval_score")
    if raw_score is not None:
        line = f"{SCORE_LABEL}: raw={raw_score:.4f}"
        expanded_score = state.get("expanded_retrieval_score")
        if expanded_score is not None:
            line += f", expanded={expanded_score:.4f}"
        print(line)
    rewritten_queries = state.get("rewritten_queries")
    if rewritten_queries is not None:
        print(f"Rewrite used: yes ({len(rewritten_queries)} variants)")
        for rewritten_query in rewritten_queries:
            print(f"  - {rewritten_query}")


def run_query(graph: CompiledStateGraph, query: str) -> None:
    """Run one query through the graph and render the final state."""
    state: PipelineState = graph.invoke({"query": query})
    print(BANNER)
    print(_response_text(state))
    _render_sources(state)
    print(DIVIDER)
    _render_diagnostics(state)
    print(BANNER)


def _run_interactive(graph: CompiledStateGraph) -> int:
    """Prompt for queries until the user exits."""
    print("Enterprise Knowledge & Support Agent")
    print("Type an empty line, 'exit', or Ctrl-C to quit.")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if query.lower() in {"", "exit", "quit"}:
            return 0
        _run_safely(graph, query)


def _run_safely(graph: CompiledStateGraph, query: str) -> int:
    """Run one query, reporting failures without leaking payloads."""
    try:
        run_query(graph, query)
    # Presentation-layer last resort: show only the exception type so no
    # provider payload or prompt fragment can reach the terminal.
    except Exception as exc:
        print(
            "ERROR: the pipeline could not process this query "
            f"({type(exc).__name__})",
            file=sys.stderr,
        )
        return 1
    return 0


def _compile_graph() -> CompiledStateGraph | None:
    """Compile the pipeline, reporting a start-up failure by type only.

    Returns:
        The compiled graph, or ``None`` when the corpus or configuration
        could not be loaded. Loader messages name filesystem paths, so
        only the exception type reaches the terminal.
    """
    try:
        return build_graph()
    # Deliberately broad: every start-up failure -- corpus, frontmatter,
    # configuration -- is the same outcome for the caller, an entry point
    # that cannot serve any request.
    except Exception as exc:
        print(
            STARTUP_ERROR_TEXT.format(name=type(exc).__name__),
            file=sys.stderr,
        )
        return None


def _run_setup_check() -> int:
    """Report corpus and credential readiness without running a query.

    Lazy credential validation is what lets the deterministic routes run
    without a key, but it also means an operator would otherwise discover
    a missing key only mid-demo. This is the explicit place to ask.

    Returns:
        ``0`` when the corpus loads, whatever the credential state: a
        missing key is a documented degraded mode, not a broken install.
        ``1`` when the corpus or configuration cannot be loaded at all.
    """
    if _compile_graph() is None:
        return 1
    print("Corpus and pipeline: ready")
    for line in _corpus_screen_lines():
        print(line)
    print(
        KEY_CONFIGURED_TEXT
        if config.has_llm_credential()
        else KEY_MISSING_TEXT
    )
    return 0


def _corpus_screen_lines() -> list[str]:
    """Describe what the ingestion screen did to the loaded corpus.

    The corpus is read a second time here instead of being pulled out of
    the compiled graph: ``--check`` runs once, and giving the graph an
    accessor for it would put a reporting need into the pipeline
    contract. Reached only after the graph compiled, so the load cannot
    fail here for a reason the caller has not already been told about.

    Returns:
        One line per document that had lines quarantined, or a single
        line stating that none did.
    """
    documents = load_screened_corpus()
    quarantined = [
        document for document in documents if document.quarantined_line_count
    ]
    if not quarantined:
        return [CORPUS_SCREEN_CLEAN_TEXT.format(count=len(documents))]
    return [
        CORPUS_SCREEN_QUARANTINED_TEXT.format(
            source_id=document.source_id,
            lines=document.quarantined_line_count,
        )
        for document in quarantined
    ]


def main() -> int:
    """Compile the graph once and dispatch the requested mode.

    A handled request -- answered, blocked, or degraded to fallback --
    exits ``0``; only a start-up failure or an unexpected presentation
    error is non-zero. A missing credential is a controlled request
    outcome, so it is not a start-up failure (remediation plan
    section 8.4).
    """
    arguments = [arg for arg in sys.argv[1:] if arg != "--interactive"]
    if "--check" in arguments:
        return _run_setup_check()

    graph = _compile_graph()
    if graph is None:
        return 1

    if arguments:
        query = " ".join(arguments).strip()
        if not query:
            raise SystemExit('Usage: python main.py "<your question>"')
        return _run_safely(graph, query)
    return _run_interactive(graph)


if __name__ == "__main__":
    raise SystemExit(main())
