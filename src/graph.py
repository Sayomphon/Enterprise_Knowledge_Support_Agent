"""LangGraph nodes, conditional edges, and routing for the pipeline.

Implements the AGENTS.md section 2 architecture: deterministic guardrail,
original TF-IDF retrieval, a supported-scope and authority-coverage gate,
a calibrated three-band score router, an adaptive rewrite branch,
policy-first evidence selection, grounded generation, and deterministic
answer-contract validation whose invalid branch routes to fallback --
never straight to ``END``. Reporter output enters the state as
``candidate_answer`` and only the validation node may promote it to the
public ``answer`` (remediation plan Finding 5). Every threshold is read
from ``src.config``; no number lives in this module.

Credentials are checked at the LLM boundary, not at start-up, so the
zero-LLM routes stay usable without a key; a credential missing at the
reporter becomes its own reason code rather than a thin-evidence verdict
(remediation plan Finding 8). Each logging call reports whether the append
succeeded, and the outcome travels in the state as ``telemetry_logged``
so the response text can stay truthful (Finding 7).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src import config
from src.agents import MissingLlmCredentialError
from src.agents.reporter import generate_answer
from src.agents.rewriter import safe_rewrite
from src.answer_renderer import render_answer
from src.evidence_selector import select_evidence
from src.fallback import ReasonCode
from src.guardrails.citation_validator import validate_answer
from src.guardrails.input_guardrail import screen_query
from src.guardrails.rewrite_validator import validate_rewrites
from src.guardrails.scope_validator import validate_scope
from src.ingestion.loader import load_documents
from src.logging_utils import log_fallback_event
from src.retrievers.base import Retriever
from src.retrievers.local_tfidf import LocalTfidfRetriever
from src.schemas import Document, PipelineState


def route_after_guardrail(
    state: PipelineState,
) -> Literal["blocked", "pass"]:
    """Send blocked requests to the refusal node, everything else onward.

    Args:
        state: Pipeline state after the guardrail node ran.

    Returns:
        The branch key consumed by the LangGraph conditional edge.
    """
    return "blocked" if state.get("route") == "blocked" else "pass"


def route_after_raw_retrieval(
    state: PipelineState,
) -> Literal["high", "medium", "low"]:
    """Select the pipeline branch from the raw retrieval score.

    Implements the three-band policy of AGENTS.md section 2:
    high-confidence queries go straight to generation, medium-band
    queries pay for one rewrite attempt, and low-band queries fall back
    without any LLM call so out-of-domain questions are never rewritten
    into the domain.

    Args:
        state: Pipeline state that must contain ``raw_retrieval_score``.

    Returns:
        The branch key consumed by the LangGraph conditional edge.

    Raises:
        KeyError: If retrieval has not populated ``raw_retrieval_score``.
    """
    score = state["raw_retrieval_score"]
    if score >= config.DIRECT_ANSWER_THRESHOLD:
        return "high"
    if score >= config.REWRITE_FLOOR:
        return "medium"
    return "low"


def route_after_expanded_retrieval(
    state: PipelineState,
) -> Literal["answer", "fallback"]:
    """Gate the rewrite branch on the expanded retrieval score.

    Args:
        state: Pipeline state containing ``expanded_retrieval_score``.

    Returns:
        ``"answer"`` when the max-pooled score clears the final
        threshold, otherwise ``"fallback"``.
    """
    if state["expanded_retrieval_score"] >= config.FINAL_ANSWER_THRESHOLD:
        return "answer"
    return "fallback"


def route_after_scope(
    state: PipelineState,
) -> Literal["high", "medium", "fallback"]:
    """Gate the score router on supported scope and policy coverage.

    The similarity score alone cannot tell an answerable question from a
    lexically similar one the corpus has no policy for, so a request
    continues only when the scope gate accepted the topic, an
    authoritative policy covers it, and the raw score clears the rewrite
    floor.

    Args:
        state: Pipeline state after the scope node ran.

    Returns:
        The branch key consumed by the LangGraph conditional edge.

    Raises:
        KeyError: If retrieval has not populated ``raw_retrieval_score``.
    """
    if state.get("scope_reason") or state.get("fallback_reason"):
        return "fallback"
    band = route_after_raw_retrieval(state)
    return "fallback" if band == "low" else band


def route_after_evidence_selection(
    state: PipelineState,
) -> Literal["report", "fallback"]:
    """Let only requests backed by authoritative policy reach the reporter.

    Args:
        state: Pipeline state after the evidence-selection node ran.

    Returns:
        ``"fallback"`` when the selector found no policy evidence for the
        request, otherwise ``"report"``.
    """
    return "fallback" if state.get("fallback_reason") else "report"


def route_after_report(
    state: PipelineState,
) -> Literal["validate", "fallback"]:
    """Skip citation validation when the reporter itself failed.

    Args:
        state: Pipeline state after the report node ran.

    Returns:
        ``"fallback"`` when the report node recorded a failure reason,
        otherwise ``"validate"``.
    """
    return "fallback" if state.get("fallback_reason") else "validate"


def route_after_validation(
    state: PipelineState,
) -> Literal["valid", "invalid"]:
    """Terminate on validated answers, degrade invalid ones to fallback.

    Args:
        state: Pipeline state after the answer validation node ran.

    Returns:
        The branch key consumed by the LangGraph conditional edge.
    """
    return "valid" if state.get("route") == "answered" else "invalid"


def _top_source_ids(state: PipelineState) -> list[str]:
    """List retrieved candidate ids, best match first, for telemetry."""
    return [
        document.source_id
        for document in state.get("retrieved_candidates", [])
    ]


def build_graph(
    retriever: Retriever | None = None,
    log_path: str | Path | None = None,
    documents: list[Document] | None = None,
) -> CompiledStateGraph:
    """Compile the support pipeline with injectable dependencies.

    Args:
        retriever: Retrieval backend; defaults to a character TF-IDF
            index built once over the validated corpus. Tests inject a
            stub so routing can be exercised offline.
        log_path: JSONL telemetry destination override; defaults to
            ``config.FALLBACK_LOG_PATH``. Tests inject a temporary path
            so they never write into the real ``logs/`` directory.
        documents: Corpus override used by tests; defaults to the
            validated corpus. Evidence selection needs the whole corpus,
            not only the retrieved candidates, because a canonical policy
            may sit outside the top-k.

    Returns:
        The compiled state graph, ready for ``invoke``.
    """
    if documents is None:
        documents = load_documents()
    if retriever is None:
        retriever = LocalTfidfRetriever(documents)
    documents_by_id = {document.source_id: document for document in documents}

    def input_guardrail_node(state: PipelineState) -> dict[str, object]:
        """Screen the raw query deterministically before any other work."""
        result = screen_query(state["query"])
        if not result.ok:
            return {"route": "blocked", "guardrail_reason": result.reason}
        # Every later stage sees only the normalized form of the query.
        return {"query": result.normalized_query}

    def refuse_node(state: PipelineState) -> dict[str, object]:
        """Log the blocked request; the refusal text itself is fixed."""
        write = log_fallback_event(
            query=state["query"],
            reason=state.get("guardrail_reason", ReasonCode.PROMPT_INJECTION),
            raw_retrieval_score=None,
            expanded_retrieval_score=None,
            top_sources=[],
            rewritten_queries=[],
            log_path=log_path,
        )
        return {"telemetry_logged": write.ok}

    def retrieve_original_node(state: PipelineState) -> dict[str, object]:
        """Retrieve with the original query and record the top-1 score."""
        results = retriever.search([state["query"]], config.TOP_K)
        top_score = results[0].score if results else 0.0
        return {
            "retrieved_candidates": results,
            "raw_retrieval_score": top_score,
        }

    def validate_scope_node(state: PipelineState) -> dict[str, object]:
        """Check the topic and its policy coverage before any LLM call."""
        decision = validate_scope(state["query"])
        updates: dict[str, object] = {
            "scope_score": decision.score,
            "scope_topics": list(decision.topics),
        }
        if not decision.supported:
            updates["scope_reason"] = decision.reason
            # A clearly out-of-domain question already falls back on the
            # low band and keeps its historical reason code. The scope
            # gate earns its reason code on in-domain hard negatives,
            # whose score would otherwise buy them an answer.
            if state["raw_retrieval_score"] >= config.REWRITE_FLOOR:
                updates["fallback_reason"] = ReasonCode.UNSUPPORTED_TOPIC.value
            return updates
        # Coverage is checked before the rewrite branch on purpose: if no
        # active policy covers the topic, a rewrite cannot conjure one and
        # the request would only pay for an LLM call before falling back.
        coverage = select_evidence(
            state["retrieved_candidates"], documents_by_id, decision.topics
        )
        if not coverage.ok:
            updates["fallback_reason"] = coverage.reason
        return updates

    def rewrite_node(state: PipelineState) -> dict[str, object]:
        """Rewrite the medium-band query, then enforce intent on the result.

        Only validated candidates enter the state, so a rejected variant
        can reach neither the retriever nor the JSONL log: it is model
        output about the employee's question, and one reason to reject it
        is that it may have absorbed an injected instruction.
        """
        candidates, rewrite_failed = safe_rewrite(state["query"])
        validation = validate_rewrites(
            state["query"], candidates, state.get("scope_topics", [])
        )
        return {
            "route": "rewrite",
            "rewritten_queries": list(validation.accepted_queries),
            "rewrite_failed": rewrite_failed,
            "rewrite_rejected": validation.reason is not None,
        }

    def retrieve_expanded_node(state: PipelineState) -> dict[str, object]:
        """Retrieve with the original plus rewritten queries, max-pooled.

        The original query is always included, so a failed rewrite can
        never make retrieval worse than the original-only baseline.
        """
        queries = [state["query"], *state.get("rewritten_queries", [])]
        results = retriever.search(queries, config.TOP_K)
        top_score = results[0].score if results else 0.0
        return {
            "retrieved_candidates": results,
            "expanded_retrieval_score": top_score,
        }

    def select_evidence_node(state: PipelineState) -> dict[str, object]:
        """Reduce the candidates to policy-first, citable answer evidence."""
        selection = select_evidence(
            state["retrieved_candidates"],
            documents_by_id,
            state.get("scope_topics", []),
        )
        if not selection.ok:
            return {"fallback_reason": selection.reason}
        return {
            "answer_evidence": list(selection.answer_evidence),
            "authoritative_source_ids": list(selection.authoritative_ids),
        }

    def report_node(state: PipelineState) -> dict[str, object]:
        """Draft the grounded answer; provider failures degrade.

        The result is written to ``candidate_answer``, never to
        ``answer``: what the model produced is a proposal until the
        validator has checked every claim against this request's
        evidence.
        """
        updates: dict[str, object] = {}
        if state.get("route") != "rewrite":
            updates["route"] = "direct_answer"
        try:
            updates["candidate_answer"] = generate_answer(
                state["query"], state["answer_evidence"]
            )
        # Caught before the broad handler so the two stay distinguishable:
        # an unconfigured service is an operator problem, and telling the
        # employee to contact HR about missing evidence would send them
        # after a policy that was never consulted (Finding 8).
        except MissingLlmCredentialError:
            updates["fallback_reason"] = ReasonCode.LLM_NOT_CONFIGURED.value
        # Deliberately broad at the LLM boundary: surviving the request
        # via the fallback path outranks diagnosing the provider error
        # here. Only the exception type is recorded -- messages could
        # carry provider payloads or prompt fragments.
        except Exception as exc:
            print(
                "report_node: degraded to fallback after "
                f"{type(exc).__name__}",
                file=sys.stderr,
            )
            updates["fallback_reason"] = ReasonCode.REPORTER_FAILURE.value
        return updates

    def validate_citations_node(state: PipelineState) -> dict[str, object]:
        """Check the candidate against this request's answer evidence.

        This is the only node that may write the public ``answer``, and
        it writes it from the deterministic renderer rather than from the
        model text, so every citation an employee sees was validated
        first.
        """
        candidate = state["candidate_answer"]
        evidence_ids = {
            document.source_id for document in state["answer_evidence"]
        }
        result = validate_answer(
            candidate,
            evidence_ids,
            state.get("authoritative_source_ids", []),
        )
        if result.ok:
            return {
                "route": "answered",
                "answer": render_answer(candidate),
                "valid_citations": list(result.citations),
            }
        return {"fallback_reason": result.reason}

    def fallback_node(state: PipelineState) -> dict[str, object]:
        """Log the degradation with its reason and finish the request."""
        explicit_reason = state.get("fallback_reason")
        if explicit_reason:
            reason = explicit_reason
        elif state.get("rewrite_rejected"):
            # Separable on purpose: the evidence was thin because every
            # rewrite drifted, not because the corpus lacks the answer.
            reason = ReasonCode.REWRITE_REJECTED.value
        elif "expanded_retrieval_score" in state:
            reason = ReasonCode.REWRITE_LOW_RETRIEVAL_SCORE.value
        else:
            reason = ReasonCode.LOW_RETRIEVAL_SCORE.value
        write = log_fallback_event(
            query=state["query"],
            reason=reason,
            raw_retrieval_score=state.get("raw_retrieval_score"),
            expanded_retrieval_score=state.get("expanded_retrieval_score"),
            top_sources=_top_source_ids(state),
            rewritten_queries=state.get("rewritten_queries", []),
            log_path=log_path,
        )
        # Clearing the candidate is part of the fallback, not tidiness:
        # a rejected draft must not survive in the state that the CLI,
        # the Streamlit app, or a future caller receives from `invoke`.
        # The write outcome travels beside the reason, never inside it: a
        # full disk did not change why this request lost its answer, and
        # the presentation layer needs both facts to stay honest.
        return {
            "route": "fallback",
            "fallback_reason": reason,
            "candidate_answer": None,
            "telemetry_logged": write.ok,
        }

    builder = StateGraph(PipelineState)
    builder.add_node("input_guardrail", input_guardrail_node)
    builder.add_node("refuse", refuse_node)
    builder.add_node("retrieve_original", retrieve_original_node)
    builder.add_node("validate_scope", validate_scope_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("retrieve_expanded", retrieve_expanded_node)
    builder.add_node("select_evidence", select_evidence_node)
    builder.add_node("report", report_node)
    builder.add_node("validate_citations", validate_citations_node)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "input_guardrail")
    builder.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {"blocked": "refuse", "pass": "retrieve_original"},
    )
    builder.add_edge("retrieve_original", "validate_scope")
    builder.add_conditional_edges(
        "validate_scope",
        route_after_scope,
        {
            "high": "select_evidence",
            "medium": "rewrite",
            "fallback": "fallback",
        },
    )
    builder.add_edge("rewrite", "retrieve_expanded")
    builder.add_conditional_edges(
        "retrieve_expanded",
        route_after_expanded_retrieval,
        {"answer": "select_evidence", "fallback": "fallback"},
    )
    builder.add_conditional_edges(
        "select_evidence",
        route_after_evidence_selection,
        {"report": "report", "fallback": "fallback"},
    )
    builder.add_conditional_edges(
        "report",
        route_after_report,
        {"validate": "validate_citations", "fallback": "fallback"},
    )
    builder.add_conditional_edges(
        "validate_citations",
        route_after_validation,
        {"valid": END, "invalid": "fallback"},
    )
    builder.add_edge("refuse", END)
    builder.add_edge("fallback", END)
    return builder.compile()
