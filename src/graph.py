"""LangGraph nodes, conditional edges, and routing for the pipeline.

Implements the AGENTS.md section 2 architecture: deterministic guardrail,
original TF-IDF retrieval, a supported-scope and authority-coverage gate,
a calibrated three-band score router, a deterministic alias expansion
that resolves most medium-band queries before any model is asked, an
adaptive rewrite branch behind it, policy-first evidence selection,
grounded generation, and deterministic answer-contract validation whose
invalid branch routes to fallback -- never straight to ``END``. Reporter output enters the state as
``candidate_answer`` and only the validation node may promote it to the
public ``answer`` (remediation plan Finding 5). Every threshold is read
from ``src.config``; no number lives in this module.

One request deadline covers both LLM boundaries: each is offered only
what is left of ``config.REQUEST_DEADLINE_SECONDS``, the optional rewrite
is skipped once nothing is left, and a reporter reached that late
degrades as a service state rather than starting a call the employee
will not wait for.

Credentials are checked at the LLM boundary, not at start-up, so the
zero-LLM routes stay usable without a key; a credential missing at the
reporter becomes its own reason code rather than a thin-evidence verdict
(remediation plan Finding 8). Each logging call reports whether the append
succeeded, and the outcome travels in the state as ``telemetry_logged``
so the response text can stay truthful (Finding 7).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

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
from src.guardrails.input_guardrail import matched_rule, screen_query
from src.guardrails.rewrite_validator import validate_rewrites
from src.guardrails.scope_validator import (
    classify_expense_eligibility,
    validate_scope,
)
from src.ingestion.loader import load_documents
from src.logging_utils import log_fallback_event
from src.query_expansion import alias_expansion_variants, label_alias_matches
from src.retrievers.base import Retriever
from src.retrievers.local_tfidf import LocalTfidfRetriever
from src.schemas import Document, GroundedAnswer, PipelineState, RetrievedDocument


class Rewriter(Protocol):
    """The rewrite seam: a query in, candidates and a failure flag out."""

    def __call__(
        self, query: str, *, budget_seconds: float | None = None
    ) -> tuple[list[str], str | None]:
        """Propose search variants for one medium-band query.

        ``budget_seconds`` is what is left of the request deadline; an
        implementation that reaches no provider may ignore it.
        """


class Reporter(Protocol):
    """The generation seam: a query and its evidence in, a draft out."""

    def __call__(
        self,
        query: str,
        retrieved: Sequence[RetrievedDocument],
        *,
        budget_seconds: float | None = None,
    ) -> GroundedAnswer:
        """Draft an unvalidated candidate answer from the evidence."""


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


def route_after_alias_expansion(
    state: PipelineState,
) -> Literal["answer", "rewrite", "fallback"]:
    """Spend an LLM call only when the alias catalog was not enough.

    The medium band exists because the original score is inconclusive,
    and the deterministic expansion often resolves it: the scope gate has
    already named the topics, and their aliases are the corpus's own
    wording for what the employee wrote informally. When that clears the
    final threshold the request is answered with no model involved at
    all, which also makes the route reproducible.

    Args:
        state: Pipeline state after the alias-expansion node ran.

    Returns:
        The branch key consumed by the LangGraph conditional edge.
    """
    if state.get("fallback_reason"):
        return "fallback"
    if state["expanded_retrieval_score"] >= config.FINAL_ANSWER_THRESHOLD:
        return "answer"
    return "rewrite"


def route_after_rewrite(
    state: PipelineState,
) -> Literal["expand", "fallback"]:
    """Skip expanded retrieval when no rewrite survived validation.

    With no accepted rewrite the expanded search is byte-identical to the
    alias expansion that already ran, so its score equals
    ``expanded_retrieval_score`` -- which is below
    ``FINAL_ANSWER_THRESHOLD``, or this branch would not have been taken.
    The verdict is therefore already decided, and running the search
    again would only pay for it.

    Args:
        state: Pipeline state after the rewrite node ran.

    Returns:
        The branch key consumed by the LangGraph conditional edge.
    """
    return "expand" if state.get("rewritten_queries") else "fallback"


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


def _elapsed_seconds(state: PipelineState) -> float | None:
    """Return how long this request has been running, or ``None``.

    Args:
        state: Pipeline state, normally carrying the monotonic stamp the
            first node wrote.

    Returns:
        Seconds since the stamp, or ``None`` when a caller invoked a
        node directly and there is no stamp to measure from. Callers
        treat that as "no deadline", which is the same behaviour these
        boundaries had before the deadline existed.
    """
    started = state.get("request_started_at")
    if started is None:
        return None
    return time.monotonic() - float(started)


def _remaining_budget(state: PipelineState) -> float | None:
    """Return what is left of the request deadline, or ``None``.

    The two LLM boundaries are sequential on the same synchronous
    request, so their own timeouts bound one stalled call each and say
    nothing about the request that contains both. This is the shared
    ceiling: whatever it returns is what the next boundary may spend.

    Args:
        state: Pipeline state after the guardrail node stamped it.

    Returns:
        Remaining seconds, which may be zero or negative when the
        deadline has passed, or ``None`` when the request is untimed.
    """
    elapsed = _elapsed_seconds(state)
    if elapsed is None:
        return None
    return config.REQUEST_DEADLINE_SECONDS - elapsed


def _latency_ms(state: PipelineState) -> int | None:
    """Return the request's elapsed milliseconds for telemetry."""
    elapsed = _elapsed_seconds(state)
    return None if elapsed is None else int(elapsed * 1000)


def _degraded(node_name: str, exc: Exception, reason: ReasonCode) -> dict[str, object]:
    """Turn a crash inside a deterministic node into a fallback reason.

    The deterministic stages have no provider behind them, so an exception
    here is an index, corpus, or programming fault rather than a verdict.
    It still must not escape the graph: AGENTS.md section 4, invariant 9
    requires every degraded request to leave a reason code in the log, and
    an exception leaves none. Only the exception type is recorded, matching
    the LLM boundaries -- a message can carry a filesystem path or a
    fragment of corpus text.

    Args:
        node_name: Node that failed, for the stderr diagnostic.
        exc: The caught exception; only its type is reported.
        reason: Stable reason code for this stage.

    Returns:
        The state update that routes the request to the fallback node.
    """
    print(
        f"{node_name}: degraded to fallback after {type(exc).__name__}",
        file=sys.stderr,
    )
    return {"fallback_reason": reason.value}


def load_screened_corpus() -> list[Document]:
    """Load the corpus through the ingestion screen the pipeline uses.

    Which screen guards ingestion is a pipeline decision, so it is made
    here rather than at each entry point: the CLI's ``--check`` reports
    what the screen did, and a second copy of this wiring there could
    report a corpus the graph never ran on.

    Returns:
        The validated corpus, with instruction-shaped lines already
        quarantined out of the chat transcripts.

    Raises:
        CorpusValidationError: If the corpus is malformed, or a policy
            document carries instruction-shaped text.
    """
    return load_documents(content_screen=matched_rule)


def build_graph(
    retriever: Retriever | None = None,
    log_path: str | Path | None = None,
    documents: list[Document] | None = None,
    rewriter: Rewriter | None = None,
    reporter: Reporter | None = None,
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
        rewriter: Rewrite seam; defaults to the LLM-backed
            ``safe_rewrite``. The evaluation harness injects a
            cache-backed stub so it can measure THIS graph rather than a
            second copy of its routing.
        reporter: Generation seam; defaults to the LLM-backed
            ``generate_answer``. Injected for the same reason: the two
            LLM seams are the only thing that stopped the harness from
            invoking the real pipeline offline.

    Returns:
        The compiled state graph, ready for ``invoke``.
    """
    rewrite = rewriter if rewriter is not None else safe_rewrite
    report = reporter if reporter is not None else generate_answer
    if documents is None:
        documents = load_screened_corpus()
    if retriever is None:
        retriever = LocalTfidfRetriever(documents)
    documents_by_id = {document.source_id: document for document in documents}

    def input_guardrail_node(state: PipelineState) -> dict[str, object]:
        """Screen the raw query deterministically before any other work.

        This is also where the request clock starts. It is stamped
        before the screen runs rather than after, so the deadline covers
        everything the employee waits for, not everything after the part
        that is already fast.
        """
        started_at = time.monotonic()
        result = screen_query(state["query"])
        if not result.ok:
            # The normalized form travels on the blocked branch too. It is
            # what the refusal node logs, so the sink records stripped text
            # bounded by MAX_QUERY_CHARS rather than the raw payload -- and
            # a non-string input becomes the empty string here instead of
            # reaching ``json.dumps`` intact.
            return {
                "route": "blocked",
                "guardrail_reason": result.reason,
                "query": result.normalized_query,
                "request_started_at": started_at,
            }
        # Every later stage sees only the normalized form of the query.
        return {
            "query": result.normalized_query,
            "request_started_at": started_at,
        }

    def refuse_node(state: PipelineState) -> dict[str, object]:
        """Log the blocked request; the refusal text itself is fixed."""
        write = log_fallback_event(
            query=state["query"],
            reason=state.get("guardrail_reason", ReasonCode.PROMPT_INJECTION),
            raw_retrieval_score=None,
            expanded_retrieval_score=None,
            top_sources=[],
            rewritten_queries=[],
            latency_ms=_latency_ms(state),
            llm_calls=state.get("llm_calls", 0),
            log_path=log_path,
        )
        return {"telemetry_logged": write.ok}

    def retrieve_original_node(state: PipelineState) -> dict[str, object]:
        """Retrieve with the original query and record the top-1 score."""
        try:
            results = retriever.search([state["query"]], config.TOP_K)
        except Exception as exc:
            # The empty candidate list and zero score are not a verdict;
            # they exist so the routers downstream have the keys they read
            # before the explicit reason sends the request to fallback.
            return {
                "retrieved_candidates": [],
                "raw_retrieval_score": 0.0,
                **_degraded(
                    "retrieve_original_node",
                    exc,
                    ReasonCode.RETRIEVAL_FAILURE,
                ),
            }
        top_score = results[0].score if results else 0.0
        return {
            "retrieved_candidates": results,
            "raw_retrieval_score": top_score,
        }

    def validate_scope_node(state: PipelineState) -> dict[str, object]:
        """Check the topic and its policy coverage before any LLM call."""
        if state.get("fallback_reason"):
            # Retrieval already failed. Re-deciding scope on an empty
            # candidate list would overwrite that cause with
            # "no_authoritative_evidence", which is the symptom.
            return {}
        try:
            decision = validate_scope(state["query"])
            # Inside the same guard as the gate above, not after it: the
            # two are one deterministic scope stage, and a crash in
            # either must leave a reason code rather than escape
            # ``invoke`` (AGENTS.md section 4, invariant 9).
            eligibility = classify_expense_eligibility(
                state["query"], decision.topics
            )
        except Exception as exc:
            return _degraded(
                "validate_scope_node", exc, ReasonCode.EVIDENCE_FAILURE
            )
        updates: dict[str, object] = {
            "scope_score": decision.score,
            "scope_topics": list(decision.topics),
        }
        if not decision.supported:
            updates["scope_reason"] = decision.reason
            if decision.reason == ReasonCode.AMBIGUOUS_TOPIC:
                # Unlike the verdict below, this one is not gated on the
                # band. An under-specified question is short and vague,
                # so it nearly always scores low, and behind the band
                # gate this code could never be recorded at all. It also
                # earns the exception on its own terms: it changes what
                # the employee is told -- name the leave type -- rather
                # than only what the log calls the same refusal.
                updates["fallback_reason"] = ReasonCode.AMBIGUOUS_TOPIC.value
            # A clearly out-of-domain question already falls back on the
            # low band and keeps its historical reason code. The scope
            # gate earns its reason code on in-domain hard negatives,
            # whose score would otherwise buy them an answer.
            elif state["raw_retrieval_score"] >= config.REWRITE_FLOOR:
                updates["fallback_reason"] = ReasonCode.UNSUPPORTED_TOPIC.value
        elif eligibility == "uncovered":
            # Unlike the verdicts above, this one is deliberately NOT
            # gated on the band: an eligibility question scores high
            # precisely because the process policy shares its wording,
            # so a rule that only fired on low scores would never fire
            # at all. The reason is written before the router reads it,
            # which is what keeps a high score from buying an answer the
            # corpus never granted (AGENTS.md section 4, invariant 4c).
            updates["fallback_reason"] = (
                ReasonCode.UNCOVERED_EXPENSE_ITEM.value
            )
        # Policy coverage is NOT checked here. It used to be, to save the
        # rewrite branch an LLM call when no policy covered the topic --
        # but ``select_evidence`` filters the RETRIEVED CANDIDATES, not
        # the corpus, and a rewrite changes which documents are retrieved.
        # The check therefore refused medium-band questions whose policy
        # merely sat outside the original top-k, logging a corpus-gap
        # reason for a question the corpus answers. It now runs once, in
        # ``select_evidence_node``, against whichever candidate set that
        # branch actually produced.
        return updates

    def expand_deterministic_node(state: PipelineState) -> dict[str, object]:
        """Search the topic aliases before paying for a model rewrite.

        The variants are configuration data derived from the topics the
        scope gate already accepted, so they are not put through the
        rewrite validator: that validator exists to prove a MODEL did not
        change the employee's intent, and there is no model here. The
        original query stays first in the search, so max-pooling makes
        this expansion incapable of scoring worse than the raw retrieval
        it replaces.
        """
        variants = alias_expansion_variants(
            state["query"], state.get("scope_topics", [])
        )
        queries = [state["query"], *variants]
        try:
            results = retriever.search(queries, config.TOP_K)
        except Exception as exc:
            # The zero score would send the router to "rewrite"; the
            # explicit reason overrides it into the fallback branch, so a
            # crashed index never buys an LLM call.
            return {
                "alias_expansion_queries": variants,
                "expanded_retrieval_score": 0.0,
                **_degraded(
                    "expand_deterministic_node",
                    exc,
                    ReasonCode.RETRIEVAL_FAILURE,
                ),
            }
        top_score = results[0].score if results else 0.0
        return {
            "alias_expansion_queries": variants,
            "retrieved_candidates": label_alias_matches(
                results, variants, state["query"]
            ),
            "expanded_retrieval_score": top_score,
        }

    def rewrite_node(state: PipelineState) -> dict[str, object]:
        """Rewrite the medium-band query, then enforce intent on the result.

        Only validated candidates enter the state, so a rejected variant
        can reach neither the retriever nor the JSONL log: it is model
        output about the employee's question, and one reason to reject it
        is that it may have absorbed an injected instruction.

        The rewrite is the optional half of a medium-band request, so it
        is also the half the request deadline gives up first: with no
        budget left the call is not made at all, and the request degrades
        to the alias-expanded retrieval it already has (AGENTS.md
        section 4, invariant 8).
        """
        budget = _remaining_budget(state)
        if budget is not None and budget <= 0:
            print(
                "rewrite_node: skipped, request deadline exhausted",
                file=sys.stderr,
            )
            return {
                "route": "rewrite",
                "rewritten_queries": [],
                "rewrite_rejected": False,
                # Its own reason, not the low-score one: the corpus was
                # never asked a second question, so reporting thin
                # evidence would describe an experiment that did not run.
                "rewrite_failure_reason": (
                    ReasonCode.REQUEST_DEADLINE_EXCEEDED.value
                ),
            }
        candidates, failure_reason = rewrite(
            state["query"], budget_seconds=budget
        )
        validation = validate_rewrites(
            state["query"], candidates, state.get("scope_topics", [])
        )
        updates: dict[str, object] = {
            "route": "rewrite",
            "rewritten_queries": list(validation.accepted_queries),
            "rewrite_rejected": validation.reason is not None,
            "llm_calls": state.get("llm_calls", 0) + 1,
        }
        if failure_reason is not None:
            updates["rewrite_failure_reason"] = failure_reason
        return updates

    def retrieve_expanded_node(state: PipelineState) -> dict[str, object]:
        """Retrieve with the original, the aliases and the rewrites.

        The original query is always included, so a failed rewrite can
        never make retrieval worse than the original-only baseline; the
        alias variants ride along for the same reason, so a rewrite that
        is worse than the catalog cannot undo what the catalog already
        found.
        """
        alias_queries = list(state.get("alias_expansion_queries", []))
        queries = [
            state["query"],
            *alias_queries,
            *state.get("rewritten_queries", []),
        ]
        try:
            results = retriever.search(queries, config.TOP_K)
        except Exception as exc:
            # The zero score makes the expanded router choose "fallback";
            # the explicit reason is what the log actually records.
            return {
                "expanded_retrieval_score": 0.0,
                **_degraded(
                    "retrieve_expanded_node",
                    exc,
                    ReasonCode.RETRIEVAL_FAILURE,
                ),
            }
        top_score = results[0].score if results else 0.0
        return {
            "retrieved_candidates": label_alias_matches(
                results, alias_queries, state["query"]
            ),
            "expanded_retrieval_score": top_score,
        }

    def select_evidence_node(state: PipelineState) -> dict[str, object]:
        """Reduce the candidates to policy-first, citable answer evidence."""
        try:
            selection = select_evidence(
                state["retrieved_candidates"],
                documents_by_id,
                state.get("scope_topics", []),
            )
        except Exception as exc:
            return _degraded(
                "select_evidence_node", exc, ReasonCode.EVIDENCE_FAILURE
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
        budget = _remaining_budget(state)
        if budget is not None and budget <= 0:
            # Unlike the rewrite, this boundary produces the answer, so
            # skipping it ends the request -- as a service state, which
            # is what an exhausted deadline is.
            print(
                "report_node: skipped, request deadline exhausted",
                file=sys.stderr,
            )
            updates["fallback_reason"] = (
                ReasonCode.REQUEST_DEADLINE_EXCEEDED.value
            )
            return updates
        try:
            updates["candidate_answer"] = report(
                state["query"],
                state["answer_evidence"],
                budget_seconds=budget,
            )
            updates["llm_calls"] = state.get("llm_calls", 0) + 1
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
            # The bodies travel with the ids because provenance alone
            # cannot see a wrong figure under a right citation. The
            # original query goes with them so a number the employee
            # wrote themselves is not treated as invented.
            evidence_texts={
                document.source_id: document.content
                for document in state["answer_evidence"]
            },
            query=state["query"],
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
        elif state.get("rewrite_failure_reason"):
            # The rewriter never ran to completion. Reporting that as a
            # low expanded score would tell an operator the corpus is
            # thin during a provider outage, and would tell the employee
            # to contact HR about a service that was never reachable.
            reason = str(state["rewrite_failure_reason"])
        elif state.get("rewrite_rejected"):
            # Separable on purpose: the evidence was thin because every
            # rewrite drifted, not because the corpus lacks the answer.
            reason = ReasonCode.REWRITE_REJECTED.value
        elif state.get("route") == "rewrite":
            # The branch marker the rewrite node already set, rather than
            # the presence of a score key. Key presence is not a fact
            # about the route: any node that ever writes that key -- even
            # as None -- would silently change what the log records.
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
            alias_query_count=len(state.get("alias_expansion_queries", [])),
            # The scope verdict travels beside the reason rather than
            # replacing it. An unsupported question whose raw score never
            # cleared the rewrite floor is logged as a low score on
            # purpose -- that is the gate that stopped it -- and these
            # two fields are what let an operator group by topic anyway,
            # without a routing rule being rewritten to suit analytics.
            scope_topics=state.get("scope_topics", []),
            scope_reason=state.get("scope_reason"),
            # Cost travels with the reason: a deadline record that does
            # not say how long the request took cannot be checked, and a
            # call count separates a rewrite that ran from one that the
            # budget skipped.
            latency_ms=_latency_ms(state),
            llm_calls=state.get("llm_calls", 0),
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
    builder.add_node("expand_deterministic", expand_deterministic_node)
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
            "medium": "expand_deterministic",
            "fallback": "fallback",
        },
    )
    builder.add_conditional_edges(
        "expand_deterministic",
        route_after_alias_expansion,
        {
            "answer": "select_evidence",
            "rewrite": "rewrite",
            "fallback": "fallback",
        },
    )
    builder.add_conditional_edges(
        "rewrite",
        route_after_rewrite,
        {"expand": "retrieve_expanded", "fallback": "fallback"},
    )
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
