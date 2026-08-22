"""Offline evaluation harness for retrieval routing, guardrail, citations.

Usage:
    python eval/run_eval.py --set calibration
    python eval/run_eval.py --set heldout
    python eval/run_eval.py --set heldout_v2
    python eval/run_eval.py --set guardrail
    python eval/run_eval.py --set contracts
    python eval/run_eval.py --set calibration --distribution
    python eval/run_eval.py --set calibration --perturb
    python eval/run_eval.py --set calibration --ngram 2,4
    python eval/run_eval.py --set calibration --title-weight 2
    python eval/run_eval.py --set guardrail --strict

Without ``--strict`` this is a reporting command: it prints every metric
and exits 0 even when a case contradicts its label, which is what a
threshold sweep needs. ``--strict`` turns the same run into a gate that
exits 1 on any labelled failure, so review and CI can depend on it.

No LLM is called anywhere in this harness. Medium-band expansion uses
``eval/cached_rewrites.json``; a query absent from the cache is evaluated
as a failed rewrite (original-query-only expansion), mirroring the
runtime's graceful degradation, and is reported as a cache miss.

The contract fixtures -- citations and rewrite pairs -- have their own
``--set contracts`` report. A retrieval run still scores them, because
``--strict`` has always gated on them, but it prints only their failures:
the same rates appearing under every retrieval set made one measurement
read as two.

Metric definitions:
    Retrieval Hit@3          -- answerable cases whose final retrieval
                                contains at least one expected source.
    Retrieval Recall@1       -- answerable cases whose best-placed
                                expected source ranks top of a
                                full-corpus search over the same queries
                                the pipeline ran. Hit@3 tolerates a
                                two-source case that found one of them at
                                rank 3; this does not.
    Retrieval MRR            -- mean reciprocal rank of that same source.
                                A case whose search surfaced no expected
                                source contributes 0.
    Answer-route Selection Precision
                             -- of cases predicted "answered", the share
                                that are labelled answerable AND hit an
                                expected source (answering from wrong
                                documents counts as a miss). It scores
                                route selection, not answer correctness:
                                nothing offline reads the answer text.
    Answer-route Coverage    -- labelled-answerable cases actually routed
                                to "answered". Always reported beside
                                Precision: a pipeline that answers almost
                                nothing scores perfect precision.
    False Fallback Rate      -- labelled-answerable cases the pipeline
                                refused. It is 1 - Coverage, reported in
                                its own right because the cost of this
                                pipeline's caution is the number a
                                reviewer should see without arithmetic.
    OOD Fallback Accuracy    -- labelled-fallback cases in the "ood"
                                category, that is questions outside
                                HR/Finance entirely, predicted fallback.
    Unsupported In-domain Fallback Accuracy
                             -- labelled-fallback cases in the
                                "unsupported" category, that is HR/Finance
                                questions the corpus has no policy for.
    Overall Fallback Accuracy
                             -- every labelled-fallback case, whatever
                                its category. It is the union of the two
                                rates above and is reported separately so
                                neither one silently borrows the other's
                                denominator.
    Authoritative Evidence Coverage Rate
                             -- cases predicted "answered" whose evidence
                                contains at least one policy document.
    Rewrite Recovery Rate    -- answerable cases entering the medium band
                                that end up predicted "answered".
    Injection Block Rate     -- attack cases blocked by the guardrail.
    Benign Pass Rate         -- benign lookalike cases passing it.
    Rewrite Intent Preservation Rate
                             -- labelled rewrite pairs in
                                eval/rewrite_cases.json whose accept or
                                reject verdict matches the label.
    Citation Provenance Validity Rate -- labelled candidate answers in
                                eval/citation_cases.json where the
                                validator's verdict matches the label.
    Claim Source Coverage Rate
                             -- claims across those candidates whose
                                source ids are non-empty and entirely
                                inside the case's answer evidence. It
                                describes the labelled set, which mixes
                                grounded and deliberately ungrounded
                                claims; live reporter coverage is not
                                measured offline.
    Invalid Candidate Leakage Rate
                             -- rejected candidates that still produced
                                public answer text under the runtime
                                promote rule. Target is zero.

``--set answers --live`` is the one exception to "no LLM is called
anywhere". Every metric above scores routing or a contract through a stub
reporter, so none of them reads an answer; this set runs the real
pipeline against fact anchors taken from the corpus and is therefore the
only thing here that can fail because an ANSWER was wrong. It costs money,
so it asks before spending it.

    Fact Recall              -- required corpus facts that appeared in
                                the rendered answer.
    Fact-Citation Alignment  -- found facts stated in a claim that cites
                                a document actually containing that fact.
                                A right number under the wrong citation
                                fails here.
    Alien Number Rate        -- numbers in the answer that appear in
                                neither the evidence nor the question.
                                Target is zero.
    Forbidden Fact Rate      -- claims the corpus deliberately does not
                                support, such as confirming an expense
                                case the chat transcript left open.
                                Target is zero.
    Correct Refusal Rate     -- cases the corpus cannot answer that were
                                refused rather than answered.
    Route Stability          -- cases whose route was identical across
                                every repeat run.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
# Allow `python eval/run_eval.py` from the repository root without
# installing the project as a package.
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402

from src.fallback import ReasonCode  # noqa: E402
from src.graph import build_graph, route_after_raw_retrieval  # noqa: E402
from src.guardrails.citation_validator import validate_answer  # noqa: E402
from src.guardrails.input_guardrail import (  # noqa: E402
    matched_rule,
    screen_query,
)
from src.guardrails.rewrite_validator import (  # noqa: E402
    numeric_anchors,
    validate_rewrites,
)
from src.ingestion.loader import load_documents  # noqa: E402
from src.retrievers.local_tfidf import LocalTfidfRetriever  # noqa: E402
from src.schemas import (  # noqa: E402
    AnswerClaim,
    Document,
    GroundedAnswer,
    RetrievedDocument,
)

# Routing every case through the real graph means every fallback attempts a
# telemetry write. The run gets its own throwaway sink so an evaluation can
# never append to the deployment's log.
_EVAL_LOG_DIR = tempfile.TemporaryDirectory(prefix="eval-telemetry-")
_EVAL_LOG_SINK = Path(_EVAL_LOG_DIR.name) / "fallback_queries.jsonl"

# The stub reporter's claim text. It is never scored -- the harness
# measures routing and evidence, not wording -- but it must be non-blank
# and free of citation markup to satisfy the answer contract.
EVAL_CLAIM_TEXT = "evaluation harness placeholder claim"

# The citation set describes an answer contract, so its probe query only
# has to resolve to a supported topic; which topic is immaterial.
CITATION_PROBE_QUERY = "ลาพักร้อนได้กี่วัน"
CITATION_PROBE_TOPIC = "annual_leave"

# Typo-perturbation sweep. The seed is fixed and written down because a
# robustness curve nobody else can reproduce is an anecdote with a
# decimal point; the levels are how many single-character edits one
# probe carries, so the curve has a clean 0-edit control at its left.
_PERTURBATION_SEED = 42
_PERTURBATION_LEVELS = (1, 2, 3)
_PERTURBATION_OPERATIONS = ("transpose", "delete", "substitute")
# Thai tone marks and the vowel signs that sit above or below a
# consonant. These are what a real mistyping swaps -- the calibration
# set's own `ลาพักรอ้น` is a misplaced tone mark -- and they are combining
# characters, so replacing one leaves a word that still looks like a word.
_THAI_MARKS = "่้๊๋ัิีึืุู็์ํ"

# The two buckets of the guardrail fixture, and the balanced minimum
# AGENTS.md section 10 requires of it. A rate computed over fewer cases
# than this is not evidence, so the harness refuses to report one.
_GUARDRAIL_CASE_TYPES = ("attack", "benign")
_MIN_GUARDRAIL_CASES_PER_TYPE = 12

RETRIEVAL_SETS = {
    "calibration": "retrieval_calibration.json",
    "heldout": "retrieval_heldout.json",
    # The second held-out split, written after the first one had been
    # read during the Task 1 audit. A reporting set that has been looked
    # at is a regression fixture, not generalisation evidence, so this
    # one was authored before any threshold work in the same round and
    # is run once (AGENTS.md section 10; remediation plan P1-7). Same
    # category mix as the original: 4 normal, 3 noisy, 3 out-of-domain,
    # 4 in-domain unsupported.
    "heldout_v2": "retrieval_heldout_v2.json",
    # Near-domain hard negatives and the benign questions they are one
    # word away from. It is a SEPARATE file rather than more calibration
    # cases so the scope catalog can be extended and re-measured without
    # touching the split the thresholds were tuned on, and separate from
    # the held-out set so that split keeps its one-run-only status.
    "near_domain": "near_domain_cases.json",
}


class EvalFixtureError(ValueError):
    """Raised when an evaluation fixture violates its schema."""


@dataclass(frozen=True)
class RetrievalCase:
    """One labelled retrieval query with an unambiguous expectation."""

    id: str
    category: str
    query: str
    expected_route: str
    expected_sources: tuple[str, ...]


def _searched_queries(state: dict) -> tuple[str, ...]:
    """List the queries behind the retrieval this request finished on.

    The rank metrics have to score the search the pipeline actually ran,
    and the medium band searches more than the employee's own words. This
    mirrors the list ``retrieve_expanded_node`` builds instead of
    inferring one from the band, so a case whose rewrite failed is ranked
    on the original query alone -- which is what it retrieved with.

    Args:
        state: The finished pipeline state.

    Returns:
        The original query first, then whatever expansion that route
        added, in the retriever's own argument order.
    """
    return (
        state["query"],
        *state.get("alias_expansion_queries", []),
        *state.get("rewritten_queries", []),
    )


@dataclass(frozen=True)
class Prediction:
    """Deterministic routing outcome for one retrieval case."""

    route: str
    band: str
    raw_score: float
    expanded_score: float | None
    retrieved_ids: tuple[str, ...]
    cache_hit: bool | None
    authoritative_ids: tuple[str, ...] = ()
    scope_topics: tuple[str, ...] = ()
    reason: str | None = None
    searched_queries: tuple[str, ...] = ()


def _load_json(file_name: str) -> object:
    """Decode one fixture file from the evaluation directory."""
    return json.loads((EVAL_DIR / file_name).read_text(encoding="utf-8"))


def load_retrieval_cases(set_name: str) -> list[RetrievalCase]:
    """Load one retrieval set, rejecting ambiguous labels outright."""
    raw_cases = _load_json(RETRIEVAL_SETS[set_name])
    cases: list[RetrievalCase] = []
    seen_ids: set[str] = set()
    for raw in raw_cases:
        case = RetrievalCase(
            id=str(raw["id"]),
            category=str(raw["category"]),
            query=str(raw["query"]),
            expected_route=str(raw["expected_route"]),
            expected_sources=tuple(raw["expected_sources"]),
        )
        if case.expected_route not in {"answered", "fallback"}:
            raise EvalFixtureError(
                f"{case.id}: expected_route must be 'answered' or "
                "'fallback' -- ambiguous labels are not allowed"
            )
        if case.expected_route == "answered" and not case.expected_sources:
            raise EvalFixtureError(
                f"{case.id}: answerable cases must name expected_sources"
            )
        if case.expected_route == "fallback" and case.expected_sources:
            raise EvalFixtureError(
                f"{case.id}: fallback cases must not name expected_sources"
            )
        if case.id in seen_ids:
            raise EvalFixtureError(f"duplicate case id {case.id!r}")
        seen_ids.add(case.id)
        cases.append(case)
    return cases


def load_rewrite_cache() -> dict[str, list[str]]:
    """Load the deterministic rewrite cache used for medium-band cases."""
    raw = _load_json("cached_rewrites.json")
    return {
        query: list(entry["queries"]) for query, entry in raw.items()
    }


def _cached_rewriter(rewrite_cache: dict[str, list[str]]):
    """Build the rewrite seam the harness injects into the real graph.

    Args:
        rewrite_cache: Query to pre-generated variants, from
            ``eval/cached_rewrites.json``.

    Returns:
        A callable with the signature of ``safe_rewrite`` that replays the
        cache instead of calling a provider. A query the cache does not
        cover is reported the way a provider failure is, so the graph
        degrades exactly as it would in production.
    """

    def rewrite(
        query: str, *, budget_seconds: float | None = None
    ) -> tuple[list[str], str | None]:
        # The budget is accepted and ignored: replaying a cache costs no
        # provider time, so trimming it would only make the harness
        # behave differently from the pipeline it measures.
        candidates = rewrite_cache.get(query)
        if not candidates:
            return [], ReasonCode.REWRITE_FAILURE.value
        return list(candidates), None

    return rewrite


def _stub_reporter(
    query: str,
    retrieved: Sequence[RetrievedDocument],
    *,
    budget_seconds: float | None = None,
) -> GroundedAnswer:
    """Stand in for the generation seam without calling a provider.

    The harness measures routing and evidence selection, not wording, so
    this returns the minimal candidate that satisfies the answer
    contract: one claim citing every authoritative id the selector chose.
    Everything downstream of it -- the citation validator, the promote
    rule, the fallback edge -- is then the production code path rather
    than a second implementation of it.

    Args:
        query: The guardrail-normalized query, unused here.
        retrieved: Answer evidence chosen by the evidence selector.
        budget_seconds: Remaining request budget, unused: this seam
            reaches no provider and so cannot exceed a deadline.

    Returns:
        A candidate answer citing the authoritative evidence.
    """
    policies = [
        document
        for document in retrieved
        if document.authority == "authoritative"
    ]
    return GroundedAnswer(
        claims=[
            AnswerClaim(
                text=EVAL_CLAIM_TEXT,
                source_ids=[document.source_id for document in policies],
                # Lifted from the first cited policy rather than written
                # here: the answer contract now requires a span that
                # really occurs in a document the claim cites, so a
                # hand-written placeholder would make every routing case
                # fall back on a rule this harness is not measuring.
                evidence_quote=_first_policy_span(policies),
            )
        ]
    )


def _first_policy_span(policies: Sequence[RetrievedDocument]) -> str:
    """Return a verbatim span of the first policy offered as evidence.

    Args:
        policies: The authoritative documents of one request, in the
            order the evidence selector produced.

    Returns:
        The document's longest opening line, which is always long enough
        to satisfy the contract's minimum, or ``""`` when there is no
        policy -- a state the graph never reaches the reporter in.
    """
    if not policies:
        return ""
    lines = [line.strip() for line in policies[0].content.splitlines()]
    return max(lines, key=len, default="")


def build_eval_graph(
    retriever: LocalTfidfRetriever,
    documents_by_id: dict[str, Document],
    rewrite_cache: dict[str, list[str]],
) -> CompiledStateGraph:
    """Compile the production graph with both LLM seams replaced.

    Args:
        retriever: Index shared across the whole run.
        documents_by_id: The corpus, for canonical-link resolution.
        rewrite_cache: Pre-generated rewrites replayed for medium-band
            cases.

    Returns:
        The same graph the CLI and the Streamlit app run, with the two
        provider boundaries swapped for offline stand-ins and telemetry
        pointed at a throwaway sink.
    """
    return build_graph(
        retriever=retriever,
        log_path=_EVAL_LOG_SINK,
        documents=list(documents_by_id.values()),
        rewriter=_cached_rewriter(rewrite_cache),
        reporter=_stub_reporter,
    )


def predict(
    case: RetrievalCase,
    graph: CompiledStateGraph,
    rewrite_cache: dict[str, list[str]],
) -> Prediction:
    """Route one case through the production graph, without any LLM.

    This invokes ``build_graph`` rather than restating its routing. A
    second copy of the router cannot regress in step with the first, so a
    harness that owns one is measuring itself: every threshold
    comparison, every gate ordering, and every reason code below now
    comes from ``src.graph``.

    Args:
        case: The labelled retrieval case.
        graph: The compiled pipeline from ``build_eval_graph``.
        rewrite_cache: Consulted only to report whether this query had a
            cached rewrite, which is a fact about the fixture rather than
            about the routing.

    Returns:
        The routing outcome read off the finished pipeline state.

    Raises:
        EvalFixtureError: If the fixture query does not pass the
            guardrail, which would make its label untestable.
    """
    guard = screen_query(case.query)
    if not guard.ok:
        raise EvalFixtureError(
            f"{case.id}: retrieval fixtures must pass the guardrail"
        )
    state = graph.invoke({"query": case.query})

    raw_score = state.get("raw_retrieval_score", 0.0)
    band = route_after_raw_retrieval({"raw_retrieval_score": raw_score})
    return Prediction(
        route="answered" if state.get("route") == "answered" else "fallback",
        band=band,
        raw_score=raw_score,
        expanded_score=state.get("expanded_retrieval_score"),
        retrieved_ids=tuple(
            document.source_id
            for document in state.get("retrieved_candidates", [])
        ),
        cache_hit=(
            guard.normalized_query in rewrite_cache
            if band == "medium"
            else None
        ),
        authoritative_ids=tuple(state.get("authoritative_source_ids", [])),
        scope_topics=tuple(state.get("scope_topics", [])),
        reason=state.get("fallback_reason"),
        searched_queries=_searched_queries(state),
    )


def _ratio(numerator: int, denominator: int) -> str:
    """Format one metric as a rate and its raw counts.

    An empty denominator reports "n/a" rather than a rate: a metric over
    no cases is not a measurement, and printing 0.000 or 1.000 for it
    would read as one.
    """
    if denominator == 0:
        return "n/a (0 cases)"
    return f"{numerator / denominator:.3f} ({numerator}/{denominator})"


def _best_expected_rank(
    case: RetrievalCase,
    prediction: Prediction,
    retriever: LocalTfidfRetriever,
    corpus_size: int,
) -> int | None:
    """Rank one case's best-placed expected source across the corpus.

    Hit@3 answers "did anything expected survive the top-k", which a
    two-source case passes on one source sitting at rank 3. Rank quality
    needs the position itself, so this repeats the case's own search with
    the cut-off removed rather than reading it off the truncated
    candidate list.

    The best-placed expected source is scored rather than the first one
    listed: ``expected_sources`` is a set of acceptable documents, not a
    priority order -- some cases name the policy first and others the
    chat that illustrates it -- so keying the metric on position 0 would
    measure fixture spelling as much as retrieval.

    Args:
        case: The labelled retrieval case.
        prediction: The routing outcome, for the queries it searched with.
        retriever: The index this run is measuring.
        corpus_size: Number of documents, used as an unbounded top-k.

    Returns:
        The 1-based rank of the highest-ranked expected source, or
        ``None`` when the search surfaced none of them.
    """
    ranked = retriever.search(list(prediction.searched_queries), corpus_size)
    expected = set(case.expected_sources)
    for position, document in enumerate(ranked, start=1):
        if document.source_id in expected:
            return position
    return None


def evaluate_retrieval(
    set_name: str,
    retriever: LocalTfidfRetriever,
    documents_by_id: dict[str, Document],
    show_distribution: bool,
    perturb: bool = False,
) -> int:
    """Run one retrieval set and print per-case rows plus the metrics.

    Returns:
        The number of strict failures in this set: every case whose route
        contradicted its label, plus answerable cases whose retrieval
        missed each expected source. The contract fixtures count too, but
        ``main`` adds them, so this stays a retrieval number.

        A labelled-answerable case that fell back counts here, even
        though it costs coverage rather than precision. The label is the
        contract: "either route is fine" is not a label this harness
        accepts, so a coverage miss must fail the gate rather than hide
        inside a rate. The held-out set currently carries one such case,
        which is why a strict held-out run exits non-zero against an
        otherwise healthy pipeline.
    """
    cases = load_retrieval_cases(set_name)
    rewrite_cache = load_rewrite_cache()
    graph = build_eval_graph(retriever, documents_by_id, rewrite_cache)
    predictions = {
        case.id: predict(case, graph, rewrite_cache) for case in cases
    }

    print(f"== retrieval set: {set_name} ({len(cases)} cases) ==")
    print(
        "thresholds: REWRITE_FLOOR="
        f"{config.REWRITE_FLOOR}, DIRECT_ANSWER_THRESHOLD="
        f"{config.DIRECT_ANSWER_THRESHOLD}, FINAL_ANSWER_THRESHOLD="
        f"{config.FINAL_ANSWER_THRESHOLD}"
    )
    for case in cases:
        prediction = predictions[case.id]
        expanded = (
            f" expanded={prediction.expanded_score:.4f}"
            if prediction.expanded_score is not None
            else ""
        )
        cache_note = ""
        if prediction.band == "medium" and not prediction.cache_hit:
            cache_note = " [rewrite-cache MISS: original-only expansion]"
        verdict = "ok" if prediction.route == case.expected_route else "WRONG"
        reason_note = f" reason={prediction.reason}" if prediction.reason else ""
        print(
            f"  {case.id:18s} {case.category:11s} raw="
            f"{prediction.raw_score:.4f}{expanded} band="
            f"{prediction.band:6s} predicted={prediction.route:8s} "
            f"expected={case.expected_route:8s} {verdict}{reason_note}"
            f"{cache_note}"
        )

    if show_distribution:
        print("-- score distribution by category (raw top-1) --")
        by_category: dict[str, list[float]] = {}
        for case in cases:
            by_category.setdefault(case.category, []).append(
                predictions[case.id].raw_score
            )
        for category, scores in sorted(by_category.items()):
            ordered = sorted(scores)
            print(
                f"  {category:9s} n={len(ordered)} min={ordered[0]:.4f} "
                f"max={ordered[-1]:.4f} "
                f"all={[round(score, 4) for score in ordered]}"
            )

    answerable = [c for c in cases if c.expected_route == "answered"]
    fallback_labelled = [c for c in cases if c.expected_route == "fallback"]

    hits = sum(
        bool(
            set(c.expected_sources)
            & set(predictions[c.id].retrieved_ids[: config.TOP_K])
        )
        for c in answerable
    )
    predicted_answered = [
        c for c in cases if predictions[c.id].route == "answered"
    ]
    precise = sum(
        c.expected_route == "answered"
        and bool(
            set(c.expected_sources)
            & set(predictions[c.id].retrieved_ids[: config.TOP_K])
        )
        for c in predicted_answered
    )
    # Each fallback category is scored against its own denominator. The
    # two categories test different defences -- the low band stops an
    # out-of-domain question, the scope gate stops an in-domain one the
    # corpus has no policy for -- so pooling them let a set with five
    # unsupported cases report a nine-case "OOD" rate that no out-of-domain
    # measurement backed.
    out_of_domain = [c for c in fallback_labelled if c.category == "ood"]
    ood_correct = sum(
        predictions[c.id].route == "fallback" for c in out_of_domain
    )
    unsupported = [c for c in fallback_labelled if c.category == "unsupported"]
    unsupported_correct = sum(
        predictions[c.id].route == "fallback" for c in unsupported
    )
    fallback_correct = sum(
        predictions[c.id].route == "fallback" for c in fallback_labelled
    )
    ranks = {
        c.id: _best_expected_rank(
            c, predictions[c.id], retriever, len(documents_by_id)
        )
        for c in answerable
    }
    recall_at_1 = sum(rank == 1 for rank in ranks.values())
    reciprocal_ranks = sum(
        1.0 / rank for rank in ranks.values() if rank is not None
    )
    answered_with_policy = sum(
        bool(predictions[c.id].authoritative_ids) for c in predicted_answered
    )
    covered = sum(
        predictions[c.id].route == "answered" for c in answerable
    )
    medium_answerable = [
        c for c in answerable if predictions[c.id].band == "medium"
    ]
    recovered = sum(
        predictions[c.id].route == "answered" for c in medium_answerable
    )

    print("-- metrics --")
    print(f"  Retrieval Hit@{config.TOP_K}:      {_ratio(hits, len(answerable))}")
    print(f"  Retrieval Recall@1:     {_ratio(recall_at_1, len(answerable))}")
    print(
        "  Retrieval MRR:          "
        + (
            "n/a (0 cases)"
            if not answerable
            else f"{reciprocal_ranks / len(answerable):.3f} "
            f"(best expected source, n={len(answerable)})"
        )
    )
    print(
        "  Answer-route Selection Precision: "
        f"{_ratio(precise, len(predicted_answered))}"
    )
    print(
        "      share of cases routed \"answered\" that were labelled "
        "answerable and hit an expected source -- route selection, not "
        "answer correctness"
    )
    print(
        "  Answer-route Coverage:  "
        f"{_ratio(covered, len(answerable))}"
    )
    print(
        "  False Fallback Rate:    "
        f"{_ratio(len(answerable) - covered, len(answerable))}"
    )
    print(
        "  OOD Fallback Accuracy:  "
        f"{_ratio(ood_correct, len(out_of_domain))}"
    )
    print(
        "  Unsupported In-domain Fallback Accuracy: "
        f"{_ratio(unsupported_correct, len(unsupported))}"
    )
    print(
        "  Overall Fallback Accuracy: "
        f"{_ratio(fallback_correct, len(fallback_labelled))}"
    )
    print(
        "  Authoritative Evidence Coverage Rate: "
        f"{_ratio(answered_with_policy, len(predicted_answered))}"
    )
    print(
        "  Rewrite Recovery Rate:  "
        f"{_ratio(recovered, len(medium_answerable))}"
    )

    if perturb:
        evaluate_perturbations(cases, graph, rewrite_cache)

    route_failures = sum(
        predictions[c.id].route != c.expected_route for c in cases
    )
    # An answerable case that reached "answered" without any expected
    # source is answering from the wrong documents, which the printed
    # Precision already discounts; counting it here makes strict mode
    # fail on it rather than only reporting it.
    source_failures = sum(
        1
        for c in answerable
        if not set(c.expected_sources)
        & set(predictions[c.id].retrieved_ids[: config.TOP_K])
    )
    return route_failures + source_failures


def _perturb_query(query: str, edits: int, rng: random.Random) -> str:
    """Apply ``edits`` single-character typos to one query.

    The three operations are the mistakes a Thai keyboard actually
    produces: two characters swapped, one dropped, and a tone mark or
    vowel sign replaced by a neighbouring one. Substitution is tried
    first because it is the most characteristic of Thai input and the
    least destructive; a query with no mark to swap gets a
    transposition instead, so every requested edit is really applied.

    Args:
        query: The fixture query, unmodified.
        edits: How many single-character typos to apply.
        rng: Seeded generator, so a sweep is reproducible by anyone.

    Returns:
        The perturbed query. A query too short to edit is returned
        unchanged rather than emptied, which would test the guardrail
        instead of the index.
    """
    characters = list(query)
    for _ in range(edits):
        if len(characters) < 2:
            break
        operation = rng.choice(_PERTURBATION_OPERATIONS)
        if operation == "substitute":
            positions = [
                index
                for index, character in enumerate(characters)
                if character in _THAI_MARKS
            ]
            if positions:
                position = rng.choice(positions)
                characters[position] = rng.choice(
                    [
                        mark
                        for mark in _THAI_MARKS
                        if mark != characters[position]
                    ]
                )
                continue
            operation = "transpose"
        index = rng.randrange(len(characters) - 1)
        if operation == "transpose":
            characters[index], characters[index + 1] = (
                characters[index + 1],
                characters[index],
            )
        else:
            del characters[index]
    return "".join(characters)


def evaluate_perturbations(
    cases: Sequence[RetrievalCase],
    graph: CompiledStateGraph,
    rewrite_cache: dict[str, list[str]],
) -> None:
    """Print the Hit@3 degradation curve under injected typos.

    The design claim behind character n-grams is that a misspelling
    still shares most of its fragments with the correct wording. Two
    fixture cases demonstrating it are an anecdote; this measures it, by
    running the same answerable queries again with one, two and three
    typos injected under a fixed seed.

    It reports and never gates. The perturbed queries carry no labels of
    their own -- they inherit the expected sources of the case they came
    from -- and a random edit that genuinely destroys a question is a
    fact about the edit, not a regression to fail a build on.

    Args:
        cases: The set's cases. Only labelled-answerable ones in the
            ``normal`` category are used: Hit@3 needs expected sources,
            and the ``noisy`` cases are already misspelled, so their
            level-0 column would not be the clean baseline the curve is
            measured against.
        graph: The compiled pipeline, shared with the main run.
        rewrite_cache: Passed to ``predict`` unchanged; a perturbed
            medium-band query is a cache miss by construction.
    """
    baseline = [
        c
        for c in cases
        if c.expected_route == "answered" and c.category == "normal"
    ]
    if not baseline:
        print(
            "-- typo perturbation: no correctly-spelled answerable "
            "cases in this set --"
        )
        return
    rng = random.Random(_PERTURBATION_SEED)
    print(
        f"-- typo-perturbation robustness (seed {_PERTURBATION_SEED}, "
        f"n={len(baseline)} correctly-spelled answerable cases, one "
        "probe per case per level) --"
    )
    print(
        "   reporting only, and a small sample: it says whether the "
        "index degrades gracefully, not by how much. Perturbed queries "
        "are absent from the rewrite cache by construction, so a "
        "medium-band probe expands on aliases and the original alone"
    )
    for level in (0, *_PERTURBATION_LEVELS):
        hits = 0
        answered = 0
        rejected = 0
        lost: list[str] = []
        for case in baseline:
            query = (
                case.query
                if level == 0
                else _perturb_query(case.query, level, rng)
            )
            probe = RetrievalCase(
                id=f"{case.id}~p{level}",
                category=case.category,
                query=query,
                expected_route=case.expected_route,
                expected_sources=case.expected_sources,
            )
            try:
                prediction = predict(probe, graph, rewrite_cache)
            except EvalFixtureError:
                # An edit that trips the input screen is a fact about
                # the edit. It counts as neither a hit nor a silent
                # pass, so it is reported on its own.
                rejected += 1
                continue
            if set(case.expected_sources) & set(
                prediction.retrieved_ids[: config.TOP_K]
            ):
                hits += 1
            else:
                lost.append(case.id)
            if prediction.route == "answered":
                answered += 1
        scored = len(baseline) - rejected
        note = f" guardrail-rejected={rejected}" if rejected else ""
        print(
            f"  perturbations={level}  Hit@{config.TOP_K}: "
            f"{_ratio(hits, scored)}  routed answered: "
            f"{_ratio(answered, scored)}{note}"
        )
        if lost:
            print(f"      lost every expected source: {', '.join(lost)}")


def _validated_guardrail_cases(raw_cases: object) -> list[dict]:
    """Reject a guardrail fixture that cannot support a security claim.

    The retrieval loader validates its fixtures strictly and this one did
    not, so two silent failures were possible: a record whose ``type``
    was misspelled landed in neither bucket and left both denominators
    untouched, and an empty file produced "n/a (0 cases)" with zero
    failures -- a green security gate over no coverage at all.

    Args:
        raw_cases: Decoded contents of ``guardrail_cases.json``.

    Returns:
        The validated records.

    Raises:
        EvalFixtureError: If a record is malformed, a ``type`` is not one
            of the two buckets, an id repeats, or either bucket falls
            below the balanced minimum AGENTS.md section 10 requires.
    """
    if not isinstance(raw_cases, list):
        raise EvalFixtureError("guardrail_cases.json must hold a list")
    seen_ids: set[str] = set()
    for case in raw_cases:
        if not isinstance(case, dict):
            raise EvalFixtureError("guardrail case must be an object")
        missing = {"id", "type", "query"} - set(case)
        if missing:
            raise EvalFixtureError(
                f"guardrail case is missing {sorted(missing)}"
            )
        if case["type"] not in _GUARDRAIL_CASE_TYPES:
            raise EvalFixtureError(
                f"{case['id']}: type must be one of "
                f"{sorted(_GUARDRAIL_CASE_TYPES)}, got {case['type']!r}"
            )
        if case["id"] in seen_ids:
            raise EvalFixtureError(f"duplicate case id {case['id']!r}")
        seen_ids.add(case["id"])
    for bucket in _GUARDRAIL_CASE_TYPES:
        count = sum(case["type"] == bucket for case in raw_cases)
        if count < _MIN_GUARDRAIL_CASES_PER_TYPE:
            raise EvalFixtureError(
                f"guardrail set needs at least "
                f"{_MIN_GUARDRAIL_CASES_PER_TYPE} {bucket} cases, got "
                f"{count}"
            )
    return raw_cases


def evaluate_guardrail() -> int:
    """Run the guardrail set and print both precision-critical rates.

    Returns:
        The number of security failures: attacks the screen let through
        plus benign lookalikes it refused. Both halves count, because a
        guardrail that blocks real questions is as unusable as one that
        misses attacks.
    """
    raw_cases = _validated_guardrail_cases(_load_json("guardrail_cases.json"))
    attacks = [c for c in raw_cases if c["type"] == "attack"]
    benign = [c for c in raw_cases if c["type"] == "benign"]
    print(
        f"== guardrail set ({len(attacks)} attack / {len(benign)} benign) =="
    )

    blocked = 0
    for case in attacks:
        result = screen_query(case["query"])
        # Naming the rule that fired keeps a widened pattern traceable to
        # the attack shape it was meant to cover.
        outcome = (
            f"blocked by {matched_rule(case['query'])}"
            if not result.ok
            else "PASSED (miss)"
        )
        blocked += not result.ok
        print(f"  {case['id']:15s} attack  -> {outcome}")
    passed = 0
    for case in benign:
        result = screen_query(case["query"])
        outcome = (
            "pass"
            if result.ok
            else f"BLOCKED ({result.reason}: {matched_rule(case['query'])})"
        )
        passed += result.ok
        print(f"  {case['id']:15s} benign  -> {outcome}")

    print("-- metrics --")
    print(f"  Injection Block Rate: {_ratio(blocked, len(attacks))}")
    print(f"  Benign Pass Rate:     {_ratio(passed, len(benign))}")
    return (len(attacks) - blocked) + (len(benign) - passed)


def _validated_citation_cases(raw_cases: object) -> list[dict]:
    """Reject a citation fixture the harness cannot score honestly.

    The set gained per-case document bodies when the answer contract
    started checking the figures a claim states, and a body filed under
    an id the case does not cite would be scored against nothing while
    looking deliberate in the file.

    Args:
        raw_cases: Decoded contents of ``citation_cases.json``.

    Returns:
        The validated records.

    Raises:
        EvalFixtureError: If a record is malformed, an id repeats, a
            rejection case names no expected reason, or the authority
            and evidence-body maps reference ids the case does not list
            as evidence.
    """
    if not isinstance(raw_cases, list):
        raise EvalFixtureError("citation_cases.json must hold a list")
    seen_ids: set[str] = set()
    for case in raw_cases:
        if not isinstance(case, dict):
            raise EvalFixtureError("citation case must be an object")
        missing = {
            "id",
            "claims",
            "evidence_ids",
            "authoritative_ids",
            "expected_valid",
        } - set(case)
        if missing:
            raise EvalFixtureError(
                f"citation case is missing {sorted(missing)}"
            )
        case_id = case["id"]
        if case_id in seen_ids:
            raise EvalFixtureError(f"duplicate case id {case_id!r}")
        seen_ids.add(case_id)
        evidence_ids = set(case["evidence_ids"])
        unknown_authority = set(case["authoritative_ids"]) - evidence_ids
        if unknown_authority:
            raise EvalFixtureError(
                f"{case_id}: authoritative ids {sorted(unknown_authority)} "
                "are not in evidence_ids"
            )
        unknown_bodies = set(case.get("evidence_texts", {})) - evidence_ids
        if unknown_bodies:
            raise EvalFixtureError(
                f"{case_id}: evidence_texts names {sorted(unknown_bodies)}, "
                "which the case does not list as evidence"
            )
        if not case["expected_valid"] and not case.get("expected_reason"):
            # Without it the comparison below scores the case against
            # ``None`` and can only pass by accident.
            raise EvalFixtureError(
                f"{case_id}: a rejection case must name expected_reason"
            )
    return raw_cases


def _citation_case_query(case: dict) -> str:
    """Return the employee question one citation fixture answers.

    Most cases describe a contract rather than a question, so they share
    the probe query; a case about a figure the employee supplied
    themselves carries its own.
    """
    return str(case.get("query", CITATION_PROBE_QUERY))


def evaluate_citations(report: bool = True) -> int:
    """Score the answer-contract validator against its labelled cases.

    Args:
        report: Whether to print the rates. A retrieval run scores this
            fixture because ``--strict`` has always gated on it, but
            printing the same three rates under every set made one
            measurement look like several. Mismatches print either way:
            a strict failure has to stay diagnosable.

    Returns:
        The number of strict failures: labelled verdicts the validator
        disagreed with, plus any rejected candidate that still rendered
        public text. Claim Source Coverage is reported but never counted,
        because the labelled set deliberately mixes grounded and
        ungrounded claims -- it describes the fixture, not a defect.
    """
    raw_cases = _validated_citation_cases(_load_json("citation_cases.json"))
    correct = 0
    grounded_claims = 0
    total_claims = 0
    rejected = 0
    leaked = 0
    for case in raw_cases:
        candidate = _candidate_answer(case)
        evidence_ids = set(case["evidence_ids"])
        result = validate_answer(
            candidate,
            evidence_ids,
            set(case["authoritative_ids"]),
            # The bodies come from the same corpus builder the graph
            # probe below uses, so the fixture cannot describe one set of
            # documents to the validator and another to the pipeline.
            evidence_texts={
                document.source_id: document.content
                for document in _citation_case_corpus(case)
            },
            query=_citation_case_query(case),
        )
        matches = result.ok == case["expected_valid"] and (
            case["expected_valid"]
            or result.reason == case.get("expected_reason")
        )
        correct += matches
        if not matches:
            print(
                f"  citation case {case['id']} MISMATCH: "
                f"ok={result.ok} reason={result.reason}"
            )
        total_claims += len(candidate.claims)
        for claim in candidate.claims:
            cited = set(claim.normalized_source_ids())
            grounded_claims += bool(cited) and cited <= evidence_ids
        if not result.ok:
            rejected += 1
            leaked += bool(_promoted_answer(candidate, case))
    if report:
        print(
            "  Citation Provenance Validity Rate: "
            f"{_ratio(correct, len(raw_cases))}"
        )
        print(
            "  Claim Source Coverage Rate:        "
            f"{_ratio(grounded_claims, total_claims)}"
        )
        print(
            "  Invalid Candidate Leakage Rate:    "
            f"{_ratio(leaked, rejected)}"
        )
    return (len(raw_cases) - correct) + leaked


def _promoted_answer(candidate: GroundedAnswer, case: dict) -> str:
    """Return the public answer the RUNTIME produces for one candidate.

    The previous version of this helper took the validator's own verdict
    and returned "" whenever that verdict was False. Since it was only
    ever called on rejected candidates, it could not return anything but
    "": the leakage metric was structurally incapable of being non-zero
    and would have reported 0 even with the promote rule deleted from
    ``validate_citations_node``. It now drives the real graph with a
    reporter that hands back this exact candidate, so the number measures
    the pipeline's promote rule rather than restating it.

    Args:
        candidate: The candidate answer the fixture describes.
        case: The fixture record, for the evidence behind that candidate.

    Returns:
        ``state["answer"]`` as the pipeline would show it, or an empty
        string when the request degraded instead.
    """
    documents = _citation_case_corpus(case)
    graph = build_graph(
        retriever=_FixedRetriever(documents),
        log_path=_EVAL_LOG_SINK,
        documents=documents,
        # No rewrite and no failure reason: the probe query sits in the
        # high band, so this seam is never reached. It still has to
        # satisfy the ``Rewriter`` protocol -- the second element is a
        # reason code or ``None``, and a bare ``True`` would have been
        # written straight into the log the day the probe or a threshold
        # moved.
        rewriter=lambda query, *, budget_seconds=None: ([], None),
        reporter=(
            lambda query, retrieved, *, budget_seconds=None: candidate
        ),
    )
    state = graph.invoke({"query": _citation_case_query(case)})
    return str(state.get("answer", ""))


class _FixedRetriever:
    """Return the whole supplied corpus, top-scored, for any query.

    The citation set describes an answer contract, not a retrieval
    outcome, so this hands the pipeline exactly the evidence the fixture
    names and lets the real scope, evidence and validation stages run.
    """

    def __init__(self, documents: list[Document]) -> None:
        self._results = [
            RetrievedDocument(
                source_id=document.source_id,
                title=document.title,
                source_type=document.source_type,
                content=document.content,
                score=min(config.DIRECT_ANSWER_THRESHOLD + 0.1, 1.0),
                authority=document.authority,
                status=document.status,
                topics=document.topics,
                canonical_source_ids=document.canonical_source_ids,
                matched_query=CITATION_PROBE_QUERY,
                matched_query_type="original",
            )
            for document in documents
        ]

    def search(
        self, queries: Sequence[str], top_k: int
    ) -> list[RetrievedDocument]:
        """Ignore the query and return the fixture's evidence."""
        return list(self._results[:top_k])


def _citation_case_corpus(case: dict) -> list[Document]:
    """Build the corpus one citation fixture implies.

    Args:
        case: The fixture record.

    Returns:
        One document per id in ``evidence_ids``, carrying policy
        authority when the fixture lists it in ``authoritative_ids`` and
        chat authority otherwise. Bodies come from the case's optional
        ``evidence_texts`` map, because the answer contract reads them:
        a case about a figure has to say which document states it. The
        placeholder body stands in for the cases that describe structure
        or provenance and name no figure at all. Every document covers
        the probe topic, so the scope and evidence gates admit them and
        the run reaches the validation node under test.
    """
    authoritative = set(case["authoritative_ids"])
    bodies = case.get("evidence_texts", {})
    return [
        Document(
            source_id=source_id,
            title=source_id,
            source_type="policy" if source_id in authoritative else "chat",
            content=str(bodies.get(source_id, EVAL_CLAIM_TEXT)),
            authority=(
                "authoritative"
                if source_id in authoritative
                else "supplementary"
            ),
            status="active",
            topics=(CITATION_PROBE_TOPIC,),
            canonical_source_ids=(
                ()
                if source_id in authoritative
                else tuple(sorted(authoritative))
            ),
        )
        for source_id in case["evidence_ids"]
    ]


def _candidate_answer(case: dict) -> GroundedAnswer:
    """Rebuild one labelled candidate answer from its fixture record."""
    return GroundedAnswer(
        claims=[
            AnswerClaim(
                text=str(claim["text"]),
                source_ids=list(claim["source_ids"]),
                evidence_quote=str(claim.get("evidence_quote", "")),
            )
            for claim in case["claims"]
        ],
        insufficient_evidence=bool(case.get("insufficient_evidence", False)),
    )


def evaluate_rewrite_pairs(report: bool = True) -> int:
    """Score the rewrite validator against its labelled pairs.

    Args:
        report: Whether to print the rate; see ``evaluate_citations``.

    Returns:
        The number of pairs whose accept/reject verdict contradicted the
        label.
    """
    raw_cases = _load_json("rewrite_cases.json")
    correct = 0
    for case in raw_cases:
        result = validate_rewrites(
            case["original"], [case["rewrite"]], case["original_topics"]
        )
        verdict = "accept" if result.accepted_queries else "reject"
        correct += verdict == case["expected"]
        if verdict != case["expected"]:
            print(
                f"  rewrite case {case['id']} MISMATCH: verdict={verdict} "
                f"expected={case['expected']}"
            )
    if report:
        print(
            "  Rewrite Intent Preservation Rate:   "
            f"{_ratio(correct, len(raw_cases))}"
        )
    return len(raw_cases) - correct


def evaluate_contracts(report: bool = True) -> int:
    """Score both validator fixtures in one place.

    The citation and rewrite sets describe contracts rather than
    retrieval, so they no longer belong inside a retrieval report. They
    are still gated by every retrieval run, which is why the caller can
    ask for the failure count without the printed rates.

    Args:
        report: Whether to print the header and the rates.

    Returns:
        Strict failures across both fixtures.
    """
    if report:
        print("== contract fixtures: citations, rewrite pairs ==")
        print("-- metrics --")
    return evaluate_citations(report) + evaluate_rewrite_pairs(report)


ANSWER_CASE_FILE = "answer_cases.json"

# The four kinds of question the answer set covers. They are listed so a
# misspelled category cannot quietly create a fifth bucket nobody scores.
_ANSWER_CATEGORIES = (
    "normal",
    "multi_condition",
    "ambiguous_chat",
    "correct_refusal",
)

# Upper bound on LLM calls one case-run can make: one rewrite on the
# medium band plus one report. Used only for the cost estimate printed
# before a live run, never as a limit.
_MAX_CALLS_PER_RUN = 2

# The renderer's own citation markup. It has to come off the answer before
# numbers are counted: "[HR-001]" is emitted deterministically from ids the
# validator already approved, so scoring its digits as model output made the
# first live run report 38 of 51 answers as containing an invented number
# when every one of them was a source id. Bounded quantifiers, like every
# other pattern in this repository.
_CITATION_MARKUP = re.compile(r"\[[A-Za-z]{1,10}-\d{1,6}\]")


@dataclass(frozen=True)
class FactAnchor:
    """One deterministic fact an answer must, or must not, contain.

    Attributes:
        fact: Human-readable name, used in the report.
        patterns: Compiled alternatives; any match counts as present.
        source_id: The document expected to carry this fact, or ``None``
            for a forbidden fact, which by definition has none.
    """

    fact: str
    patterns: tuple[re.Pattern[str], ...]
    source_id: str | None = None


@dataclass(frozen=True)
class AnswerCase:
    """One labelled question with the facts its answer is judged on."""

    id: str
    category: str
    query: str
    required_facts: tuple[FactAnchor, ...] = ()
    forbidden_facts: tuple[FactAnchor, ...] = ()
    required_policy_sources: tuple[str, ...] = ()
    expected_insufficient: bool = False


@dataclass
class AnswerRun:
    """What one execution of one case produced."""

    route: str
    answer: str
    citations: tuple[str, ...] = ()
    claims: tuple[tuple[str, tuple[str, ...]], ...] = ()
    evidence: tuple[tuple[str, str], ...] = ()
    reason: str | None = None
    latency_seconds: float = 0.0
    found_facts: tuple[str, ...] = field(default_factory=tuple)
    aligned_facts: tuple[str, ...] = field(default_factory=tuple)
    forbidden_hits: tuple[str, ...] = field(default_factory=tuple)
    alien_numbers: tuple[str, ...] = field(default_factory=tuple)


def _compiled_patterns(raw: object, case_id: str) -> tuple[re.Pattern[str], ...]:
    """Compile one anchor's alternatives, rejecting an empty list.

    Args:
        raw: The fixture's ``patterns`` value.
        case_id: Case id, for the error message.

    Returns:
        The compiled patterns.

    Raises:
        EvalFixtureError: If the list is empty or a pattern is invalid. An
            anchor with no pattern matches nothing and would silently
            score every answer as a miss.
    """
    if not isinstance(raw, list) or not raw:
        raise EvalFixtureError(f"{case_id}: patterns must be a non-empty list")
    compiled: list[re.Pattern[str]] = []
    for pattern in raw:
        try:
            compiled.append(re.compile(str(pattern)))
        except re.error as exc:
            raise EvalFixtureError(
                f"{case_id}: invalid pattern {pattern!r} ({exc})"
            ) from None
    return tuple(compiled)


def _anchors(raw: object, case_id: str, key: str) -> tuple[FactAnchor, ...]:
    """Build the anchors of one fixture field."""
    if not isinstance(raw, list):
        raise EvalFixtureError(f"{case_id}: {key} must be a list")
    anchors: list[FactAnchor] = []
    for entry in raw:
        if not isinstance(entry, dict) or "fact" not in entry:
            raise EvalFixtureError(f"{case_id}: {key} entry needs a 'fact'")
        anchors.append(
            FactAnchor(
                fact=str(entry["fact"]),
                patterns=_compiled_patterns(entry.get("patterns"), case_id),
                source_id=(
                    str(entry["source_id"]) if "source_id" in entry else None
                ),
            )
        )
    return tuple(anchors)


def load_answer_cases(
    documents_by_id: dict[str, Document] | None = None,
) -> list[AnswerCase]:
    """Load the answer set, rejecting a fixture the corpus cannot support.

    Args:
        documents_by_id: The corpus. When supplied, every required fact is
            checked against the document it names, so the set cannot ask
            for a fact this corpus does not carry -- which would report a
            fixture bug as a model failure, and would tempt whoever reads
            the report to "fix" the model.

    Returns:
        The validated cases.

    Raises:
        EvalFixtureError: On a duplicate id, an unknown category, a
            malformed anchor, or a required fact absent from its own
            source document.
    """
    raw_cases = _load_json(ANSWER_CASE_FILE)
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvalFixtureError(f"{ANSWER_CASE_FILE} must hold a non-empty list")
    cases: list[AnswerCase] = []
    seen_ids: set[str] = set()
    for raw in raw_cases:
        missing = {"id", "category", "query"} - set(raw)
        if missing:
            raise EvalFixtureError(f"answer case is missing {sorted(missing)}")
        case_id = str(raw["id"])
        if case_id in seen_ids:
            raise EvalFixtureError(f"duplicate case id {case_id!r}")
        seen_ids.add(case_id)
        if raw["category"] not in _ANSWER_CATEGORIES:
            raise EvalFixtureError(
                f"{case_id}: category must be one of "
                f"{sorted(_ANSWER_CATEGORIES)}, got {raw['category']!r}"
            )
        case = AnswerCase(
            id=case_id,
            category=str(raw["category"]),
            query=str(raw["query"]),
            required_facts=_anchors(
                raw.get("required_facts", []), case_id, "required_facts"
            ),
            forbidden_facts=_anchors(
                raw.get("forbidden_facts", []), case_id, "forbidden_facts"
            ),
            required_policy_sources=tuple(
                raw.get("required_policy_sources", [])
            ),
            expected_insufficient=bool(raw.get("expected_insufficient", False)),
        )
        if case.expected_insufficient and case.required_facts:
            raise EvalFixtureError(
                f"{case_id}: a case expected to be refused cannot also "
                "require facts in its answer"
            )
        if documents_by_id is not None:
            _require_facts_in_corpus(case, documents_by_id)
        cases.append(case)
    return cases


def _require_facts_in_corpus(
    case: AnswerCase, documents_by_id: dict[str, Document]
) -> None:
    """Reject a required fact its own source document does not carry."""
    for anchor in case.required_facts:
        document = documents_by_id.get(str(anchor.source_id))
        if document is None:
            raise EvalFixtureError(
                f"{case.id}: fact {anchor.fact!r} names unknown source "
                f"{anchor.source_id!r}"
            )
        if not _matches(anchor, f"{document.title}\n{document.content}"):
            raise EvalFixtureError(
                f"{case.id}: fact {anchor.fact!r} does not appear in "
                f"{anchor.source_id}; the corpus cannot support it"
            )


def _matches(anchor: FactAnchor, text: str) -> bool:
    """Report whether any of one anchor's patterns occurs in the text."""
    return any(pattern.search(text) for pattern in anchor.patterns)


def score_answer(case: AnswerCase, run: AnswerRun) -> AnswerRun:
    """Apply every deterministic check to one finished run.

    Nothing here asks a model to judge a model. The checks are string and
    number containment against the corpus, so the same answer scores the
    same way on any machine and the score can be re-derived from the
    transcript months later.

    Args:
        case: The labelled case.
        run: The run to score, with its route, answer and evidence filled
            in.

    Returns:
        The same run with its per-check findings populated.
    """
    evidence_text = "\n".join(content for _, content in run.evidence)
    found = tuple(
        anchor.fact
        for anchor in case.required_facts
        if _matches(anchor, run.answer)
    )
    aligned = tuple(
        anchor.fact
        for anchor in case.required_facts
        if _fact_is_aligned(anchor, run)
    )
    forbidden = tuple(
        anchor.fact
        for anchor in case.forbidden_facts
        if _matches(anchor, run.answer)
    )
    known = numeric_anchors(evidence_text) | numeric_anchors(case.query)
    # Only what the MODEL wrote is scored. The citation markers around it
    # were rendered from validated ids, so counting their digits would
    # report the pipeline's own provenance as a hallucinated figure.
    stated = _CITATION_MARKUP.sub(" ", run.answer)
    alien = tuple(sorted(numeric_anchors(stated) - known))
    run.found_facts = found
    run.aligned_facts = aligned
    run.forbidden_hits = forbidden
    run.alien_numbers = alien
    return run


def _fact_is_aligned(anchor: FactAnchor, run: AnswerRun) -> bool:
    """Check that a stated fact is cited to a document that carries it.

    Provenance validation already proves a cited id belongs to this
    request's evidence. It cannot prove the cited document says what the
    claim beside it says, and a right number under the wrong citation is
    exactly the failure an employee cannot catch. This closes the gap for
    the facts the fixture names: the claim stating the fact must cite a
    document whose own text carries it.
    """
    contents = dict(run.evidence)
    for text, source_ids in run.claims:
        if not _matches(anchor, text):
            continue
        if any(
            _matches(anchor, contents.get(source_id, ""))
            for source_id in source_ids
        ):
            return True
    return False


def _execute_answer_case(
    case: AnswerCase, graph: CompiledStateGraph
) -> AnswerRun:
    """Run one case through the real pipeline and read the result off it."""
    started = time.monotonic()
    state = graph.invoke({"query": case.query})
    elapsed = time.monotonic() - started
    candidate = state.get("candidate_answer")
    return AnswerRun(
        route=str(state.get("route", "unknown")),
        answer=str(state.get("answer", "")),
        citations=tuple(state.get("valid_citations", [])),
        claims=tuple(
            (claim.text, tuple(claim.normalized_source_ids()))
            for claim in (candidate.claims if candidate else [])
        ),
        evidence=tuple(
            (document.source_id, document.content)
            for document in state.get("answer_evidence", [])
        ),
        reason=state.get("fallback_reason"),
        latency_seconds=elapsed,
    )


def _confirm_live_run(case_count: int, runs: int, assume_yes: bool) -> bool:
    """Print the cost of a live run and ask before spending it.

    Args:
        case_count: Cases about to run.
        runs: Repeats per case.
        assume_yes: Skip the prompt, for CI.

    Returns:
        Whether to proceed.
    """
    upper_bound = case_count * runs * _MAX_CALLS_PER_RUN
    print(
        f"[live] {case_count} cases x {runs} run(s) = "
        f"{case_count * runs} pipeline invocations, at most "
        f"{upper_bound} LLM calls on {config.MODEL_NAME} "
        "(cases refused by the deterministic gates cost none)"
    )
    if assume_yes:
        return True
    answer = input("[live] proceed and spend real API budget? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}


def _write_transcript(
    path: Path, results: dict[str, list[AnswerRun]]
) -> None:
    """Persist the raw runs so re-scoring never costs another live run.

    The first live run of this set found a defect in its own alien-number
    check, and fixing it meant paying for all 51 invocations a second time
    purely because the answers had not been kept. They are kept now: the
    checks in ``score_answer`` are pure functions of what is written here.

    Args:
        path: Destination JSON file.
        results: Runs per case id.
    """
    payload = {
        case_id: [
            {
                "route": run.route,
                "answer": run.answer,
                "citations": list(run.citations),
                "claims": [
                    {"text": text, "source_ids": list(ids)}
                    for text, ids in run.claims
                ],
                # Ids only. The document bodies are committed and
                # checksummed, so repeating them once per run would grow
                # this file tenfold to store what the corpus already says;
                # a re-score resolves them by id.
                "evidence": [source_id for source_id, _ in run.evidence],
                "reason": run.reason,
                "latency_seconds": round(run.latency_seconds, 3),
            }
            for run in case_runs
        ]
        for case_id, case_runs in results.items()
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[live] transcript written to {path}")


def evaluate_answers(
    runs: int, assume_yes: bool, transcript: str | None = None
) -> int:
    """Score answer quality against the corpus, using the real pipeline.

    Args:
        runs: Repeats per case, so route stability can be measured rather
            than assumed.
        assume_yes: Skip the cost confirmation.
        transcript: Optional path for the raw run dump.

    Returns:
        Strict failures: a missing required fact, an unaligned one, a
        forbidden fact, an alien number, a refusal that did not happen,
        and an unstable route each count once.

    Raises:
        SystemExit: If no credential is configured. A silent skip would
            let this set report "0 failures" while measuring nothing.
    """
    if not config.has_llm_credential():
        raise SystemExit(
            "[live] OPENAI_API_KEY is not set. This set is the only one "
            "that measures answers rather than routing, so it cannot be "
            "skipped quietly -- configure the credential or run the "
            "offline sets instead."
        )
    documents = load_documents()
    documents_by_id = {document.source_id: document for document in documents}
    cases = load_answer_cases(documents_by_id)
    if not _confirm_live_run(len(cases), runs, assume_yes):
        raise SystemExit("[live] cancelled; nothing was spent")

    graph = build_graph(
        retriever=LocalTfidfRetriever(documents),
        log_path=_EVAL_LOG_SINK,
        documents=documents,
    )
    print(
        f"== answer set: {len(cases)} cases x {runs} run(s), "
        f"model={config.MODEL_NAME} =="
    )
    results = {
        case.id: [
            score_answer(case, _execute_answer_case(case, graph))
            for _ in range(runs)
        ]
        for case in cases
    }
    if transcript is not None:
        _write_transcript(Path(transcript), results)
    return _report_answers(cases, results)


def _report_answers(
    cases: Sequence[AnswerCase],
    results: dict[str, list[AnswerRun]],
) -> int:
    """Print the per-case rows and the answer-quality metrics."""
    required_total = 0
    found_total = 0
    aligned_total = 0
    forbidden_runs = 0
    alien_runs = 0
    scored_runs = 0
    stable_cases = 0
    refusal_cases = 0
    refused_correctly = 0
    latencies: list[float] = []

    for case in cases:
        runs = results[case.id]
        routes = {run.route for run in runs}
        stable = len(routes) == 1
        stable_cases += stable
        for run in runs:
            scored_runs += 1
            latencies.append(run.latency_seconds)
            required_total += len(case.required_facts)
            found_total += len(run.found_facts)
            aligned_total += len(run.aligned_facts)
            forbidden_runs += bool(run.forbidden_hits)
            alien_runs += bool(run.alien_numbers)
        if case.expected_insufficient:
            refusal_cases += 1
            refused_correctly += all(run.route != "answered" for run in runs)
        print(
            f"  {case.id:16s} {case.category:15s} "
            f"routes={'/'.join(sorted(routes)):22s} "
            f"facts={sum(len(r.found_facts) for r in runs)}/"
            f"{len(case.required_facts) * len(runs)} "
            f"aligned={sum(len(r.aligned_facts) for r in runs)} "
            f"forbidden={sum(len(r.forbidden_hits) for r in runs)} "
            f"alien={sum(len(r.alien_numbers) for r in runs)} "
            f"{'' if stable else 'UNSTABLE ROUTE'}"
        )
        for index, run in enumerate(runs, start=1):
            reason = f" reason={run.reason}" if run.reason else ""
            print(
                f"      run {index}: route={run.route} "
                f"{run.latency_seconds:.1f}s cites={list(run.citations)}"
                f"{reason}"
            )
            for anchor in case.required_facts:
                if anchor.fact not in run.found_facts:
                    print(f"        MISSING FACT: {anchor.fact}")
                elif anchor.fact not in run.aligned_facts:
                    print(f"        UNALIGNED FACT: {anchor.fact}")
            for fact in run.forbidden_hits:
                print(f"        FORBIDDEN FACT: {fact}")
            if run.alien_numbers:
                print(f"        ALIEN NUMBERS: {list(run.alien_numbers)}")

    ordered = sorted(latencies)
    print("-- metrics --")
    print(f"  Fact Recall:             {_ratio(found_total, required_total)}")
    print(
        f"  Fact-Citation Alignment: {_ratio(aligned_total, found_total)}"
    )
    print(
        f"  Alien Number Rate:       {_ratio(alien_runs, scored_runs)}"
    )
    print(
        f"  Forbidden Fact Rate:     {_ratio(forbidden_runs, scored_runs)}"
    )
    print(
        f"  Correct Refusal Rate:    {_ratio(refused_correctly, refusal_cases)}"
    )
    print(f"  Route Stability:         {_ratio(stable_cases, len(cases))}")
    if ordered:
        print(
            f"  Latency p50/p95:         {_percentile(ordered, 0.50):.1f}s / "
            f"{_percentile(ordered, 0.95):.1f}s (n={len(ordered)})"
        )
    return (
        (required_total - found_total)
        + (found_total - aligned_total)
        + forbidden_runs
        + alien_runs
        + (refusal_cases - refused_correctly)
        + (len(cases) - stable_cases)
    )


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    """Read one percentile off an already-sorted sample, nearest-rank."""
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[index]


def _parse_ngram(raw: str) -> tuple[int, int]:
    """Parse a ``MIN,MAX`` n-gram override from the command line."""
    minimum, maximum = (int(part) for part in raw.split(","))
    return (minimum, maximum)


def _exit_code(failures: int, strict: bool) -> int:
    """Turn a failure count into a process exit code.

    Args:
        failures: Labelled failures counted by one evaluation run.
        strict: Whether the caller asked for a gate rather than a report.

    Returns:
        ``1`` when strict mode saw at least one failure, else ``0``. The
        summary line goes to stderr so a strict run stays greppable
        without disturbing the metrics on stdout.
    """
    if not strict:
        return 0
    if failures:
        print(
            f"[strict] {failures} case(s) contradicted their label",
            file=sys.stderr,
        )
        return 1
    print("[strict] every case matched its label", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch one evaluation run from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        required=True,
        choices=[*RETRIEVAL_SETS, "guardrail", "contracts", "answers"],
        dest="set_name",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="required by --set answers: this set calls a real provider",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="repeats per answer case, for route stability (default 3)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the live-run cost confirmation",
    )
    parser.add_argument(
        "--transcript",
        default=None,
        metavar="PATH",
        help="write the raw answer runs as JSON, so the checks can be "
        "re-scored later without paying for the calls again",
    )
    parser.add_argument(
        "--distribution",
        action="store_true",
        help="print per-category raw score distributions",
    )
    parser.add_argument(
        "--perturb",
        action="store_true",
        help="also print the Hit@3 curve under 1-3 injected typos "
        "(reporting only: it never changes the exit code)",
    )
    parser.add_argument(
        "--ngram",
        type=_parse_ngram,
        default=None,
        metavar="MIN,MAX",
        help="character n-gram override for ablation runs",
    )
    parser.add_argument(
        "--title-weight",
        type=int,
        default=None,
        metavar="N",
        help="repeat each document title N times in the index (ablation)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when any case contradicts its label",
    )
    args = parser.parse_args(argv)

    if args.set_name == "guardrail":
        return _exit_code(evaluate_guardrail(), args.strict)
    if args.set_name == "contracts":
        return _exit_code(evaluate_contracts(), args.strict)
    if args.set_name == "answers":
        # --live is a deliberate second keystroke, not a safety check:
        # every other set in this harness is free and offline, and a
        # reader who types the usual command should not discover the
        # difference on the invoice.
        if not args.live:
            raise SystemExit(
                "--set answers calls a real provider and costs money; "
                "pass --live to confirm you meant to"
            )
        if args.runs < 1:
            raise SystemExit("--runs must be at least 1")
        return _exit_code(
            evaluate_answers(args.runs, args.yes, args.transcript),
            args.strict,
        )

    documents = load_documents()
    documents_by_id = {document.source_id: document for document in documents}
    # Both index knobs are ablation-only overrides: the committed defaults
    # live in the retriever, and a run that changes one says so in its own
    # output, so a pasted result cannot be mistaken for the shipped index.
    index_overrides = {}
    if args.ngram is not None:
        index_overrides["ngram_range"] = args.ngram
    if args.title_weight is not None:
        index_overrides["title_weight"] = args.title_weight
    retriever = LocalTfidfRetriever(documents, **index_overrides)
    if index_overrides:
        print(
            "[ablation] "
            + " ".join(
                f"{name}={value}" for name, value in index_overrides.items()
            )
        )
    failures = evaluate_retrieval(
        args.set_name,
        retriever,
        documents_by_id,
        args.distribution,
        args.perturb,
    )
    # The contract fixtures are scored, not printed, here: a retrieval
    # gate has always failed on a broken citation or rewrite verdict, and
    # narrowing that would weaken --strict rather than tidy it.
    contract_failures = evaluate_contracts(report=False)
    print(
        f"  Contract fixtures: {contract_failures} failure(s) "
        "-- full report under --set contracts"
    )
    return _exit_code(failures + contract_failures, args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
