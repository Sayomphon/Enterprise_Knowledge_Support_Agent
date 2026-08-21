"""Offline evaluation harness for retrieval routing, guardrail, citations.

Usage:
    python eval/run_eval.py --set calibration
    python eval/run_eval.py --set heldout
    python eval/run_eval.py --set guardrail
    python eval/run_eval.py --set calibration --distribution
    python eval/run_eval.py --set calibration --ngram 2,4
    python eval/run_eval.py --set guardrail --strict

Without ``--strict`` this is a reporting command: it prints every metric
and exits 0 even when a case contradicts its label, which is what a
threshold sweep needs. ``--strict`` turns the same run into a gate that
exits 1 on any labelled failure, so review and CI can depend on it.

No LLM is called anywhere in this harness. Medium-band expansion uses
``eval/cached_rewrites.json``; a query absent from the cache is evaluated
as a failed rewrite (original-query-only expansion), mirroring the
runtime's graceful degradation, and is reported as a cache miss.

Metric definitions:
    Retrieval Hit@3          -- answerable cases whose final retrieval
                                contains at least one expected source.
    Answer-route Precision   -- of cases predicted "answered", the share
                                that are labelled answerable AND hit an
                                expected source (answering from wrong
                                documents counts as a miss).
    Answer-route Coverage    -- labelled-answerable cases actually routed
                                to "answered". Always reported beside
                                Precision: a pipeline that answers almost
                                nothing scores perfect precision.
    OOD Fallback Accuracy    -- labelled-fallback cases predicted fallback.
    Unsupported In-domain Fallback Accuracy
                             -- labelled-fallback cases in the
                                "unsupported" category, that is HR/Finance
                                questions the corpus has no policy for.
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
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
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
from src.guardrails.rewrite_validator import validate_rewrites  # noqa: E402
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

# The two buckets of the guardrail fixture, and the balanced minimum
# AGENTS.md section 10 requires of it. A rate computed over fewer cases
# than this is not evidence, so the harness refuses to report one.
_GUARDRAIL_CASE_TYPES = ("attack", "benign")
_MIN_GUARDRAIL_CASES_PER_TYPE = 12

RETRIEVAL_SETS = {
    "calibration": "retrieval_calibration.json",
    "heldout": "retrieval_heldout.json",
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

    def rewrite(query: str) -> tuple[list[str], str | None]:
        candidates = rewrite_cache.get(query)
        if not candidates:
            return [], ReasonCode.REWRITE_FAILURE.value
        return list(candidates), None

    return rewrite


def _stub_reporter(
    query: str, retrieved: Sequence[RetrievedDocument]
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

    Returns:
        A candidate answer citing the authoritative evidence.
    """
    policy_ids = [
        document.source_id
        for document in retrieved
        if document.authority == "authoritative"
    ]
    return GroundedAnswer(
        claims=[AnswerClaim(text=EVAL_CLAIM_TEXT, source_ids=policy_ids)]
    )


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


def evaluate_retrieval(
    set_name: str,
    retriever: LocalTfidfRetriever,
    documents_by_id: dict[str, Document],
    show_distribution: bool,
) -> int:
    """Run one retrieval set and print per-case rows plus the metrics.

    Returns:
        The number of strict failures across this set and the two
        validator sets it also scores: every case whose route
        contradicted its label, answerable cases whose retrieval missed
        each expected source, and the citation and rewrite verdict
        mismatches.

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
    ood_correct = sum(
        predictions[c.id].route == "fallback" for c in fallback_labelled
    )
    unsupported = [c for c in fallback_labelled if c.category == "unsupported"]
    unsupported_correct = sum(
        predictions[c.id].route == "fallback" for c in unsupported
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
    print(
        "  Answer-route Precision: "
        f"{_ratio(precise, len(predicted_answered))}"
    )
    print(
        "  Answer-route Coverage:  "
        f"{_ratio(covered, len(answerable))}"
    )
    print(
        "  OOD Fallback Accuracy:  "
        f"{_ratio(ood_correct, len(fallback_labelled))}"
    )
    print(
        "  Unsupported In-domain Fallback Accuracy: "
        f"{_ratio(unsupported_correct, len(unsupported))}"
    )
    print(
        "  Authoritative Evidence Coverage Rate: "
        f"{_ratio(answered_with_policy, len(predicted_answered))}"
    )
    print(
        "  Rewrite Recovery Rate:  "
        f"{_ratio(recovered, len(medium_answerable))}"
    )
    citation_failures = evaluate_citations()
    rewrite_failures = evaluate_rewrite_pairs()

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
    return (
        route_failures
        + source_failures
        + citation_failures
        + rewrite_failures
    )


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


def evaluate_citations() -> int:
    """Score the answer-contract validator against its labelled cases.

    Returns:
        The number of strict failures: labelled verdicts the validator
        disagreed with, plus any rejected candidate that still rendered
        public text. Claim Source Coverage is reported but never counted,
        because the labelled set deliberately mixes grounded and
        ungrounded claims -- it describes the fixture, not a defect.
    """
    raw_cases = _load_json("citation_cases.json")
    correct = 0
    grounded_claims = 0
    total_claims = 0
    rejected = 0
    leaked = 0
    for case in raw_cases:
        candidate = _candidate_answer(case)
        evidence_ids = set(case["evidence_ids"])
        result = validate_answer(
            candidate, evidence_ids, set(case["authoritative_ids"])
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
        rewriter=lambda query: ([], True),
        reporter=lambda query, retrieved: candidate,
    )
    state = graph.invoke({"query": CITATION_PROBE_QUERY})
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
        chat authority otherwise. Every document covers the probe topic,
        so the scope and evidence gates admit them and the run reaches
        the validation node under test.
    """
    authoritative = set(case["authoritative_ids"])
    return [
        Document(
            source_id=source_id,
            title=source_id,
            source_type="policy" if source_id in authoritative else "chat",
            content=EVAL_CLAIM_TEXT,
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
            )
            for claim in case["claims"]
        ],
        insufficient_evidence=bool(case.get("insufficient_evidence", False)),
    )


def evaluate_rewrite_pairs() -> int:
    """Score the rewrite validator against its labelled pairs.

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
    print(
        "  Rewrite Intent Preservation Rate:   "
        f"{_ratio(correct, len(raw_cases))}"
    )
    return len(raw_cases) - correct


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
        choices=[*RETRIEVAL_SETS, "guardrail"],
        dest="set_name",
    )
    parser.add_argument(
        "--distribution",
        action="store_true",
        help="print per-category raw score distributions",
    )
    parser.add_argument(
        "--ngram",
        type=_parse_ngram,
        default=None,
        metavar="MIN,MAX",
        help="character n-gram override for ablation runs",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when any case contradicts its label",
    )
    args = parser.parse_args(argv)

    if args.set_name == "guardrail":
        return _exit_code(evaluate_guardrail(), args.strict)

    documents = load_documents()
    documents_by_id = {document.source_id: document for document in documents}
    if args.ngram is None:
        retriever = LocalTfidfRetriever(documents)
    else:
        retriever = LocalTfidfRetriever(documents, ngram_range=args.ngram)
        print(f"[ablation] ngram_range={args.ngram}")
    failures = evaluate_retrieval(
        args.set_name, retriever, documents_by_id, args.distribution
    )
    return _exit_code(failures, args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
