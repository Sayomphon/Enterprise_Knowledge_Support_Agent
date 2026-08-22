"""Immutable domain objects and the LangGraph pipeline state contract.

This module sits at the bottom of the dependency graph and imports nothing
from the rest of ``src`` (AGENTS.md section 3). Domain objects are frozen
dataclasses so every state transition in the pipeline is auditable; the
pipeline state is a ``TypedDict`` whose optional fields mirror the routes
that actually populate them. The two Pydantic models carry structured LLM
output into the pipeline, which is parsed at the provider boundary rather
than by hand (AGENTS.md section 6.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

SourceType = Literal["policy", "chat"]

# Evidence authority is metadata, never a model decision: policy documents
# state the rule, chat transcripts only illustrate it, so a normative
# answer may never rest on chat alone (remediation plan Finding 4).
Authority = Literal["authoritative", "supplementary"]

# Lifecycle flag; only "active" documents may support an answer.
DocumentStatus = Literal["active", "inactive"]

# Bounded topic catalog of the current corpus. It is intentionally closed:
# a question outside these topics has no policy behind it, so the pipeline
# must fall back instead of answering from lexically similar text. This is
# a prototype scope list, not a production intent taxonomy.
KnowledgeTopic = Literal[
    "reimbursement_process",
    "receipt_policy",
    "annual_leave",
    "sick_leave",
    "work_from_home",
]

KNOWLEDGE_TOPICS: tuple[KnowledgeTopic, ...] = get_args(KnowledgeTopic)

# Which of the searched queries a document matched best. Expanded
# retrieval max-pools over the original query plus its variants, so this
# records whether a high score came from what the employee actually wrote
# or from a generated one (remediation plan Finding 3). The two kinds of
# variant are kept apart because their trust differs: an "alias" variant
# is built from this repository's own topic catalog, while a "rewrite" is
# model output that had to pass the rewrite validator first.
MatchedQueryType = Literal["original", "alias", "rewrite"]

# Terminal or intermediate branch of the pipeline graph (AGENTS.md section 16).
Route = Literal[
    "blocked",
    "direct_answer",
    "rewrite",
    "fallback",
    "answered",
]


@dataclass(frozen=True)
class Document:
    """One corpus document loaded from Markdown with YAML frontmatter.

    Attributes:
        source_id: Stable citation identifier, for example ``"HR-001"``.
            This exact string appears in ``[SOURCE-ID]`` answer citations.
        title: Human-readable document title shown beside citations.
        source_type: Corpus stratum: ``"policy"`` is authoritative text,
            ``"chat"`` is noisy conversational evidence.
        content: Markdown body with the frontmatter block removed.
        authority: Evidence weight of the document. Bound to
            ``source_type`` by the loader so the two can never disagree.
        status: Lifecycle flag; retired documents stay in the corpus for
            provenance but never support an answer.
        topics: Knowledge topics the document covers, drawn from
            ``KNOWLEDGE_TOPICS``.
        canonical_source_ids: For chat documents, the policy documents
            that carry the authoritative version of the same rule. Empty
            for policy documents.
    """

    source_id: str
    title: str
    source_type: SourceType
    content: str
    authority: Authority
    status: DocumentStatus
    topics: tuple[KnowledgeTopic, ...]
    canonical_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievedDocument:
    """A corpus document paired with its retrieval score for one query.

    Attributes:
        source_id: Citation identifier of the underlying document.
        title: Title of the underlying document.
        source_type: Stratum of the underlying document.
        content: Markdown body of the underlying document.
        score: Cosine similarity between query and document. This is a
            similarity heuristic, never a probability that the answer is
            correct (AGENTS.md section 4, invariant 6). A document pulled
            in through a canonical link rather than ranked by retrieval
            carries ``0.0``; ``EvidenceSelection`` records why it is there.
        authority: Evidence weight carried over from the document.
        status: Lifecycle flag carried over from the document.
        topics: Knowledge topics carried over from the document.
        canonical_source_ids: Canonical policy ids carried over from the
            document, used to resolve chat evidence to its policy.
        matched_query: The query that produced ``score``, or ``None`` for
            a document resolved through a canonical link.
        matched_query_type: Whether ``matched_query`` was the employee's
            own query or a generated rewrite. ``None`` alongside a
            ``None`` ``matched_query``.
    """

    source_id: str
    title: str
    source_type: SourceType
    content: str
    score: float
    authority: Authority
    status: DocumentStatus
    topics: tuple[KnowledgeTopic, ...]
    canonical_source_ids: tuple[str, ...] = ()
    matched_query: str | None = None
    matched_query_type: MatchedQueryType | None = None


@dataclass(frozen=True)
class ScopeDecision:
    """Outcome of deterministic supported-scope validation for one query.

    Attributes:
        supported: True when the query maps to a supported knowledge
            topic and may continue towards an answer.
        topics: Supported topics the query resolved to, sorted for
            reproducible logging and evidence matching.
        score: Best alias similarity behind the decision, a lexical
            heuristic on the same [0, 1] scale as retrieval scores.
        reason: Rejection reason code when not ``supported``, else
            ``None``.
    """

    supported: bool
    topics: tuple[KnowledgeTopic, ...] = ()
    score: float = 0.0
    reason: str | None = None


@dataclass(frozen=True)
class EvidenceSelection:
    """Outcome of deterministic policy-first evidence selection.

    Attributes:
        answer_evidence: Documents the reporter may cite, authoritative
            policy first, supplementary chat last.
        authoritative_ids: Ids of the policy documents in
            ``answer_evidence``; a normative answer needs at least one.
        supplementary_ids: Ids of the chat documents in
            ``answer_evidence``, kept for context only.
        ok: True when at least one active, topic-matching policy document
            backs the request.
        reason: Rejection reason code when not ``ok``, else ``None``.
    """

    answer_evidence: tuple[RetrievedDocument, ...] = ()
    authoritative_ids: tuple[str, ...] = ()
    supplementary_ids: tuple[str, ...] = ()
    ok: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class LogWriteResult:
    """Outcome of one append to the JSONL telemetry sink.

    The writer stays best-effort -- a failed append must never break a
    user request -- but the caller now learns whether the append actually
    happened, so the response text can stop claiming that a question was
    recorded when it was not (remediation plan Finding 7).

    Attributes:
        ok: True when the record reached the sink.
        error_type: Exception class name of a caught filesystem error,
            else ``None``. Only the type is carried: an ``OSError``
            message names the path it failed on, and a log destination is
            deployment detail that must not reach a user or a caller.
    """

    ok: bool
    error_type: str | None = None


@dataclass(frozen=True)
class LogReadResult:
    """Outcome of one bounded read of the JSONL telemetry sink.

    Attributes:
        records: Decoded JSONL objects, oldest first, capped at the
            requested limit.
        skipped_lines: Count of lines that did not decode, including a
            partial final line left by an interrupted append. The count
            is reportable; the malformed text itself never is.
    """

    records: tuple[dict[str, object], ...] = ()
    skipped_lines: int = 0


@dataclass(frozen=True)
class RewriteValidationResult:
    """Outcome of deterministic intent-preservation checks on rewrites.

    Attributes:
        accepted_queries: Candidates that may be sent to the retriever,
            in the model's preference order.
        rejected_queries: Candidates withheld from retrieval. They are
            diagnostics only and must never be logged or executed.
        reason: Reason code when every candidate was rejected, else
            ``None``.
    """

    accepted_queries: tuple[str, ...] = ()
    rejected_queries: tuple[str, ...] = ()
    reason: str | None = None


class AnswerClaim(BaseModel):
    """One factual statement of a candidate answer with its evidence.

    Attributes:
        text: The claim in the language of the employee's question,
            written without any ``[SOURCE-ID]`` markup: citations are
            rendered deterministically from ``source_ids`` after
            validation, so malformed model-written brackets cannot reach
            the employee.
        source_ids: Ids of the answer-evidence documents that support
            this exact claim. A factual claim with an empty list is a
            contract violation, not an uncited sentence to be tolerated.
    """

    text: str
    source_ids: list[str] = Field(default_factory=list)

    def normalized_source_ids(self) -> tuple[str, ...]:
        """Return the cited ids stripped, in the model's own order.

        The validator and the renderer must agree byte for byte on what
        the model cited, so the single normalization rule lives with the
        contract instead of being repeated at both ends.

        Returns:
            The stripped ids. Duplicates and blanks are preserved: they
            are contract violations for the validator to report, not
            noise for this method to hide.
        """
        return tuple(source_id.strip() for source_id in self.source_ids)


# Upper bound on the claims one answer may carry. It is a contract with
# the provider rather than a calibrated threshold: the bound travels into
# the JSON schema sent with the request, so the model is told the limit
# instead of being trimmed afterwards. Six is what the demo answers
# needed -- a nine-claim answer took 29.6 seconds to generate and read
# like a policy dump rather than a reply -- and an answer that genuinely
# needs more is a sign the question should be split.
MAX_ANSWER_CLAIMS = 6


class GroundedAnswer(BaseModel):
    """Structured reporter output, before any validation has run.

    This is the *candidate* answer: it lives in ``candidate_answer`` and
    may never be shown to anyone. Only the validator may promote it into
    the public ``answer`` field (remediation plan Finding 5).

    Attributes:
        claims: The factual statements the model wants to make, in
            reading order, at most ``MAX_ANSWER_CLAIMS`` of them. Empty
            when the model reports insufficient evidence. A provider that
            returns more than the cap fails to parse, which the reporter
            boundary degrades to a fallback -- the alternative, silently
            keeping the first six, would render a truncated answer as a
            complete one.
        insufficient_evidence: Set by the model when the supplied
            evidence cannot answer the question. It must then make no
            claims at all; the request degrades to the fallback response
            instead of to a hedged half-answer.
    """

    claims: list[AnswerClaim] = Field(
        default_factory=list, max_length=MAX_ANSWER_CLAIMS
    )
    insufficient_evidence: bool = False


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of deterministic validation of a candidate answer.

    Attributes:
        ok: True when the candidate satisfies the answer contract: every
            claim is non-empty, carries at least one source id, cites
            only this request's answer evidence with at least one policy
            document behind it, and states no figure that is absent from
            both its own citations and the employee's question.
        citations: Sorted, deduplicated ids cited across all claims when
            ``ok``.
        reason: Rejection reason code when not ``ok``, else ``None``.
            One of ``"missing_citation"``, ``"fabricated_citation"``,
            ``"invalid_answer_structure"``, ``"insufficient_reporter_evidence"``,
            ``"no_authoritative_evidence"`` or
            ``"unsupported_numeric_claim"``.
    """

    ok: bool
    citations: tuple[str, ...] = ()
    reason: str | None = None


class PipelineState(TypedDict):
    """Shared state flowing through the LangGraph pipeline.

    Only ``query`` exists on every route. Every other field is
    ``NotRequired`` because each route populates a different subset: a
    blocked request never carries retrieval results, and a fallback never
    carries an answer. Keeping the contract partial matches the runtime
    reality instead of forcing placeholder values into unrelated routes.
    """

    query: str

    # Monotonic timestamp stamped by the first node, from which every
    # boundary derives what is left of REQUEST_DEADLINE_SECONDS. A
    # monotonic reading rather than a wall clock: the budget is an
    # elapsed duration, and the system clock may step during a request.
    request_started_at: NotRequired[float]
    # How many provider calls this request actually spent. Telemetry
    # only -- no route reads it -- so an operator can tell a two-call
    # medium-band request from one the alias catalog settled.
    llm_calls: NotRequired[int]

    # Routing outcome and guardrail verdict.
    route: NotRequired[Route]
    guardrail_reason: NotRequired[str]

    # Retrieval results shared by the answer-bearing routes.
    retrieved_candidates: NotRequired[list[RetrievedDocument]]
    raw_retrieval_score: NotRequired[float]

    # Deterministic supported-scope verdict for the original query. It is
    # decided once, before the rewrite branch: the rewrite validator
    # already refuses any candidate that leaves these topics, so a
    # re-decision after expansion could only agree with this one.
    scope_topics: NotRequired[list[str]]
    scope_score: NotRequired[float]
    scope_reason: NotRequired[str]

    # Deterministic search variants built from the resolved topics'
    # aliases, before any model is asked for one. Populated on the medium
    # band only, and empty when no alias matched. They are configuration
    # data rather than model output, so they do not pass through the
    # rewrite validator -- which is also why they are kept in their own
    # field instead of being mixed into ``rewritten_queries``.
    alias_expansion_queries: NotRequired[list[str]]

    # Populated only on the medium-band rewrite path.
    rewritten_queries: NotRequired[list[str]]
    # Why the rewriter produced nothing, or absent when it succeeded. It
    # is a reason code rather than a boolean because the fallback node
    # writes it straight into the log: a provider outage and a missing
    # credential are different operator facts, and neither is "the corpus
    # lacked an answer".
    rewrite_failure_reason: NotRequired[str]
    rewrite_rejected: NotRequired[bool]
    expanded_retrieval_score: NotRequired[float]

    # Policy-first evidence actually offered to the reporter.
    answer_evidence: NotRequired[list[RetrievedDocument]]
    authoritative_source_ids: NotRequired[list[str]]

    # Unvalidated Reporter output. It is deliberately separate from
    # ``answer``: nothing outside the validation node may read it, and no
    # presentation layer may render it (remediation plan Finding 5). The
    # fallback node sets it back to ``None``, so a rejected draft cannot
    # leave the graph inside the state returned to the caller.
    candidate_answer: NotRequired[GroundedAnswer | None]

    # Populated only when the Reporter produced a validated answer.
    answer: NotRequired[str]
    valid_citations: NotRequired[list[str]]

    # Populated only when the request degraded to the fallback response.
    fallback_reason: NotRequired[str]

    # Whether this request's blocked/fallback event actually reached the
    # JSONL sink. It is telemetry health, never an answer failure reason:
    # evidence routing can be perfectly correct while the disk is full.
    # The response selector reads it so the user-facing text cannot claim
    # a question was recorded when the append failed (Finding 7).
    telemetry_logged: NotRequired[bool]
