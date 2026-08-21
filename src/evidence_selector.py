"""Deterministic policy-first selection of the evidence a reporter may cite.

Retrieval optimises for recall and happily ranks a chat transcript first,
because informal wording matches informal wording. Answering is a
different question: a normative rule may only be stated on the authority
of a policy document, so this module separates the retrieved candidates
from the evidence that actually reaches the reporter.

The rules are metadata-driven and contain no model call: chat evidence is
resolved to the policy it points at, inactive documents are dropped, and
a request with no authoritative policy behind it is refused rather than
answered from chat alone (remediation plan Finding 4).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from src.schemas import Document, EvidenceSelection, RetrievedDocument

# Matches ``ReasonCode.NO_AUTHORITATIVE_EVIDENCE``; kept as a literal so
# this module stays a pure function over domain objects.
_NO_AUTHORITATIVE_EVIDENCE = "no_authoritative_evidence"


def select_evidence(
    candidates: Sequence[RetrievedDocument],
    documents_by_id: Mapping[str, Document],
    topics: Sequence[str] = (),
) -> EvidenceSelection:
    """Choose the answer evidence for one request from its candidates.

    Args:
        candidates: Retrieved documents for this request, best match
            first. Both strata are accepted; ranking is not authority.
        documents_by_id: The complete corpus indexed by ``source_id``,
            used to pull in a canonical policy that retrieval itself did
            not rank into the top-k.
        topics: Supported topics resolved for the query. When empty, no
            topic filter is applied, which is the behaviour before the
            scope validator runs.

    Returns:
        The selection. ``ok`` is False with reason
        ``"no_authoritative_evidence"`` when no active policy document
        covers the request, in which case ``answer_evidence`` is empty so
        no caller can accidentally answer from the rejected evidence.
    """
    wanted_topics = set(topics)
    active = [
        candidate for candidate in candidates if candidate.status == "active"
    ]
    in_scope = [
        candidate
        for candidate in active
        if not wanted_topics or wanted_topics & set(candidate.topics)
    ]

    authoritative: dict[str, RetrievedDocument] = {
        candidate.source_id: candidate
        for candidate in in_scope
        if candidate.authority == "authoritative"
    }
    chat_candidates = [
        candidate
        for candidate in in_scope
        if candidate.authority != "authoritative"
    ]
    for candidate in chat_candidates:
        for canonical_id in candidate.canonical_source_ids:
            resolved = _resolved_policy(
                canonical_id, documents_by_id, wanted_topics
            )
            if resolved is not None and resolved.source_id not in authoritative:
                authoritative[resolved.source_id] = resolved

    if not authoritative:
        return EvidenceSelection(ok=False, reason=_NO_AUTHORITATIVE_EVIDENCE)

    # Chat evidence never travels alone: it is kept only when the policy
    # that governs the same rule is in the answer evidence too, so the
    # reporter always sees the authoritative version beside it and a
    # contradicting chat line cannot be the only source of a claim.
    supplementary = [
        candidate
        for candidate in chat_candidates
        if any(
            canonical_id in authoritative
            for canonical_id in candidate.canonical_source_ids
        )
    ]

    ordered_authoritative = _ranked(authoritative.values())
    ordered_supplementary = _ranked(supplementary)
    return EvidenceSelection(
        answer_evidence=(*ordered_authoritative, *ordered_supplementary),
        authoritative_ids=tuple(
            document.source_id for document in ordered_authoritative
        ),
        supplementary_ids=tuple(
            document.source_id for document in ordered_supplementary
        ),
        ok=True,
    )


def _resolved_policy(
    canonical_id: str,
    documents_by_id: Mapping[str, Document],
    wanted_topics: set[str],
) -> RetrievedDocument | None:
    """Turn one canonical link into citable evidence, or reject it.

    Args:
        canonical_id: Policy id declared by a chat document.
        documents_by_id: The complete corpus indexed by ``source_id``.
        wanted_topics: Supported topics of the request; empty means no
            topic filter.

    Returns:
        The policy document as evidence with score ``0.0``, because it
        was pulled in by a metadata link rather than ranked by retrieval,
        or ``None`` when the target is unknown, retired, or off topic.
    """
    document = documents_by_id.get(canonical_id)
    if document is None or document.status != "active":
        return None
    if document.authority != "authoritative":
        return None
    if wanted_topics and not wanted_topics & set(document.topics):
        return None
    return RetrievedDocument(
        source_id=document.source_id,
        title=document.title,
        source_type=document.source_type,
        content=document.content,
        score=0.0,
        authority=document.authority,
        status=document.status,
        topics=document.topics,
        canonical_source_ids=document.canonical_source_ids,
    )


def _ranked(
    documents: Iterable[RetrievedDocument],
) -> tuple[RetrievedDocument, ...]:
    """Order one evidence stratum by score, tie-broken by source id."""
    return tuple(
        sorted(documents, key=lambda item: (-item.score, item.source_id))
    )
