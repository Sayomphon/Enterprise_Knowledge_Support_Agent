"""Deterministic validation of the reporter's candidate answer contract.

A pure function with no LLM and no I/O. It enforces four separate things
about the structure the reporter returned, and reports which one failed:

    - the structure is coherent: claims exist, carry text, and do not
      contradict the model's own "insufficient evidence" flag
    - every claim carries at least one source id, so coverage is per
      claim rather than per answer
    - every cited id belongs to the answer evidence of THIS request, so a
      chat document that retrieval surfaced but the evidence selector
      rejected can never be cited
    - every claim rests on at least one authoritative policy id, so a
      rule is never stated on the authority of a chat transcript alone

What it still does NOT prove is entailment: a claim citing a valid policy
can be a wrong reading of that policy. Claim-to-source coverage is an
auditability guarantee, not a correctness one, and the README must say so
plainly (AGENTS.md section 7).
"""

from __future__ import annotations

import re
from collections.abc import Collection

from src.schemas import AnswerClaim, GroundedAnswer, ValidationResult

# Must agree with the source_id grammar enforced by the corpus loader, so
# every loadable document is citable and every citable id is loadable.
CITATION_PATTERN = re.compile(r"\[([A-Z]{2,5}-\d{3})\]")


def validate_answer(
    candidate: GroundedAnswer,
    evidence_ids: Collection[str],
    authoritative_ids: Collection[str],
) -> ValidationResult:
    """Decide whether a candidate answer may become the public answer.

    Args:
        candidate: Unvalidated structured output from the reporter.
        evidence_ids: Source ids of the answer evidence for this request.
            Trust this set, never the model (AGENTS.md section 7, output
            validation).
        authoritative_ids: The subset of ``evidence_ids`` carrying policy
            authority, as decided by the evidence selector.

    Returns:
        ``ok=True`` with the sorted, deduplicated ids cited across all
        claims when every rule holds. Otherwise ``ok=False`` with the
        reason of the first rule that failed:
        ``"insufficient_reporter_evidence"`` (the model declared the
        evidence insufficient), ``"invalid_answer_structure"`` (no
        claims, blank claim, self-contradicting flag, repeated id, or
        model-written citation markup), ``"missing_citation"`` (a claim
        with no source), ``"fabricated_citation"`` (an id outside this
        request's evidence), or ``"no_authoritative_evidence"`` (a claim
        resting on chat alone).
    """
    known_ids = set(evidence_ids)
    policy_ids = set(authoritative_ids)
    if candidate.insufficient_evidence:
        # A model that both reports insufficient evidence and makes
        # claims has contradicted itself; neither half can be trusted.
        if candidate.claims:
            return ValidationResult(
                ok=False, reason="invalid_answer_structure"
            )
        return ValidationResult(
            ok=False, reason="insufficient_reporter_evidence"
        )
    if not candidate.claims:
        return ValidationResult(ok=False, reason="invalid_answer_structure")

    cited: set[str] = set()
    for claim in candidate.claims:
        reason = _claim_rejection_reason(claim, known_ids, policy_ids)
        if reason is not None:
            # One unsupported claim invalidates the whole answer: the
            # employee would read the rendered lines as one statement,
            # so partial validity cannot be shown as a valid answer.
            return ValidationResult(ok=False, reason=reason)
        cited.update(claim.normalized_source_ids())
    return ValidationResult(ok=True, citations=tuple(sorted(cited)))


def _claim_rejection_reason(
    claim: AnswerClaim,
    known_ids: set[str],
    policy_ids: set[str],
) -> str | None:
    """Return the reason one claim fails the contract, or ``None``.

    Args:
        claim: One claim of the candidate answer.
        known_ids: Answer-evidence ids of this request.
        policy_ids: Authoritative subset of ``known_ids``.

    Returns:
        The reason code of the first violated rule, or ``None`` when the
        claim is acceptable.
    """
    if not claim.text.strip():
        return "invalid_answer_structure"
    if CITATION_PATTERN.search(claim.text):
        # Citations are rendered from validated ids, so markup inside the
        # claim text is either a duplicate or an unvalidated id posing as
        # one. Both are structure defects, not citations.
        return "invalid_answer_structure"
    source_ids = claim.normalized_source_ids()
    if not source_ids:
        return "missing_citation"
    if "" in source_ids:
        return "invalid_answer_structure"
    if len(set(source_ids)) != len(source_ids):
        # A repeated id would render as "[FIN-001] [FIN-001]" and hints
        # the structure was assembled rather than reasoned about.
        return "invalid_answer_structure"
    if not set(source_ids) <= known_ids:
        return "fabricated_citation"
    if not set(source_ids) & policy_ids:
        return "no_authoritative_evidence"
    return None
