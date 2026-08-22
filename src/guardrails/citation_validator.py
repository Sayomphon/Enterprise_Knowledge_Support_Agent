"""Deterministic validation of the reporter's candidate answer contract.

A pure function with no LLM and no I/O. It enforces five separate things
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
    - every number and clock time a claim states appears in a document
      that claim cites, or in the employee's own question

What it still does NOT prove is entailment: a claim citing a valid policy
can be a wrong reading of that policy, and the numeric rule proves only
that a stated figure exists in the cited text, never that it was applied
to the right condition. Claim-to-source coverage is an auditability
guarantee, not a correctness one, and the README must say so plainly
(AGENTS.md section 7).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping

from src.guardrails.rewrite_validator import numeric_anchors
from src.schemas import AnswerClaim, GroundedAnswer, ValidationResult

# Deliberately wider than the loader's source_id grammar. This pattern does
# not decide what is citable; it detects text that *poses* as a citation, so
# it must also catch the near-misses a model produces -- a different case, a
# fourth digit, round brackets, padding spaces. Matching only the exact
# loadable shape would let "[hr-001]" and "(HR-777)" through to the employee
# as unvalidated source ids. A false positive here costs an answer (the
# request degrades to fallback); a false negative costs the grounding
# guarantee, so the pattern errs wide.
CITATION_PATTERN = re.compile(r"[\[(]\s*[A-Za-z]{2,5}\s*-\s*\d{1,5}\s*[\])]")


def validate_answer(
    candidate: GroundedAnswer,
    evidence_ids: Collection[str],
    authoritative_ids: Collection[str],
    *,
    evidence_texts: Mapping[str, str],
    query: str = "",
) -> ValidationResult:
    """Decide whether a candidate answer may become the public answer.

    Args:
        candidate: Unvalidated structured output from the reporter.
        evidence_ids: Source ids of the answer evidence for this request.
            Trust this set, never the model (AGENTS.md section 7, output
            validation).
        authoritative_ids: The subset of ``evidence_ids`` carrying policy
            authority, as decided by the evidence selector.
        evidence_texts: Body text of each answer-evidence document, by
            source id. It is keyword-only and has no default because a
            caller that omitted it would silently disable the numeric
            rule, and a guard nobody notices is worse than no guard.
        query: The employee's own question, whose numbers a claim may
            legitimately repeat back.

    Returns:
        ``ok=True`` with the sorted, deduplicated ids cited across all
        claims when every rule holds. Otherwise ``ok=False`` with the
        reason of the first rule that failed:
        ``"insufficient_reporter_evidence"`` (the model declared the
        evidence insufficient), ``"invalid_answer_structure"`` (no
        claims, blank claim, a line break inside a claim,
        self-contradicting flag, repeated id, or model-written citation
        markup), ``"missing_citation"`` (a claim
        with no source), ``"fabricated_citation"`` (an id outside this
        request's evidence), ``"no_authoritative_evidence"`` (a claim
        resting on chat alone), or ``"unsupported_numeric_claim"`` (a
        figure that appears in neither the cited documents nor the
        question).
    """
    known_ids = set(evidence_ids)
    policy_ids = set(authoritative_ids)
    # Computed once per request rather than once per claim: the anchors
    # of a document do not depend on which claim cites it.
    anchors_by_id = {
        source_id: numeric_anchors(content)
        for source_id, content in evidence_texts.items()
    }
    query_anchors = numeric_anchors(query)
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
        reason = _claim_rejection_reason(
            claim, known_ids, policy_ids, anchors_by_id, query_anchors
        )
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
    anchors_by_id: Mapping[str, set[str]],
    query_anchors: set[str],
) -> str | None:
    """Return the reason one claim fails the contract, or ``None``.

    Args:
        claim: One claim of the candidate answer.
        known_ids: Answer-evidence ids of this request.
        policy_ids: Authoritative subset of ``known_ids``.
        anchors_by_id: Numeric anchors of each evidence document.
        query_anchors: Numeric anchors of the employee's question.

    Returns:
        The reason code of the first violated rule, or ``None`` when the
        claim is acceptable.
    """
    if not claim.text.strip():
        return "invalid_answer_structure"
    if "\n" in claim.text or "\r" in claim.text:
        # The renderer emits one line per claim and appends that claim's
        # citations at the end of it, so a claim carrying its own line
        # break would render leading lines that state a rule and carry no
        # citation at all -- uncited output in the public answer, which
        # AGENTS.md section 4 forbids.
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
    stated_anchors = numeric_anchors(claim.text)
    if stated_anchors:
        # The expensive hallucination in an HR/Finance answer is not an
        # invented source -- provenance already catches that -- but a
        # right-looking citation under a wrong figure: "ลาได้ 15 วัน"
        # citing the policy that says 10. Every number and clock time in
        # the claim must therefore appear in a document this claim cites,
        # or in what the employee asked, which is what keeps "ลา 2 วันได้
        # ไหม" answerable without the answer repeating a figure no
        # document holds. The rule is deliberately per claim, not per
        # answer: pooling the anchors of every cited document would let a
        # figure borrowed from an unrelated claim's source pass.
        #
        # It proves presence, not correct application -- 10 days of leave
        # quoted against the wrong seniority still passes -- and it will
        # reject an arithmetic result the corpus does not state
        # literally. Both are deliberate: this set is the last rule, so
        # everything it rejects has already satisfied provenance, and a
        # fallback is cheaper than a confidently wrong number.
        supported = set(query_anchors)
        for source_id in source_ids:
            supported |= anchors_by_id.get(source_id, set())
        if not stated_anchors <= supported:
            return "unsupported_numeric_claim"
    return None
