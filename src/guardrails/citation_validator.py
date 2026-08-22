"""Deterministic validation of the reporter's candidate answer contract.

A pure function with no LLM and no I/O. It enforces six separate things
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
    - every claim quotes text that really occurs in one of the POLICY
      documents it cites, so the answer rests on a span of the rule
      rather than on the model's summary of it
    - every number and clock time a claim states appears inside that
      quote, so a figure cannot be borrowed from elsewhere in a long
      document, or from the employee's own question

The span rule is what separates provenance from support. The rules above
it prove that a cited id was really in this request's evidence; the span
proves that the cited document contains the words the claim rests on, so
"annual leave is 30 days" citing the sick-leave policy is refused instead
of rendered. It also closes indirect injection at the same point: a quote
found only in a chat transcript is not a policy span, so a claim built
from a poisoned transcript fails even when it co-cites a real policy id.

What it still does NOT prove is entailment: a quote can be lifted out of
its condition -- ten days of leave quoted against the wrong seniority
still passes -- so this is presence of the wording, not correctness of
the reading. The README must say so plainly (AGENTS.md section 7).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping

from src.guardrails.normalization import fold_for_contract_checks
from src.guardrails.rewrite_validator import numeric_anchors
from src.guardrails.text_similarity import normalize_for_matching
from src.schemas import (
    MIN_EVIDENCE_QUOTE_CHARS,
    AnswerClaim,
    GroundedAnswer,
    ValidationResult,
)

# Deliberately wider than the loader's source_id grammar. This pattern does
# not decide what is citable; it detects text that *poses* as a citation, so
# it must also catch the near-misses a model produces -- a different case, a
# fourth digit, round brackets, padding spaces. Matching only the exact
# loadable shape would let "[hr-001]" and "(HR-777)" through to the employee
# as unvalidated source ids. A false positive here costs an answer (the
# request degrades to fallback); a false negative costs the grounding
# guarantee, so the pattern errs wide.
#
# It is matched against folded text rather than raw claim text, which is
# what extends the same width to the bracket and dash spellings an ASCII
# pattern cannot see: "[ZZ-999]" written with fullwidth or CJK brackets
# renders identically to the employee.
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
            caller that omitted it would silently disable the span and
            numeric rules, and a guard nobody notices is worse than no
            guard.
        query: The employee's own question. It is recorded on the result
            of no rule: a figure the employee wrote is not evidence that
            the corpus states it, and accepting one let a question
            supply the number its own answer then quoted back.

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
        resting on chat alone), ``"unsupported_claim_span"`` (a quote
        absent from every policy the claim cites), or
        ``"unsupported_numeric_claim"`` (a figure absent from that
        quote).
    """
    known_ids = set(evidence_ids)
    policy_ids = set(authoritative_ids)
    # Folded once per request rather than once per claim: what a document
    # says does not depend on which claim cites it. Only the POLICY
    # bodies are folded, because only they may back a span -- a quote
    # found in a chat transcript is the shape indirect injection takes,
    # and co-citing a real policy id must not launder it.
    policy_texts = {
        source_id: normalize_for_matching(fold_for_contract_checks(content))
        for source_id, content in evidence_texts.items()
        if source_id in policy_ids
    }
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
            claim, known_ids, policy_ids, policy_texts
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
    policy_texts: Mapping[str, str],
) -> str | None:
    """Return the reason one claim fails the contract, or ``None``.

    Args:
        claim: One claim of the candidate answer.
        known_ids: Answer-evidence ids of this request.
        policy_ids: Authoritative subset of ``known_ids``.
        policy_texts: Folded body of each POLICY document, by source id.

    Returns:
        The reason code of the first violated rule, or ``None`` when the
        claim is acceptable. Provenance rules are reported before support
        rules: an id that was never in this request's evidence is the
        stronger statement about what went wrong.
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
    if CITATION_PATTERN.search(fold_for_contract_checks(claim.text)):
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
    return _unsupported_span_reason(claim, source_ids, policy_texts)


def _unsupported_span_reason(
    claim: AnswerClaim,
    source_ids: tuple[str, ...],
    policy_texts: Mapping[str, str],
) -> str | None:
    """Check one claim against the policy text it says it rests on.

    Everything above this point proves PROVENANCE: the cited id was in
    this request's evidence and at least one of them carries policy
    authority. None of it can see whether that document says what the
    claim says, which is how "annual leave is 30 days" citing the
    sick-leave policy passed. The quote is what makes that checkable
    without a model: the claim names the words it rests on, and they
    either occur in a cited policy or they do not.

    Args:
        claim: One claim that already satisfied every provenance rule.
        source_ids: The claim's normalized cited ids.
        policy_texts: Folded body of each POLICY document, by source id.

    Returns:
        ``"unsupported_claim_span"`` when the quote is missing or absent
        from every cited policy, ``"unsupported_numeric_claim"`` when the
        claim states a figure the quote does not, else ``None``.
    """
    quote = claim.evidence_quote.strip()
    if len(quote) < MIN_EVIDENCE_QUOTE_CHARS:
        # A missing quote and a two-character one fail together on
        # purpose: "30" occurs in almost any policy body, so a quote
        # short enough to match by accident proves nothing at all.
        return "unsupported_claim_span"
    folded_quote = normalize_for_matching(fold_for_contract_checks(quote))
    if not any(
        folded_quote in policy_texts.get(source_id, "")
        for source_id in source_ids
    ):
        # Searched only in the policies THIS claim cites, never pooled
        # across the request: a quote found under a document the claim
        # did not name is not evidence for the citation it did write.
        return "unsupported_claim_span"
    stated_anchors = numeric_anchors(claim.text)
    if not stated_anchors <= numeric_anchors(quote):
        # Anchored to the quote rather than to the whole document, which
        # is the difference between "this figure is somewhere in a
        # thirty-line policy" and "this figure is in the sentence the
        # claim rests on". The employee's own question is deliberately
        # not a source here either: a number they wrote is not evidence
        # that the corpus states it, and treating it as one let a
        # question supply the figure its answer then quoted back.
        return "unsupported_numeric_claim"
    return None
