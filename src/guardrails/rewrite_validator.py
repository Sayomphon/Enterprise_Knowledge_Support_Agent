"""Deterministic intent-preservation checks on generated rewrites.

The rewriter prompt asks the model to preserve intent; this module is
what actually enforces it. Without it a rewrite can quietly change the
question -- turning an out-of-scope topic into an in-scope one, or a
500-baht limit into 5,000 -- and the expanded retrieval score would then
look convincing for a question nobody asked.

Four independent signals decide each candidate, because no single one is
trustworthy on its own: the injection screen re-runs on model output,
numeric and time anchors may not change, the resolved topic may not grow
beyond the original question's topics, and enough of the original wording
must survive. Rejected candidates are never executed and never logged --
they are model output about a user query, and one rejection reason for
rejecting them is that they may carry injected instructions.

The anchor rule is one-sided by design: a candidate may not ADD a figure
the employee never wrote, and it may drop one. Dropping is what a
generalizing rewrite does, the original query is searched beside every
rewrite, and the reporter answers from the original -- so the result
counts the dropped anchors as a diagnostic instead of refusing them.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from src import config
from src.guardrails.input_guardrail import screen_query
from src.guardrails.normalization import fold_for_contract_checks
from src.guardrails.scope_validator import validate_scope
from src.guardrails.text_similarity import overlap
from src.schemas import RewriteValidationResult

# Matches ``ReasonCode.REWRITE_REJECTED``; kept as a literal so this
# module stays a pure function over strings.
_REWRITE_REJECTED = "rewrite_rejected"

# Bounded quantifiers only, as in the input guardrail: these patterns run
# on model output, which is untrusted and may be adversarially long.
_TIME_PATTERN = re.compile(r"[0-9]{1,2}[:.][0-9]{2}")
_NUMBER_PATTERN = re.compile(r"[0-9]{1,3}(?:,[0-9]{3}){1,4}|[0-9]{1,12}")


def validate_rewrites(
    original_query: str,
    candidates: Sequence[str],
    original_topics: Sequence[str] = (),
    continuity_threshold: float | None = None,
) -> RewriteValidationResult:
    """Select the rewrite candidates that may reach the retriever.

    Args:
        original_query: The guardrail-normalized user query.
        candidates: Raw rewrite candidates from the model, in its own
            preference order.
        original_topics: Supported topics resolved for the original
            query by the scope gate. A candidate may narrow this set but
            never introduce a topic the employee did not ask about.
        continuity_threshold: Lexical continuity override used by tests
            and the calibration sweep; defaults to
            ``config.REWRITE_CONTINUITY_THRESHOLD``.

    Returns:
        The accepted candidates in model order, the rejected ones for
        diagnostics, how many of the original query's anchors the
        accepted ones dropped, and reason ``"rewrite_rejected"`` when
        nothing survived. An empty candidate list is not a rejection: it
        means the model produced nothing, which the rewriter already
        reports.
    """
    threshold = (
        continuity_threshold
        if continuity_threshold is not None
        else config.REWRITE_CONTINUITY_THRESHOLD
    )
    original_normalized = unicodedata.normalize("NFC", original_query).strip()
    original_anchors = numeric_anchors(original_normalized)
    allowed_topics = set(original_topics)

    accepted: list[str] = []
    rejected: list[str] = []
    seen = {original_normalized}
    for candidate in candidates:
        cleaned = unicodedata.normalize("NFC", str(candidate)).strip()
        if not cleaned or cleaned in seen:
            # Blank and duplicate candidates are not intent drift; they
            # simply add nothing, because the original is always searched.
            rejected.append(cleaned)
            continue
        seen.add(cleaned)
        if _is_acceptable(
            cleaned,
            original_normalized,
            original_anchors,
            allowed_topics,
            threshold,
        ):
            accepted.append(cleaned)
        else:
            rejected.append(cleaned)

    return RewriteValidationResult(
        accepted_queries=tuple(accepted),
        rejected_queries=tuple(rejected),
        dropped_anchor_count=_dropped_anchor_count(
            original_anchors, accepted
        ),
        reason=_REWRITE_REJECTED if candidates and not accepted else None,
    )


def _dropped_anchor_count(
    original_anchors: set[str], accepted: Sequence[str]
) -> int:
    """Count the original anchors that an accepted rewrite left out.

    Anchor DELETION is tolerated on purpose while anchor INVENTION is
    rejected: the rewrites this repository ships in
    ``eval/cached_rewrites.json`` generalize on purpose -- "two days of
    annual leave, where do I click" becomes "how to file annual leave in
    the HR Portal" -- and enforcing equality would refuse them and cost
    the medium band its recovery rate. It stays measurable rather than
    merely asserted, so that a wrong answer traced to a rewrite can be
    checked against this number instead of debated.

    Args:
        original_anchors: Numeric anchors of the employee's own query.
        accepted: Candidates that passed every intent-preservation rule.

    Returns:
        How many distinct original anchors are missing from at least one
        accepted candidate. Bounded by ``len(original_anchors)``, and
        zero whenever every accepted candidate carried them all.
    """
    dropped: set[str] = set()
    for candidate in accepted:
        dropped |= original_anchors - numeric_anchors(candidate)
    return len(dropped)


def numeric_anchors(text: str) -> set[str]:
    """Extract the numbers and clock times a rewrite must not change.

    Times are collected first so that ``22:00`` is one anchor rather than
    the two numbers 22 and 00, and thousands separators are removed so
    that ``5,000`` and ``5000`` are recognised as the same value.

    Args:
        text: Query, rewrite candidate, claim, or document body.

    Returns:
        The set of normalized numeric anchors. The text is folded first,
        so a figure spelled in Thai or fullwidth numerals is read as the
        amount it shows rather than as no figure at all: an extractor
        that sees only ASCII digits would compare an empty set against
        the original and let a changed amount through unnoticed.
    """
    folded = fold_for_contract_checks(text)
    times = set(_TIME_PATTERN.findall(folded))
    remainder = _TIME_PATTERN.sub(" ", folded)
    numbers = {
        match.replace(",", "") for match in _NUMBER_PATTERN.findall(remainder)
    }
    return times | numbers


def _is_acceptable(
    candidate: str,
    original_query: str,
    original_anchors: set[str],
    allowed_topics: set[str],
    threshold: float,
) -> bool:
    """Apply every intent-preservation rule to one candidate.

    Args:
        candidate: Normalized rewrite candidate.
        original_query: Normalized original query.
        original_anchors: Numeric anchors of the original query.
        allowed_topics: Topics the original query resolved to.
        threshold: Minimum lexical continuity.

    Returns:
        True when the candidate preserved the user's intent under every
        rule; False as soon as one rule rejects it.
    """
    screened = screen_query(candidate)
    if not screened.ok:
        # Layer B of the injection defence: model output is screened with
        # the same rules as user input, so a rewrite that absorbed an
        # embedded instruction never reaches retrieval.
        return False
    if not numeric_anchors(candidate) <= original_anchors:
        # A number the employee never wrote is a changed question, not a
        # cleaned one -- 500 baht must not become 5,000.
        return False
    candidate_scope = validate_scope(candidate)
    if not candidate_scope.supported:
        # An unsupported candidate resolves to an EMPTY topic set, and the
        # subset test below would accept it vacuously. That empty set is
        # exactly what the scope gate returns for the topics the corpus has
        # no policy for, so without this check a rewrite may swap the leave
        # type -- sick leave for maternity leave -- which AGENTS.md
        # section 4, invariant 7 forbids by name.
        return False
    if not set(candidate_scope.topics) <= allowed_topics:
        return False
    return overlap(original_query, candidate) >= threshold
