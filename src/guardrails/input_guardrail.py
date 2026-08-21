"""Deterministic pre-retrieval screen for incoming queries.

Runs before any retrieval or LLM call and uses no model: type, emptiness,
and length checks first, then Unicode normalization, then a catalog of
named high-precision prompt-injection rules in English and Thai.

Two representations of the query exist here on purpose. The pipeline
receives ``normalized_query``, NFC-normalized user text that retrieval
and the rewriter can work with. The rules match the hardened foldings
from ``guardrail_match_variants`` instead: NFKC, case folding, separator
punctuation turned into spaces, and zero-width characters handled BOTH
ways -- deleted in one folding and spaced in the other -- because an
attacker chooses where to hide them and neither treatment covers both
placements. A rule fires if either folding hits, so ``Ig<ZWSP>nore``,
``ignore<ZWSP>previous`` and ``ignore.previous.instructions`` all reach
the pattern that the plain wording would hit. The hardened forms are
never returned to the pipeline, because NFKC rewrites compatibility
characters that Thai corpus text may rely on.

Precision is deliberately favoured over recall: blocking a benign
enterprise question costs more here than letting a novel attack phrasing
through, because downstream layers (evidence encoding, claim and citation
validation) still contain what the guardrail misses. Regex screening is a
prototype safeguard, not defence-in-depth.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from src import config

# "prompt_injection" matches the JSONL reason-code enum in AGENTS.md
# section 8; the three validation codes extend that enum for requests
# rejected before retrieval can even start.
GuardrailReason = Literal[
    "invalid_query_type",
    "empty_query",
    "query_too_long",
    "prompt_injection",
]

# Invisible format characters an attacker can drop into a query to break a
# pattern without changing what the employee's screen shows. They admit two
# placements and no single folding covers both: DELETING one rejoins
# "ig<ZWSP>nore" but glues "ignore<ZWSP>previous" into a single token that
# no longer offers the whitespace gap every English rule requires, while
# folding it to a space does exactly the reverse. Both foldings are built,
# and a rule matches if either one hits.
_FORMAT_CHARACTERS = "​‌‍⁠﻿­"

# Punctuation an attacker can use as a word separator. Folding it to a
# space costs nothing in this domain -- "e-receipt" simply becomes
# "e receipt" for matching -- and closes "ignore.previous.instructions".
_SEPARATOR_CHARACTERS = "._*/\\|~+,;:‐‑‒–—―-"

# NFKC folds fullwidth and other compatibility spellings into plain
# ASCII, but it also splits Thai SARA AM into nikhahit + sara aa, and NFC
# does not put it back because that character is a composition exclusion.
# Recomposing it keeps every Thai rule written in the ordinary spelling
# while still gaining the compatibility folding, and it canonicalizes a
# query that arrives already decomposed.
_THAI_SARA_AM = "\u0e33"
_THAI_DECOMPOSED_SARA_AM = "\u0e4d\u0e32"

_MATCH_TRANSLATION = {
    **{ord(character): None for character in _FORMAT_CHARACTERS},
    **{ord(character): " " for character in _SEPARATOR_CHARACTERS},
}
# The same table with format characters folded to a space instead. Neither
# folding covers both placements on its own, so rules are matched against
# both (see ``guardrail_match_variants``).
_SPACED_MATCH_TRANSLATION = {
    ord(character): " "
    for character in _FORMAT_CHARACTERS + _SEPARATOR_CHARACTERS
}


@dataclass(frozen=True)
class InjectionRule:
    """One named injection pattern in the guardrail catalog.

    Attributes:
        rule_id: Stable identifier of the attack shape. Every rule is
            paired in tests/test_guardrail.py with an attack case and a
            benign lookalike that shares its vocabulary, so widening a
            rule cannot silently start refusing enterprise questions.
        pattern: Compiled pattern, matched against the hardened text
            produced by ``guardrail_match_text``.
    """

    rule_id: str
    pattern: re.Pattern[str]


# Bounded quantifiers only -- unbounded repetition or nested groups would
# expose the guardrail to catastrophic backtracking (ReDoS). Patterns are
# written in lower case because the match text is already case-folded;
# IGNORECASE is kept so a future rule written with capitals still works.
INJECTION_RULES: tuple[InjectionRule, ...] = (
    InjectionRule(
        # English "ignore/disregard/forget (all|the|your) previous
        # instructions", with the synonyms an attacker reaches for once
        # "instructions" is blocked. The determiner slot repeats up to
        # twice, bounded, so "all of the previous instructions" is covered
        # without the unbounded repetition this module forbids: the
        # article is the single most common word in this attack and its
        # absence made the flagship rule miss the canonical phrasing.
        # "policy" is deliberately absent: it is the domain noun of this
        # corpus, so blocking it would refuse real questions.
        rule_id="english_ignore_previous_instructions",
        pattern=re.compile(
            r"(?:ignore|disregard|forget|skip)\s{1,5}"
            r"(?:(?:all|any|the|your|these|those)(?:\s{1,5}of)?\s{1,5}){0,2}"
            r"(?:previous|prior|earlier|above|system|these|your)\s{1,5}"
            r"(?:instructions?|rules?|directions?|prompts?|guidelines?)",
            re.IGNORECASE,
        ),
    ),
    InjectionRule(
        # English "reveal/show/print (the|your) system prompt", extended
        # to the developer-message wording of the same request.
        rule_id="english_reveal_system_prompt",
        pattern=re.compile(
            r"(?:reveal|show|print|display|expose|disclose|repeat"
            r"|tell|output|give|state|share)\s{1,5}"
            r"(?:me\s{1,5})?(?:your\s{1,5}|the\s{1,5})?"
            r"(?:system|developer)\s{1,5}"
            r"(?:prompts?|messages?|instructions?)",
            re.IGNORECASE,
        ),
    ),
    InjectionRule(
        # English "reveal the hidden/original prompt". The noun is
        # restricted to "prompt": an HR or Finance question never asks
        # for one, while "the original instructions" is ordinary
        # expense-workflow wording and must keep passing.
        rule_id="english_reveal_hidden_prompt",
        pattern=re.compile(
            r"(?:reveal|show|print|display|expose|disclose|repeat"
            r"|tell|output|give|state|share)\s{1,5}"
            r"(?:me\s{1,5})?(?:your\s{1,5}|the\s{1,5})?"
            r"(?:hidden|initial|original|internal|secret)\s{1,5}prompts?",
            re.IGNORECASE,
        ),
    ),
    InjectionRule(
        # English "disclose your internal/original instructions". The
        # possessive is required for exactly the reason above: it is what
        # separates asking the assistant about itself from asking for a
        # document.
        rule_id="english_reveal_own_instructions",
        pattern=re.compile(
            r"(?:reveal|show|print|display|expose|disclose|repeat"
            r"|tell|output|give|state|share)\s{1,5}"
            r"(?:me\s{1,5})?your\s{1,5}"
            r"(?:hidden|initial|original|internal|secret)\s{1,5}"
            r"instructions?",
            re.IGNORECASE,
        ),
    ),
    InjectionRule(
        # English "what is your system prompt". The three rules above all
        # key on an imperative verb, so the interrogative form of the same
        # request slipped past every one of them. The noun is restricted
        # to the assistant's own configuration -- an employee asks what
        # the leave *policy* is, never what a *prompt* is.
        rule_id="english_ask_for_system_prompt",
        pattern=re.compile(
            r"what(?:\s{1,5}(?:exactly|precisely))?\s{1,5}"
            r"(?:is|are|was|were)\s{1,5}"
            # "the system prompt" is fair game -- that phrase is not
            # enterprise vocabulary. "the original instructions" is, so
            # the second branch demands the possessive, exactly as
            # english_reveal_own_instructions does.
            r"(?:(?:your|the)\s{1,5}(?:system|developer)\s{1,5}"
            r"(?:prompts?|messages?|instructions?)"
            r"|your\s{1,5}(?:initial|original|hidden|secret|internal)"
            r"\s{1,5}(?:prompts?|instructions?))",
            re.IGNORECASE,
        ),
    ),
    InjectionRule(
        # English "override/replace the system instructions/rules".
        rule_id="english_override_system_instructions",
        pattern=re.compile(
            r"(?:override|replace|supersede)\s{1,5}"
            r"(?:the\s{1,5}|all\s{1,5}|your\s{1,5})?system\s{1,5}"
            r"(?:instructions?|prompts?|rules?)",
            re.IGNORECASE,
        ),
    ),
    InjectionRule(
        # English "bypass (the|your) guardrails/restrictions/filters".
        rule_id="english_bypass_guardrails",
        pattern=re.compile(
            r"bypass\s{1,5}(?:the\s{1,5}|your\s{1,5})?"
            r"(?:guardrails?|restrictions?|filters?|safety)",
            re.IGNORECASE,
        ),
    ),
    InjectionRule(
        # Thai "forget/disregard the previous, entire, or system
        # instructions". (?!ซื้อ) excludes Thai purchase orders
        # ("khamsang sue") so procurement questions never match.
        rule_id="thai_forget_previous_instructions",
        pattern=re.compile(
            r"(?:ลืม|เพิกเฉย|ไม่ต้องสนใจ)[^\n]{0,10}"
            # 6, not 30: the attack writes the tail straight after the
            # noun ("khamsang kon-na"), while "I forgot ALL the travel
            # expense rules" needs a 15-character business noun phrase in
            # between. The wide gap was letting that declarative sentence
            # match, which is a refusal of a real question.
            r"(?:คำสั่ง(?!ซื้อ)|ข้อกำหนด)[^\n]{0,6}"
            r"(?:ก่อนหน้า|ทั้งหมด|ระบบ)"
        ),
    ),
    InjectionRule(
        # Thai "reveal/show/tell the system prompt or system instructions".
        # "kot rabop" (system rules) is deliberately not a disclosure
        # target here: employees ask about the HR Portal's rules in those
        # words. The attack shape that uses it -- replacing or cancelling
        # those rules -- is caught by the two control-phrase rules below.
        rule_id="thai_reveal_system_prompt",
        pattern=re.compile(
            r"(?:เปิดเผย|แสดง|บอก|ขอดู)[^\n]{0,10}"
            r"(?:system\s{0,3}prompt|คำสั่งระบบ(?!การ)|พรอมป์ต์ระบบ|พรอมต์ระบบ"
            r"|คำสั่งตั้งต้น|พรอมต์ที่ซ่อนอยู่)",
            re.IGNORECASE,
        ),
    ),
    InjectionRule(
        # Thai "reveal the developer message". Spelled out rather than
        # loosened to "พัฒนา", so a question about a notice from the
        # development team is not mistaken for an attack.
        rule_id="thai_reveal_developer_message",
        pattern=re.compile(
            r"(?:เปิดเผย|แสดง|บอก|ขอดู)[^\n]{0,10}"
            r"(?:ข้อความนักพัฒนา|ข้อความผู้พัฒนา|developer\s{0,3}message)",
            re.IGNORECASE,
        ),
    ),
    InjectionRule(
        # Thai "use this instruction instead of the system rules", the
        # shape that hides behind a valid HR question.
        rule_id="thai_replace_system_rules",
        pattern=re.compile(
            # "instead of" must name the system. Without that anchor the
            # rule also matched "can I use the sick-leave rule instead",
            # which is an ordinary HR question.
            r"ใช้[^\n]{0,10}(?:คำสั่ง(?!ซื้อ)|กฎ|ข้อกำหนด)[^\n]{0,8}"
            r"แทน[^\n]{0,10}ระบบ"
        ),
    ),
    InjectionRule(
        # Thai "cancel / do not follow the existing system rules".
        rule_id="thai_cancel_existing_rules",
        pattern=re.compile(
            # The tail must reach "rabop" (system): "cancel the OLD rule"
            # alone is what an employee asks about a leave policy, and
            # gap tuning cannot separate the two -- both write the tail
            # immediately after the noun.
            r"(?:ยกเลิก|ไม่ต้องทำตาม|ห้ามทำตาม)[^\n]{0,10}"
            r"(?:คำสั่ง(?!ซื้อ)|กฎ|ข้อกำหนด)"
            r"(?:เดิม|ก่อนหน้า|ทั้งหมด)?[^\n]{0,6}ระบบ"
        ),
    ),
    InjectionRule(
        # Thai "bypass/evade the system's restrictions or protections".
        rule_id="thai_bypass_system_restrictions",
        pattern=re.compile(
            r"(?:ข้าม|เลี่ยง|หลบ)[^\n]{0,10}"
            r"(?:guardrails?|ข้อจำกัดของระบบ|การป้องกันของระบบ|ระบบป้องกัน)",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class GuardrailResult:
    """Structured verdict for one screened query.

    Attributes:
        ok: True when the query may proceed to retrieval.
        normalized_query: NFC-normalized, stripped query text; every later
            pipeline stage must use this form, never the raw input. Empty
            when the raw input was not a string.
        reason: Reason code explaining a blocked query, ``None`` when the
            query passed.
    """

    ok: bool
    normalized_query: str
    reason: GuardrailReason | None = None


def guardrail_match_text(query: str) -> str:
    """Fold a query into the hardened form the injection rules match.

    Args:
        query: NFC-normalized query text.

    Returns:
        The NFKC-normalized, case-folded text with Thai SARA AM
        recomposed, format characters removed, separator punctuation
        turned into spaces, and runs of whitespace collapsed. This form
        exists only for matching: it is never returned to the pipeline,
        because NFKC rewrites compatibility characters that the Thai
        corpus and its queries may depend on.
    """
    return _fold(query, _MATCH_TRANSLATION)


def guardrail_match_variants(query: str) -> tuple[str, str]:
    """Return both hardened foldings every rule must be matched against.

    Args:
        query: NFC-normalized query text.

    Returns:
        The folding that deletes format characters and the folding that
        turns them into spaces, in that order. Matching both closes an
        attacker's choice of where to hide the invisible character; it
        costs one extra pass over a string already bounded by
        ``MAX_QUERY_CHARS``, and the two are identical whenever the query
        contains no format character at all.
    """
    return (
        _fold(query, _MATCH_TRANSLATION),
        _fold(query, _SPACED_MATCH_TRANSLATION),
    )


def _fold(query: str, translation: Mapping[int, str | None]) -> str:
    """Apply one hardening table to a query."""
    folded = unicodedata.normalize("NFKC", query).replace(
        _THAI_DECOMPOSED_SARA_AM, _THAI_SARA_AM
    )
    folded = folded.translate(translation)
    return " ".join(folded.casefold().split())


def screen_query(
    query: object,
    max_query_chars: int | None = None,
) -> GuardrailResult:
    """Screen one raw query before retrieval or any LLM call.

    Args:
        query: Raw user input. Typed ``object`` because this boundary must
            reject non-string payloads itself instead of trusting callers.
        max_query_chars: Length-limit override used by tests; defaults to
            ``config.MAX_QUERY_CHARS``.

    Returns:
        The structured verdict. When ``ok`` is False, ``reason`` names the
        rejection cause for logging and the fallback response.
    """
    limit = (
        max_query_chars
        if max_query_chars is not None
        else config.MAX_QUERY_CHARS
    )
    if not isinstance(query, str):
        return GuardrailResult(
            ok=False, normalized_query="", reason="invalid_query_type"
        )
    # NFC before anything else: composed and decomposed spellings of the
    # same visible Thai text must behave identically (AGENTS.md section 7).
    normalized = unicodedata.normalize("NFC", query).strip()
    if not normalized:
        return GuardrailResult(
            ok=False, normalized_query="", reason="empty_query"
        )
    if len(normalized) > limit:
        return GuardrailResult(
            ok=False, normalized_query=normalized, reason="query_too_long"
        )
    if matched_rule(normalized) is not None:
        return GuardrailResult(
            ok=False,
            normalized_query=normalized,
            reason="prompt_injection",
        )
    return GuardrailResult(ok=True, normalized_query=normalized)


def matched_rule(query: str) -> str | None:
    """Return the id of the first injection rule a query triggers.

    Exposed separately from ``screen_query`` so tests and the guardrail
    evaluation can report which rule fired, without the reason code
    having to carry that detail into the JSONL log.

    Args:
        query: NFC-normalized query text.

    Returns:
        The ``rule_id`` of the first matching rule, or ``None`` when the
        query matches no rule.
    """
    match_texts = guardrail_match_variants(query)
    for rule in INJECTION_RULES:
        if any(rule.pattern.search(text) for text in match_texts):
            return rule.rule_id
    return None
