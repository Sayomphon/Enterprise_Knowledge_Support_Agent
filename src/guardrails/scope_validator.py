"""Deterministic supported-scope gate for incoming questions.

A retrieval score measures how similar a question looks to a document,
not whether the corpus contains its answer. A maternity-leave question
shares most of its wording with the sick-leave policy, clears the direct
threshold, and would be answered from the wrong rule. This module asks
the separate question the score cannot answer: is the topic one this
corpus actually covers?

Two alias catalogs drive the decision. The supported catalog mirrors the
five topics of the current corpus; the unsupported catalog names the
HR/Finance topics employees plausibly ask about that this corpus has no
policy for, so they are refused explicitly instead of being answered from
whatever ranked first. Both are configuration data for an eight-document
prototype, not a production intent taxonomy: they need maintenance
whenever the corpus scope changes.
"""

from __future__ import annotations

from src import config
from src.guardrails.text_similarity import (
    character_ngrams,
    containment_of_ngrams,
    normalize_for_matching,
)
from src.schemas import KnowledgeTopic, ScopeDecision

# Matches ``ReasonCode.UNSUPPORTED_TOPIC``; kept as a literal so this
# module stays a pure function over the query text.
_UNSUPPORTED_TOPIC = "unsupported_topic"

# Aliases are drawn from the corpus vocabulary itself, including the
# informal spellings the chat transcripts use, so slang and typos resolve
# to the same topic as the formal wording. System names such as "Expense
# Portal" are deliberately absent: they share too many character n-grams
# with the unrelated "HR Portal", and every question about them already
# names the action -- claiming, filing, leave -- that identifies the topic.
SUPPORTED_TOPIC_ALIASES: dict[KnowledgeTopic, tuple[str, ...]] = {
    "reimbursement_process": (
        "เบิกค่าใช้จ่าย",
        "ยื่นเบิก",
        "ขอเบิก",
        "เบิกเงิน",
        "เบิกตัง",
        "ค่าเดินทาง",
        "ค่าแท็กซี่",
        "ค่าที่จอดรถ",
        "ขอคืนเงิน",
        "เคลมค่าใช้จ่าย",
        "taxi",
        "reimbursement",
    ),
    "receipt_policy": (
        "ใบเสร็จ",
        "ใบเสด",
        "ใบกำกับภาษี",
        "สลิปโอนเงิน",
        "แบบฟอร์มรับรองค่าใช้จ่าย",
        "e-receipt",
        "receipt",
    ),
    "annual_leave": (
        "ลาพักร้อน",
        "ลาพักผ่อน",
        "วันลาคงเหลือ",
        "วันลาประจำปี",
        "ลาหยุดยาว",
        "annual leave",
    ),
    "sick_leave": (
        "ลาป่วย",
        "ใบรับรองแพทย์",
        "sick leave",
    ),
    "work_from_home": (
        "ทำงานจากที่บ้าน",
        "ทำงานที่บ้าน",
        "work from home",
        "wfh",
    ),
}

# In-domain topics with no policy in this corpus. They are listed
# explicitly because lexical similarity alone would route them to a
# neighbouring policy: maternity leave to sick leave, meal expenses to
# the reimbursement process, salary questions to the finance documents.
UNSUPPORTED_TOPIC_ALIASES: dict[str, tuple[str, ...]] = {
    "maternity_leave": ("ลาคลอด", "maternity leave"),
    "ordination_leave": ("ลาบวช", "ลาอุปสมบท"),
    "marriage_leave": ("ลาแต่งงาน", "ลาสมรส"),
    "resignation": ("ลาออก", "ใบลาออก", "resignation"),
    "payroll_date": ("เงินเดือนออก", "วันจ่ายเงินเดือน", "payday"),
    "salary": (
        "เงินเดือน",
        "ฐานเงินเดือน",
        "ปรับเงินเดือน",
        "salary",
    ),
    "bonus": ("โบนัส", "bonus"),
    "medical_reimbursement": (
        "ค่ารักษาพยาบาล",
        "เบิกค่ารักษา",
        "ประกันสุขภาพ",
        "ค่าทำฟัน",
    ),
    "meal_reimbursement": (
        "ค่าอาหาร",
        "ค่าข้าว",
        "เบี้ยเลี้ยงอาหาร",
        "meal allowance",
    ),
    "hotel_reimbursement": (
        "ค่าโรงแรม",
        "ค่าที่พัก",
        "accommodation",
    ),
}


def _prepared(
    catalog: dict[str, tuple[str, ...]],
) -> dict[str, tuple[tuple[str, set[str]], ...]]:
    """Fragment every alias once, at import, instead of once per query.

    The catalogs are module constants, so their n-gram sets are constant
    too; rebuilding them per call was the bulk of this gate's cost, and
    the gate runs once per query plus once per rewrite candidate.

    Args:
        catalog: Topic to its alias tuple.

    Returns:
        The same mapping with each alias paired with its fragments.
    """
    return {
        topic: tuple(
            (normalized, character_ngrams(normalized))
            for normalized in (
                normalize_for_matching(alias) for alias in aliases
            )
        )
        for topic, aliases in catalog.items()
    }


_SUPPORTED_ALIAS_NGRAMS = _prepared(SUPPORTED_TOPIC_ALIASES)
_UNSUPPORTED_ALIAS_NGRAMS = _prepared(UNSUPPORTED_TOPIC_ALIASES)


def validate_scope(
    query: str, match_threshold: float | None = None
) -> ScopeDecision:
    """Decide whether one query falls inside the supported knowledge scope.

    An unsupported alias wins ties against a supported one: a question
    that mentions both an unsupported topic and a supported keyword --
    "does maternity leave need a medical certificate" -- is about the
    topic with no policy behind it, and answering it from the sick-leave
    rule would be a confident wrong answer.

    Args:
        query: The guardrail-normalized user query.
        match_threshold: Alias similarity override used by tests and the
            calibration sweep; defaults to
            ``config.SCOPE_MATCH_THRESHOLD``.

    Returns:
        The decision. Unsupported and under-specified queries both carry
        reason ``"unsupported_topic"`` with an empty topic tuple; a
        supported query carries every topic that cleared the threshold,
        sorted, plus the best alias score behind them.
    """
    threshold = (
        match_threshold
        if match_threshold is not None
        else config.SCOPE_MATCH_THRESHOLD
    )
    normalized_query = normalize_for_matching(query)
    query_ngrams = character_ngrams(normalized_query)
    supported_scores = {
        topic: _best_alias_score(
            normalized_query, query_ngrams, prepared_aliases
        )
        for topic, prepared_aliases in _SUPPORTED_ALIAS_NGRAMS.items()
    }
    unsupported_score = max(
        (
            _best_alias_score(
                normalized_query, query_ngrams, prepared_aliases
            )
            for prepared_aliases in _UNSUPPORTED_ALIAS_NGRAMS.values()
        ),
        default=0.0,
    )
    best_supported_score = max(supported_scores.values(), default=0.0)

    if unsupported_score >= threshold and (
        unsupported_score >= best_supported_score
    ):
        return ScopeDecision(
            supported=False,
            score=unsupported_score,
            reason=_UNSUPPORTED_TOPIC,
        )

    matched_topics = tuple(
        sorted(
            topic
            for topic, score in supported_scores.items()
            if score >= threshold
        )
    )
    if not matched_topics:
        # No alias reached the threshold: either the question is out of
        # domain, or it is too under-specified to name a topic, such as
        # asking how many days of leave without saying which leave.
        return ScopeDecision(
            supported=False,
            score=best_supported_score,
            reason=_UNSUPPORTED_TOPIC,
        )
    return ScopeDecision(
        supported=True,
        topics=matched_topics,
        score=best_supported_score,
    )


def _best_alias_score(
    normalized_query: str,
    query_ngrams: set[str],
    prepared_aliases: tuple[tuple[str, set[str]], ...],
) -> float:
    """Score one topic by its best-matching alias inside the query."""
    return max(
        (
            containment_of_ngrams(
                alias, alias_ngrams, normalized_query, query_ngrams
            )
            for alias, alias_ngrams in prepared_aliases
        ),
        default=0.0,
    )
