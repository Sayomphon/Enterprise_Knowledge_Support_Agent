"""Deterministic supported-scope gate for incoming questions.

A retrieval score measures how similar a question looks to a document,
not whether the corpus contains its answer. A maternity-leave question
shares most of its wording with the sick-leave policy, clears the direct
threshold, and would be answered from the wrong rule. This module asks
the separate question the score cannot answer: is the topic one this
corpus actually covers?

A refusal is then split once more. "How many days of leave can I take"
has no answer because it never says which leave; "ordination leave" has
none because the corpus has no such policy. Both refuse, and only one of
them is worth asking the employee to finish, so the gate reports them
under separate reason codes.

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

# Match ``ReasonCode.UNSUPPORTED_TOPIC`` and ``ReasonCode.AMBIGUOUS_TOPIC``;
# kept as literals so this module stays a pure function over the query text.
_UNSUPPORTED_TOPIC = "unsupported_topic"
_AMBIGUOUS_TOPIC = "ambiguous_topic"

# How many topics a query must touch before "under-specified" is a better
# description of it than "unsupported". Two is the definition of the word
# rather than a calibrated cut-off -- one partial match is a coincidence,
# two is a question that named a concept the corpus covers without saying
# which one -- so it lives here beside the rule instead of in the config
# file, where every other value is a measured threshold.
_AMBIGUOUS_MIN_TOPICS = 2

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
        # The two phrases FIN-002 itself opens with. They earn their place
        # twice: an employee who asks about "financial evidence" without
        # saying "receipt" resolves to this topic, and the deterministic
        # expansion searches the document's own wording, which is what
        # lifts a paraphrased receipt question over the answer threshold
        # without any index or threshold change (2026-08-22 ablation).
        "หลักฐานทางการเงิน",
        "เอกสารประกอบการเบิก",
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
    # The groups below close a verified gap rather than an imagined one:
    # an eligibility question about an expense item the corpus has no
    # rule for ("can I claim my phone bill through the Expense Portal")
    # scored above the direct threshold on the reimbursement PROCESS
    # policy, which describes how to file a claim and says nothing about
    # which items qualify. The process wording carried the question past
    # the gate, and the answer would have been a confident yes drawn from
    # a document that never granted it.
    "phone_reimbursement": (
        "ค่าโทรศัพท์",
        "ค่าโทรศัพท์มือถือ",
        "ค่ามือถือ",
    ),
    "internet_reimbursement": (
        "ค่าอินเทอร์เน็ต",
        "ค่าเน็ต",
        "ค่า wifi",
    ),
    "office_equipment": (
        "ค่าอุปกรณ์สำนักงาน",
        "อุปกรณ์ทำงาน",
        "ค่าโต๊ะเก้าอี้",
    ),
    "training_expense": (
        "ค่าอบรม",
        "ค่าสัมมนา",
        "ค่าคอร์สเรียน",
    ),
    # Per diem is a daily allowance, not the meal receipt case above; it
    # gets its own group so a future policy can be added for one without
    # implying the other.
    "per_diem": (
        "เบี้ยเลี้ยง",
        "per diem",
    ),
    "business_leave": ("ลากิจ",),
    # Deliberately narrow. "kha-nam-man" (fuel) must not be widened
    # towards "kha-doen-thang" (travel), which is a SUPPORTED alias of
    # the reimbursement process: taxi fares after overtime are covered,
    # and an alias that swallowed the travel wording would refuse them.
    "fuel_mileage": (
        "ค่าน้ำมัน",
        "ค่าน้ำมันรถ",
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
    query: str,
    match_threshold: float | None = None,
    ambiguous_min_score: float | None = None,
) -> ScopeDecision:
    """Decide whether one query falls inside the supported knowledge scope.

    An unsupported alias wins ties against a supported one: a question
    that mentions both an unsupported topic and a supported keyword --
    "does maternity leave need a medical certificate" -- is about the
    topic with no policy behind it, and answering it from the sick-leave
    rule would be a confident wrong answer.

    A query that resolves no topic is separated once more, into one the
    corpus has no policy for and one that is merely under-specified:
    "how many days of leave can I take" names no leave type, and telling
    that employee to contact HR wastes a question the assistant could
    have answered had it been asked precisely. The split is a wording
    decision, never a routing one -- both verdicts refuse the request.

    Args:
        query: The guardrail-normalized user query.
        match_threshold: Alias similarity override used by tests and the
            calibration sweep; defaults to
            ``config.SCOPE_MATCH_THRESHOLD``.
        ambiguous_min_score: Under-specified floor override used by the
            same callers; defaults to
            ``config.SCOPE_AMBIGUOUS_MIN_SCORE``.

    Returns:
        The decision. A supported query carries every topic that cleared
        the threshold, sorted, plus the best alias score behind them. A
        refused query carries an empty topic tuple and reason
        ``"ambiguous_topic"`` when it touched at least two topics above
        the under-specified floor without resolving any, else
        ``"unsupported_topic"``. The topic tuple stays empty in both
        cases: a near-match is not a resolved topic, and populating it
        would put topics the gate refused into the alias expansion and
        the telemetry alike.
    """
    threshold = (
        match_threshold
        if match_threshold is not None
        else config.SCOPE_MATCH_THRESHOLD
    )
    ambiguous_floor = (
        ambiguous_min_score
        if ambiguous_min_score is not None
        else config.SCOPE_AMBIGUOUS_MIN_SCORE
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
        # asking how many days of leave without saying which leave. The
        # second-best score is what tells them apart, so it is counted
        # rather than the best one: an out-of-domain query can brush a
        # single topic by accident, but brushing two of them without
        # resolving either is what an unfinished question looks like.
        near_topics = sum(
            score >= ambiguous_floor for score in supported_scores.values()
        )
        return ScopeDecision(
            supported=False,
            score=best_supported_score,
            reason=(
                _AMBIGUOUS_TOPIC
                if near_topics >= _AMBIGUOUS_MIN_TOPICS
                else _UNSUPPORTED_TOPIC
            ),
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
