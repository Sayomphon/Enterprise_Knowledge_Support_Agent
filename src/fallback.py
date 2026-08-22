"""Fixed user-facing responses and the fallback reason-code enum.

Thai strings are user-facing message constants and are allowed here by
AGENTS.md section 6.1; they must never be inlined into pipeline logic.
The response texts are fixed so refusal and fallback behaviour stays
byte-identical across requests and trivially testable.

Two pure classifications of a reason code live here beside the texts
they explain: ``is_service_failure`` decides which body an employee
reads, and ``reason_family`` groups the same codes for whoever reads a
hundred of them. Both are derived from one set so a code cannot be
answered as an outage while being counted as something else.
"""

from __future__ import annotations

import enum

from src.schemas import KNOWLEDGE_TOPICS, KnowledgeTopic


class ReasonCode(enum.StrEnum):
    """Enumerated causes written to the JSONL fallback log.

    The first six values mirror the enum in AGENTS.md section 8. The
    remaining values extend it, as that section instructs, instead of
    inventing ad-hoc strings:

        - three for requests rejected by deterministic input validation
          before retrieval
        - three for a stage that failed rather than decided: the reporter
          boundary, the retriever, and the scope/evidence stage. Each has
          its own code so a crash inside a stage degrades to a logged
          fallback instead of escaping the graph, and so an operator can
          tell a stage outage from an evidence verdict
        - three for the deterministic contracts added by the remediation
          plan: an unsupported topic, a supported topic with no policy
          behind it, and a rewrite whose intent drifted away from the
          original question
        - three for the answer contract, separating a structurally
          broken candidate from a reporter that honestly reported its
          evidence as insufficient, and from one whose claim stated a
          figure its own citations do not contain
        - one for an LLM route reached without configured credentials,
          which is a service state and must never be reported to the
          employee as thin evidence (remediation plan Finding 8)
        - one for a boundary reached with none of the request deadline
          left, which is the same kind of fact: the stage did not run,
          so the corpus was never asked
        - one for a question too under-specified to name a topic, split
          out of ``unsupported_topic`` because the corpus may well hold
          the answer and only the employee can say which one they meant
    """

    PROMPT_INJECTION = "prompt_injection"
    LOW_RETRIEVAL_SCORE = "low_retrieval_score"
    REWRITE_LOW_RETRIEVAL_SCORE = "rewrite_low_retrieval_score"
    MISSING_CITATION = "missing_citation"
    FABRICATED_CITATION = "fabricated_citation"
    REWRITE_FAILURE = "rewrite_failure"
    INVALID_QUERY_TYPE = "invalid_query_type"
    EMPTY_QUERY = "empty_query"
    QUERY_TOO_LONG = "query_too_long"
    REPORTER_FAILURE = "reporter_failure"
    RETRIEVAL_FAILURE = "retrieval_failure"
    EVIDENCE_FAILURE = "evidence_failure"
    UNSUPPORTED_TOPIC = "unsupported_topic"
    NO_AUTHORITATIVE_EVIDENCE = "no_authoritative_evidence"
    REWRITE_REJECTED = "rewrite_rejected"
    INVALID_ANSWER_STRUCTURE = "invalid_answer_structure"
    INSUFFICIENT_REPORTER_EVIDENCE = "insufficient_reporter_evidence"
    UNSUPPORTED_NUMERIC_CLAIM = "unsupported_numeric_claim"
    LLM_NOT_CONFIGURED = "llm_not_configured"
    REQUEST_DEADLINE_EXCEEDED = "request_deadline_exceeded"
    AMBIGUOUS_TOPIC = "ambiguous_topic"


REFUSAL_TEXT = (
    "ขออภัย ระบบไม่สามารถดำเนินการคำขอที่พยายามเปลี่ยนแปลง"
    "คำสั่งหรือข้อกำหนดการทำงานของระบบได้\n"
    "หากต้องการสอบถามข้อมูลเกี่ยวกับขั้นตอนการทำงานภายในองค์กร "
    "สามารถถามใหม่ได้ค่ะ"
)

INVALID_QUERY_TEXT = (
    "ขออภัย ระบบไม่สามารถประมวลผลคำถามนี้ได้\n"
    "กรุณาพิมพ์คำถามเป็นข้อความที่ไม่ว่างเปล่า "
    "และมีความยาวไม่เกินที่ระบบกำหนด แล้วลองใหม่อีกครั้งค่ะ"
)

# The degraded responses are assembled from a body and a telemetry
# notice rather than written out four times, so the two bodies and the
# two notices each exist once and every combination stays byte-identical
# across requests. Which notice is appended is a fact about this request,
# not a template: claiming a question was recorded when the append failed
# is the dishonesty Finding 7 exists to remove.
# What each supported topic is called in the employee's own words. The
# mapping is keyed by ``KnowledgeTopic`` rather than written out as one
# sentence so a topic added to the catalog cannot quietly go unmentioned
# here: a test asserts the two agree, and this is the only place either
# surface names what the assistant can answer.
SUPPORTED_TOPIC_HINTS: dict[KnowledgeTopic, str] = {
    "reimbursement_process": "การเบิกค่าใช้จ่าย",
    "receipt_policy": "ใบเสร็จและหลักฐานการจ่ายเงิน",
    "annual_leave": "การลาพักร้อน",
    "sick_leave": "การลาป่วย",
    "work_from_home": "การทำงานจากที่บ้าน (WFH)",
}
# Named in the catalog's own order so the line is byte-identical across
# requests, like every other fixed text here.
_SUPPORTED_TOPIC_LINE = "หัวข้อที่ระบบตอบได้: " + ", ".join(
    SUPPORTED_TOPIC_HINTS[topic] for topic in KNOWLEDGE_TOPICS
)
# The list rides on the shared body rather than on one surface, so the
# CLI and the Streamlit card gain it together. It is what turns a refusal
# into something the employee can act on: "no evidence" tells them the
# assistant failed, while the same sentence plus the covered topics tells
# them what to ask instead.
_INSUFFICIENT_EVIDENCE_BODY = (
    "ขออภัย ยังไม่พบข้อมูลที่มีหลักฐานเพียงพอจากฐานความรู้ขององค์กร\n"
    f"{_SUPPORTED_TOPIC_LINE}\n"
    "กรุณาลองระบุรายละเอียดเพิ่มเติม หรือติดต่อ HR/Finance โดยตรง"
)
# A missing credential is a service state, so the text says so instead of
# sending the employee to HR over evidence that was never consulted. It
# names no variable, path, or provider (remediation plan Finding 8).
_SERVICE_UNAVAILABLE_BODY = (
    "ขออภัย ขณะนี้ระบบผู้ช่วยตอบคำถามด้วย AI ยังไม่พร้อมให้บริการ\n"
    "กรุณาลองใหม่อีกครั้งภายหลัง หรือติดต่อผู้ดูแลระบบขององค์กร"
)
# An under-specified question is not a gap in the corpus, so it is not
# answered with the corpus-gap text: the answer probably exists and the
# question has to name which one it wants. The prompt names the two
# families of choice the catalog actually distinguishes and no number,
# because the employee cannot act on a threshold and naming one would
# publish the gate's calibration.
_AMBIGUOUS_TOPIC_BODY = (
    "คำถามนี้ยังกว้างเกินกว่าจะเลือกเอกสารที่ตรงได้\n"
    "กรุณาระบุให้ชัดขึ้น เช่น ประเภทการลา (ลาพักร้อน / ลาป่วย) "
    "หรือประเภทค่าใช้จ่ายที่ต้องการเบิก"
)
_LOGGED_NOTICE = "คำถามนี้ถูกบันทึกไว้เพื่อใช้ปรับปรุงระบบแล้ว"
_LOG_UNAVAILABLE_NOTICE = (
    "ระบบไม่สามารถบันทึกคำถามเพื่อการวิเคราะห์ได้ในขณะนี้"
)

FALLBACK_TEXT = f"{_INSUFFICIENT_EVIDENCE_BODY}\n{_LOGGED_NOTICE}"
FALLBACK_TEXT_UNLOGGED = (
    f"{_INSUFFICIENT_EVIDENCE_BODY}\n{_LOG_UNAVAILABLE_NOTICE}"
)
SERVICE_UNAVAILABLE_TEXT = f"{_SERVICE_UNAVAILABLE_BODY}\n{_LOGGED_NOTICE}"
SERVICE_UNAVAILABLE_TEXT_UNLOGGED = (
    f"{_SERVICE_UNAVAILABLE_BODY}\n{_LOG_UNAVAILABLE_NOTICE}"
)
AMBIGUOUS_TOPIC_TEXT = f"{_AMBIGUOUS_TOPIC_BODY}\n{_LOGGED_NOTICE}"
AMBIGUOUS_TOPIC_TEXT_UNLOGGED = (
    f"{_AMBIGUOUS_TOPIC_BODY}\n{_LOG_UNAVAILABLE_NOTICE}"
)

# Reason codes that name a stage which FAILED rather than an evidence
# verdict it reached. A missing credential was the first one recognised
# here; a provider error, a crashed index and a crashed evidence stage
# are the same kind of fact about the service, and telling an employee
# that a policy could not be found is false when no policy was ever
# consulted. ``rewrite_rejected`` is deliberately absent: there the
# rewriter worked and the deterministic validator refused its output,
# which is the pipeline deciding rather than failing.
_SERVICE_FAILURE_REASONS = frozenset(
    {
        ReasonCode.LLM_NOT_CONFIGURED.value,
        ReasonCode.REPORTER_FAILURE.value,
        ReasonCode.RETRIEVAL_FAILURE.value,
        ReasonCode.EVIDENCE_FAILURE.value,
        ReasonCode.REWRITE_FAILURE.value,
        ReasonCode.REQUEST_DEADLINE_EXCEEDED.value,
    }
)


def is_service_failure(reason: str | None) -> bool:
    """Report whether a fallback reason describes a broken stage.

    The presentation layers ask this instead of comparing against one
    reason code each, so the boundary between "the corpus had no answer"
    and "a stage could not run" is defined once, beside the texts it
    selects.

    Args:
        reason: Reason code recorded for a degraded request, or ``None``
            when the request did not degrade.

    Returns:
        True when the request degraded because a stage failed rather
        than because its evidence was found wanting.
    """
    return reason in _SERVICE_FAILURE_REASONS


def is_ambiguous_topic(reason: str | None) -> bool:
    """Report whether a fallback reason describes an unfinished question.

    Asked by the presentation layers for the same reason as
    ``is_service_failure``: the card around the text must describe the
    same outcome the text does, and a question that only needs to be
    narrowed is neither an outage nor a gap in the corpus.

    Args:
        reason: Reason code recorded for a degraded request, or ``None``
            when the request did not degrade.

    Returns:
        True when the request degraded because the question named no
        single topic.
    """
    return reason == ReasonCode.AMBIGUOUS_TOPIC


class ReasonFamily(enum.StrEnum):
    """Analytics grouping over ``ReasonCode``.

    A reason code answers "why did this one request degrade"; a family
    answers "who should look at a hundred of them". The two are kept
    apart on purpose: the codes are the routing and logging contract and
    never collapse, while the families exist so a console or a warehouse
    query can rank causes without hard-coding a list of codes that a new
    member would silently fall out of.

    The four names each imply a different owner: a knowledge gap is a
    document somebody has to write, a service failure is an operational
    incident, a rejected input is a security or input-validation event,
    and a validation failure is a defect in what the model produced
    against a contract this repository owns.
    """

    KNOWLEDGE_GAP = "knowledge_gap"
    SERVICE_FAILURE = "service_failure"
    SECURITY_OR_INVALID_INPUT = "security_or_invalid_input"
    VALIDATION_FAILURE = "validation_failure"


# Every reason code, grouped. The service family is derived from the set
# the wording selector already uses rather than restated, so a code can
# never be answered with the service text while being reported under a
# different family.
_REASON_FAMILIES: dict[str, ReasonFamily] = {
    **{
        reason: ReasonFamily.SERVICE_FAILURE
        for reason in _SERVICE_FAILURE_REASONS
    },
    ReasonCode.PROMPT_INJECTION.value: (
        ReasonFamily.SECURITY_OR_INVALID_INPUT
    ),
    ReasonCode.INVALID_QUERY_TYPE.value: (
        ReasonFamily.SECURITY_OR_INVALID_INPUT
    ),
    ReasonCode.EMPTY_QUERY.value: ReasonFamily.SECURITY_OR_INVALID_INPUT,
    ReasonCode.QUERY_TOO_LONG.value: ReasonFamily.SECURITY_OR_INVALID_INPUT,
    ReasonCode.LOW_RETRIEVAL_SCORE.value: ReasonFamily.KNOWLEDGE_GAP,
    ReasonCode.REWRITE_LOW_RETRIEVAL_SCORE.value: ReasonFamily.KNOWLEDGE_GAP,
    ReasonCode.UNSUPPORTED_TOPIC.value: ReasonFamily.KNOWLEDGE_GAP,
    ReasonCode.NO_AUTHORITATIVE_EVIDENCE.value: ReasonFamily.KNOWLEDGE_GAP,
    # An under-specified question is grouped with the knowledge gaps
    # rather than with the rejected inputs: the input was valid, and a
    # run of them says the catalog's topics are not visible enough to
    # employees, which is a documentation job like the others here.
    ReasonCode.AMBIGUOUS_TOPIC.value: ReasonFamily.KNOWLEDGE_GAP,
    ReasonCode.MISSING_CITATION.value: ReasonFamily.VALIDATION_FAILURE,
    ReasonCode.FABRICATED_CITATION.value: ReasonFamily.VALIDATION_FAILURE,
    ReasonCode.INVALID_ANSWER_STRUCTURE.value: (
        ReasonFamily.VALIDATION_FAILURE
    ),
    # The reporter's own "I cannot answer from this" is grouped with the
    # contract failures rather than with the knowledge gaps: it is a
    # model judgement checked by the validator, not a measurement of the
    # corpus, and treating it as corpus evidence would let a bad day at
    # the provider read as a missing document.
    ReasonCode.INSUFFICIENT_REPORTER_EVIDENCE.value: (
        ReasonFamily.VALIDATION_FAILURE
    ),
    ReasonCode.UNSUPPORTED_NUMERIC_CLAIM.value: (
        ReasonFamily.VALIDATION_FAILURE
    ),
    # Here the rewriter worked and the deterministic validator refused
    # its output, which is this repository's contract rejecting model
    # output -- the same shape as an invalid answer, not a stage outage.
    ReasonCode.REWRITE_REJECTED.value: ReasonFamily.VALIDATION_FAILURE,
}


def reason_family(reason: str | None) -> ReasonFamily | None:
    """Group one reason code for reporting.

    Args:
        reason: Reason code recorded for a blocked or degraded request.

    Returns:
        The family the code belongs to, or ``None`` for ``None`` and for
        any string that is not a member of ``ReasonCode``. An unknown
        code is reported as unknown rather than bucketed into a default
        family, because a silent default is how a new code disappears
        from the counts it should be raising.
    """
    if reason is None:
        return None
    return _REASON_FAMILIES.get(str(reason))


def refusal_text_for(reason: str) -> str:
    """Select the fixed blocked-response text for a guardrail reason.

    Args:
        reason: Guardrail reason code for the blocked request.

    Returns:
        The injection refusal for ``prompt_injection``; the generic
        invalid-input text for every other pre-retrieval rejection.
    """
    if reason == ReasonCode.PROMPT_INJECTION:
        return REFUSAL_TEXT
    return INVALID_QUERY_TEXT


def response_text_for_state(
    route: str | None,
    guardrail_reason: str | None,
    fallback_reason: str | None = None,
    *,
    telemetry_logged: bool = True,
) -> str | None:
    """Select the fixed user-facing text for a finished pipeline route.

    Shared by every presentation layer (CLI and Streamlit) so the mapping
    from route to fixed text exists exactly once.

    Args:
        route: Final ``route`` value from the pipeline state.
        guardrail_reason: Guardrail reason code for blocked requests.
        fallback_reason: Reason code recorded for a degraded request.
            It selects between three bodies and nothing finer: a service
            state (``_SERVICE_FAILURE_REASONS``) tells the employee to
            try again, an under-specified question asks them to name
            what they meant, and every remaining evidence outcome shares
            one text, because naming the individual verdicts would turn
            internal routing into user-facing noise the employee cannot
            act on. The three differ in what the employee should DO
            next, which is the only distinction worth spending a
            separate message on.
        telemetry_logged: Whether this request's event reached the JSONL
            sink. ``False`` selects the wording that does not claim the
            question was recorded. The graph sets the flag on every route
            that attempts a write, so the default covers only routes that
            never log.

    Returns:
        The fixed refusal or degraded-response text, or ``None`` when the
        route carries a validated answer that should be shown instead.
    """
    if route == "blocked":
        # The refusal texts make no logging claim, so they are honest
        # under either outcome and need no variant.
        return refusal_text_for(guardrail_reason or "")
    if route != "fallback":
        return None
    if is_service_failure(fallback_reason):
        return (
            SERVICE_UNAVAILABLE_TEXT
            if telemetry_logged
            else SERVICE_UNAVAILABLE_TEXT_UNLOGGED
        )
    if is_ambiguous_topic(fallback_reason):
        return (
            AMBIGUOUS_TOPIC_TEXT
            if telemetry_logged
            else AMBIGUOUS_TOPIC_TEXT_UNLOGGED
        )
    return FALLBACK_TEXT if telemetry_logged else FALLBACK_TEXT_UNLOGGED
