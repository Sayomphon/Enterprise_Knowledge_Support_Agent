"""Fixed user-facing responses and the fallback reason-code enum.

Thai strings are user-facing message constants and are allowed here by
AGENTS.md section 6.1; they must never be inlined into pipeline logic.
The response texts are fixed so refusal and fallback behaviour stays
byte-identical across requests and trivially testable.
"""

from __future__ import annotations

import enum


class ReasonCode(enum.StrEnum):
    """Enumerated causes written to the JSONL fallback log.

    The first six values mirror the enum in AGENTS.md section 8. The
    remaining values extend it, as that section instructs, instead of
    inventing ad-hoc strings: three for requests rejected by
    deterministic input validation before retrieval, one for a
    reporter-boundary failure so that a provider outage still routes to
    fallback with a stable code, three for the deterministic contracts
    added by the remediation plan -- an unsupported topic, a supported
    topic with no policy behind it, and a rewrite whose intent drifted
    away from the original question -- two for the answer contract,
    separating a structurally broken candidate from a reporter that
    honestly reported its evidence as insufficient, and one for an LLM
    route reached without configured credentials, which is a service
    state and must never be reported to the employee as thin evidence
    (remediation plan Finding 8).
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
    UNSUPPORTED_TOPIC = "unsupported_topic"
    NO_AUTHORITATIVE_EVIDENCE = "no_authoritative_evidence"
    REWRITE_REJECTED = "rewrite_rejected"
    INVALID_ANSWER_STRUCTURE = "invalid_answer_structure"
    INSUFFICIENT_REPORTER_EVIDENCE = "insufficient_reporter_evidence"
    LLM_NOT_CONFIGURED = "llm_not_configured"


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
_INSUFFICIENT_EVIDENCE_BODY = (
    "ขออภัย ยังไม่พบข้อมูลที่มีหลักฐานเพียงพอจากฐานความรู้ขององค์กร\n"
    "กรุณาลองระบุรายละเอียดเพิ่มเติม หรือติดต่อ HR/Finance โดยตรง"
)
# A missing credential is a service state, so the text says so instead of
# sending the employee to HR over evidence that was never consulted. It
# names no variable, path, or provider (remediation plan Finding 8).
_SERVICE_UNAVAILABLE_BODY = (
    "ขออภัย ขณะนี้ระบบผู้ช่วยตอบคำถามด้วย AI ยังไม่พร้อมให้บริการ\n"
    "กรุณาลองใหม่อีกครั้งภายหลัง หรือติดต่อผู้ดูแลระบบขององค์กร"
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
            Only ``llm_not_configured`` changes the wording: every other
            reason is an evidence outcome the employee cannot act on
            differently, and naming them would turn internal routing into
            user-facing noise.
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
    if fallback_reason == ReasonCode.LLM_NOT_CONFIGURED:
        return (
            SERVICE_UNAVAILABLE_TEXT
            if telemetry_logged
            else SERVICE_UNAVAILABLE_TEXT_UNLOGGED
        )
    return FALLBACK_TEXT if telemetry_logged else FALLBACK_TEXT_UNLOGGED
