"""Unit tests for fixed response texts and their route mapping."""

import unittest

from src.fallback import (
    FALLBACK_TEXT,
    FALLBACK_TEXT_UNLOGGED,
    INVALID_QUERY_TEXT,
    REFUSAL_TEXT,
    SERVICE_UNAVAILABLE_TEXT,
    SERVICE_UNAVAILABLE_TEXT_UNLOGGED,
    ReasonCode,
    refusal_text_for,
    response_text_for_state,
)

# The sentence the degraded texts may print only when the append to the
# JSONL sink actually succeeded.
LOGGED_CLAIM = "คำถามนี้ถูกบันทึกไว้เพื่อใช้ปรับปรุงระบบแล้ว"


class TestRefusalTextSelection(unittest.TestCase):
    """Blocked requests map to exactly one fixed text per reason."""

    def test_prompt_injection_gets_the_injection_refusal(self) -> None:
        self.assertEqual(
            refusal_text_for(ReasonCode.PROMPT_INJECTION), REFUSAL_TEXT
        )

    def test_input_validation_reasons_get_the_invalid_query_text(
        self,
    ) -> None:
        for reason in ("empty_query", "query_too_long", "invalid_query_type"):
            with self.subTest(reason=reason):
                self.assertEqual(
                    refusal_text_for(reason), INVALID_QUERY_TEXT
                )


class TestResponseTextForState(unittest.TestCase):
    """The shared route-to-text mapping used by CLI and Streamlit."""

    def test_blocked_route_returns_the_refusal_text(self) -> None:
        self.assertEqual(
            response_text_for_state("blocked", "prompt_injection"),
            REFUSAL_TEXT,
        )

    def test_fallback_route_returns_the_fallback_text(self) -> None:
        self.assertEqual(
            response_text_for_state("fallback", None), FALLBACK_TEXT
        )

    def test_answered_route_returns_none_so_the_answer_is_shown(
        self,
    ) -> None:
        self.assertIsNone(response_text_for_state("answered", None))


class TestLoggingHonesty(unittest.TestCase):
    """The degraded text may claim a recorded question only when true."""

    def test_successful_logging_keeps_the_recorded_claim(self) -> None:
        text = response_text_for_state(
            "fallback", None, ReasonCode.LOW_RETRIEVAL_SCORE, telemetry_logged=True
        )

        self.assertEqual(text, FALLBACK_TEXT)
        self.assertIn(LOGGED_CLAIM, text)

    def test_failed_logging_replaces_the_claim_with_the_true_statement(
        self,
    ) -> None:
        text = response_text_for_state(
            "fallback",
            None,
            ReasonCode.LOW_RETRIEVAL_SCORE,
            telemetry_logged=False,
        )

        self.assertEqual(text, FALLBACK_TEXT_UNLOGGED)
        self.assertNotIn(LOGGED_CLAIM, text)

    def test_both_variants_keep_the_same_guidance_body(self) -> None:
        # Only the telemetry sentence differs: a logging failure is not a
        # reason to tell the employee something different about evidence.
        self.assertEqual(
            FALLBACK_TEXT.rsplit("\n", 1)[0],
            FALLBACK_TEXT_UNLOGGED.rsplit("\n", 1)[0],
        )

    def test_blocked_text_makes_no_logging_claim_either_way(self) -> None:
        for logged in (True, False):
            with self.subTest(telemetry_logged=logged):
                text = response_text_for_state(
                    "blocked",
                    ReasonCode.PROMPT_INJECTION,
                    telemetry_logged=logged,
                )
                self.assertEqual(text, REFUSAL_TEXT)
                self.assertNotIn(LOGGED_CLAIM, text)


class TestServiceUnavailableMapping(unittest.TestCase):
    """A missing credential is a service state, not thin evidence."""

    def test_llm_not_configured_selects_the_service_text(self) -> None:
        text = response_text_for_state(
            "fallback", None, ReasonCode.LLM_NOT_CONFIGURED
        )

        self.assertEqual(text, SERVICE_UNAVAILABLE_TEXT)
        self.assertNotEqual(text, FALLBACK_TEXT)

    def test_service_text_honours_the_logging_outcome_too(self) -> None:
        text = response_text_for_state(
            "fallback",
            None,
            ReasonCode.LLM_NOT_CONFIGURED,
            telemetry_logged=False,
        )

        self.assertEqual(text, SERVICE_UNAVAILABLE_TEXT_UNLOGGED)
        self.assertNotIn(LOGGED_CLAIM, text)

    def test_service_text_never_names_the_credential_or_provider(
        self,
    ) -> None:
        for text in (
            SERVICE_UNAVAILABLE_TEXT,
            SERVICE_UNAVAILABLE_TEXT_UNLOGGED,
        ):
            with self.subTest(text=text[:24]):
                for secret_word in ("OPENAI", "API", "key", "sk-"):
                    self.assertNotIn(secret_word, text)

    def test_every_other_reason_keeps_the_evidence_wording(self) -> None:
        evidence_reasons = (
            ReasonCode.LOW_RETRIEVAL_SCORE,
            ReasonCode.UNSUPPORTED_TOPIC,
            ReasonCode.NO_AUTHORITATIVE_EVIDENCE,
            ReasonCode.FABRICATED_CITATION,
            ReasonCode.REPORTER_FAILURE,
        )
        for reason in evidence_reasons:
            with self.subTest(reason=reason):
                self.assertEqual(
                    response_text_for_state("fallback", None, reason),
                    FALLBACK_TEXT,
                )


if __name__ == "__main__":
    unittest.main()
