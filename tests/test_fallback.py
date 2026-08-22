"""Unit tests for fixed response texts and their route mapping."""

import ast
import unittest
from pathlib import Path
from typing import get_args

from src.fallback import (
    FALLBACK_TEXT,
    FALLBACK_TEXT_UNLOGGED,
    INVALID_QUERY_TEXT,
    REFUSAL_TEXT,
    SERVICE_UNAVAILABLE_TEXT,
    SERVICE_UNAVAILABLE_TEXT_UNLOGGED,
    ReasonCode,
    is_service_failure,
    refusal_text_for,
    response_text_for_state,
)
from src.guardrails.input_guardrail import GuardrailReason

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
    """A failed stage is a service state, not thin evidence.

    The distinction is the point of the wording: an employee told that
    no policy was found goes and asks HR about a rule the pipeline never
    read, while an employee told the service is unavailable tries again.
    """

    #: Reasons naming a stage that could not run at all.
    SERVICE_REASONS = (
        ReasonCode.LLM_NOT_CONFIGURED,
        ReasonCode.REPORTER_FAILURE,
        ReasonCode.RETRIEVAL_FAILURE,
        ReasonCode.EVIDENCE_FAILURE,
        ReasonCode.REWRITE_FAILURE,
        ReasonCode.REQUEST_DEADLINE_EXCEEDED,
    )

    #: Reasons naming a verdict the pipeline actually reached.
    EVIDENCE_REASONS = (
        ReasonCode.LOW_RETRIEVAL_SCORE,
        ReasonCode.REWRITE_LOW_RETRIEVAL_SCORE,
        ReasonCode.UNSUPPORTED_TOPIC,
        ReasonCode.NO_AUTHORITATIVE_EVIDENCE,
        ReasonCode.FABRICATED_CITATION,
        ReasonCode.MISSING_CITATION,
        ReasonCode.INVALID_ANSWER_STRUCTURE,
        ReasonCode.INSUFFICIENT_REPORTER_EVIDENCE,
        ReasonCode.UNSUPPORTED_NUMERIC_CLAIM,
        ReasonCode.REWRITE_REJECTED,
    )

    #: Reasons the guardrail writes on the blocked route, which has its
    #: own fixed texts and never reaches the fallback wording at all.
    BLOCKED_REASONS = (
        ReasonCode.PROMPT_INJECTION,
        ReasonCode.EMPTY_QUERY,
        ReasonCode.QUERY_TOO_LONG,
        ReasonCode.INVALID_QUERY_TYPE,
    )

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

    def test_every_stage_failure_selects_the_service_text(self) -> None:
        for reason in self.SERVICE_REASONS:
            with self.subTest(reason=reason):
                self.assertEqual(
                    response_text_for_state("fallback", None, reason),
                    SERVICE_UNAVAILABLE_TEXT,
                )

    def test_stage_failures_never_send_the_employee_to_hr(self) -> None:
        # The evidence text ends by naming HR/Finance. A provider outage
        # must not, because there is no policy question to ask them.
        for reason in self.SERVICE_REASONS:
            with self.subTest(reason=reason):
                text = response_text_for_state("fallback", None, reason)
                self.assertNotIn("HR/Finance", text)

    def test_every_evidence_reason_keeps_the_evidence_wording(self) -> None:
        for reason in self.EVIDENCE_REASONS:
            with self.subTest(reason=reason):
                self.assertEqual(
                    response_text_for_state("fallback", None, reason),
                    FALLBACK_TEXT,
                )

    def test_is_service_failure_agrees_with_the_selected_text(self) -> None:
        # The helper is what the Streamlit notice card asks, so a card
        # that shows the outage icon and a body about missing evidence
        # would be this assertion failing rather than a UI review.
        for reason in self.SERVICE_REASONS + self.EVIDENCE_REASONS:
            with self.subTest(reason=reason):
                text = response_text_for_state("fallback", None, reason)
                self.assertEqual(
                    is_service_failure(reason),
                    text == SERVICE_UNAVAILABLE_TEXT,
                )

    def test_no_reason_code_is_left_unclassified(self) -> None:
        # Guards the guard: a reason code added later must be sorted
        # into a wording deliberately, not inherit the evidence text by
        # being forgotten here.
        classified = set(self.SERVICE_REASONS) | set(
            self.EVIDENCE_REASONS
        ) | set(self.BLOCKED_REASONS)

        self.assertEqual(classified, set(ReasonCode))

    def test_an_unknown_reason_is_not_treated_as_an_outage(self) -> None:
        self.assertFalse(is_service_failure(None))
        self.assertFalse(is_service_failure("some_future_reason"))


class TestReasonCodeLiterals(unittest.TestCase):
    """Every reason string in ``src`` must be a member of the enum.

    Four deterministic modules keep their reason codes as bare literals
    so they stay pure functions that import nothing from ``fallback``,
    and the guardrail restates four more as a ``Literal`` type. Comments
    claim each one "matches ReasonCode", but nothing bound the two:
    renaming a member would change what the graph logs while those
    modules kept returning the old string, and no test would fail. This
    is that binding.
    """

    #: Modules that return reason codes without importing the enum.
    LITERAL_SOURCES = (
        "src/guardrails/citation_validator.py",
        "src/guardrails/scope_validator.py",
        "src/guardrails/rewrite_validator.py",
        "src/evidence_selector.py",
    )

    @staticmethod
    def _reason_literals(relative: str) -> set[str]:
        """Collect the strings one module actually returns as reasons.

        Only ``return "..."`` statements and module-level private string
        constants count. Dictionary keys and rule identifiers are
        snake_case too, and scanning every constant would sweep them in.
        """
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and isinstance(
                node.value, ast.Constant
            ):
                if isinstance(node.value.value, str):
                    found.add(node.value.value)
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and any(
                    isinstance(target, ast.Name)
                    and target.id.startswith("_")
                    and target.id.isupper()
                    for target in node.targets
                )
            ):
                found.add(node.value.value)
        return found

    def test_every_returned_reason_literal_is_a_reason_code(self) -> None:
        known = {code.value for code in ReasonCode}
        for relative in self.LITERAL_SOURCES:
            literals = self._reason_literals(relative)
            self.assertTrue(literals, f"{relative} returned no literal")
            for literal in literals:
                with self.subTest(module=relative, literal=literal):
                    self.assertIn(literal, known)

    def test_every_guardrail_reason_is_a_reason_code(self) -> None:
        known = {code.value for code in ReasonCode}

        for reason in get_args(GuardrailReason):
            with self.subTest(reason=reason):
                self.assertIn(reason, known)

    def test_the_binding_would_catch_a_renamed_member(self) -> None:
        # Guards the guard: if the collector ever stopped finding
        # anything, the assertions above would pass vacuously.
        literals = self._reason_literals(
            "src/guardrails/citation_validator.py"
        )

        self.assertIn("fabricated_citation", literals)
        self.assertNotIn("annual_leave", literals)


if __name__ == "__main__":
    unittest.main()
