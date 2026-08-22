"""Unit tests for fixed response texts and their route mapping."""

import ast
import unittest
from pathlib import Path
from typing import get_args

from src.fallback import (
    AMBIGUOUS_TOPIC_TEXT,
    AMBIGUOUS_TOPIC_TEXT_UNLOGGED,
    FALLBACK_TEXT,
    FALLBACK_TEXT_UNLOGGED,
    INVALID_QUERY_TEXT,
    REFUSAL_TEXT,
    SERVICE_UNAVAILABLE_TEXT,
    SERVICE_UNAVAILABLE_TEXT_UNLOGGED,
    SUPPORTED_TOPIC_HINTS,
    ReasonCode,
    ReasonFamily,
    _REASON_FAMILIES,
    is_ambiguous_topic,
    is_service_failure,
    reason_family,
    refusal_text_for,
    response_text_for_state,
)
from src.guardrails.input_guardrail import GuardrailReason
from src.schemas import KNOWLEDGE_TOPICS

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


class TestSupportedTopicSuggestions(unittest.TestCase):
    """A refusal names what the assistant CAN answer.

    The list lives on the shared body so the CLI and the Streamlit card
    gain it together, and it is generated from the topic catalog so it
    cannot fall behind it.
    """

    def test_every_supported_topic_has_a_hint(self) -> None:
        # The binding that keeps the sentence honest: a topic added to
        # the corpus catalog with no hint here would leave the fallback
        # advertising four of five things the assistant can do.
        self.assertEqual(set(SUPPORTED_TOPIC_HINTS), set(KNOWLEDGE_TOPICS))

    def test_the_fallback_text_names_each_one(self) -> None:
        for topic, hint in SUPPORTED_TOPIC_HINTS.items():
            with self.subTest(topic=topic):
                self.assertIn(hint, FALLBACK_TEXT)

    def test_both_logging_variants_carry_the_list(self) -> None:
        for text in (FALLBACK_TEXT, FALLBACK_TEXT_UNLOGGED):
            with self.subTest(text=text[:24]):
                self.assertIn(SUPPORTED_TOPIC_HINTS["annual_leave"], text)

    def test_a_stage_failure_does_not_advertise_topics(self) -> None:
        # Nothing was searched, so listing what the corpus covers would
        # describe an outage as a coverage question.
        self.assertNotIn(
            SUPPORTED_TOPIC_HINTS["annual_leave"], SERVICE_UNAVAILABLE_TEXT
        )


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

    #: Reasons with a body of their own, being neither an outage nor a
    #: statement that the corpus lacks the answer.
    AMBIGUOUS_REASONS = (ReasonCode.AMBIGUOUS_TOPIC,)

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
        classified = (
            set(self.SERVICE_REASONS)
            | set(self.EVIDENCE_REASONS)
            | set(self.AMBIGUOUS_REASONS)
            | set(self.BLOCKED_REASONS)
        )

        self.assertEqual(classified, set(ReasonCode))

    def test_an_unknown_reason_is_not_treated_as_an_outage(self) -> None:
        self.assertFalse(is_service_failure(None))
        self.assertFalse(is_service_failure("some_future_reason"))


class TestAmbiguousTopicMapping(unittest.TestCase):
    """An unfinished question gets its own answer, not the corpus gap.

    The three degraded bodies differ in what the employee should do
    next: try again, name what they meant, or go and ask HR. Merging
    this one back into the evidence text would send somebody to HR over
    a question the assistant could have answered.
    """

    def test_the_ambiguous_reason_selects_its_own_text(self) -> None:
        text = response_text_for_state(
            "fallback", None, ReasonCode.AMBIGUOUS_TOPIC
        )

        self.assertEqual(text, AMBIGUOUS_TOPIC_TEXT)
        self.assertNotEqual(text, FALLBACK_TEXT)
        self.assertNotEqual(text, SERVICE_UNAVAILABLE_TEXT)

    def test_it_honours_the_logging_outcome_like_the_others(self) -> None:
        text = response_text_for_state(
            "fallback",
            None,
            ReasonCode.AMBIGUOUS_TOPIC,
            telemetry_logged=False,
        )

        self.assertEqual(text, AMBIGUOUS_TOPIC_TEXT_UNLOGGED)
        self.assertNotIn(LOGGED_CLAIM, text)

    def test_it_asks_for_a_narrower_question_instead_of_naming_hr(
        self,
    ) -> None:
        # The corpus may well hold the answer, so the employee is asked
        # to finish the question rather than to leave the assistant.
        self.assertNotIn("HR/Finance", AMBIGUOUS_TOPIC_TEXT)

    def test_it_reveals_no_threshold(self) -> None:
        # The gate's calibration is not user-facing: a message naming a
        # score teaches an employee to game it and tells them nothing
        # they can act on (AGENTS.md section 7).
        for figure in ("0.11", "0.40", "score", "threshold"):
            with self.subTest(figure=figure):
                self.assertNotIn(figure, AMBIGUOUS_TOPIC_TEXT)

    def test_the_predicate_agrees_with_the_selected_text(self) -> None:
        for code in ReasonCode:
            with self.subTest(reason=code):
                text = response_text_for_state("fallback", None, code)
                self.assertEqual(
                    is_ambiguous_topic(code), text == AMBIGUOUS_TOPIC_TEXT
                )

    def test_an_unfinished_question_is_not_reported_as_an_outage(
        self,
    ) -> None:
        self.assertFalse(is_service_failure(ReasonCode.AMBIGUOUS_TOPIC))


class TestReasonFamilies(unittest.TestCase):
    """Grouping reason codes for reporting, without changing routing.

    The families exist so a console or a warehouse query can rank causes
    without hard-coding a list of codes. That only holds while the
    mapping is total: a code with no family would vanish from every
    grouped count while still degrading real requests.
    """

    def test_every_reason_code_has_a_family(self) -> None:
        for code in ReasonCode:
            with self.subTest(reason=code):
                self.assertIsNotNone(reason_family(code))

    def test_the_mapping_names_no_code_that_does_not_exist(self) -> None:
        # The other half of exhaustiveness: a renamed member would
        # otherwise leave its old spelling behind as a family for a
        # reason nothing can ever record.
        self.assertEqual(
            set(_REASON_FAMILIES), {code.value for code in ReasonCode}
        )

    def test_no_family_is_left_without_a_reason_code(self) -> None:
        self.assertEqual(set(_REASON_FAMILIES.values()), set(ReasonFamily))

    def test_the_service_family_is_exactly_the_service_wording(
        self,
    ) -> None:
        # The two classifications answer different questions and must
        # not disagree: a request told "the service is unavailable"
        # cannot be counted as a knowledge gap.
        for code in ReasonCode:
            with self.subTest(reason=code):
                self.assertEqual(
                    reason_family(code) is ReasonFamily.SERVICE_FAILURE,
                    is_service_failure(code),
                )

    def test_the_blocked_reasons_are_the_input_family(self) -> None:
        for code in TestServiceUnavailableMapping.BLOCKED_REASONS:
            with self.subTest(reason=code):
                self.assertIs(
                    reason_family(code),
                    ReasonFamily.SECURITY_OR_INVALID_INPUT,
                )

    def test_an_unknown_code_is_reported_as_unknown(self) -> None:
        # Not bucketed into a default: a silent default is how a code
        # added later disappears from the counts it should raise.
        self.assertIsNone(reason_family(None))
        self.assertIsNone(reason_family("some_future_reason"))


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
