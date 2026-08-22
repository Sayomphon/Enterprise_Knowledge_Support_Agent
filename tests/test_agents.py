"""Unit tests for the shared LLM client boundary.

The credential check lives here, at the boundary, so the deterministic
routes keep working without a key (remediation plan Finding 8). No client
is ever constructed in these tests: the provider constructor is patched.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from src.agents import (
    MissingLlmCredentialError,
    _effective_timeout,
    credential_fingerprint,
    get_llm,
)
from src.config import LLM_TIMEOUT_SECONDS

KEY_VARIABLE = "OPENAI_API_KEY"
FAKE_KEY = "sk-test-not-a-real-key"


class TestBoundaryBudget(unittest.TestCase):
    """A boundary may never be given more time than the request has left."""

    def test_no_budget_leaves_the_configured_timeout_alone(self) -> None:
        self.assertEqual(_effective_timeout(30.0, None), 30.0)

    def test_a_smaller_budget_replaces_the_configured_timeout(self) -> None:
        self.assertEqual(_effective_timeout(30.0, 12.7), 12.0)

    def test_a_larger_budget_does_not_extend_the_boundary(self) -> None:
        self.assertEqual(_effective_timeout(10.0, 40.0), 10.0)

    def test_whole_seconds_keep_the_client_cache_small(self) -> None:
        # Requests differing by milliseconds must share one cached
        # client rather than each building their own.
        self.assertEqual(
            _effective_timeout(30.0, 12.10), _effective_timeout(30.0, 12.99)
        )

    def test_a_vanishing_budget_never_becomes_an_impossible_timeout(
        self,
    ) -> None:
        # The graph skips a boundary whose budget is gone; if one is
        # reached anyway, a sub-second timeout would fail every call.
        self.assertEqual(_effective_timeout(30.0, 0.4), 1.0)

    def test_the_budget_reaches_the_constructed_client(self) -> None:
        with mock.patch("src.agents.ChatOpenAI") as client:
            with mock.patch.dict(os.environ, {KEY_VARIABLE: FAKE_KEY}):
                import src.agents as agents

                agents._build_llm.cache_clear()
                self.addCleanup(agents._build_llm.cache_clear)
                get_llm(budget_seconds=8.0)

        self.assertEqual(client.call_args.kwargs["timeout"], 8.0)
        self.assertLess(
            client.call_args.kwargs["timeout"], LLM_TIMEOUT_SECONDS
        )


class TestCredentialBoundary(unittest.TestCase):
    """Missing credentials are refused before a client exists."""

    def setUp(self) -> None:
        # The builder is cached per model name; a client cached by another
        # test must not decide this one's outcome.
        import src.agents as agents

        agents._build_llm.cache_clear()
        self.addCleanup(agents._build_llm.cache_clear)

    def test_missing_key_raises_the_narrow_credential_error(self) -> None:
        with mock.patch.dict(os.environ, {KEY_VARIABLE: ""}):
            with self.assertRaises(MissingLlmCredentialError):
                get_llm()

    def test_blank_key_is_treated_as_missing(self) -> None:
        with mock.patch.dict(os.environ, {KEY_VARIABLE: "   "}):
            with self.assertRaises(MissingLlmCredentialError):
                get_llm()

    def test_no_client_is_constructed_without_a_credential(self) -> None:
        with mock.patch("src.agents.ChatOpenAI") as client:
            with mock.patch.dict(os.environ, {KEY_VARIABLE: ""}):
                with self.assertRaises(MissingLlmCredentialError):
                    get_llm()

        self.assertEqual(client.call_count, 0)

    def test_rotated_credential_retires_the_cached_client(self) -> None:
        # The credential check alone only detects a key being REMOVED. A
        # rotated or revoked key leaves has_llm_credential() true, so a
        # cache keyed on the model name alone kept handing back a client
        # built with the dead secret and every request degraded as
        # reporter_failure until the process restarted.
        # The patched constructor returns one shared mock, so identity
        # cannot distinguish the two clients: what is under test is
        # whether a SECOND client was constructed at all.
        with mock.patch("src.agents.ChatOpenAI") as client:
            with mock.patch.dict(os.environ, {KEY_VARIABLE: "sk-old-value"}):
                get_llm("gpt-4o-mini")
            with mock.patch.dict(os.environ, {KEY_VARIABLE: "sk-new-value"}):
                get_llm("gpt-4o-mini")

        self.assertEqual(client.call_count, 2)

    def test_same_credential_still_reuses_one_client(self) -> None:
        with mock.patch("src.agents.ChatOpenAI") as client:
            with mock.patch.dict(os.environ, {KEY_VARIABLE: "sk-value"}):
                get_llm("gpt-4o-mini")
                get_llm("gpt-4o-mini")

        self.assertEqual(client.call_count, 1)

    def test_the_fingerprint_never_contains_the_credential(self) -> None:
        secret = "sk-a-very-secret-value"
        with mock.patch.dict(os.environ, {KEY_VARIABLE: secret}):
            fingerprint = credential_fingerprint()

        self.assertTrue(fingerprint)
        self.assertNotIn(secret, fingerprint)
        self.assertNotIn(secret[3:], fingerprint)

    def test_error_text_names_the_variable_but_never_a_value(self) -> None:
        with mock.patch.dict(os.environ, {KEY_VARIABLE: ""}):
            with self.assertRaises(MissingLlmCredentialError) as caught:
                get_llm()

        message = str(caught.exception)
        self.assertIn(KEY_VARIABLE, message)
        self.assertNotIn(FAKE_KEY, message)

    def test_configured_key_builds_one_client_per_model_name(self) -> None:
        with mock.patch("src.agents.ChatOpenAI") as client:
            with mock.patch.dict(os.environ, {KEY_VARIABLE: FAKE_KEY}):
                first = get_llm("gpt-5-mini")
                second = get_llm("gpt-5-mini")
                get_llm("gpt-4o-mini")

        self.assertIs(first, second)
        self.assertEqual(client.call_count, 2)

    def test_credential_state_is_re_read_on_every_call(self) -> None:
        # A cached client must not outlive the credential it was built
        # with, or the boundary stops reporting the current environment.
        with mock.patch("src.agents.ChatOpenAI"):
            with mock.patch.dict(os.environ, {KEY_VARIABLE: FAKE_KEY}):
                get_llm()
            with mock.patch.dict(os.environ, {KEY_VARIABLE: ""}):
                with self.assertRaises(MissingLlmCredentialError):
                    get_llm()

    def test_gpt5_models_are_built_without_a_temperature_argument(
        self,
    ) -> None:
        # gpt-5 family models accept only the default temperature.
        with mock.patch("src.agents.ChatOpenAI") as client:
            with mock.patch.dict(os.environ, {KEY_VARIABLE: FAKE_KEY}):
                get_llm("gpt-5-mini")

        self.assertNotIn("temperature", client.call_args.kwargs)

    def test_other_models_are_built_with_the_configured_temperature(
        self,
    ) -> None:
        with mock.patch("src.agents.ChatOpenAI") as client:
            with mock.patch.dict(os.environ, {KEY_VARIABLE: FAKE_KEY}):
                get_llm("gpt-4o-mini")

        self.assertIn("temperature", client.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
