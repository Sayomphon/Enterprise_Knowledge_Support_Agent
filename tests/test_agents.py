"""Unit tests for the shared LLM client boundary.

The credential check lives here, at the boundary, so the deterministic
routes keep working without a key (remediation plan Finding 8). No client
is ever constructed in these tests: the provider constructor is patched.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from src.agents import MissingLlmCredentialError, get_llm

KEY_VARIABLE = "OPENAI_API_KEY"
FAKE_KEY = "sk-test-not-a-real-key"


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
