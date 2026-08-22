"""Unit tests for environment-driven configuration parsing and validation.

The reload-based tests re-import ``src.config`` under a patched
environment because validation runs at import time by design: a bad value
must stop the process before any pipeline component can use it.
"""

import importlib
import os
import unittest
from unittest import mock

import src.config as config_module

# Every reload-sensitive variable is pinned explicitly so a developer's
# local .env can never change what these tests observe.
_BASELINE_ENV = {
    "MODEL_NAME": "",
    "REWRITE_FLOOR": "0.10",
    "DIRECT_ANSWER_THRESHOLD": "0.30",
    "FINAL_ANSWER_THRESHOLD": "0.30",
    "TOP_K": "3",
    "MAX_QUERY_CHARS": "500",
    "CORPUS_DIR": "data/docs",
    "ENABLE_OPS_VIEW": "false",
}


class TestEnvHelpers(unittest.TestCase):
    """The shared read-one-setting helpers define the parsing contract."""

    def test_unset_and_blank_values_fall_back_to_the_default(self) -> None:
        with mock.patch.dict(os.environ, {"EXAMPLE_SETTING": "   "}):
            self.assertEqual(
                config_module._env_str("EXAMPLE_SETTING", "fallback"),
                "fallback",
            )
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                config_module._env_str("EXAMPLE_SETTING", "fallback"),
                "fallback",
            )

    def test_set_values_are_stripped_not_replaced(self) -> None:
        with mock.patch.dict(os.environ, {"EXAMPLE_SETTING": "  value  "}):
            self.assertEqual(
                config_module._env_str("EXAMPLE_SETTING", "fallback"),
                "value",
            )

    def test_numeric_helpers_parse_valid_input(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"EXAMPLE_FLOAT": "0.25", "EXAMPLE_INT": "7"},
        ):
            self.assertEqual(
                config_module._env_float("EXAMPLE_FLOAT", "0.0"), 0.25
            )
            self.assertEqual(config_module._env_int("EXAMPLE_INT", "0"), 7)

    def test_malformed_float_error_names_the_variable_and_value(self) -> None:
        with mock.patch.dict(os.environ, {"EXAMPLE_FLOAT": "abc"}):
            with self.assertRaisesRegex(
                ValueError, "EXAMPLE_FLOAT.*'abc'"
            ):
                config_module._env_float("EXAMPLE_FLOAT", "0.0")

    def test_malformed_int_error_names_the_variable_and_value(self) -> None:
        with mock.patch.dict(os.environ, {"EXAMPLE_INT": "1.5"}):
            with self.assertRaisesRegex(ValueError, "EXAMPLE_INT.*'1.5'"):
                config_module._env_int("EXAMPLE_INT", "0")

    def test_flag_helper_accepts_the_usual_boolean_spellings(self) -> None:
        for literal, expected in (
            ("true", True),
            ("TRUE", True),
            ("1", True),
            ("on", True),
            ("false", False),
            ("0", False),
            ("off", False),
        ):
            with self.subTest(literal=literal):
                with mock.patch.dict(os.environ, {"EXAMPLE_FLAG": literal}):
                    self.assertIs(
                        config_module._env_flag("EXAMPLE_FLAG", "false"),
                        expected,
                    )

    def test_unknown_flag_value_is_rejected_rather_than_assumed(
        self,
    ) -> None:
        # A privacy flag that silently resolves a typo to the permissive
        # value is worse than one that refuses to start.
        with mock.patch.dict(os.environ, {"EXAMPLE_FLAG": "treu"}):
            with self.assertRaisesRegex(ValueError, "EXAMPLE_FLAG.*'treu'"):
                config_module._env_flag("EXAMPLE_FLAG", "false")


class TestCredentialPresence(unittest.TestCase):
    """Secrets are reported by presence only, and read at call time."""

    def test_missing_and_blank_keys_report_as_unconfigured(self) -> None:
        for value in ("", "   "):
            with self.subTest(value=repr(value)):
                with mock.patch.dict(
                    os.environ, {"OPENAI_API_KEY": value}
                ):
                    self.assertFalse(config_module.has_llm_credential())
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(config_module.has_llm_credential())

    def test_configured_key_reports_as_configured(self) -> None:
        with mock.patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test-not-a-real-key"}
        ):
            self.assertTrue(config_module.has_llm_credential())

    def test_the_module_exposes_no_credential_value(self) -> None:
        # Presence is all the pipeline needs; a module-level constant
        # holding the value could be logged or rendered by accident.
        secret = "sk-test-not-a-real-key"
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": secret}):
            values = [
                getattr(config_module, name)
                for name in dir(config_module)
                if not name.startswith("__")
            ]

        strings = [value for value in values if isinstance(value, str)]
        self.assertNotIn(secret, strings)


class TestImportTimeValidation(unittest.TestCase):
    """Malformed settings must stop the import, not linger until runtime."""

    def _reload_expecting_error(self, overrides: dict[str, str], match: str) -> None:
        """Reload the module under a patched environment, expecting failure."""
        env = {**_BASELINE_ENV, **overrides}
        self.addCleanup(self._restore_module)
        with mock.patch.dict(os.environ, env):
            with self.assertRaisesRegex(ValueError, match):
                importlib.reload(config_module)

    @staticmethod
    def _restore_module() -> None:
        """Re-import with the real environment so later tests see sane values."""
        importlib.reload(config_module)

    def test_threshold_above_one_is_rejected(self) -> None:
        self._reload_expecting_error(
            {"DIRECT_ANSWER_THRESHOLD": "1.5"},
            "DIRECT_ANSWER_THRESHOLD",
        )

    def test_negative_threshold_is_rejected(self) -> None:
        self._reload_expecting_error(
            {"REWRITE_FLOOR": "-0.1"},
            "REWRITE_FLOOR",
        )

    def test_rewrite_floor_above_direct_threshold_is_rejected(self) -> None:
        self._reload_expecting_error(
            {"REWRITE_FLOOR": "0.9", "DIRECT_ANSWER_THRESHOLD": "0.2"},
            "REWRITE_FLOOR must not exceed",
        )

    def test_final_threshold_below_direct_threshold_is_rejected(
        self,
    ) -> None:
        # The expanded score is max-pooled over the original query plus
        # its rewrites, so it is never below the raw score. A final gate
        # under the direct gate would therefore let the rewrite branch
        # answer a query the direct branch already refused on the very
        # same number -- and the graph relies on the opposite to skip an
        # expansion that cannot change the verdict.
        self._reload_expecting_error(
            {
                "DIRECT_ANSWER_THRESHOLD": "0.30",
                "FINAL_ANSWER_THRESHOLD": "0.20",
            },
            "FINAL_ANSWER_THRESHOLD must not sit below",
        )

    def test_ambiguous_floor_at_the_match_threshold_is_rejected(
        self,
    ) -> None:
        # The under-specified band lies between the two: at or above the
        # match threshold a topic is resolved rather than ambiguous, so
        # the band would be empty and the reason code unreachable.
        self._reload_expecting_error(
            {
                "SCOPE_MATCH_THRESHOLD": "0.40",
                "SCOPE_AMBIGUOUS_MIN_SCORE": "0.40",
            },
            "SCOPE_AMBIGUOUS_MIN_SCORE must sit below",
        )

    def test_ambiguous_floor_outside_the_score_range_is_rejected(
        self,
    ) -> None:
        self._reload_expecting_error(
            {"SCOPE_AMBIGUOUS_MIN_SCORE": "-0.1"},
            "SCOPE_AMBIGUOUS_MIN_SCORE",
        )

    def test_non_positive_request_deadline_is_rejected(self) -> None:
        # A zero deadline would skip both boundaries of every request,
        # turning a misconfiguration into a service that answers nothing.
        self._reload_expecting_error(
            {"REQUEST_DEADLINE_SECONDS": "0"}, "REQUEST_DEADLINE_SECONDS"
        )

    def test_non_positive_top_k_is_rejected(self) -> None:
        self._reload_expecting_error({"TOP_K": "0"}, "TOP_K")

    def test_non_positive_max_query_chars_is_rejected(self) -> None:
        self._reload_expecting_error(
            {"MAX_QUERY_CHARS": "-5"}, "MAX_QUERY_CHARS"
        )

    def test_unparseable_ops_view_flag_is_rejected(self) -> None:
        self._reload_expecting_error(
            {"ENABLE_OPS_VIEW": "maybe"}, "ENABLE_OPS_VIEW"
        )

    def test_ops_view_defaults_to_off_when_unset(self) -> None:
        self.addCleanup(self._restore_module)
        environment = {**_BASELINE_ENV, "ENABLE_OPS_VIEW": ""}
        with mock.patch.dict(os.environ, environment):
            module = importlib.reload(config_module)
            self.assertFalse(module.ENABLE_OPS_VIEW)

    def test_relative_corpus_dir_is_anchored_to_the_project_root(self) -> None:
        self.addCleanup(self._restore_module)
        with mock.patch.dict(
            os.environ, {**_BASELINE_ENV, "CORPUS_DIR": "data/docs"}
        ):
            module = importlib.reload(config_module)
            expected_root = module._PROJECT_ROOT
            self.assertEqual(
                module.CORPUS_DIR, str(expected_root / "data" / "docs")
            )


if __name__ == "__main__":
    unittest.main()
