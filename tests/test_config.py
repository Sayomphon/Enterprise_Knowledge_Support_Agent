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

    def test_non_positive_top_k_is_rejected(self) -> None:
        self._reload_expecting_error({"TOP_K": "0"}, "TOP_K")

    def test_non_positive_max_query_chars_is_rejected(self) -> None:
        self._reload_expecting_error(
            {"MAX_QUERY_CHARS": "-5"}, "MAX_QUERY_CHARS"
        )

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
