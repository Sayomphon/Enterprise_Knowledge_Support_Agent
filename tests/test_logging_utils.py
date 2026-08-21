"""Unit tests for the JSONL telemetry writer and its bounded reader.

Every test writes into a temporary directory: the real ``logs/`` sink is
runtime data and must never be touched by the suite (AGENTS.md section 9).
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import config
from src.logging_utils import (
    log_fallback_event,
    read_persistent_events,
    read_recent_events,
)

QUERY = "ขั้นตอนการเบิกค่าแท็กซี่หลังทำ OT ต้องทำอย่างไร"
REASON = "low_retrieval_score"
LOG_SCHEMA_KEYS = [
    "timestamp",
    "query",
    "reason",
    "raw_retrieval_score",
    "expanded_retrieval_score",
    "top_sources",
    "rewritten_queries",
    "alias_query_count",
]


def _fixed_clock() -> datetime:
    """Return a fixed timezone-aware timestamp for reproducible records."""
    return datetime(2026, 8, 21, 9, 30, tzinfo=timezone(timedelta(hours=7)))


class TestLogWrite(unittest.TestCase):
    """Appending one event, and what the caller learns about it."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)
        self.log_path = self.directory / "fallback_queries.jsonl"

    def _append(self, **overrides: object):
        """Append one event with the shared defaults."""
        arguments: dict[str, object] = {
            "query": QUERY,
            "reason": REASON,
            "raw_retrieval_score": 0.0706,
            "expanded_retrieval_score": None,
            "top_sources": ["FIN-001"],
            "rewritten_queries": [],
            "log_path": self.log_path,
            "now": _fixed_clock,
        }
        arguments.update(overrides)
        return log_fallback_event(**arguments)

    def _records(self) -> list[dict[str, object]]:
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def test_successful_write_reports_success_and_writes_one_line(
        self,
    ) -> None:
        result = self._append()

        self.assertTrue(result.ok)
        self.assertIsNone(result.error_type)
        self.assertEqual(len(self._records()), 1)

    def test_record_follows_the_schema_with_a_timezone_aware_timestamp(
        self,
    ) -> None:
        self._append()

        record = self._records()[0]
        self.assertEqual(list(record.keys()), LOG_SCHEMA_KEYS)
        self.assertIsNotNone(
            datetime.fromisoformat(str(record["timestamp"])).tzinfo
        )

    def test_appending_never_rewrites_earlier_records(self) -> None:
        self._append()
        self._append(reason="prompt_injection")

        reasons = [record["reason"] for record in self._records()]
        self.assertEqual(reasons, [REASON, "prompt_injection"])

    def test_missing_parent_directory_is_created(self) -> None:
        self.log_path = self.directory / "nested" / "fallback.jsonl"

        self.assertTrue(self._append().ok)

    def test_unwritable_path_reports_failure_without_raising(self) -> None:
        # A path whose parent is a regular file cannot be created, which
        # forces the append to fail the way a full or read-only disk does.
        blocker = self.directory / "occupied"
        blocker.write_text("occupied", encoding="utf-8")
        self.log_path = blocker / "fallback.jsonl"

        with contextlib.redirect_stderr(io.StringIO()):
            result = self._append()

        self.assertFalse(result.ok)
        self.assertIsNotNone(result.error_type)

    def test_unserialisable_query_reports_failure_without_raising(
        self,
    ) -> None:
        # screen_query rejects a non-string, but the refusal path still
        # hands it to this writer. json.dumps answers with TypeError, and
        # a best-effort writer must absorb that like any filesystem error
        # rather than let it kill the request.
        result = self._append(query={"not": "serialisable"})

        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "TypeError")

    def test_lone_surrogate_reports_failure_without_raising(self) -> None:
        # Arrives from argv under surrogateescape; utf-8 cannot encode it.
        result = self._append(query="taxi \ud800 claim")

        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "UnicodeEncodeError")

    def test_oversized_query_is_truncated_and_marked(self) -> None:
        # The sink is append-only with no rotation, and a query rejected
        # for being too long is still logged, so the bound lives here.
        self._append(query="ก" * 300_000)

        logged = self._records()[0]["query"]
        self.assertLess(len(logged), 2_000)
        self.assertTrue(logged.endswith("...[truncated]"))

    def test_query_within_the_bound_is_stored_verbatim(self) -> None:
        result = self._append(query=QUERY)

        self.assertTrue(result.ok)
        self.assertEqual(self._records()[0]["query"], QUERY)

    def test_failure_output_names_the_exception_type_only(self) -> None:
        blocker = self.directory / "occupied"
        blocker.write_text("occupied", encoding="utf-8")
        self.log_path = blocker / "fallback.jsonl"

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = self._append()

        message = stderr.getvalue()
        self.assertIn("logging_utils", message)
        self.assertIn(str(result.error_type), message)
        # Neither the destination nor the employee's question may appear
        # in an error line that a shared terminal will keep.
        self.assertNotIn(str(self.log_path), message)
        self.assertNotIn(QUERY, message)
        self.assertNotIn(str(self.log_path), str(result))

    def test_only_allowlisted_fields_can_reach_the_sink(self) -> None:
        # The keyword-only signature is the enforcement: a prompt, an
        # answer, or a provider payload has no parameter to arrive on.
        with self.assertRaises(TypeError):
            log_fallback_event(
                query=QUERY,
                reason=REASON,
                raw_retrieval_score=None,
                expanded_retrieval_score=None,
                top_sources=[],
                rewritten_queries=[],
                system_prompt="never",
                log_path=self.log_path,
            )


class TestBoundedLogRead(unittest.TestCase):
    """Reading back the newest rows without loading the whole sink."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log_path = Path(tmp.name) / "fallback_queries.jsonl"

    def _write_events(self, count: int) -> None:
        """Write ``count`` well-formed records numbered from zero."""
        lines = [
            json.dumps({"query": f"q{index}", "reason": REASON})
            for index in range(count)
        ]
        self.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_missing_sink_reads_as_empty_rather_than_failing(self) -> None:
        result = read_recent_events(10, log_path=self.log_path)

        self.assertEqual(result.records, ())
        self.assertEqual(result.skipped_lines, 0)

    def test_reader_returns_the_newest_rows_oldest_first(self) -> None:
        self._write_events(20)

        result = read_recent_events(5, log_path=self.log_path)

        self.assertEqual(
            [record["query"] for record in result.records],
            ["q15", "q16", "q17", "q18", "q19"],
        )

    def test_reader_returns_everything_when_the_sink_is_smaller(
        self,
    ) -> None:
        self._write_events(3)

        result = read_recent_events(50, log_path=self.log_path)

        self.assertEqual(len(result.records), 3)
        self.assertEqual(result.records[0]["query"], "q0")

    def test_reader_spans_more_than_one_backward_block(self) -> None:
        # Forces several backward reads so the block-boundary handling is
        # exercised rather than assumed.
        import src.logging_utils as logging_utils

        original = logging_utils._TAIL_BLOCK_BYTES
        logging_utils._TAIL_BLOCK_BYTES = 32
        self.addCleanup(
            setattr, logging_utils, "_TAIL_BLOCK_BYTES", original
        )
        self._write_events(40)

        result = read_recent_events(4, log_path=self.log_path)

        self.assertEqual(
            [record["query"] for record in result.records],
            ["q36", "q37", "q38", "q39"],
        )

    def test_partial_final_line_is_counted_not_rendered(self) -> None:
        self._write_events(3)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write('{"query": "interrupted appen')

        result = read_recent_events(10, log_path=self.log_path)

        self.assertEqual(len(result.records), 3)
        self.assertEqual(result.skipped_lines, 1)
        self.assertNotIn(
            "interrupted", json.dumps(result.records, ensure_ascii=False)
        )

    def test_malformed_middle_line_is_skipped_without_losing_the_rest(
        self,
    ) -> None:
        self.log_path.write_text(
            '{"query": "q0"}\nnot json at all\n{"query": "q1"}\n',
            encoding="utf-8",
        )

        result = read_recent_events(10, log_path=self.log_path)

        self.assertEqual(
            [record["query"] for record in result.records], ["q0", "q1"]
        )
        self.assertEqual(result.skipped_lines, 1)

    def test_non_object_line_is_skipped(self) -> None:
        # Valid JSON is not automatically a record; the audit table can
        # only render objects.
        self.log_path.write_text(
            '{"query": "q0"}\n[1, 2, 3]\n', encoding="utf-8"
        )

        result = read_recent_events(10, log_path=self.log_path)

        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.skipped_lines, 1)

    def test_non_positive_limit_reads_nothing(self) -> None:
        self._write_events(5)

        self.assertEqual(
            read_recent_events(0, log_path=self.log_path).records, ()
        )

    def test_written_events_are_readable_by_the_bounded_reader(self) -> None:
        # Writer and reader share one file format; asserting them
        # together is what keeps that true.
        for index in range(3):
            log_fallback_event(
                query=f"{QUERY} {index}",
                reason=REASON,
                raw_retrieval_score=0.05,
                expanded_retrieval_score=None,
                top_sources=["FIN-001"],
                rewritten_queries=[],
                log_path=self.log_path,
                now=_fixed_clock,
            )

        result = read_recent_events(2, log_path=self.log_path)

        self.assertEqual(len(result.records), 2)
        self.assertEqual(result.records[-1]["query"], f"{QUERY} 2")


class TestPersistentViewGate(unittest.TestCase):
    """The sink is not read at all unless the demo flag is enabled."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log_path = Path(tmp.name) / "fallback_queries.jsonl"
        self.log_path.write_text(
            json.dumps({"query": QUERY, "reason": REASON}) + "\n",
            encoding="utf-8",
        )

    def _with_flag(self, enabled: bool):
        """Patch the demo flag for one test."""
        patcher = unittest.mock.patch.object(
            config, "ENABLE_OPS_VIEW", enabled
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_flag_is_off_by_default(self) -> None:
        # The committed default keeps past sessions' questions out of the
        # running app; enabling it is a deliberate local act.
        self.assertFalse(config._env_flag("ENABLE_OPS_VIEW", "false"))

    def test_disabled_flag_reads_nothing_from_an_existing_sink(
        self,
    ) -> None:
        self._with_flag(False)

        result = read_persistent_events(50, log_path=self.log_path)

        self.assertEqual(result.records, ())
        self.assertEqual(result.skipped_lines, 0)

    def test_enabled_flag_reads_the_bounded_window(self) -> None:
        self._with_flag(True)

        result = read_persistent_events(50, log_path=self.log_path)

        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0]["query"], QUERY)

    def test_disabled_flag_never_opens_the_sink(self) -> None:
        # "Read then hide" would still pull other sessions' questions
        # into this process; the gate must be before the read.
        self._with_flag(False)
        with unittest.mock.patch(
            "src.logging_utils.read_recent_events"
        ) as reader:
            read_persistent_events(50, log_path=self.log_path)

        self.assertEqual(reader.call_count, 0)


class TestRepositoryHygiene(unittest.TestCase):
    """The sink and the secrets file stay out of version control."""

    def test_env_file_and_jsonl_logs_are_git_ignored(self) -> None:
        ignore_rules = (
            Path(__file__).resolve().parents[1] / ".gitignore"
        ).read_text(encoding="utf-8")

        self.assertIn(".env", ignore_rules)
        self.assertIn("logs/*.jsonl", ignore_rules)


if __name__ == "__main__":
    unittest.main()
