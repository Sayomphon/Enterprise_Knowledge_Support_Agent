"""Unit tests for the JSONL telemetry writer and its bounded reader.

Every test writes into a temporary directory: the real ``logs/`` sink is
runtime data and must never be touched by the suite (AGENTS.md section 9).
"""

from __future__ import annotations

import contextlib
import io
import json
import stat
import subprocess
import tempfile
import threading
import time
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import config
from src.logging_utils import (
    log_fallback_event,
    read_persistent_events,
    read_recent_events,
    redact_query,
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
    "scope_topics",
    "scope_reason",
    "latency_ms",
    "llm_calls",
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

    def test_scope_fields_default_to_an_absent_verdict(self) -> None:
        # A blocked request never reaches the scope gate, so its record
        # says "no verdict" rather than omitting the keys: a JSONL
        # reader should not have to distinguish a missing key from a
        # gate that ran and resolved nothing.
        self._append()

        record = self._records()[0]
        self.assertEqual(record["scope_topics"], [])
        self.assertIsNone(record["scope_reason"])

    def test_the_scope_verdict_is_recorded_beside_the_reason(self) -> None:
        # The pair this field exists for: an in-domain question stopped
        # by its raw score keeps `low_retrieval_score` as the reason,
        # and the topic it did resolve is still readable.
        self._append(
            reason="low_retrieval_score",
            scope_topics=["annual_leave", "sick_leave"],
            scope_reason="unsupported_topic",
        )

        record = self._records()[0]
        self.assertEqual(record["reason"], "low_retrieval_score")
        self.assertEqual(
            record["scope_topics"], ["annual_leave", "sick_leave"]
        )
        self.assertEqual(record["scope_reason"], "unsupported_topic")

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


class TestQueryRedaction(unittest.TestCase):
    """What the sink may keep of an employee's own words.

    The file holds raw questions across every session, and a question
    about sick leave or a reimbursement can carry an identity number, a
    bank account, a phone number or an address. Redaction runs at the
    writer because that is the single choke point into the sink
    (AGENTS.md section 7, "Log privacy").
    """

    def test_a_thai_national_id_never_reaches_the_record(self) -> None:
        redacted = redact_query("เลขบัตร 1234567890123 ใช้เบิกได้ไหม")

        self.assertNotIn("1234567890123", redacted)
        self.assertIn("[ID]", redacted)

    def test_a_bank_account_number_is_replaced(self) -> None:
        redacted = redact_query("โอนเข้าบัญชี 1234567890 ได้ไหม")

        self.assertNotIn("1234567890", redacted)
        self.assertIn("[ACCT]", redacted)

    def test_a_mobile_number_is_replaced(self) -> None:
        redacted = redact_query("ติดต่อกลับที่ 0812345678 ได้ไหม")

        self.assertNotIn("0812345678", redacted)
        self.assertIn("[TEL]", redacted)

    def test_an_email_address_is_replaced(self) -> None:
        redacted = redact_query("ส่งใบเสร็จไปที่ somchai.k@example.co.th")

        self.assertNotIn("somchai.k@example.co.th", redacted)
        self.assertIn("[EMAIL]", redacted)

    def test_an_identifier_in_thai_digits_is_masked_too(self) -> None:
        # Python's \d is Unicode-aware, so the length rules cover an id
        # typed in Thai or fullwidth digits without the text being folded
        # first -- folding would store a question nobody asked.
        for digits in ("๑๒๓๔๕๖๗๘๙๐๑๒๓", "１２３４５６７８９０１２３"):
            with self.subTest(digits=digits):
                self.assertEqual(
                    redact_query(f"เลขบัตร {digits} ใช้เบิกได้ไหม"),
                    "เลขบัตร [ID] ใช้เบิกได้ไหม",
                )

    def test_a_thai_digit_phone_number_is_masked_as_an_account(
        self,
    ) -> None:
        # The phone rule anchors on a literal ASCII zero, so this one
        # falls to the account rule: the wrong label, but not in the
        # clear. Asserted rather than left to be discovered.
        self.assertEqual(
            redact_query("ติดต่อกลับที่ ๐๘๑๒๓๔๕๖๗๘ ได้ไหม"),
            "ติดต่อกลับที่ [ACCT] ได้ไหม",
        )

    def test_amounts_in_thai_digits_are_left_alone(self) -> None:
        self.assertEqual(
            redact_query("เบิก ๕๐๐ บาท ลา ๑๐ วัน"), "เบิก ๕๐๐ บาท ลา ๑๐ วัน"
        )

    def test_an_ordinary_question_is_left_alone(self) -> None:
        # Over-redaction has a cost of its own: the sink exists to debug
        # retrieval, and a record whose amounts and deadlines have been
        # masked cannot do that. Amounts, day counts and document ids
        # are the whole subject of these questions.
        for query in (
            "เบิกค่าแท็กซี่ 500 บาท ต้องใช้ใบเสร็จไหม",
            "ลาพักร้อนได้กี่วันต่อปี ต้องแจ้งล่วงหน้า 3 วันไหม",
            "ยื่นเบิกภายใน 30 วันตามเอกสาร FIN-001 ใช่ไหม",
            "How many annual leave days do I get?",
        ):
            with self.subTest(query=query):
                self.assertEqual(redact_query(query), query)

    def test_redaction_keeps_the_surrounding_question_readable(self) -> None:
        redacted = redact_query("เบอร์ 0812345678 เบิกค่าแท็กซี่ 500 บาท")

        self.assertIn("เบิกค่าแท็กซี่ 500 บาท", redacted)

    def test_the_written_record_carries_the_redacted_query(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log_path = Path(tmp.name) / "fallback_queries.jsonl"

        log_fallback_event(
            query="เลขบัตร 1234567890123 อีเมล somchai@example.com",
            reason=REASON,
            raw_retrieval_score=None,
            expanded_retrieval_score=None,
            top_sources=[],
            rewritten_queries=["เลขบัตร 1234567890123 ขอเบิก"],
            log_path=log_path,
            now=_fixed_clock,
        )

        written = log_path.read_text(encoding="utf-8")
        self.assertNotIn("1234567890123", written)
        self.assertNotIn("somchai@example.com", written)
        # A rewrite is the same question in the model's words, so it can
        # carry the same identifiers the employee typed.
        self.assertIn("[ID]", json.loads(written)["rewritten_queries"][0])

    def test_redaction_stays_bounded_on_adversarial_input(self) -> None:
        # Same ReDoS discipline as the guardrail: bounded quantifiers,
        # asserted against input built to make a greedy pattern backtrack
        # (AGENTS.md section 7).
        hostile = ("9" * 5_000 + "@" + "a." * 2_000) * 5

        started = time.perf_counter()
        redact_query(hostile)

        self.assertLess(time.perf_counter() - started, 1.0)


class TestSinkPermissions(unittest.TestCase):
    """The sink is private to the user the process runs as."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name) / "logs"
        self.log_path = self.directory / "fallback_queries.jsonl"

    def _append(self) -> None:
        log_fallback_event(
            query=QUERY,
            reason=REASON,
            raw_retrieval_score=None,
            expanded_retrieval_score=None,
            top_sources=[],
            rewritten_queries=[],
            log_path=self.log_path,
            now=_fixed_clock,
        )

    def _mode(self, path: Path) -> int:
        return stat.S_IMODE(path.stat().st_mode)

    def test_a_new_sink_is_readable_only_by_its_owner(self) -> None:
        self._append()

        self.assertEqual(self._mode(self.log_path), 0o600)

    def test_a_new_log_directory_is_private_too(self) -> None:
        self._append()

        self.assertEqual(self._mode(self.directory), 0o700)

    def test_a_world_readable_sink_is_narrowed_on_the_next_write(
        self,
    ) -> None:
        # The committed default used to be 0644, so an existing sink has
        # to be corrected rather than only new ones created correctly.
        self.directory.mkdir(parents=True)
        self.log_path.touch(mode=0o644)

        self._append()

        self.assertEqual(self._mode(self.log_path), 0o600)


class TestSinkRotation(unittest.TestCase):
    """Retention: an append-only file with no ceiling is not a policy."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log_path = Path(tmp.name) / "fallback_queries.jsonl"
        self._configure(max_bytes=400, backups=2)

    def _configure(self, *, max_bytes: int, backups: int) -> None:
        for name, value in (
            ("LOG_MAX_BYTES", max_bytes),
            ("LOG_BACKUP_COUNT", backups),
        ):
            patch = unittest.mock.patch.object(config, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def _append(self, query: str = QUERY) -> None:
        log_fallback_event(
            query=query,
            reason=REASON,
            raw_retrieval_score=None,
            expanded_retrieval_score=None,
            top_sources=[],
            rewritten_queries=[],
            log_path=self.log_path,
            now=_fixed_clock,
        )

    def _backup(self, index: int) -> Path:
        return self.log_path.with_name(f"{self.log_path.name}.{index}")

    def test_a_full_sink_rolls_over_before_the_next_record(self) -> None:
        for _ in range(4):
            self._append()

        self.assertTrue(self._backup(1).exists())
        self.assertGreaterEqual(self._backup(1).stat().st_size, 400)

    def test_the_newest_record_lands_in_the_live_sink(self) -> None:
        for _ in range(4):
            self._append()
        self._append(query="คำถามล่าสุด")

        self.assertIn(
            "คำถามล่าสุด", self.log_path.read_text(encoding="utf-8")
        )

    def test_retention_drops_the_oldest_page(self) -> None:
        for index in range(24):
            self._append(query=f"{QUERY} {index}")

        self.assertTrue(self._backup(2).exists())
        self.assertFalse(self._backup(3).exists())

    def test_a_rotated_sink_stays_private(self) -> None:
        for _ in range(4):
            self._append()

        self.assertEqual(
            stat.S_IMODE(self._backup(1).stat().st_mode), 0o600
        )

    def test_rotation_is_off_when_no_ceiling_is_configured(self) -> None:
        self._configure(max_bytes=0, backups=2)

        for _ in range(6):
            self._append()

        self.assertFalse(self._backup(1).exists())

    def test_every_record_survives_two_concurrent_writers(self) -> None:
        # One record is one write of well under a page, so O_APPEND keeps
        # the lines whole; the assertion is that they all still decode.
        self._configure(max_bytes=0, backups=0)
        barrier = threading.Barrier(2)

        def writer(tag: str) -> None:
            barrier.wait()
            for index in range(20):
                self._append(query=f"{tag}-{index}")

        threads = [
            threading.Thread(target=writer, args=(tag,))
            for tag in ("a", "b")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 40)
        for line in lines:
            json.loads(line)


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

    def test_rotated_pages_are_git_ignored_too(self) -> None:
        # Rotation renames the sink to "....jsonl.1", which the original
        # "logs/*.jsonl" rule does not match -- so adding rotation would
        # otherwise have made a page of real employee questions
        # committable.
        root = Path(__file__).resolve().parents[1]
        page = root / "logs" / "fallback_queries.jsonl.1"

        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", str(page)],
            cwd=root,
            check=False,
        )

        self.assertEqual(ignored.returncode, 0)


if __name__ == "__main__":
    unittest.main()
