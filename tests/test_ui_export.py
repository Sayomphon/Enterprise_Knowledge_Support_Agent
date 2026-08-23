"""Unit tests for the console's exports and its cross-session view.

The console offers every section's own data in three formats, and reads
requests from every session of the browser tab rather than from the one
the assistant happens to be showing. Both are presentation concerns, so
they are asserted here against the pure builders in ``ui.formatting`` and
against ``ui.runtime`` with a stubbed session state -- no Streamlit
runtime, and no live pipeline.
"""

from __future__ import annotations

import csv
import io
import json
import unittest
from unittest import mock

from src import config
from src.fallback import ReasonCode, ReasonFamily
from ui import runtime
from ui.formatting import (
    _corpus_export_rows,
    _evaluation_export_rows,
    _overview_export_rows,
    _overview_stats,
    _request_key,
    _rows_to_csv,
    _rows_to_jsonl,
    _rows_to_markdown,
    _runtime_export_rows,
    _session_export_rows,
)


def _request(
    request_id: str = "Q-001",
    session_id: str | None = None,
    timestamp: str = "2026-08-23T09:00:05+07:00",
    route: str = "answered",
    **state_overrides,
) -> dict:
    """Build one request record as ``ui.runtime`` stores it.

    A stored record carries no ``session_id``: ``_all_requests`` attaches
    that when it aggregates, so the tests that check aggregation must
    start from a record without it.
    """
    state = {
        "query": "ลาป่วยกี่วันต้องมีใบรับรองแพทย์",
        "route": route,
        "raw_retrieval_score": config.DIRECT_ANSWER_THRESHOLD + 0.05,
        "llm_calls": 1,
    }
    state.update(state_overrides)
    record = {
        "request_id": request_id,
        "timestamp": timestamp,
        "query": state["query"],
        "state": state,
        "latency_seconds": 7.111,
    }
    if session_id is not None:
        record["session_id"] = session_id
    return record


class TestSerialisers(unittest.TestCase):
    """One row shape, three serialisations of it."""

    ROWS = [
        {"a": 1, "b": ["x", "y"]},
        {"a": None, "c": "comma, quote \" and pipe |"},
    ]

    def test_csv_header_is_the_union_of_every_row_key(self) -> None:
        # The exports mix row shapes on purpose -- KPI rows beside triage
        # rows -- so a column only later rows carry must survive.
        reader = csv.reader(io.StringIO(_rows_to_csv(self.ROWS)))
        self.assertEqual(next(reader), ["a", "b", "c"])

    def test_a_list_value_is_joined_and_a_missing_one_is_blank(
        self,
    ) -> None:
        rows = list(csv.DictReader(io.StringIO(_rows_to_csv(self.ROWS))))

        self.assertEqual(rows[0]["b"], "x; y")
        self.assertEqual(rows[1]["a"], "")

    def test_a_comma_or_quote_stays_inside_one_csv_field(self) -> None:
        rows = list(csv.DictReader(io.StringIO(_rows_to_csv(self.ROWS))))

        self.assertEqual(rows[1]["c"], 'comma, quote " and pipe |')

    def test_a_pipe_cannot_split_a_markdown_cell(self) -> None:
        markdown = _rows_to_markdown("Demo", self.ROWS)

        self.assertIn("pipe \\|", markdown)
        for line in markdown.splitlines():
            if line.startswith("|"):
                self.assertEqual(line.count("|") - line.count("\\|"), 4)

    def test_a_formula_shaped_query_cannot_execute_in_a_spreadsheet(
        self,
    ) -> None:
        # Every route is exported, so a blocked injection attempt is what
        # an operator downloads: the cell must reach Excel as text.
        rows = [
            {"query": "=cmd|'/c calc'!A0"},
            {"query": "+1+1"},
            {"query": "@SUM(1,2)"},
            {"query": '=HYPERLINK("http://evil.example","click")'},
        ]
        exported = list(csv.DictReader(io.StringIO(_rows_to_csv(rows))))

        for row in exported:
            self.assertTrue(row["query"].startswith("'"))

    def test_a_negative_reading_stays_a_number(self) -> None:
        # The guard keys on the leading character, so a score or a delta
        # must not be turned into text by it.
        rows = [{"score": -0.42, "delta": "-5"}]
        exported = list(csv.DictReader(io.StringIO(_rows_to_csv(rows))))

        self.assertEqual(exported[0]["score"], "-0.42")
        self.assertEqual(exported[0]["delta"], "-5")

    def test_jsonl_is_one_object_per_row_with_types_kept(self) -> None:
        lines = _rows_to_jsonl(self.ROWS).splitlines()

        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["b"], ["x", "y"])
        self.assertIsNone(json.loads(lines[1])["a"])

    def test_an_empty_export_is_empty_rather_than_a_stray_header(
        self,
    ) -> None:
        self.assertEqual(_rows_to_csv([]), "")
        self.assertEqual(_rows_to_jsonl([]), "")
        self.assertIn("No rows", _rows_to_markdown("Demo", []))


class TestOverviewExport(unittest.TestCase):
    """The overview export states what the overview panel shows."""

    def test_the_stats_count_each_outcome_once(self) -> None:
        requests = [
            _request("Q-001"),
            _request("Q-002", route="rewrite"),
            _request(
                "Q-003",
                route="fallback",
                fallback_reason=ReasonCode.LOW_RETRIEVAL_SCORE.value,
                llm_calls=0,
            ),
            _request(
                "Q-004",
                route="blocked",
                guardrail_reason=ReasonCode.PROMPT_INJECTION.value,
                llm_calls=0,
            ),
        ]

        stats = _overview_stats(requests)

        self.assertEqual(stats["requests"], 4)
        self.assertEqual(stats["answered"], 2)
        self.assertEqual(stats["fallback"], 1)
        self.assertEqual(stats["blocked"], 1)
        self.assertEqual(stats["llm_calls"], 2)
        self.assertEqual(stats["avg_latency_seconds"], 7.111)

    def test_an_empty_tab_reports_no_latency_instead_of_zero(self) -> None:
        # Zero seconds would read as an instant answer rather than as
        # "nothing has been asked yet".
        self.assertIsNone(_overview_stats([])["avg_latency_seconds"])

    def test_the_export_carries_the_cards_and_the_triage_rows(self) -> None:
        stats = _overview_stats([_request()])
        triage = [
            (
                ReasonCode.LOW_RETRIEVAL_SCORE.value,
                2,
                "เบิกตังค่า Taxi ได้ปะ",
                ReasonFamily.KNOWLEDGE_GAP,
            )
        ]

        rows = _overview_export_rows(stats, triage, sessions=3)

        metrics = {row["metric"] for row in rows}
        self.assertIn("sessions", metrics)
        self.assertIn("answered", metrics)
        self.assertIn("avg_latency_seconds", metrics)
        self.assertIn(ReasonCode.LOW_RETRIEVAL_SCORE.value, metrics)
        triage_row = next(
            row for row in rows if row["group"] == "unanswered"
        )
        self.assertEqual(triage_row["value"], 2)
        self.assertEqual(triage_row["family"], ReasonFamily.KNOWLEDGE_GAP.value)


class TestSectionExports(unittest.TestCase):
    """Evaluation, Knowledge Base and Runtime export what they render."""

    def test_evaluation_rows_cover_metrics_and_fixtures(self) -> None:
        groups = [
            {
                "name": "guardrail",
                "cases": None,
                "metrics": [("Injection Block Rate", 28, 28, "")],
                "remarks": [],
            }
        ]
        fixtures = [("guardrail_cases.json", 56, "attack 28 · benign 28")]

        rows = _evaluation_export_rows(groups, fixtures)

        self.assertEqual(rows[0]["group"], "guardrail")
        self.assertEqual((rows[0]["passed"], rows[0]["total"]), (28, 28))
        self.assertEqual(rows[1]["metric"], "guardrail_cases.json")
        self.assertEqual(rows[1]["total"], 56)

    def test_the_corpus_export_lists_the_index_not_the_documents(
        self,
    ) -> None:
        document = mock.Mock(
            source_id="FIN-001",
            title="ขั้นตอนการเบิกค่าใช้จ่าย",
            source_type="policy",
            content="ยื่นเบิกผ่าน Expense Portal ภายใน 30 วัน",
        )

        rows = _corpus_export_rows([document])

        self.assertEqual(set(rows[0]), {"source_id", "title", "source_type"})
        self.assertNotIn("Expense Portal", json.dumps(rows, ensure_ascii=False))

    def test_the_runtime_export_carries_presence_never_a_credential(
        self,
    ) -> None:
        runtime_snapshot = {
            "llm": {"model_name": "gpt-5-mini", "api_key": "configured"},
            "logging": {"events": ["blocked", "fallback"]},
        }

        rows = _runtime_export_rows(runtime_snapshot)

        api_key_row = next(row for row in rows if row["key"] == "api_key")
        self.assertEqual(api_key_row["value"], "configured")
        self.assertEqual(
            next(row for row in rows if row["key"] == "events")["value"],
            "blocked; fallback",
        )


class TestRequestKeys(unittest.TestCase):
    """A request id alone stops being unique once sessions accumulate."""

    def test_the_same_id_in_two_sessions_gets_two_keys(self) -> None:
        first = _request("Q-001", session_id="RAG-20260823-0900")
        second = _request("Q-001", session_id="RAG-20260823-0930")

        self.assertNotEqual(_request_key(first), _request_key(second))

    def test_the_export_row_names_the_session_it_came_from(self) -> None:
        rows = _session_export_rows(
            [_request("Q-001", session_id="RAG-20260823-0900")]
        )

        self.assertEqual(rows[0]["session_id"], "RAG-20260823-0900")


class TestExportRedaction(unittest.TestCase):
    """The export masks the identifier shapes the sink masks.

    A downloaded file is a second copy of the same employee questions,
    and the Query Logs export puts these rows beside sink rows under the
    same field names (AGENTS.md section 7).
    """

    def test_an_identifier_in_the_query_is_masked(self) -> None:
        rows = _session_export_rows(
            [_request(query="เบอร์ 0812345678 ขอผมใช้เบิกได้ไหม")]
        )

        self.assertEqual(rows[0]["query"], "เบอร์ [TEL] ขอผมใช้เบิกได้ไหม")

    def test_a_rewrite_is_masked_on_the_same_terms(self) -> None:
        rows = _session_export_rows(
            [
                _request(
                    rewritten_queries=["ติดต่อ hr@corp.co.th เรื่องเบิกค่าเดินทาง"]
                )
            ]
        )

        self.assertEqual(
            rows[0]["rewritten_queries"],
            ["ติดต่อ [EMAIL] เรื่องเบิกค่าเดินทาง"],
        )

    def test_an_ordinary_question_is_left_untouched(self) -> None:
        # Redaction is a bounded safeguard, not a PII classifier: day
        # counts and document ids stay intact so the row can still
        # explain a retrieval.
        rows = _session_export_rows([_request()])

        self.assertEqual(rows[0]["query"], "ลาป่วยกี่วันต้องมีใบรับรองแพทย์")


class TestCrossSessionAggregation(unittest.TestCase):
    """``_all_requests`` is the console's view of the whole browser tab."""

    def setUp(self) -> None:
        self.state: dict = {
            "sessions": [
                {
                    "session_id": "RAG-20260823-0900",
                    "started_at": "2026-08-23T09:00:00+07:00",
                    "history": [
                        _request("Q-001", timestamp="2026-08-23T09:00:05+07:00")
                    ],
                },
                {
                    "session_id": "RAG-20260823-0930",
                    "started_at": "2026-08-23T09:30:00+07:00",
                    "history": [
                        _request(
                            "Q-001",
                            session_id="RAG-20260823-0930",
                            timestamp="2026-08-23T09:30:07+07:00",
                        )
                    ],
                },
            ],
            "active_session": "RAG-20260823-0930",
        }
        patcher = mock.patch.object(runtime.st, "session_state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_every_session_contributes_its_requests(self) -> None:
        rows = runtime._all_requests()

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [row["session_id"] for row in rows],
            ["RAG-20260823-0900", "RAG-20260823-0930"],
        )

    def test_rows_are_ordered_by_when_they_ran(self) -> None:
        timestamps = [row["timestamp"] for row in runtime._all_requests()]

        self.assertEqual(timestamps, sorted(timestamps))

    def test_aggregating_does_not_mutate_the_stored_records(self) -> None:
        runtime._all_requests()

        stored = self.state["sessions"][0]["history"][0]
        self.assertNotIn("session_id", stored)

    def test_the_active_session_still_sees_only_its_own_history(
        self,
    ) -> None:
        # The assistant reads one transcript; only the console aggregates.
        active = runtime._active_session()

        self.assertEqual(len(active["history"]), 1)
        self.assertEqual(active["session_id"], "RAG-20260823-0930")


if __name__ == "__main__":
    unittest.main()
