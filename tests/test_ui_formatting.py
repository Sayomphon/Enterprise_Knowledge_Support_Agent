"""Unit tests for the pure formatting layer of the Streamlit UI.

``app.py`` used to hold every one of these functions beside a
module-level ``main()`` call, which made the file impossible to import
and therefore impossible to test: the score bands, the axis geometry and
the node trace were verified by looking at screenshots. They now live in
``ui/formatting.py``, take a state and return markup, and are asserted
here against the same calibrated thresholds the graph routes on.
"""

from __future__ import annotations

import unittest

from src import config
from src.schemas import RetrievedDocument
from ui.formatting import (
    _axis_position,
    _band_word,
    _citation_numbers,
    _display_route,
    _format_score,
    _gating_score,
    _log_matches,
    _score_band,
    _trace_rows,
)
from ui.labels import SCORE_BAR_CEILING


def _policy_evidence() -> RetrievedDocument:
    """Build the one piece of evidence a trace row needs to name."""
    return RetrievedDocument(
        source_id="FIN-001",
        title="Expense process",
        source_type="policy",
        content="ยื่นเบิกผ่าน Expense Portal ภายใน 30 วัน",
        score=config.DIRECT_ANSWER_THRESHOLD + 0.05,
        authority="authoritative",
        status="active",
        topics=("reimbursement_process",),
    )


def _answered_state(**overrides) -> dict:
    """Build the state of a request that reached a validated answer."""
    state = {
        "query": "ขั้นตอนการเบิกค่าแท็กซี่หลังทำ OT",
        "route": "answered",
        "raw_retrieval_score": config.DIRECT_ANSWER_THRESHOLD + 0.05,
        "answer": "ยื่นผ่าน Expense Portal ภายใน 30 วัน [FIN-001]",
        "valid_citations": ["FIN-001", "CHAT-001"],
        "answer_evidence": [],
        "scope_topics": ["reimbursement_process"],
    }
    state.update(overrides)
    return state


class TestScoreBands(unittest.TestCase):
    """Colour follows the calibrated thresholds, never a UI cut-off."""

    def test_band_boundaries_match_the_configured_thresholds(self) -> None:
        cases = (
            (config.DIRECT_ANSWER_THRESHOLD, "green"),
            (config.DIRECT_ANSWER_THRESHOLD - 1e-9, "amber"),
            (config.REWRITE_FLOOR, "amber"),
            (config.REWRITE_FLOOR - 1e-9, "red"),
        )
        for score, expected in cases:
            with self.subTest(score=score):
                self.assertEqual(_score_band(score), expected)

    def test_the_band_word_agrees_with_the_band_colour(self) -> None:
        # The console prints a word and draws a dot from the same score;
        # they must not be able to disagree.
        pairs = {
            "green": _band_word(config.DIRECT_ANSWER_THRESHOLD),
            "amber": _band_word(config.REWRITE_FLOOR),
            "red": _band_word(0.0),
        }
        self.assertEqual(len(set(pairs.values())), 3)

    def test_a_missing_score_is_rendered_as_a_dash(self) -> None:
        self.assertEqual(_format_score(None), "–")


class TestAxisGeometry(unittest.TestCase):
    """The threshold strip is a drawing, so it must stay inside its box."""

    def test_zero_sits_at_the_left_edge(self) -> None:
        self.assertEqual(_axis_position(0.0), 0.0)

    def test_the_ceiling_sits_at_the_right_edge(self) -> None:
        self.assertEqual(_axis_position(SCORE_BAR_CEILING), 100.0)

    def test_scores_beyond_the_ceiling_are_clamped(self) -> None:
        self.assertEqual(_axis_position(1.0), 100.0)

    def test_a_negative_score_cannot_leave_the_strip(self) -> None:
        # Cosine similarity over non-negative vectors cannot be negative,
        # but the clamp is what keeps a future retriever from drawing
        # outside the panel rather than failing visibly.
        self.assertEqual(_axis_position(-0.5), 0.0)


class TestCitationNumbering(unittest.TestCase):
    """Inline chips are traceable to the source list under the answer."""

    def test_numbering_follows_the_validated_citation_order(self) -> None:
        numbers = _citation_numbers(
            _answered_state(valid_citations=["FIN-001", "CHAT-001"])
        )

        self.assertEqual(numbers, {"FIN-001": 1, "CHAT-001": 2})

    def test_a_request_without_citations_numbers_nothing(self) -> None:
        self.assertEqual(_citation_numbers({"route": "fallback"}), {})


class TestRouteDisplay(unittest.TestCase):
    """The console label names the route the graph actually took."""

    def test_each_route_gets_its_own_label(self) -> None:
        cases = (
            ({"route": "blocked"}, "Blocked"),
            ({"route": "fallback"}, "Fallback"),
            (_answered_state(rewritten_queries=["เบิกค่าแท็กซี่"]), "Rewrite"),
            (_answered_state(), "Direct"),
        )
        for state, expected in cases:
            with self.subTest(expected=expected):
                label, _, _ = _display_route(state)
                self.assertEqual(label, expected)

    def test_the_gating_score_prefers_the_expanded_one(self) -> None:
        # The expanded score is what admitted a medium-band answer, so it
        # is the number the console must show beside the thresholds.
        state = _answered_state(
            raw_retrieval_score=0.14, expanded_retrieval_score=0.23
        )

        self.assertEqual(_gating_score(state), 0.23)

    def test_a_blocked_request_has_no_gating_score(self) -> None:
        self.assertIsNone(_gating_score({"route": "blocked"}))


class TestTraceRows(unittest.TestCase):
    """The node trace describes the route rather than the happy path."""

    def _nodes(self, state: dict) -> list[str]:
        return [row[0] for row in _trace_rows(state)]

    def test_every_graph_node_appears_on_every_route(self) -> None:
        # The panel shows the shape of the pipeline, so a node that was
        # never reached is listed as skipped rather than omitted.
        answered = self._nodes(_answered_state())
        blocked = self._nodes(
            {"route": "blocked", "guardrail_reason": "prompt_injection"}
        )

        self.assertEqual(answered, blocked)
        for node in ("input_guardrail", "retrieve_original", "report"):
            self.assertIn(node, answered)

    def test_a_blocked_request_marks_retrieval_as_never_called(self) -> None:
        rows = {row[0]: row for row in _trace_rows({"route": "blocked"})}

        self.assertEqual(rows["retrieve_original"][1], "skip")

    def test_an_unspent_reporter_call_is_not_shown_as_billed(self) -> None:
        # A request that reaches the reporter with no deadline left is
        # refused before the provider is touched; labelling that row as
        # an LLM call would overstate what the request cost.
        state = {
            "route": "fallback",
            "raw_retrieval_score": config.DIRECT_ANSWER_THRESHOLD + 0.05,
            "answer_evidence": [_policy_evidence()],
            "fallback_reason": "request_deadline_exceeded",
        }

        report_row = next(
            row for row in _trace_rows(state) if row[0] == "report"
        )

        self.assertIn("not called", report_row[3])


class TestLogFiltering(unittest.TestCase):
    """The console search reads the record, never the pipeline."""

    def test_an_empty_filter_matches_every_record(self) -> None:
        self.assertTrue(_log_matches({"query": "x", "reason": "y"}, ""))

    def test_the_filter_matches_the_reason_code_as_well_as_the_query(
        self,
    ) -> None:
        record = {"query": "ลาพักร้อน", "reason": "unsupported_topic"}

        self.assertTrue(_log_matches(record, "unsupported"))
        self.assertTrue(_log_matches(record, "ลาพัก"))
        self.assertFalse(_log_matches(record, "fabricated"))


if __name__ == "__main__":
    unittest.main()
