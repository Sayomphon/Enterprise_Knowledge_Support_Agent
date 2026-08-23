"""Tests for the order in which the assistant draws a turn.

A grounded answer can take half a minute. Until the question and the
running status were drawn before the pipeline call, the page sat
unchanged for that whole time and then produced the question and its
answer together, which reads as a reload rather than as a conversation.
The order is behaviour, so it is asserted here rather than looked at.
"""

from __future__ import annotations

import contextlib
import unittest
from unittest import mock

from ui import assistant


class _RecordingStreamlit:
    """Minimal ``st`` stand-in that records the calls the flow makes."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.session_state: dict = {}

    @contextlib.contextmanager
    def container(self, key: str | None = None):
        """Record the status block opening."""
        self.calls.append(f"container:{key}")
        yield

    @contextlib.contextmanager
    def spinner(self, text: str):
        """Record the status text the employee sees while waiting."""
        self.calls.append(f"spinner:{text}")
        yield

    def rerun(self) -> None:
        """Record the rerun that turns the pending turn into history."""
        self.calls.append("rerun")


class _InterruptingStreamlit(_RecordingStreamlit):
    """``st`` stand-in whose status block is cut short mid-request.

    Streamlit stops a running script at its next enqueue once a fresh
    interaction arrives, and leaving the status block is such a point:
    clicking the theme toggle or New Session while the spinner is up
    ends the run before the outcome is recorded.
    """

    @contextlib.contextmanager
    def spinner(self, text: str):
        """Record the status, then end the run on the way out."""
        self.calls.append(f"spinner:{text}")
        yield
        raise RuntimeError("rerun requested while the request was in flight")


class TestOptimisticTurn(unittest.TestCase):
    """The question and the status precede the run that answers them."""

    def setUp(self) -> None:
        self.calls: list[str] = []
        stub = _RecordingStreamlit(self.calls)
        patches = [
            mock.patch.object(assistant, "st", stub),
            mock.patch.object(
                assistant,
                "_render_user_bubble",
                side_effect=lambda *_: self.calls.append("bubble"),
            ),
            mock.patch.object(
                assistant,
                "_invoke_graph",
                side_effect=lambda query: (
                    self.calls.append("invoke"),
                    ({"query": query, "route": "fallback"}, 0.25, {"x": 0.1}),
                )[1],
            ),
            mock.patch.object(
                assistant,
                "_record_request",
                side_effect=lambda *_: self.calls.append("record"),
            ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_the_question_is_drawn_before_the_pipeline_is_called(
        self,
    ) -> None:
        assistant._ask("ขั้นตอนการเบิกค่าแท็กซี่หลังทำ OT ต้องทำอย่างไร")

        self.assertLess(
            self.calls.index("bubble"), self.calls.index("invoke")
        )

    def test_the_status_is_shown_while_the_pipeline_runs(self) -> None:
        assistant._ask("ลาป่วยกี่วันต้องมีใบรับรองแพทย์")

        spinner = next(
            call for call in self.calls if call.startswith("spinner:")
        )
        self.assertEqual(
            spinner, f"spinner:{assistant.THINKING_LABEL}"
        )
        self.assertLess(
            self.calls.index(spinner), self.calls.index("invoke")
        )

    def test_the_outcome_is_recorded_and_the_page_reruns(self) -> None:
        assistant._ask("ลาพักร้อนต้องแจ้งล่วงหน้ากี่วัน")

        self.assertEqual(
            self.calls[-2:],
            ["record", "rerun"],
        )


class TestPendingHandoff(unittest.TestCase):
    """The queued question outlives a run that does not finish.

    ``araya_pending`` is the only copy of the question while the graph
    is running: the composer has already been rerun away and nothing is
    in history yet. Clearing it before the outcome is recorded loses the
    request outright, so it is cleared after.
    """

    QUESTION = "ขั้นตอนการเบิกค่าแท็กซี่หลังทำ OT ต้องทำอย่างไร"

    def _run(self, stub: _RecordingStreamlit) -> None:
        """Drive ``_ask`` against one ``st`` stand-in."""
        patches = [
            mock.patch.object(assistant, "st", stub),
            mock.patch.object(assistant, "_render_user_bubble"),
            mock.patch.object(
                assistant,
                "_invoke_graph",
                return_value=({"query": self.QUESTION}, 0.25, {}),
            ),
            mock.patch.object(
                assistant,
                "_record_request",
                side_effect=lambda *_: stub.calls.append("record"),
            ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        stub.session_state["araya_pending"] = self.QUESTION

    def test_a_recorded_turn_clears_the_queued_question(self) -> None:
        stub = _RecordingStreamlit([])
        self._run(stub)

        assistant._ask(self.QUESTION)

        self.assertNotIn("araya_pending", stub.session_state)
        self.assertIn("record", stub.calls)

    def test_an_interrupted_run_keeps_the_question_for_the_next_one(
        self,
    ) -> None:
        stub = _InterruptingStreamlit([])
        self._run(stub)

        with self.assertRaises(RuntimeError):
            assistant._ask(self.QUESTION)

        self.assertEqual(
            stub.session_state["araya_pending"], self.QUESTION
        )
        self.assertNotIn("record", stub.calls)


if __name__ == "__main__":
    unittest.main()
