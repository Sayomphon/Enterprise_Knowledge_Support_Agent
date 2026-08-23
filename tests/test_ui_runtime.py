"""Containment tests for the seam between the UI and the pipeline.

``ui/runtime.py`` is where a Streamlit script run enters the graph. Every
node inside the graph degrades to a logged fallback, but the call itself
is still a seam: a failure in the graph runtime, or in a router reached
with a state it did not expect, would otherwise surface as a Streamlit
traceback -- and a traceback on that page can carry a filesystem path, a
prompt fragment or a provider payload (AGENTS.md section 4, invariant
10). These tests drive that seam with a graph that raises.

The seam streams the graph rather than invoking it, so that the console
can report where a request spent its time without the pipeline carrying a
per-node timing field; the timing half of that contract is asserted here
too.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from src.fallback import (
    ReasonCode,
    is_service_failure,
    response_text_for_state,
)
from ui.runtime import _invoke_graph

QUERY = "ขั้นตอนการเบิกค่าแท็กซี่หลังทำ OT ต้องทำอย่างไร"
# The message carries everything invariant 10 forbids on the page: a
# filesystem path, a fragment of the system prompt, and a provider
# payload. The tests assert none of it reaches the employee or stderr.
LEAKY_MESSAGE = (
    "graph exploded reading /srv/secret/prompt.txt: "
    "'You are an HR assistant' -> {'choices': [{'text': 'sk-live-123'}]}"
)


class FailingGraph:
    """Compiled-graph stand-in whose run raises."""

    def stream(self, payload: dict, **kwargs):
        """Fail the way an unhandled runtime error would."""
        raise RuntimeError(LEAKY_MESSAGE)
        yield  # pragma: no cover - generator marker, never reached


class TimedGraph:
    """Compiled-graph stand-in that reports two nodes and a final state."""

    def __init__(self, final_state: dict) -> None:
        self.final_state = final_state

    def stream(self, payload: dict, **kwargs):
        """Yield the update/value pairs a real streamed run produces."""
        yield "updates", {"input_guardrail": {"route": "answered"}}
        yield "values", {**payload, "route": "answered"}
        yield "updates", {"retrieve_original": {"raw_retrieval_score": 0.3}}
        yield "values", self.final_state


class TestGraphSeamContainment(unittest.TestCase):
    """What the employee gets when the graph itself fails."""

    def setUp(self) -> None:
        seam = mock.patch(
            "ui.runtime._graph_and_retriever",
            return_value=(FailingGraph(), None, None),
        )
        seam.start()
        self.addCleanup(seam.stop)
        self.stderr = io.StringIO()

    def _invoke(self):
        with contextlib.redirect_stderr(self.stderr):
            return _invoke_graph(QUERY)

    def test_an_unhandled_failure_returns_a_state_instead_of_raising(
        self,
    ) -> None:
        state, latency, node_seconds = self._invoke()

        self.assertEqual(state["route"], "fallback")
        self.assertEqual(
            state["fallback_reason"], ReasonCode.EVIDENCE_FAILURE.value
        )
        self.assertGreaterEqual(latency, 0.0)

    def test_the_failure_state_reads_as_a_service_outage(self) -> None:
        # Not as thin evidence: nothing was retrieved, so telling the
        # employee the corpus lacked an answer would send them after a
        # policy that was never consulted (remediation plan Finding 8).
        state, _, _ = self._invoke()

        self.assertTrue(is_service_failure(state["fallback_reason"]))
        self.assertFalse(state["telemetry_logged"])

    def test_the_employee_text_carries_none_of_the_exception(self) -> None:
        state, _, _ = self._invoke()

        text = response_text_for_state(
            state["route"],
            state.get("guardrail_reason"),
            state["fallback_reason"],
            telemetry_logged=state["telemetry_logged"],
        )

        self.assertIsNotNone(text)
        for secret in ("srv", "prompt", "sk-live", "RuntimeError"):
            self.assertNotIn(secret, text)

    def test_stderr_names_the_exception_type_and_nothing_else(self) -> None:
        self._invoke()

        printed = self.stderr.getvalue()
        self.assertIn("RuntimeError", printed)
        for secret in ("srv", "sk-live", "You are an HR assistant"):
            self.assertNotIn(secret, printed)

    def test_the_seam_writes_no_telemetry_of_its_own(self) -> None:
        # Logging is the graph's job; the UI layer holds no business
        # logic and must not invent a record for a request the pipeline
        # never finished (AGENTS.md section 3).
        with mock.patch("src.logging_utils.log_fallback_event") as writer:
            self._invoke()

        self.assertEqual(writer.call_count, 0)

    def test_a_failed_run_still_returns_a_timing_mapping(self) -> None:
        # The console reads this per request; a missing key there would
        # be a second failure on top of the one being reported.
        _, _, node_seconds = self._invoke()

        self.assertEqual(node_seconds, {})


class TestNodeTiming(unittest.TestCase):
    """The seam times each node without changing the pipeline contract."""

    FINAL = {
        "query": QUERY,
        "route": "answered",
        "raw_retrieval_score": 0.3,
        "answer": "ยื่นผ่าน Expense Portal [FIN-001]",
    }

    def setUp(self) -> None:
        seam = mock.patch(
            "ui.runtime._graph_and_retriever",
            return_value=(TimedGraph(self.FINAL), None, None),
        )
        seam.start()
        self.addCleanup(seam.stop)

    def test_each_reported_node_gets_its_own_measurement(self) -> None:
        _, _, node_seconds = _invoke_graph(QUERY)

        self.assertEqual(
            sorted(node_seconds), ["input_guardrail", "retrieve_original"]
        )
        for node, seconds in node_seconds.items():
            with self.subTest(node=node):
                self.assertGreaterEqual(seconds, 0.0)

    def test_a_node_that_never_ran_is_absent_rather_than_zero(self) -> None:
        # Zero would read as a node that ran instantly, which is exactly
        # the wrong thing to tell an operator reading a slow request.
        _, _, node_seconds = _invoke_graph(QUERY)

        self.assertNotIn("report", node_seconds)
        self.assertNotIn("fallback", node_seconds)

    def test_the_final_state_is_the_last_streamed_value(self) -> None:
        state, latency, _ = _invoke_graph(QUERY)

        self.assertEqual(state, self.FINAL)
        self.assertGreaterEqual(latency, 0.0)

    def test_the_node_times_stay_within_the_measured_total(self) -> None:
        _, latency, node_seconds = _invoke_graph(QUERY)

        self.assertLessEqual(sum(node_seconds.values()), latency + 1e-6)


if __name__ == "__main__":
    unittest.main()
