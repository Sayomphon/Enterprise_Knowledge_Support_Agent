"""Containment tests for the seam between the UI and the pipeline.

``ui/runtime.py`` is where a Streamlit script run enters the graph. Every
node inside the graph degrades to a logged fallback, but the call itself
is still a seam: a failure in the graph runtime, or in a router reached
with a state it did not expect, would otherwise surface as a Streamlit
traceback -- and a traceback on that page can carry a filesystem path, a
prompt fragment or a provider payload (AGENTS.md section 4, invariant
10). These tests drive that seam with a graph that raises.
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
    """Compiled-graph stand-in whose ``invoke`` raises."""

    def invoke(self, payload: dict) -> dict:
        """Fail the way an unhandled runtime error would."""
        raise RuntimeError(LEAKY_MESSAGE)


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
        state, latency = self._invoke()

        self.assertEqual(state["route"], "fallback")
        self.assertEqual(
            state["fallback_reason"], ReasonCode.EVIDENCE_FAILURE.value
        )
        self.assertGreaterEqual(latency, 0.0)

    def test_the_failure_state_reads_as_a_service_outage(self) -> None:
        # Not as thin evidence: nothing was retrieved, so telling the
        # employee the corpus lacked an answer would send them after a
        # policy that was never consulted (remediation plan Finding 8).
        state, _ = self._invoke()

        self.assertTrue(is_service_failure(state["fallback_reason"]))
        self.assertFalse(state["telemetry_logged"])

    def test_the_employee_text_carries_none_of_the_exception(self) -> None:
        state, _ = self._invoke()

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


if __name__ == "__main__":
    unittest.main()
