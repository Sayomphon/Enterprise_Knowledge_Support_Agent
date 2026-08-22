"""One real request against the real provider, run only on request.

Everything else in this repository mocks the two LLM seams, which proves
the routing and the contracts but never that a real provider returns
something this pipeline can accept. This is that check: one high-band
question, one Reporter call, and the answer contract asserted on the
result.

It is opt-in twice over -- an environment flag and a configured
credential -- because it costs money and reaches the network, and
because the default suite must stay runnable with neither (AGENTS.md
section 9). Skipping is the expected outcome; a skipped run is not a
passed one, so the result is recorded in eval/RESULTS.md when it is
actually run.

    RUN_LIVE_SMOKE=1 python -m unittest tests.live.test_live_smoke -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src import config
from src.graph import build_graph
from src.schemas import PipelineState

# A question the calibration set scores in the high band (raw 0.2752),
# so it reaches the Reporter directly: one provider call, no rewrite,
# and a route that does not depend on live rewrite quality. A
# medium-band query would make this test flaky for a reason that has
# nothing to do with the contract it checks.
SMOKE_QUERY = "ทำงานจากที่บ้านได้สัปดาห์ละกี่วัน"

_ENABLED = os.getenv("RUN_LIVE_SMOKE") == "1"


@unittest.skipUnless(
    _ENABLED and config.has_llm_credential(),
    "live smoke test: set RUN_LIVE_SMOKE=1 with a configured credential",
)
class TestLiveSmoke(unittest.TestCase):
    """The answer contract, verified against a real provider once."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the one request, so its cost is paid once per class."""
        cls._log_dir = tempfile.TemporaryDirectory(prefix="live-smoke-")
        log_path = Path(cls._log_dir.name) / "fallback_queries.jsonl"
        graph = build_graph(log_path=log_path)
        cls.state: PipelineState = graph.invoke({"query": SMOKE_QUERY})

    @classmethod
    def tearDownClass(cls) -> None:
        cls._log_dir.cleanup()

    def test_the_request_reaches_a_validated_answer(self) -> None:
        self.assertEqual(self.state["route"], "answered")
        self.assertTrue(self.state["answer"].strip())

    def test_every_citation_belongs_to_this_request_s_evidence(
        self,
    ) -> None:
        evidence_ids = {
            document.source_id
            for document in self.state["answer_evidence"]
        }
        citations = set(self.state["valid_citations"])

        self.assertTrue(citations)
        self.assertTrue(citations <= evidence_ids)

    def test_the_rendered_answer_carries_its_citation_markup(self) -> None:
        # The renderer emits the markup from validated ids, so this is
        # what an employee actually sees rather than what the model
        # wrote.
        for source_id in self.state["valid_citations"]:
            with self.subTest(source_id=source_id):
                self.assertIn(f"[{source_id}]", self.state["answer"])

    def test_the_answer_rests_on_at_least_one_policy(self) -> None:
        self.assertTrue(self.state["authoritative_source_ids"])

    def test_no_unvalidated_draft_leaves_the_graph(self) -> None:
        # The candidate/answer split is the contract this test is most
        # able to break, because only a live run produces a real
        # candidate at all.
        self.assertNotIn("fallback_reason", self.state)
        self.assertEqual(self.state.get("llm_calls"), 1)


if __name__ == "__main__":
    unittest.main()
