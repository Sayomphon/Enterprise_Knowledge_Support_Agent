"""Unit tests for deterministic rendering of a validated answer.

Rendering is the only place where ``[SOURCE-ID]`` markup is produced, so
these tests pin the output format and its determinism: the same
candidate must always render byte-identical text.
"""

import unittest

from src.answer_renderer import render_answer
from src.schemas import AnswerClaim, GroundedAnswer

FIRST_CLAIM = "ยื่นเบิกผ่าน Expense Portal ภายใน 30 วัน"
SECOND_CLAIM = "ต้องแนบใบเสร็จหรือเอกสารประกอบตามนโยบาย"


class TestRenderAnswer(unittest.TestCase):
    """Format, ordering, and stability of the public answer text."""

    def test_single_claim_renders_text_then_citation(self) -> None:
        candidate = GroundedAnswer(
            claims=[AnswerClaim(text=FIRST_CLAIM, source_ids=["FIN-001"])]
        )

        self.assertEqual(
            render_answer(candidate), f"{FIRST_CLAIM} [FIN-001]"
        )

    def test_each_claim_renders_on_its_own_line(self) -> None:
        candidate = GroundedAnswer(
            claims=[
                AnswerClaim(text=FIRST_CLAIM, source_ids=["FIN-001"]),
                AnswerClaim(
                    text=SECOND_CLAIM, source_ids=["FIN-002", "FIN-001"]
                ),
            ]
        )

        self.assertEqual(
            render_answer(candidate),
            f"{FIRST_CLAIM} [FIN-001]\n"
            f"{SECOND_CLAIM} [FIN-001] [FIN-002]",
        )

    def test_citation_order_does_not_depend_on_model_order(self) -> None:
        # Two candidates that differ only in the order the model listed
        # its ids must render identically, so answers stay diffable.
        first = GroundedAnswer(
            claims=[
                AnswerClaim(
                    text=FIRST_CLAIM, source_ids=["FIN-002", "FIN-001"]
                )
            ]
        )
        second = GroundedAnswer(
            claims=[
                AnswerClaim(
                    text=FIRST_CLAIM, source_ids=["FIN-001", "FIN-002"]
                )
            ]
        )

        self.assertEqual(render_answer(first), render_answer(second))

    def test_claim_text_is_stripped(self) -> None:
        candidate = GroundedAnswer(
            claims=[
                AnswerClaim(
                    text=f"  {FIRST_CLAIM}  ", source_ids=[" FIN-001 "]
                )
            ]
        )

        self.assertEqual(
            render_answer(candidate), f"{FIRST_CLAIM} [FIN-001]"
        )


if __name__ == "__main__":
    unittest.main()
