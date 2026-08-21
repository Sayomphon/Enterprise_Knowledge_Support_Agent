"""Unit tests for deterministic answer-contract validation.

The validator decides whether a candidate answer may be promoted to the
public answer. Coverage is per claim, so these tests exercise one broken
claim inside an otherwise valid candidate as well as whole-structure
failures; Thai claim text lives in named constants so the test logic
stays English-only.
"""

import unittest

from src.guardrails.citation_validator import validate_answer
from src.schemas import AnswerClaim, GroundedAnswer

EVIDENCE_IDS = {"FIN-001", "FIN-002", "CHAT-001"}
AUTHORITATIVE_IDS = {"FIN-001", "FIN-002"}

CLAIM_TEXT = "ยื่นเบิกผ่าน Expense Portal ภายใน 30 วัน"
SECOND_CLAIM_TEXT = "ต้องแนบใบเสร็จหรือเอกสารประกอบตามนโยบาย"
CHAT_CLAIM_TEXT = "เพื่อนร่วมงานบอกว่าเบิกย้อนหลังได้ 60 วัน"


def _answer(*claims: AnswerClaim, insufficient: bool = False) -> GroundedAnswer:
    """Build a candidate answer from the given claims."""
    return GroundedAnswer(
        claims=list(claims), insufficient_evidence=insufficient
    )


def _validate(candidate: GroundedAnswer):
    """Validate a candidate against the shared evidence of these tests."""
    return validate_answer(candidate, EVIDENCE_IDS, AUTHORITATIVE_IDS)


class TestValidAnswers(unittest.TestCase):
    """Candidates that satisfy every rule of the contract."""

    def test_single_grounded_claim_passes(self) -> None:
        result = _validate(
            _answer(AnswerClaim(text=CLAIM_TEXT, source_ids=["FIN-001"]))
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.citations, ("FIN-001",))
        self.assertIsNone(result.reason)

    def test_citations_are_sorted_and_deduplicated_across_claims(
        self,
    ) -> None:
        result = _validate(
            _answer(
                AnswerClaim(text=CLAIM_TEXT, source_ids=["FIN-002"]),
                AnswerClaim(
                    text=SECOND_CLAIM_TEXT,
                    source_ids=["FIN-002", "FIN-001"],
                ),
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.citations, ("FIN-001", "FIN-002"))

    def test_chat_source_beside_a_policy_source_passes(self) -> None:
        # Chat may accompany the policy that governs the same rule; what
        # the contract forbids is a claim resting on chat alone.
        result = _validate(
            _answer(
                AnswerClaim(
                    text=CLAIM_TEXT, source_ids=["FIN-001", "CHAT-001"]
                )
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.citations, ("CHAT-001", "FIN-001"))

    def test_surrounding_whitespace_in_ids_is_tolerated(self) -> None:
        result = _validate(
            _answer(AnswerClaim(text=CLAIM_TEXT, source_ids=[" FIN-001 "]))
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.citations, ("FIN-001",))


class TestClaimLevelRejection(unittest.TestCase):
    """One unsupported claim invalidates the whole candidate answer."""

    def test_claim_without_any_source_is_rejected(self) -> None:
        result = _validate(
            _answer(AnswerClaim(text=CLAIM_TEXT, source_ids=[]))
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "missing_citation")

    def test_one_uncited_claim_among_valid_ones_rejects_the_answer(
        self,
    ) -> None:
        result = _validate(
            _answer(
                AnswerClaim(text=CLAIM_TEXT, source_ids=["FIN-001"]),
                AnswerClaim(text=SECOND_CLAIM_TEXT, source_ids=[]),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "missing_citation")

    def test_fabricated_id_is_rejected(self) -> None:
        result = _validate(
            _answer(AnswerClaim(text=CLAIM_TEXT, source_ids=["ZZ-999"]))
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "fabricated_citation")

    def test_real_document_not_selected_as_evidence_is_rejected(
        self,
    ) -> None:
        # HR-003 exists in the corpus, but provenance is per request:
        # only ids in this request's answer evidence may be cited.
        result = _validate(
            _answer(AnswerClaim(text=CLAIM_TEXT, source_ids=["HR-003"]))
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "fabricated_citation")

    def test_one_fabricated_id_among_valid_ones_still_rejects(self) -> None:
        result = _validate(
            _answer(
                AnswerClaim(text=CLAIM_TEXT, source_ids=["FIN-001"]),
                AnswerClaim(
                    text=SECOND_CLAIM_TEXT, source_ids=["AB-123"]
                ),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "fabricated_citation")

    def test_chat_only_claim_is_rejected(self) -> None:
        result = _validate(
            _answer(
                AnswerClaim(text=CHAT_CLAIM_TEXT, source_ids=["CHAT-001"])
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "no_authoritative_evidence")


class TestStructureRejection(unittest.TestCase):
    """Structural defects are separable from citation defects."""

    def test_empty_claim_list_without_the_flag_is_rejected(self) -> None:
        result = _validate(_answer())

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "invalid_answer_structure")

    def test_blank_claim_text_is_rejected(self) -> None:
        result = _validate(
            _answer(AnswerClaim(text="   ", source_ids=["FIN-001"]))
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "invalid_answer_structure")

    def test_model_written_citation_markup_is_rejected(self) -> None:
        # The renderer owns citation markup; a bracket inside claim text
        # is either a duplicate or an unvalidated id posing as one.
        result = _validate(
            _answer(
                AnswerClaim(
                    text=f"{CLAIM_TEXT} [FIN-001]", source_ids=["FIN-001"]
                )
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "invalid_answer_structure")

    def test_repeated_id_inside_one_claim_is_rejected(self) -> None:
        result = _validate(
            _answer(
                AnswerClaim(
                    text=CLAIM_TEXT, source_ids=["FIN-001", "FIN-001"]
                )
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "invalid_answer_structure")

    def test_blank_source_id_is_rejected_as_structure(self) -> None:
        result = _validate(
            _answer(AnswerClaim(text=CLAIM_TEXT, source_ids=["  "]))
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "invalid_answer_structure")

    def test_insufficient_evidence_without_claims_is_reported_separately(
        self,
    ) -> None:
        result = _validate(_answer(insufficient=True))

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "insufficient_reporter_evidence")

    def test_insufficient_evidence_with_claims_is_a_contradiction(
        self,
    ) -> None:
        result = _validate(
            _answer(
                AnswerClaim(text=CLAIM_TEXT, source_ids=["FIN-001"]),
                insufficient=True,
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "invalid_answer_structure")


if __name__ == "__main__":
    unittest.main()
