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

# Bodies of the documents behind those ids. The numeric rule reads the
# document text rather than trusting the claim, so the figures a valid
# claim may state have to exist here first.
EVIDENCE_TEXTS = {
    "FIN-001": (
        "ยื่นเบิกผ่าน Expense Portal ภายใน 30 วันนับจากวันที่จ่าย "
        "ค่าแท็กซี่หลังเวลา 22:00 น. เบิกได้ตามจริง"
    ),
    "FIN-002": (
        "กรณีใบเสร็จหาย ใช้แบบฟอร์มรับรองแทนได้ในวงเงินไม่เกิน 500 บาท "
        "วงเงินโครงการรวมไม่เกิน 5,000 บาทต่อปี"
    ),
    "CHAT-001": "เพื่อนร่วมงานบอกว่าเบิกย้อนหลังได้ 60 วัน",
}

CLAIM_TEXT = "ยื่นเบิกผ่าน Expense Portal ภายใน 30 วัน"
SECOND_CLAIM_TEXT = "ต้องแนบใบเสร็จหรือเอกสารประกอบตามนโยบาย"
CHAT_CLAIM_TEXT = "เพื่อนร่วมงานบอกว่าเบิกย้อนหลังได้ 60 วัน"
# One claim carrying its own line break. The renderer appends citations at
# the end of a claim, so this would show its first line uncited.
MULTILINE_CLAIM_TEXT = (
    "พนักงานต้องแนบใบเสร็จทุกครั้ง\nค่าแท็กซี่ล่วงเวลาเบิกได้ไม่เกิน 500 บาท"
)
# Citation markup the model may write in a shape the loader would never
# emit. Each must still be rejected as markup, not mistaken for prose.
LOWERCASE_MARKUP_TEXT = "ตามระเบียบ [fin-001] ต้องแนบใบเสร็จทุกครั้ง"
ROUND_BRACKET_MARKUP_TEXT = "ตามระเบียบ (FIN-777) ต้องแนบใบเสร็จทุกครั้ง"
FOUR_DIGIT_MARKUP_TEXT = "ตามระเบียบ [FIN-0012] ต้องแนบใบเสร็จทุกครั้ง"
# Benign lookalikes paired with the rules above: ordinary punctuation and a
# hyphenated product word must never be read as citation markup.
BENIGN_PUNCTUATION_TEXT = (
    "ยื่นภายใน 30 วัน (นับจากวันที่จ่าย) และใช้ e-receipt แทนใบเสร็จกระดาษได้"
)


def _answer(*claims: AnswerClaim, insufficient: bool = False) -> GroundedAnswer:
    """Build a candidate answer from the given claims."""
    return GroundedAnswer(
        claims=list(claims), insufficient_evidence=insufficient
    )


def _validate(candidate: GroundedAnswer, query: str = ""):
    """Validate a candidate against the shared evidence of these tests.

    Args:
        candidate: The candidate answer under test.
        query: The employee's question, for the cases that check a
            figure the employee supplied themselves. Most cases ask
            nothing numeric, so the default is the empty question.
    """
    return validate_answer(
        candidate,
        EVIDENCE_IDS,
        AUTHORITATIVE_IDS,
        evidence_texts=EVIDENCE_TEXTS,
        query=query,
    )


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
        # The claim that names a figure cites the document that states
        # it; this case is about the citation set, not the numeric rule.
        result = _validate(
            _answer(
                AnswerClaim(text=SECOND_CLAIM_TEXT, source_ids=["FIN-002"]),
                AnswerClaim(
                    text=CLAIM_TEXT,
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

    def test_claim_containing_a_line_break_is_rejected(self) -> None:
        # The renderer joins claims with "\n" and appends each claim's
        # citations at its end, so a claim that breaks its own line would
        # render a rule with no [SOURCE-ID] behind it.
        result = _validate(
            _answer(
                AnswerClaim(
                    text=MULTILINE_CLAIM_TEXT, source_ids=["FIN-001"]
                )
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "invalid_answer_structure")

    def test_citation_markup_near_misses_are_rejected(self) -> None:
        # A model that writes the id in a shape the loader would never
        # produce is still writing markup, and the employee reads it as a
        # source id the validator never checked.
        for text in (
            LOWERCASE_MARKUP_TEXT,
            ROUND_BRACKET_MARKUP_TEXT,
            FOUR_DIGIT_MARKUP_TEXT,
        ):
            with self.subTest(text=text):
                result = _validate(
                    _answer(AnswerClaim(text=text, source_ids=["FIN-001"]))
                )

                self.assertFalse(result.ok)
                self.assertEqual(result.reason, "invalid_answer_structure")

    def test_ordinary_punctuation_is_not_citation_markup(self) -> None:
        # Paired with the rule above: widening the markup pattern must not
        # start refusing answers that merely use brackets or a hyphen.
        result = _validate(
            _answer(
                AnswerClaim(
                    text=BENIGN_PUNCTUATION_TEXT, source_ids=["FIN-001"]
                )
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.citations, ("FIN-001",))

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


class TestNumericAnchorRule(unittest.TestCase):
    """A right citation under a wrong figure is still a wrong answer.

    Provenance and coverage both pass on "ลาได้ 15 วัน" citing the policy
    that says 10, which is the cheapest hallucination to produce and the
    most expensive one to act on in an HR/Finance answer.
    """

    def test_figure_absent_from_the_cited_document_is_rejected(
        self,
    ) -> None:
        result = _validate(
            _answer(
                AnswerClaim(
                    text="ยื่นเบิกผ่าน Expense Portal ภายใน 45 วัน",
                    source_ids=["FIN-001"],
                )
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "unsupported_numeric_claim")

    def test_figure_present_in_the_cited_document_passes(self) -> None:
        result = _validate(
            _answer(AnswerClaim(text=CLAIM_TEXT, source_ids=["FIN-001"]))
        )

        self.assertTrue(result.ok)

    def test_clock_time_is_checked_as_one_anchor(self) -> None:
        supported = _validate(
            _answer(
                AnswerClaim(
                    text="เบิกค่าแท็กซี่ได้เมื่อทำงานหลังเวลา 22:00 น.",
                    source_ids=["FIN-001"],
                )
            )
        )
        invented = _validate(
            _answer(
                AnswerClaim(
                    text="เบิกค่าแท็กซี่ได้เมื่อทำงานหลังเวลา 21:00 น.",
                    source_ids=["FIN-001"],
                )
            )
        )

        self.assertTrue(supported.ok)
        self.assertFalse(invented.ok)
        self.assertEqual(invented.reason, "unsupported_numeric_claim")

    def test_a_figure_the_employee_wrote_may_be_repeated_back(
        self,
    ) -> None:
        # Without this exemption the assistant could not answer "ลา 2
        # วันได้ไหม" by naming the two days the employee asked about.
        result = _validate(
            _answer(
                AnswerClaim(
                    text="การลา 2 วันต้องยื่นผ่านระบบตามขั้นตอนปกติ",
                    source_ids=["FIN-001"],
                )
            ),
            query="ลา 2 วันต้องทำอย่างไร",
        )

        self.assertTrue(result.ok)

    def test_thousand_separators_do_not_change_a_figure(self) -> None:
        # The evidence writes 5,000 and the claim writes 5000; they are
        # the same amount and the anchors normalize to the same string.
        result = _validate(
            _answer(
                AnswerClaim(
                    text="วงเงินโครงการรวมไม่เกิน 5000 บาทต่อปี",
                    source_ids=["FIN-002"],
                )
            )
        )

        self.assertTrue(result.ok)

    def test_figure_from_an_uncited_document_is_still_rejected(
        self,
    ) -> None:
        # The 500-baht limit is in FIN-002, which this claim does not
        # cite. Pooling every document's anchors would let a claim borrow
        # a figure from a source it never named.
        result = _validate(
            _answer(
                AnswerClaim(
                    text="ใบเสร็จหายรับรองแทนได้ไม่เกิน 500 บาท",
                    source_ids=["FIN-001"],
                )
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "unsupported_numeric_claim")

    def test_a_claim_without_figures_is_unaffected(self) -> None:
        result = _validate(
            _answer(
                AnswerClaim(text=SECOND_CLAIM_TEXT, source_ids=["FIN-001"])
            )
        )

        self.assertTrue(result.ok)

    def test_provenance_rules_are_still_reported_first(self) -> None:
        # A fabricated id AND an invented figure: the answer must be
        # rejected for the id, because that is the stronger statement
        # about what went wrong and the reason an operator acts on.
        result = _validate(
            _answer(
                AnswerClaim(
                    text="ยื่นเบิกภายใน 45 วัน", source_ids=["ZZ-999"]
                )
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "fabricated_citation")


if __name__ == "__main__":
    unittest.main()
