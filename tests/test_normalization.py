"""Unit tests for the canonical folding the contract checks share.

The folding exists because an attacker picks the spelling: a Thai
numeral, a fullwidth digit, or a CJK bracket renders like the ASCII
original on an employee's screen while missing every pattern written in
ASCII. Each test therefore pairs a lookalike spelling with the plain one
it must fold to, and the Thai cases guard the direction the folding
could break instead: NFKC decomposes SARA AM, so a corpus word must
survive the fold unchanged (AGENTS.md section 9).
"""

import unittest

from src.guardrails.normalization import fold_for_contract_checks, nfkc_fold

# The same visible word before and after NFKC would tear SARA AM apart.
THAI_SARA_AM_WORD = "คำถามเรื่องการทำงาน"


class TestNfkcFold(unittest.TestCase):
    """The shared NFKC primitive must not damage Thai corpus wording."""

    def test_thai_sara_am_survives_the_compatibility_fold(self) -> None:
        self.assertEqual(nfkc_fold(THAI_SARA_AM_WORD), THAI_SARA_AM_WORD)

    def test_a_decomposed_sara_am_is_recomposed(self) -> None:
        decomposed = THAI_SARA_AM_WORD.replace("ำ", "ํา")

        self.assertEqual(nfkc_fold(decomposed), THAI_SARA_AM_WORD)

    def test_fullwidth_letters_fold_to_ascii(self) -> None:
        self.assertEqual(nfkc_fold("ＨＲ"), "HR")


class TestDigitFolding(unittest.TestCase):
    """Every decimal spelling of a figure must fold to ASCII digits."""

    def test_thai_numerals_fold_to_ascii(self) -> None:
        self.assertEqual(fold_for_contract_checks("ลาได้ ๑๐ วัน"), "ลาได้ 10 วัน")

    def test_fullwidth_numerals_fold_to_ascii(self) -> None:
        self.assertEqual(fold_for_contract_checks("ลาได้ １０ วัน"), "ลาได้ 10 วัน")

    def test_arabic_indic_numerals_fold_to_ascii(self) -> None:
        self.assertEqual(fold_for_contract_checks("٥٠٠ บาท"), "500 บาท")

    def test_ascii_digits_are_left_alone(self) -> None:
        self.assertEqual(
            fold_for_contract_checks("ไม่เกิน 5,000 บาท"), "ไม่เกิน 5,000 บาท"
        )


class TestPunctuationFolding(unittest.TestCase):
    """Bracket and dash lookalikes must fold to the ASCII spelling."""

    def test_fullwidth_brackets_fold_to_ascii(self) -> None:
        self.assertEqual(fold_for_contract_checks("［ZZ－９９９］"), "[ZZ-999]")

    def test_cjk_brackets_fold_to_ascii(self) -> None:
        for spelling in ("【HR-001】", "〔HR-001〕", "〖HR-001〗", "⟦HR-001⟧"):
            with self.subTest(spelling=spelling):
                self.assertEqual(
                    fold_for_contract_checks(spelling), "[HR-001]"
                )

    def test_unicode_dashes_fold_to_ascii(self) -> None:
        for dash in ("‐", "‑", "–", "—", "―", "−"):
            with self.subTest(dash=dash):
                self.assertEqual(
                    fold_for_contract_checks(f"[HR{dash}001]"), "[HR-001]"
                )

    def test_thai_text_without_lookalikes_is_unchanged(self) -> None:
        # The fold runs on every claim and every document body, so a
        # corpus sentence that contains none of the shapes above must
        # come back byte for byte.
        sentence = "ยื่นเบิกผ่าน Expense Portal ภายใน 30 วัน"

        self.assertEqual(fold_for_contract_checks(sentence), sentence)


if __name__ == "__main__":
    unittest.main()
