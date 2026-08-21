"""Behavioural tests for the character TF-IDF retriever.

All tests run offline against the real ``data/docs`` corpus or a small
synthetic corpus. Thai query literals live in named constants so the test
logic itself stays English-only (AGENTS.md section 6.1).
"""

from __future__ import annotations

import unittest

from src.ingestion.loader import load_documents
from src.retrievers.base import Retriever
from src.retrievers.local_tfidf import LocalTfidfRetriever
from src.schemas import Document

# Exact in-domain phrasings whose wording closely matches one document.
EXACT_QUERY_EXPECTATIONS = (
    ("ขั้นตอนการเบิกค่าแท็กซี่หลังทำ OT ต้องทำอย่างไร", "FIN-001"),
    ("พนักงานมีสิทธิ์ลาพักร้อนกี่วัน", "HR-001"),
    ("ระเบียบการลาป่วย", "HR-002"),
    ("นโยบายการทำงานจากที่บ้าน", "HR-003"),
)

# The same annual-leave question, once misspelled and once spelled
# correctly; both must land on the same document.
TYPO_QUERY = "ลาพักรอ้น 2 วันกดตรงไหนอะ"
CORRECT_SPELLING_QUERY = "ลาพักร้อน 2 วันกดตรงไหน"
ANNUAL_LEAVE_SOURCE_IDS = {"HR-001", "CHAT-002"}

# Queries with no support anywhere in the HR/Finance corpus.
OUT_OF_DOMAIN_QUERIES = (
    "Bitcoin วันนี้ราคาเท่าไหร่",
    "แมนยูแข่งกี่โมงคืนนี้",
    "วิธีทำต้มยำกุ้งให้อร่อย",
)

MULTI_QUERY_REIMBURSEMENT = "เบิกค่าแท็กซี่"
MULTI_QUERY_LEAVE = "ลาพักร้อน"

SYNTHETIC_BODY = "Identical body text used to force a score tie."


def _policy_document(
    source_id: str, title: str, content: str = SYNTHETIC_BODY
) -> Document:
    """Build one synthetic policy document with valid authority metadata."""
    return Document(
        source_id=source_id,
        title=title,
        source_type="policy",
        content=content,
        authority="authoritative",
        status="active",
        topics=("annual_leave",),
    )


def _corpus_retriever() -> LocalTfidfRetriever:
    """Build a retriever over the real corpus for one test class."""
    return LocalTfidfRetriever(load_documents())


class TestCorpusRetrieval(unittest.TestCase):
    """Retrieval quality over the shipped mock corpus."""

    retriever: LocalTfidfRetriever

    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = _corpus_retriever()

    def test_exact_thai_query_hits_the_expected_document(self) -> None:
        for query, expected_source_id in EXACT_QUERY_EXPECTATIONS:
            with self.subTest(query=query):
                results = self.retriever.search([query], top_k=3)
                self.assertEqual(results[0].source_id, expected_source_id)

    def test_typo_query_hits_the_same_document_as_correct_spelling(
        self,
    ) -> None:
        typo_top = self.retriever.search([TYPO_QUERY], top_k=1)[0]
        correct_top = self.retriever.search(
            [CORRECT_SPELLING_QUERY], top_k=1
        )[0]

        self.assertEqual(typo_top.source_id, correct_top.source_id)
        self.assertIn(typo_top.source_id, ANNUAL_LEAVE_SOURCE_IDS)

    def test_out_of_domain_scores_stay_below_exact_in_domain_scores(
        self,
    ) -> None:
        # The retriever only reports scores; the routing threshold between
        # these two groups is chosen later by calibration. This test pins
        # the precondition calibration relies on: the groups must not
        # overlap for clearly-phrased queries.
        in_domain_top_scores = [
            self.retriever.search([query], top_k=1)[0].score
            for query, _ in EXACT_QUERY_EXPECTATIONS
        ]
        out_of_domain_top_scores = [
            self.retriever.search([query], top_k=1)[0].score
            for query in OUT_OF_DOMAIN_QUERIES
        ]

        self.assertGreater(
            min(in_domain_top_scores), max(out_of_domain_top_scores)
        )

    def test_top_k_ordering_is_deterministic_across_runs(self) -> None:
        query = EXACT_QUERY_EXPECTATIONS[0][0]
        first = self.retriever.search([query], top_k=8)
        second = self.retriever.search([query], top_k=8)
        fresh_instance = _corpus_retriever().search([query], top_k=8)

        def as_pairs(results: list) -> list[tuple[str, float]]:
            return [(result.source_id, result.score) for result in results]

        self.assertEqual(as_pairs(first), as_pairs(second))
        self.assertEqual(as_pairs(first), as_pairs(fresh_instance))
        scores = [result.score for result in first]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_multi_query_scores_are_max_pooled_per_document(self) -> None:
        queries = [MULTI_QUERY_REIMBURSEMENT, MULTI_QUERY_LEAVE]
        pooled = {
            result.source_id: result.score
            for result in self.retriever.search(queries, top_k=8)
        }
        singles: dict[str, float] = {}
        for query in queries:
            for result in self.retriever.search([query], top_k=8):
                singles[result.source_id] = max(
                    singles.get(result.source_id, 0.0), result.score
                )

        self.assertEqual(set(pooled), set(singles))
        for source_id, score in pooled.items():
            with self.subTest(source_id=source_id):
                self.assertAlmostEqual(score, singles[source_id])

    def test_top_k_caps_the_result_length(self) -> None:
        query = EXACT_QUERY_EXPECTATIONS[0][0]
        self.assertEqual(len(self.retriever.search([query], top_k=3)), 3)
        self.assertEqual(len(self.retriever.search([query], top_k=50)), 8)

    def test_single_query_search_marks_every_hit_as_original(self) -> None:
        query = EXACT_QUERY_EXPECTATIONS[0][0]

        results = self.retriever.search([query], top_k=3)

        for result in results:
            with self.subTest(source_id=result.source_id):
                self.assertEqual(result.matched_query, query)
                self.assertEqual(result.matched_query_type, "original")

    def test_expanded_search_records_which_query_earned_the_score(
        self,
    ) -> None:
        # The rewrite names a document the original query never mentions,
        # so its provenance must point at the rewrite, not the original.
        results = self.retriever.search(
            [MULTI_QUERY_REIMBURSEMENT, MULTI_QUERY_LEAVE], top_k=8
        )

        by_id = {result.source_id: result for result in results}
        self.assertEqual(by_id["HR-001"].matched_query, MULTI_QUERY_LEAVE)
        self.assertEqual(by_id["HR-001"].matched_query_type, "rewrite")
        self.assertEqual(
            by_id["FIN-001"].matched_query, MULTI_QUERY_REIMBURSEMENT
        )
        self.assertEqual(by_id["FIN-001"].matched_query_type, "original")

    def test_retriever_satisfies_the_retriever_protocol(self) -> None:
        self.assertIsInstance(self.retriever, Retriever)

    def test_retriever_exposes_its_ngram_configuration(self) -> None:
        self.assertEqual(self.retriever.ngram_range, (2, 5))
        override = LocalTfidfRetriever(
            [_policy_document("ZZ-001", "Title")],
            ngram_range=(3, 5),
        )
        self.assertEqual(override.ngram_range, (3, 5))


class TestRetrieverContract(unittest.TestCase):
    """Input validation and deterministic tie-breaking."""

    def test_constructor_rejects_an_empty_corpus(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one document"):
            LocalTfidfRetriever([])

    def test_search_rejects_empty_queries(self) -> None:
        retriever = LocalTfidfRetriever([_policy_document("ZZ-001", "Title")])
        with self.assertRaisesRegex(ValueError, "at least one query"):
            retriever.search([], top_k=1)

    def test_search_rejects_non_positive_top_k(self) -> None:
        retriever = LocalTfidfRetriever([_policy_document("ZZ-001", "Title")])
        with self.assertRaisesRegex(ValueError, "top_k"):
            retriever.search(["query"], top_k=0)

    def test_tied_scores_break_by_ascending_source_id(self) -> None:
        # Two documents with identical searchable text always tie exactly,
        # so their relative order is decided purely by the tie-break rule.
        # They are constructed in descending id order to prove sorting does
        # not depend on insertion order.
        documents = [
            _policy_document("ZZ-002", "Same title"),
            _policy_document("ZZ-001", "Same title"),
            _policy_document(
                "AA-001", "Unrelated", "Completely different."
            ),
        ]
        retriever = LocalTfidfRetriever(documents)

        results = retriever.search(["identical body text"], top_k=3)

        tied = [r for r in results if r.source_id.startswith("ZZ")]
        self.assertEqual(len(tied), 2)
        self.assertEqual(tied[0].score, tied[1].score)
        self.assertEqual(
            [tied[0].source_id, tied[1].source_id], ["ZZ-001", "ZZ-002"]
        )


if __name__ == "__main__":
    unittest.main()
