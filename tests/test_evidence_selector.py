"""Behavioural tests for policy-first answer-evidence selection.

Two levels are covered: the real corpus, where a chat transcript often
outranks the policy it paraphrases, and synthetic candidates, where
authority, lifecycle status, and topic metadata can be varied one at a
time. Everything runs offline with no LLM (AGENTS.md section 9).
"""

from __future__ import annotations

import unittest

from src.evidence_selector import select_evidence
from src.guardrails.citation_validator import validate_answer
from src.ingestion.loader import load_documents
from src.retrievers.local_tfidf import LocalTfidfRetriever
from src.schemas import (
    AnswerClaim,
    Document,
    GroundedAnswer,
    RetrievedDocument,
)

# Informal phrasings whose closest lexical match is a chat transcript.
SLANG_REIMBURSEMENT_QUERY = "เบิกตังค่า taxi ได้ปะ"
TYPO_LEAVE_QUERY = "ลาพักรอ้น 2 วันกดตรงไหนอะ"

CHAT_CANDIDATE = GroundedAnswer(
    claims=[
        AnswerClaim(
            text="เบิกผ่าน Expense Portal ได้เลย", source_ids=["CHAT-001"]
        )
    ]
)


def _chat_document(
    source_id: str = "CHAT-901",
    canonical_source_ids: tuple[str, ...] = ("HR-901",),
    score: float = 0.30,
) -> RetrievedDocument:
    """Build one supplementary chat candidate."""
    return RetrievedDocument(
        source_id=source_id,
        title="Chat transcript",
        source_type="chat",
        content="Chat body.",
        score=score,
        authority="supplementary",
        status="active",
        topics=("annual_leave",),
        canonical_source_ids=canonical_source_ids,
    )


def _policy_document(
    source_id: str = "HR-901",
    score: float = 0.20,
    status: str = "active",
    topics: tuple[str, ...] = ("annual_leave",),
) -> RetrievedDocument:
    """Build one authoritative policy candidate."""
    return RetrievedDocument(
        source_id=source_id,
        title="Leave policy",
        source_type="policy",
        content="Policy body.",
        score=score,
        authority="authoritative",
        status=status,
        topics=topics,
    )


def _corpus_document(candidate: RetrievedDocument) -> Document:
    """Mirror one candidate back into its corpus document."""
    return Document(
        source_id=candidate.source_id,
        title=candidate.title,
        source_type=candidate.source_type,
        content=candidate.content,
        authority=candidate.authority,
        status=candidate.status,
        topics=candidate.topics,
        canonical_source_ids=candidate.canonical_source_ids,
    )


def _corpus(*candidates: RetrievedDocument) -> dict[str, Document]:
    """Index synthetic candidates as the corpus the selector reads."""
    return {
        candidate.source_id: _corpus_document(candidate)
        for candidate in candidates
    }


class TestRealCorpusSelection(unittest.TestCase):
    """A chat hit must still produce a policy-backed answer."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = load_documents()
        cls.documents_by_id = {
            document.source_id: document for document in cls.documents
        }
        cls.retriever = LocalTfidfRetriever(cls.documents)

    def _select(self, query: str):
        candidates = self.retriever.search([query], top_k=3)
        return candidates, select_evidence(candidates, self.documents_by_id)

    def test_slang_chat_hit_pulls_in_its_canonical_policy(self) -> None:
        candidates, selection = self._select(SLANG_REIMBURSEMENT_QUERY)

        self.assertEqual(candidates[0].source_id, "CHAT-001")
        self.assertTrue(selection.ok)
        self.assertIn("FIN-001", selection.authoritative_ids)
        self.assertIn("CHAT-001", selection.supplementary_ids)
        # Authoritative policy is always offered to the reporter first.
        self.assertEqual(
            selection.answer_evidence[0].authority, "authoritative"
        )

    def test_typo_leave_chat_hit_pulls_in_its_canonical_policy(self) -> None:
        candidates, selection = self._select(TYPO_LEAVE_QUERY)

        self.assertEqual(candidates[0].source_id, "CHAT-002")
        self.assertTrue(selection.ok)
        self.assertIn("HR-001", selection.authoritative_ids)

    def test_topic_filter_keeps_only_evidence_about_the_query_topic(
        self,
    ) -> None:
        candidates, _ = self._select(TYPO_LEAVE_QUERY)

        selection = select_evidence(
            candidates, self.documents_by_id, topics=("annual_leave",)
        )

        self.assertEqual(selection.authoritative_ids, ("HR-001",))
        self.assertEqual(selection.supplementary_ids, ("CHAT-002",))


class TestAuthorityRules(unittest.TestCase):
    """Authority, status, and topic rules decided on metadata alone."""

    def test_chat_only_evidence_falls_back(self) -> None:
        chat = _chat_document(canonical_source_ids=("HR-901",))
        # The canonical policy exists but is retired, so nothing
        # authoritative remains for this request.
        corpus = _corpus(chat, _policy_document(status="inactive"))

        selection = select_evidence([chat], corpus)

        self.assertFalse(selection.ok)
        self.assertEqual(selection.reason, "no_authoritative_evidence")
        self.assertEqual(selection.answer_evidence, ())

    def test_inactive_policy_candidate_is_excluded(self) -> None:
        retired = _policy_document(status="inactive")
        corpus = _corpus(retired)

        selection = select_evidence([retired], corpus)

        self.assertFalse(selection.ok)
        self.assertEqual(selection.reason, "no_authoritative_evidence")

    def test_policy_outranks_chat_regardless_of_retrieval_order(self) -> None:
        chat = _chat_document(score=0.42)
        policy = _policy_document(score=0.11)
        corpus = _corpus(chat, policy)

        selection = select_evidence([chat, policy], corpus)

        self.assertEqual(
            [document.source_id for document in selection.answer_evidence],
            ["HR-901", "CHAT-901"],
        )
        self.assertEqual(selection.authoritative_ids, ("HR-901",))
        self.assertEqual(selection.supplementary_ids, ("CHAT-901",))

    def test_chat_without_its_policy_in_evidence_is_dropped(self) -> None:
        # The chat points at a retired policy while an unrelated active
        # policy carries the request, so the chat line has no
        # authoritative counterpart and must not reach the reporter.
        chat = _chat_document(canonical_source_ids=("HR-902",))
        retired = _policy_document(source_id="HR-902", status="inactive")
        active = _policy_document(source_id="HR-901")
        corpus = _corpus(chat, retired, active)

        selection = select_evidence([chat, active], corpus)

        self.assertTrue(selection.ok)
        self.assertEqual(selection.authoritative_ids, ("HR-901",))
        self.assertEqual(selection.supplementary_ids, ())

    def test_off_topic_candidates_are_excluded(self) -> None:
        wanted = _policy_document(source_id="HR-901", topics=("annual_leave",))
        unrelated = _policy_document(
            source_id="HR-903", topics=("work_from_home",)
        )
        corpus = _corpus(wanted, unrelated)

        selection = select_evidence(
            [unrelated, wanted], corpus, topics=("annual_leave",)
        )

        self.assertEqual(selection.authoritative_ids, ("HR-901",))

    def test_canonical_policy_outside_top_k_is_pulled_in(self) -> None:
        chat = _chat_document(canonical_source_ids=("HR-901",))
        policy = _policy_document()
        corpus = _corpus(chat, policy)

        selection = select_evidence([chat], corpus)

        self.assertEqual(selection.authoritative_ids, ("HR-901",))
        resolved = selection.answer_evidence[0]
        # A metadata link is not a retrieval match, so the resolved
        # document carries no similarity score of its own.
        self.assertEqual(resolved.score, 0.0)

    def test_citation_of_a_rejected_chat_candidate_is_invalid(self) -> None:
        chat = _chat_document(
            source_id="CHAT-001", canonical_source_ids=("HR-902",)
        )
        retired = _policy_document(source_id="HR-902", status="inactive")
        active = _policy_document(source_id="HR-901")
        corpus = _corpus(chat, retired, active)

        selection = select_evidence([chat, active], corpus)
        result = validate_answer(
            CHAT_CANDIDATE,
            {
                document.source_id
                for document in selection.answer_evidence
            },
            selection.authoritative_ids,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "fabricated_citation")


if __name__ == "__main__":
    unittest.main()
