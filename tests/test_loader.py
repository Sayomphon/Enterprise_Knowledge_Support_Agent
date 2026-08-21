"""Behavioural tests for the corpus loader.

Every test runs offline against either the real ``data/docs`` corpus or a
temporary synthetic corpus, so no network access or API key is required
(AGENTS.md section 9).
"""

import os
import tempfile
import unittest
from pathlib import Path

from src.ingestion.loader import CorpusValidationError, load_documents
from src.schemas import KNOWLEDGE_TOPICS

EXPECTED_SOURCE_IDS = {
    "FIN-001",
    "FIN-002",
    "HR-001",
    "HR-002",
    "HR-003",
    "CHAT-001",
    "CHAT-002",
    "CHAT-003",
}

VALID_DOCUMENT = """---
source_id: HR-901
title: Synthetic loader test document
source_type: policy
authority: authoritative
status: active
topics:
  - annual_leave
canonical_source_ids: []
---

Body text used by the loader tests.
"""

# One chat document that points at HR-901, used by the canonical-link tests.
VALID_CHAT_DOCUMENT = """---
source_id: CHAT-901
title: Synthetic loader test chat
source_type: chat
authority: supplementary
status: active
topics:
  - annual_leave
canonical_source_ids:
  - HR-901
---

Chat body text used by the loader tests.
"""


def _write(directory: Path, name: str, text: str) -> Path:
    """Write one synthetic corpus file and return its path."""
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def _document_text(
    source_id: str = "HR-901",
    title: str = "Synthetic loader test document",
    source_type: str = "policy",
    body: str = "Body text used by the loader tests.",
    authority: str = "authoritative",
    status: str = "active",
    topics: str = "\n  - annual_leave",
    canonical_source_ids: str = " []",
) -> str:
    """Build one frontmatter document with selectively overridden fields."""
    return (
        f"---\n"
        f"source_id: {source_id}\n"
        f"title: {title}\n"
        f"source_type: {source_type}\n"
        f"authority: {authority}\n"
        f"status: {status}\n"
        f"topics:{topics}\n"
        f"canonical_source_ids:{canonical_source_ids}\n"
        f"---\n\n"
        f"{body}\n"
    )


class TestRealCorpus(unittest.TestCase):
    """The shipped mock corpus must satisfy the loader contract."""

    def test_all_eight_corpus_documents_load_with_unique_metadata(self) -> None:
        documents = load_documents()

        self.assertEqual(len(documents), 8)
        self.assertEqual(
            {document.source_id for document in documents},
            EXPECTED_SOURCE_IDS,
        )
        for document in documents:
            with self.subTest(source_id=document.source_id):
                self.assertTrue(document.title.strip())
                self.assertTrue(document.content.strip())
                self.assertIn(document.source_type, {"policy", "chat"})

    def test_corpus_contains_five_policy_and_three_chat_documents(self) -> None:
        documents = load_documents()

        by_type = {"policy": 0, "chat": 0}
        for document in documents:
            by_type[document.source_type] += 1
        self.assertEqual(by_type, {"policy": 5, "chat": 3})

    def test_documents_are_ordered_by_file_name_for_determinism(self) -> None:
        first = [document.source_id for document in load_documents()]
        second = [document.source_id for document in load_documents()]

        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))


class TestLoaderValidation(unittest.TestCase):
    """Malformed corpora must fail fast with an actionable error."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.corpus_dir = Path(self._tmp.name)

    def test_valid_synthetic_document_loads(self) -> None:
        _write(self.corpus_dir, "doc.md", VALID_DOCUMENT)

        documents = load_documents(self.corpus_dir)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].source_id, "HR-901")
        self.assertEqual(documents[0].source_type, "policy")

    def test_duplicate_source_id_fails_fast(self) -> None:
        _write(self.corpus_dir, "first.md", VALID_DOCUMENT)
        _write(self.corpus_dir, "second.md", VALID_DOCUMENT)

        with self.assertRaisesRegex(
            CorpusValidationError, "duplicate source_id"
        ) as context:
            load_documents(self.corpus_dir)
        self.assertIn("first.md", str(context.exception))

    def test_missing_title_fails_fast(self) -> None:
        _write(
            self.corpus_dir,
            "doc.md",
            _document_text(title="A title").replace("title: A title\n", ""),
        )

        with self.assertRaisesRegex(CorpusValidationError, "'title'"):
            load_documents(self.corpus_dir)

    def test_blank_title_fails_fast(self) -> None:
        _write(
            self.corpus_dir,
            "doc.md",
            _document_text(title="'  '"),
        )

        with self.assertRaisesRegex(CorpusValidationError, "'title'"):
            load_documents(self.corpus_dir)

    def test_malformed_frontmatter_fails_fast(self) -> None:
        malformed_files = {
            "missing_block.md": "No frontmatter at all.\n",
            "unclosed_block.md": "---\nsource_id: HR-901\n\nBody.\n",
            "not_a_mapping.md": "---\n- just\n- a list\n---\n\nBody.\n",
        }
        for name, text in malformed_files.items():
            with self.subTest(file=name):
                _write(self.corpus_dir, name, text)
                with self.assertRaises(CorpusValidationError):
                    load_documents(self.corpus_dir)
                (self.corpus_dir / name).unlink()

    def test_non_string_source_id_fails_fast(self) -> None:
        _write(
            self.corpus_dir,
            "doc.md",
            _document_text(source_id="901"),
        )

        with self.assertRaisesRegex(CorpusValidationError, "'source_id'"):
            load_documents(self.corpus_dir)

    def test_source_id_outside_citation_grammar_fails_fast(self) -> None:
        _write(
            self.corpus_dir,
            "doc.md",
            _document_text(source_id="hr-901"),
        )

        with self.assertRaisesRegex(CorpusValidationError, "citation grammar"):
            load_documents(self.corpus_dir)

    def test_unknown_source_type_fails_fast(self) -> None:
        _write(
            self.corpus_dir,
            "doc.md",
            _document_text(source_type="memo"),
        )

        with self.assertRaisesRegex(CorpusValidationError, "source_type"):
            load_documents(self.corpus_dir)

    def test_empty_content_fails_fast(self) -> None:
        _write(
            self.corpus_dir,
            "doc.md",
            _document_text(body="   "),
        )

        with self.assertRaisesRegex(CorpusValidationError, "content is empty"):
            load_documents(self.corpus_dir)

    def test_missing_corpus_directory_fails_fast(self) -> None:
        with self.assertRaisesRegex(CorpusValidationError, "does not exist"):
            load_documents(self.corpus_dir / "missing")

    def test_empty_corpus_directory_fails_fast(self) -> None:
        with self.assertRaisesRegex(CorpusValidationError, "no Markdown"):
            load_documents(self.corpus_dir)

    def test_symlink_escaping_the_corpus_directory_is_rejected(self) -> None:
        outside_dir = tempfile.TemporaryDirectory()
        self.addCleanup(outside_dir.cleanup)
        outside_file = _write(
            Path(outside_dir.name), "outside.md", VALID_DOCUMENT
        )
        os.symlink(outside_file, self.corpus_dir / "linked.md")

        with self.assertRaisesRegex(
            CorpusValidationError, "outside the corpus directory"
        ):
            load_documents(self.corpus_dir)


class TestAuthorityMetadata(unittest.TestCase):
    """Authority, topic, and canonical-link metadata must be trustworthy.

    Evidence selection and the supported-scope gate both read this
    metadata, so a corpus that contradicts itself has to fail at startup
    rather than at answer time (remediation plan Finding 4).
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.corpus_dir = Path(self._tmp.name)

    def test_real_corpus_carries_consistent_authority_metadata(self) -> None:
        documents = {
            document.source_id: document for document in load_documents()
        }

        for document in documents.values():
            with self.subTest(source_id=document.source_id):
                expected_authority = (
                    "authoritative"
                    if document.source_type == "policy"
                    else "supplementary"
                )
                self.assertEqual(document.authority, expected_authority)
                self.assertEqual(document.status, "active")
                self.assertTrue(document.topics)
                for topic in document.topics:
                    self.assertIn(topic, KNOWLEDGE_TOPICS)

    def test_real_corpus_chat_documents_link_to_policy(self) -> None:
        documents = {
            document.source_id: document for document in load_documents()
        }

        chat_links = {
            document.source_id: document.canonical_source_ids
            for document in documents.values()
            if document.source_type == "chat"
        }
        self.assertEqual(
            chat_links,
            {
                "CHAT-001": ("FIN-001", "FIN-002"),
                "CHAT-002": ("HR-001",),
                "CHAT-003": ("FIN-001", "FIN-002"),
            },
        )
        for canonical_ids in chat_links.values():
            for canonical_id in canonical_ids:
                self.assertEqual(
                    documents[canonical_id].authority, "authoritative"
                )

    def test_authority_contradicting_source_type_fails_fast(self) -> None:
        _write(
            self.corpus_dir,
            "doc.md",
            _document_text(
                source_id="CHAT-901",
                source_type="chat",
                authority="authoritative",
                canonical_source_ids="\n  - HR-901",
            ),
        )

        with self.assertRaisesRegex(CorpusValidationError, "contradicts"):
            load_documents(self.corpus_dir)

    def test_unknown_status_fails_fast(self) -> None:
        _write(self.corpus_dir, "doc.md", _document_text(status="draft"))

        with self.assertRaisesRegex(CorpusValidationError, "status"):
            load_documents(self.corpus_dir)

    def test_unknown_topic_fails_fast(self) -> None:
        _write(
            self.corpus_dir,
            "doc.md",
            _document_text(topics="\n  - maternity_leave"),
        )

        with self.assertRaisesRegex(CorpusValidationError, "unknown topic"):
            load_documents(self.corpus_dir)

    def test_empty_topic_list_fails_fast(self) -> None:
        _write(self.corpus_dir, "doc.md", _document_text(topics=" []"))

        with self.assertRaisesRegex(CorpusValidationError, "'topics'"):
            load_documents(self.corpus_dir)

    def test_policy_declaring_canonical_links_fails_fast(self) -> None:
        _write(self.corpus_dir, "policy.md", VALID_DOCUMENT)
        _write(
            self.corpus_dir,
            "doc.md",
            _document_text(
                source_id="HR-902", canonical_source_ids="\n  - HR-901"
            ),
        )

        with self.assertRaisesRegex(
            CorpusValidationError, "must not declare canonical_source_ids"
        ):
            load_documents(self.corpus_dir)

    def test_chat_without_canonical_link_fails_fast(self) -> None:
        _write(
            self.corpus_dir,
            "doc.md",
            _document_text(
                source_id="CHAT-901",
                source_type="chat",
                authority="supplementary",
            ),
        )

        with self.assertRaisesRegex(
            CorpusValidationError, "at least one canonical policy"
        ):
            load_documents(self.corpus_dir)

    def test_canonical_link_to_a_missing_document_fails_fast(self) -> None:
        _write(
            self.corpus_dir,
            "chat.md",
            _document_text(
                source_id="CHAT-901",
                source_type="chat",
                authority="supplementary",
                canonical_source_ids="\n  - HR-999",
            ),
        )

        with self.assertRaisesRegex(CorpusValidationError, "unknown source_id"):
            load_documents(self.corpus_dir)

    def test_canonical_link_to_another_chat_fails_fast(self) -> None:
        _write(self.corpus_dir, "chat_a.md", VALID_CHAT_DOCUMENT)
        _write(self.corpus_dir, "policy.md", VALID_DOCUMENT)
        _write(
            self.corpus_dir,
            "chat_b.md",
            _document_text(
                source_id="CHAT-902",
                source_type="chat",
                authority="supplementary",
                canonical_source_ids="\n  - CHAT-901",
            ),
        )

        with self.assertRaisesRegex(
            CorpusValidationError, "must reference authoritative policy"
        ):
            load_documents(self.corpus_dir)

    def test_inactive_policy_loads_and_keeps_its_status(self) -> None:
        # The loader records the lifecycle flag; excluding retired
        # documents from an answer is the evidence selector's job.
        _write(self.corpus_dir, "doc.md", _document_text(status="inactive"))

        documents = load_documents(self.corpus_dir)

        self.assertEqual(documents[0].status, "inactive")


if __name__ == "__main__":
    unittest.main()
