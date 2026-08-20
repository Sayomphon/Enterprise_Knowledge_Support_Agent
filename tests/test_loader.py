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
---

Body text used by the loader tests.
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
) -> str:
    """Build one frontmatter document with selectively overridden fields."""
    return (
        f"---\n"
        f"source_id: {source_id}\n"
        f"title: {title}\n"
        f"source_type: {source_type}\n"
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
            "---\nsource_id: HR-901\nsource_type: policy\n---\n\nBody.\n",
        )

        with self.assertRaisesRegex(CorpusValidationError, "'title'"):
            load_documents(self.corpus_dir)

    def test_blank_title_fails_fast(self) -> None:
        _write(
            self.corpus_dir,
            "doc.md",
            "---\nsource_id: HR-901\ntitle: '  '\nsource_type: policy\n"
            "---\n\nBody.\n",
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
            "---\nsource_id: 901\ntitle: A title\nsource_type: policy\n"
            "---\n\nBody.\n",
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
            "---\nsource_id: HR-901\ntitle: A title\nsource_type: policy\n"
            "---\n\n   \n",
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


if __name__ == "__main__":
    unittest.main()
