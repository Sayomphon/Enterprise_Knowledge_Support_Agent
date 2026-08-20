"""Load and validate the Markdown corpus into immutable documents.

The loader is the only component that reads the corpus directory. It fails
fast at startup on any malformed document, duplicate identifier, or path
that escapes the configured directory, so retrieval can never run over a
partially loaded corpus (AGENTS.md sections 6.6 and 7).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import cast

import yaml

from src import config
from src.schemas import Document, SourceType

_FRONTMATTER_DELIMITER = "---"
_REQUIRED_FIELDS = ("source_id", "title", "source_type")
_ALLOWED_SOURCE_TYPES = ("policy", "chat")

# Must match the [SOURCE-ID] citation grammar (project plan section 14) so
# every loaded document can actually be cited and validated at runtime.
_SOURCE_ID_PATTERN = re.compile(r"[A-Z]{2,5}-\d{3}\Z")


class CorpusValidationError(ValueError):
    """Raised when the corpus directory or one document violates the schema."""


def load_documents(corpus_dir: str | Path | None = None) -> list[Document]:
    """Load every ``*.md`` corpus file into validated documents.

    Args:
        corpus_dir: Corpus directory override used by tests; defaults to
            ``config.CORPUS_DIR``. The directory and every file in it are
            resolved before reading so the loader only ever touches files
            that really live inside this directory.

    Returns:
        Documents ordered by file name, so downstream indexing stays
        deterministic across platforms and filesystems.

    Raises:
        CorpusValidationError: If the directory is missing or holds no
            Markdown files, a file resolves outside the corpus directory,
            frontmatter is malformed, a required field is missing or
            invalid, content is empty, or two files share a ``source_id``.
    """
    base = Path(corpus_dir if corpus_dir is not None else config.CORPUS_DIR)
    base = base.resolve()
    if not base.is_dir():
        raise CorpusValidationError(f"Corpus directory does not exist: {base}")

    paths = sorted(base.glob("*.md"))
    if not paths:
        raise CorpusValidationError(
            f"Corpus directory contains no Markdown documents: {base}"
        )

    documents: list[Document] = []
    defined_in: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(base):
            # A symlink inside the directory may point anywhere on disk;
            # refusing it keeps every read inside the configured corpus
            # directory (AGENTS.md section 7, path safety).
            raise CorpusValidationError(
                f"{path.name}: resolves outside the corpus directory"
            )
        document = _parse_document(resolved)
        if document.source_id in defined_in:
            raise CorpusValidationError(
                f"{path.name}: duplicate source_id {document.source_id!r} "
                f"already defined by {defined_in[document.source_id]}"
            )
        defined_in[document.source_id] = path.name
        documents.append(document)
    return documents


def _parse_document(path: Path) -> Document:
    """Parse one Markdown file with YAML frontmatter into a document.

    Args:
        path: Resolved path of the corpus file to read.

    Returns:
        The validated, NFC-normalized document.

    Raises:
        CorpusValidationError: If the file is not UTF-8, the frontmatter is
            malformed, a metadata field is invalid, or the body is empty.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CorpusValidationError(f"{path.name}: not valid UTF-8") from exc

    frontmatter_text, body = _split_frontmatter(path.name, raw_text)
    metadata = _validated_metadata(path.name, frontmatter_text)
    if not body.strip():
        raise CorpusValidationError(f"{path.name}: document content is empty")

    # NFC keeps Thai combining characters in one canonical byte form, so
    # retrieval never misses a match because two files encoded the same
    # visible text differently.
    return Document(
        source_id=metadata["source_id"],
        title=metadata["title"],
        source_type=cast(SourceType, metadata["source_type"]),
        content=unicodedata.normalize("NFC", body.strip()),
    )


def _split_frontmatter(file_name: str, raw_text: str) -> tuple[str, str]:
    """Split a raw corpus file into its frontmatter block and Markdown body.

    Args:
        file_name: File name used to build actionable error messages.
        raw_text: Complete decoded file contents.

    Returns:
        A ``(frontmatter, body)`` pair, both without the delimiter lines.

    Raises:
        CorpusValidationError: If the file does not start with ``---`` or
            the frontmatter block is never closed.
    """
    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIMITER:
        raise CorpusValidationError(
            f"{file_name}: document must start with a '---' frontmatter block"
        )
    closing_index = next(
        (
            index
            for index in range(1, len(lines))
            if lines[index].strip() == _FRONTMATTER_DELIMITER
        ),
        None,
    )
    if closing_index is None:
        raise CorpusValidationError(
            f"{file_name}: frontmatter block is never closed with '---'"
        )
    frontmatter_text = "\n".join(lines[1:closing_index])
    body = "\n".join(lines[closing_index + 1 :])
    return frontmatter_text, body


def _validated_metadata(file_name: str, frontmatter_text: str) -> dict[str, str]:
    """Validate the frontmatter block against the corpus metadata schema.

    Args:
        file_name: File name used to build actionable error messages.
        frontmatter_text: Raw YAML between the frontmatter delimiters.

    Returns:
        A mapping with NFC-normalized ``source_id``, ``title``, and
        ``source_type`` values.

    Raises:
        CorpusValidationError: If the YAML does not parse to a mapping, a
            required field is missing, blank, or not a string, the
            ``source_id`` does not follow the citation grammar, or the
            ``source_type`` is unknown.
    """
    try:
        loaded = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as exc:
        raise CorpusValidationError(
            f"{file_name}: frontmatter is not valid YAML"
        ) from exc
    if not isinstance(loaded, dict):
        raise CorpusValidationError(
            f"{file_name}: frontmatter must be a YAML mapping"
        )

    metadata: dict[str, str] = {}
    for field in _REQUIRED_FIELDS:
        value = loaded.get(field)
        if not isinstance(value, str) or not value.strip():
            raise CorpusValidationError(
                f"{file_name}: frontmatter field {field!r} must be a "
                "non-empty string"
            )
        metadata[field] = unicodedata.normalize("NFC", value.strip())

    if not _SOURCE_ID_PATTERN.fullmatch(metadata["source_id"]):
        raise CorpusValidationError(
            f"{file_name}: source_id {metadata['source_id']!r} must match "
            "the citation grammar, e.g. 'HR-001'"
        )
    if metadata["source_type"] not in _ALLOWED_SOURCE_TYPES:
        raise CorpusValidationError(
            f"{file_name}: source_type must be one of: "
            f"{', '.join(_ALLOWED_SOURCE_TYPES)}"
        )
    return metadata
