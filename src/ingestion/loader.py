"""Load and validate the Markdown corpus into immutable documents.

The loader is the only component that reads the corpus directory. It fails
fast at startup on any malformed document, duplicate identifier, broken
authority metadata, or path that escapes the configured directory, so
retrieval can never run over a partially loaded corpus (AGENTS.md
sections 6.6 and 7).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import cast

import yaml

from src import config
from src.schemas import (
    KNOWLEDGE_TOPICS,
    Authority,
    Document,
    DocumentStatus,
    KnowledgeTopic,
    SourceType,
)

_FRONTMATTER_DELIMITER = "---"
_REQUIRED_FIELDS = ("source_id", "title", "source_type", "authority", "status")
_ALLOWED_SOURCE_TYPES = ("policy", "chat")
_ALLOWED_STATUSES = ("active", "inactive")

# Authority is derived from the corpus stratum, not chosen per file: a
# policy document is the rule, a chat transcript only illustrates it. The
# loader enforces the binding so no future document can silently promote
# chat evidence to authoritative (remediation plan Finding 4).
_AUTHORITY_BY_SOURCE_TYPE = {
    "policy": "authoritative",
    "chat": "supplementary",
}

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
            invalid, content is empty, two files share a ``source_id``,
            or a canonical link is broken.
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
    _validate_canonical_links(documents, defined_in)
    return documents


def _validate_canonical_links(
    documents: list[Document], defined_in: dict[str, str]
) -> None:
    """Reject canonical links that no complete corpus could satisfy.

    This runs only after every file has been parsed, because a link can
    only be checked once the complete id set is known.

    Args:
        documents: All parsed documents of the corpus.
        defined_in: Mapping from ``source_id`` to the file that defined
            it, used for actionable error messages.

    Raises:
        CorpusValidationError: If a canonical id is unknown, points at a
            document that is not authoritative policy, or points back at
            the document itself.
    """
    by_id = {document.source_id: document for document in documents}
    for document in documents:
        file_name = defined_in[document.source_id]
        for canonical_id in document.canonical_source_ids:
            target = by_id.get(canonical_id)
            if target is None:
                raise CorpusValidationError(
                    f"{file_name}: canonical_source_ids references unknown "
                    f"source_id {canonical_id!r}"
                )
            if target.source_id == document.source_id:
                raise CorpusValidationError(
                    f"{file_name}: canonical_source_ids must not reference "
                    "the document itself"
                )
            if target.authority != "authoritative":
                raise CorpusValidationError(
                    f"{file_name}: canonical_source_ids must reference "
                    f"authoritative policy, but {canonical_id!r} is "
                    f"{target.authority}"
                )


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
    topics = _validated_topics(path.name, metadata["raw"])
    canonical_source_ids = _validated_canonical_ids(
        path.name, metadata["raw"], cast(SourceType, metadata["source_type"])
    )
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
        authority=cast(Authority, metadata["authority"]),
        status=cast(DocumentStatus, metadata["status"]),
        topics=topics,
        canonical_source_ids=canonical_source_ids,
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


def _validated_metadata(file_name: str, frontmatter_text: str) -> dict:
    """Validate the scalar frontmatter fields against the metadata schema.

    Args:
        file_name: File name used to build actionable error messages.
        frontmatter_text: Raw YAML between the frontmatter delimiters.

    Returns:
        A mapping with the NFC-normalized scalar fields plus ``"raw"``,
        the parsed YAML mapping used by the list-field validators.

    Raises:
        CorpusValidationError: If the YAML does not parse to a mapping, a
            required field is missing, blank, or not a string, the
            ``source_id`` does not follow the citation grammar, the
            ``source_type`` or ``status`` is unknown, or the declared
            ``authority`` contradicts the ``source_type``.
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

    metadata: dict = {"raw": loaded}
    for field_name in _REQUIRED_FIELDS:
        value = loaded.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise CorpusValidationError(
                f"{file_name}: frontmatter field {field_name!r} must be a "
                "non-empty string"
            )
        metadata[field_name] = unicodedata.normalize("NFC", value.strip())

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
    if metadata["status"] not in _ALLOWED_STATUSES:
        raise CorpusValidationError(
            f"{file_name}: status must be one of: "
            f"{', '.join(_ALLOWED_STATUSES)}"
        )
    expected_authority = _AUTHORITY_BY_SOURCE_TYPE[metadata["source_type"]]
    if metadata["authority"] != expected_authority:
        raise CorpusValidationError(
            f"{file_name}: authority {metadata['authority']!r} contradicts "
            f"source_type {metadata['source_type']!r}, which is always "
            f"{expected_authority!r}"
        )
    return metadata


def _validated_topics(
    file_name: str, loaded: dict
) -> tuple[KnowledgeTopic, ...]:
    """Validate the declared topics against the bounded topic catalog.

    Args:
        file_name: File name used to build actionable error messages.
        loaded: Parsed frontmatter mapping.

    Returns:
        The declared topics in file order, without duplicates.

    Raises:
        CorpusValidationError: If ``topics`` is missing, not a list of
            non-empty strings, empty, duplicated, or names a topic
            outside ``KNOWLEDGE_TOPICS``.
    """
    raw_topics = loaded.get("topics")
    if not isinstance(raw_topics, list) or not raw_topics:
        raise CorpusValidationError(
            f"{file_name}: frontmatter field 'topics' must be a non-empty list"
        )
    topics: list[str] = []
    for raw_topic in raw_topics:
        if not isinstance(raw_topic, str) or not raw_topic.strip():
            raise CorpusValidationError(
                f"{file_name}: every entry of 'topics' must be a non-empty "
                "string"
            )
        topic = raw_topic.strip()
        if topic not in KNOWLEDGE_TOPICS:
            raise CorpusValidationError(
                f"{file_name}: unknown topic {topic!r}; supported topics "
                f"are: {', '.join(KNOWLEDGE_TOPICS)}"
            )
        if topic in topics:
            raise CorpusValidationError(
                f"{file_name}: duplicate topic {topic!r}"
            )
        topics.append(topic)
    return cast(tuple[KnowledgeTopic, ...], tuple(topics))


def _validated_canonical_ids(
    file_name: str, loaded: dict, source_type: SourceType
) -> tuple[str, ...]:
    """Validate the canonical policy links declared by one document.

    Membership of the referenced ids in the corpus is checked later by
    ``_validate_canonical_links``, once every file has been parsed.

    Args:
        file_name: File name used to build actionable error messages.
        loaded: Parsed frontmatter mapping.
        source_type: Corpus stratum of the document being parsed.

    Returns:
        The declared canonical ids in file order, without duplicates.

    Raises:
        CorpusValidationError: If the field is missing or malformed, a
            policy document declares canonical links, or a chat document
            declares none. Chat evidence without a policy behind it could
            only support an answer on its own, which the evidence
            selector forbids, so it is rejected at startup instead.
    """
    raw_ids = loaded.get("canonical_source_ids", [])
    if not isinstance(raw_ids, list):
        raise CorpusValidationError(
            f"{file_name}: frontmatter field 'canonical_source_ids' must be "
            "a list"
        )
    canonical_ids: list[str] = []
    for raw_id in raw_ids:
        if not isinstance(raw_id, str) or not _SOURCE_ID_PATTERN.fullmatch(
            raw_id.strip()
        ):
            raise CorpusValidationError(
                f"{file_name}: every entry of 'canonical_source_ids' must "
                "match the citation grammar, e.g. 'FIN-001'"
            )
        canonical_id = raw_id.strip()
        if canonical_id in canonical_ids:
            raise CorpusValidationError(
                f"{file_name}: duplicate canonical source id "
                f"{canonical_id!r}"
            )
        canonical_ids.append(canonical_id)

    if source_type == "policy" and canonical_ids:
        raise CorpusValidationError(
            f"{file_name}: a policy document is already authoritative and "
            "must not declare canonical_source_ids"
        )
    if source_type == "chat" and not canonical_ids:
        raise CorpusValidationError(
            f"{file_name}: a chat document must declare at least one "
            "canonical policy in 'canonical_source_ids'"
        )
    return tuple(canonical_ids)
