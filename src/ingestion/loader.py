"""Load and validate the Markdown corpus into immutable documents.

The loader is the only component that reads the corpus directory. It fails
fast at startup on any malformed document, duplicate identifier, broken
authority metadata, or path that escapes the configured directory, so
retrieval can never run over a partially loaded corpus (AGENTS.md
sections 6.6 and 7).

It also offers a seam for screening document text at ingestion, which the
graph fills with the injection screen. What that screen does depends on
the stratum, because the two carry different authority:

    - a CHAT transcript is noisy third-party text and the realistic
      carrier of an indirect injection, so a line shaped like an
      instruction is replaced with a placeholder and the rest of the
      transcript keeps the retrieval recall it was loaded for
    - a POLICY document is the corpus's own authority, so an instruction
      inside one is a corrupt corpus and the load fails (section 6.6)

Quarantine contains what the model is shown; it is not the guarantee. A
paraphrase this finite screen misses still reaches the reporter, and what
stops it from becoming an answer is the claim contract downstream, which
requires every claim to quote a policy document.

The screen is passed in rather than imported because the guardrails sit
above this module in the dependency order (AGENTS.md section 3).
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Callable
from dataclasses import replace
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

# What replaces a quarantined line. It says what happened rather than
# deleting the line silently: the reporter sees that something was
# removed instead of reading a transcript that appears to flow, and a
# reviewer comparing the file with the loaded document can see where.
QUARANTINE_PLACEHOLDER = "[[REDACTED: instruction-shaped line]]"


class CorpusValidationError(ValueError):
    """Raised when the corpus directory or one document violates the schema."""


def load_documents(
    corpus_dir: str | Path | None = None,
    *,
    content_screen: Callable[[str], str | None] | None = None,
) -> list[Document]:
    """Load every ``*.md`` corpus file into validated documents.

    Args:
        corpus_dir: Corpus directory override used by tests; defaults to
            ``config.CORPUS_DIR``. The directory and every file in it are
            resolved before reading so the loader only ever touches files
            that really live inside this directory.
        content_screen: Optional screen applied line by line to each
            document body, returning the id of a rule the line matches or
            ``None``. Chat lines that match are quarantined; a policy
            document that matches fails the load. Omitting it loads every
            body verbatim, which is what the tests that are not about
            this screen do.

    Returns:
        Documents ordered by file name, so downstream indexing stays
        deterministic across platforms and filesystems. Chat bodies carry
        the quarantined form, which is the only form any later stage sees.

    Raises:
        CorpusValidationError: If the directory is missing or holds no
            Markdown files, a file resolves outside the corpus directory,
            frontmatter is malformed, a required field is missing or
            invalid, content is empty, two files share a ``source_id``,
            a canonical link is broken, or a policy document carries
            instruction-shaped text.
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
        if content_screen is not None:
            document = _screened(document, path.name, content_screen)
        if document.source_id in defined_in:
            raise CorpusValidationError(
                f"{path.name}: duplicate source_id {document.source_id!r} "
                f"already defined by {defined_in[document.source_id]}"
            )
        defined_in[document.source_id] = path.name
        documents.append(document)
    _validate_canonical_links(documents, defined_in)
    return documents


def sanitize_untrusted_content(
    content: str, content_screen: Callable[[str], str | None]
) -> tuple[str, int]:
    """Replace every instruction-shaped line of a body with a placeholder.

    Per line rather than per document on purpose: a chat transcript is
    loaded for the informal vocabulary that gives slang questions their
    recall, and dropping the whole file to remove one planted line would
    pay for the defence with the evidence it was defending.

    Args:
        content: The document body, already NFC-normalized.
        content_screen: Screen returning a matched rule id, or ``None``.

    Returns:
        The sanitized body and how many lines it replaced. The body is
        returned unchanged, and the count is zero, when nothing matched.
    """
    kept: list[str] = []
    quarantined = 0
    for line in content.splitlines():
        if content_screen(line) is None:
            kept.append(line)
            continue
        kept.append(QUARANTINE_PLACEHOLDER)
        quarantined += 1
    # Rejoining an untouched body would also rewrite its line endings,
    # so a body with nothing to quarantine is returned as it arrived.
    return ("\n".join(kept) if quarantined else content, quarantined)


def _screened(
    document: Document,
    file_name: str,
    content_screen: Callable[[str], str | None],
) -> Document:
    """Apply the ingestion screen appropriate to a document's stratum.

    Args:
        document: The parsed document.
        file_name: File name used to build actionable messages.
        content_screen: Screen returning a matched rule id, or ``None``.

    Returns:
        The same document for a policy, and the quarantined form for a
        chat transcript, with its replaced-line count recorded.

    Raises:
        CorpusValidationError: If a policy document carries
            instruction-shaped text.
    """
    if document.source_type == "policy":
        rule_id = content_screen(document.content)
        if rule_id is not None:
            # Fail fast rather than quarantine: this document IS the rule
            # the assistant states, so text inside it that addresses the
            # model means the authority itself is compromised, and a
            # partially trusted policy is not a thing this pipeline can
            # reason about (AGENTS.md section 6.6).
            raise CorpusValidationError(
                f"{file_name}: policy document carries instruction-shaped "
                f"text matching injection rule {rule_id}"
            )
        return document
    content, quarantined = sanitize_untrusted_content(
        document.content, content_screen
    )
    if quarantined:
        # The file and the count, never the text: the quarantined line is
        # exactly the string that must not be repeated anywhere a person
        # or a log might read it as instruction.
        print(
            f"loader: {file_name} had {quarantined} instruction-shaped "
            "line(s) quarantined before indexing",
            file=sys.stderr,
        )
    return replace(
        document, content=content, quarantined_line_count=quarantined
    )


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
            if target.status != "active":
                # Authority alone is not enough: evidence selection drops a
                # retired policy, so a chat document whose only canonical
                # link points at one becomes evidence that can never
                # support an answer. Without this check the corpus loads
                # clean and every query it serves falls back with
                # "no_authoritative_evidence" -- a runtime symptom of a
                # metadata fault the loader promises to catch at start-up.
                raise CorpusValidationError(
                    f"{file_name}: canonical_source_ids must reference an "
                    f"active policy, but {canonical_id!r} is "
                    f"{target.status}"
                )


class _StrictLoader(yaml.SafeLoader):
    """A safe loader that refuses a mapping with a repeated key.

    ``yaml.safe_load`` keeps the LAST occurrence of a duplicated key and
    says nothing. For this corpus that is a correctness fault, not a
    style one: a file declaring ``source_id`` twice loads with an id
    different from the one a reviewer reads at the top of it, the
    duplicate-id check downstream only ever sees the surviving value, and
    the mismatch then travels into retrieval ranking, citation
    validation, and the ``[ID]`` markers an employee reads.
    """

    def construct_mapping(self, node, deep: bool = False) -> dict:
        """Build one mapping, rejecting a key that appears twice."""
        seen: set = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    f"duplicate frontmatter key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep)


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
        loaded = yaml.load(frontmatter_text, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        # ``problem`` carries the useful half of a MarkedYAMLError; args[0]
        # is its context, which is None for the duplicate-key case.
        detail = getattr(exc, "problem", None)
        raise CorpusValidationError(
            f"{file_name}: frontmatter is not valid YAML"
            + (f" ({detail})" if detail else "")
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
