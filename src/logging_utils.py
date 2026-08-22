"""Append-only JSONL telemetry for blocked and fallback events.

The writer accepts only the allowlisted fields of the AGENTS.md section 8
schema: query text, reason code, the two retrieval scores, top source
ids, rewritten queries, the count of deterministic alias variants
searched, and two request-cost figures -- elapsed milliseconds and
provider calls spent. Secrets, system prompts, and provider payloads
cannot pass through this interface because the signature does not accept
them.

Writing stays best-effort -- a filesystem failure must never break a user
request -- but it is no longer silent: the writer reports the outcome so
the caller can tell the employee the truth about whether the question was
recorded (remediation plan Finding 7).

Reading back is bounded on purpose. A long-running local demo appends
without limit, and the audit view only ever shows the newest rows, so the
reader walks the file backwards from the end instead of loading it whole.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from src import config
from src.schemas import LogReadResult, LogWriteResult

# Size of one backward read while tailing the sink. Large enough that the
# newest page of records is normally one seek, small enough that a large
# log never enters memory whole.
_TAIL_BLOCK_BYTES = 65536

# Hard ceiling on the query text of one record. The bound lives at the
# writer rather than at any one caller because this is the single choke
# point into an append-only file with no rotation: a query rejected FOR
# BEING TOO LONG is still logged, and the guardrail's own limit bounds
# what the pipeline processes, not what reaches the sink. The marker keeps
# a truncated record honest about being truncated.
_MAX_LOGGED_QUERY_CHARS = 1000
_TRUNCATION_MARKER = "...[truncated]"


def log_fallback_event(
    *,
    query: str,
    reason: str,
    raw_retrieval_score: float | None,
    expanded_retrieval_score: float | None,
    top_sources: Sequence[str],
    rewritten_queries: Sequence[str],
    alias_query_count: int = 0,
    latency_ms: int | None = None,
    llm_calls: int = 0,
    log_path: str | Path | None = None,
    now: Callable[[], datetime] | None = None,
) -> LogWriteResult:
    """Append one blocked/fallback event as a single JSONL line.

    Writes are best-effort: a logging failure must never break the user
    request, so filesystem and serialization errors alike are reported on
    stderr by exception type only and then returned as a failed result
    (AGENTS.md section 8).

    Args:
        query: The user query that was blocked or fell back.
        reason: Reason code from ``src.fallback.ReasonCode``.
        raw_retrieval_score: Top-1 original-retrieval score, if reached.
        expanded_retrieval_score: Top-1 expanded score, if reached.
        top_sources: Source ids of the retrieved documents, best first.
        rewritten_queries: Rewritten query variants, empty when no
            rewrite ran. Model output only: the deterministic alias
            variants are counted, not quoted, because their text is
            recoverable from the topic catalog and repeating it in every
            medium-band record would double the size of the sink to say
            nothing new.
        alias_query_count: How many alias variants the deterministic
            expansion searched, zero outside the medium band. It is what
            distinguishes "expansion found nothing" from "no expansion
            ran" when reading a fallback back.
        latency_ms: Wall-clock milliseconds the request spent before it
            degraded, or ``None`` when the caller tracked no start. It
            is what makes a deadline-exceeded record checkable rather
            than merely asserted.
        llm_calls: Provider calls this request actually spent. Zero on
            every deterministic route, and the number an operator needs
            to tell a rewrite that ran from one that was skipped.
        log_path: Destination override used by tests so they never write
            into the real ``logs/`` directory; defaults to
            ``config.FALLBACK_LOG_PATH``.
        now: Clock override used by tests; defaults to the local system
            clock. The timestamp is always timezone-aware.

    Returns:
        A ``LogWriteResult`` recording whether the append happened. The
        caller decides what to do with a failure; this function never
        raises one at it.
    """
    path = Path(log_path if log_path is not None else config.FALLBACK_LOG_PATH)
    timestamp = now() if now is not None else datetime.now().astimezone()
    try:
        record = {
            "timestamp": timestamp.isoformat(),
            "query": _bounded_query(query),
            "reason": str(reason),
            "raw_retrieval_score": raw_retrieval_score,
            "expanded_retrieval_score": expanded_retrieval_score,
            "top_sources": list(top_sources),
            "rewritten_queries": list(rewritten_queries),
            "alias_query_count": alias_query_count,
            "latency_ms": latency_ms,
            "llm_calls": llm_calls,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Serialization belongs inside the handler, not only the filesystem
    # call: a caller may hand this writer a query the guardrail rejected
    # for not being a string, and ``json.dumps`` answers that with
    # TypeError while ``handle.write`` answers a lone surrogate with
    # UnicodeEncodeError. Both used to escape the refusal node and kill
    # the request, which is what "best-effort" exists to prevent
    # (AGENTS.md section 4, invariant 9).
    except (OSError, TypeError, ValueError) as exc:
        print(
            "logging_utils: failed to append fallback event "
            f"({type(exc).__name__})",
            file=sys.stderr,
        )
        return LogWriteResult(ok=False, error_type=type(exc).__name__)
    return LogWriteResult(ok=True)


def _bounded_query(query: str) -> str:
    """Cap one record's query text so the sink cannot grow without bound.

    Args:
        query: Query text as the caller supplied it.

    Returns:
        The text unchanged when it fits, otherwise its first
        ``_MAX_LOGGED_QUERY_CHARS`` characters followed by a marker.

    Raises:
        TypeError: If ``query`` is not a string. The AGENTS.md section 8
            schema types this field as text, and ``json.dumps`` would
            silently accept a dict or an int and write a record no reader
            of that schema can trust. Rejecting it here makes every
            non-string behave the same way -- a reported failed write --
            instead of depending on whether the payload happened to be
            JSON-serialisable. The caller's own handler catches it.
    """
    if not isinstance(query, str):
        raise TypeError("log record 'query' must be a string")
    if len(query) <= _MAX_LOGGED_QUERY_CHARS:
        return query
    return query[:_MAX_LOGGED_QUERY_CHARS] + _TRUNCATION_MARKER


def read_recent_events(
    limit: int, log_path: str | Path | None = None
) -> LogReadResult:
    """Read the newest events from the sink without loading it whole.

    The audit view shows a fixed number of newest rows, so the reader
    seeks to the end of the file and walks backwards one block at a time
    until it holds enough complete lines. A demo that has been appending
    all day therefore costs one bounded read rather than the whole file.

    Malformed lines are tolerated and counted, never rendered: an
    interrupted append leaves a partial final line, and a partial line is
    still raw query text that has no business being displayed as an error.

    Args:
        limit: Maximum number of newest records to return.
        log_path: Source override used by tests; defaults to
            ``config.FALLBACK_LOG_PATH``.

    Returns:
        A ``LogReadResult`` with the decoded records oldest first and the
        number of lines inside the read window that failed to decode.
        Older lines outside the window are neither read nor counted. A
        missing sink is an empty result, not an error: nothing has fallen
        back yet.
    """
    path = Path(log_path if log_path is not None else config.FALLBACK_LOG_PATH)
    if limit <= 0:
        return LogReadResult()
    try:
        lines = _tail_lines(path, limit)
    except OSError as exc:
        print(
            f"logging_utils: failed to read fallback log ({type(exc).__name__})",
            file=sys.stderr,
        )
        return LogReadResult()
    records: list[dict[str, object]] = []
    skipped = 0
    for line in lines:
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        # A JSONL sink line is one object; anything else is a foreign
        # line the audit table cannot render as a record.
        if isinstance(decoded, dict):
            records.append(decoded)
        else:
            skipped += 1
    return LogReadResult(records=tuple(records[-limit:]), skipped_lines=skipped)


def read_persistent_events(
    limit: int, log_path: str | Path | None = None
) -> LogReadResult:
    """Read the sink only while the operations view is enabled.

    Who may read the persistent sink is a property of the data, not of
    the page that would draw it, so the gate lives here rather than in
    the UI (AGENTS.md section 3). The sink holds raw employee questions
    from every past session, so with the flag off nothing is read at all
    instead of being read and then filtered.

    ``ENABLE_OPS_VIEW`` is a demo switch, not authorization: it is not
    authentication, grants nothing per user, and must never be described
    as RBAC (remediation plan Finding 7).

    Args:
        limit: Maximum number of newest records to return.
        log_path: Source override used by tests; defaults to
            ``config.FALLBACK_LOG_PATH``.

    Returns:
        The bounded read result, or an empty one while the flag is off.
    """
    if not config.ENABLE_OPS_VIEW:
        return LogReadResult()
    return read_recent_events(limit, log_path=log_path)


def _tail_lines(path: Path, limit: int) -> list[str]:
    """Return at most ``limit`` complete trailing lines of a JSONL file.

    Args:
        path: Sink to tail.
        limit: Maximum number of trailing lines to return.

    Returns:
        The trailing lines, oldest first, with blank lines removed. When
        the walk stops before the start of the file, the first line read
        may have been cut mid-record, so it is dropped rather than
        returned as a malformed line the reader would have to count.

    Raises:
        OSError: If the file exists but cannot be read; the caller
            converts that into an empty result.
    """
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        chunks: list[bytes] = []
        newlines = 0
        # One extra newline: the line before the oldest one wanted is what
        # proves that oldest line is complete.
        while position > 0 and newlines <= limit:
            read_size = min(_TAIL_BLOCK_BYTES, position)
            position -= read_size
            handle.seek(position)
            block = handle.read(read_size)
            newlines += block.count(b"\n")
            chunks.append(block)
    tail = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
    lines = [line for line in tail.splitlines() if line.strip()]
    if position > 0 and lines:
        lines = lines[1:]
    return lines[-limit:]
