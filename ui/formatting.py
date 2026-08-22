"""Pure presentation functions of the UI: values in, markup out.

Every function here is deterministic and free of Streamlit calls and
session state, which is what makes them the part of the interface a test
can hold: score bands are asserted against the calibrated thresholds,
the axis position against its clamp, citation numbering against claim
order, and the trace rows against the route that produced them.

The rendering modules call these and pass the result to ``st``; nothing
in this module renders, and nothing in it decides pipeline behaviour.
"""

from __future__ import annotations

import html
from datetime import datetime

from src import config
from src.schemas import (
    PipelineState,
    RetrievedDocument,
)

from ui.labels import (
    BRAND_NAME,
    EVIDENCE_LABEL,
    EVIDENCE_STRENGTH_WORDS,
    GUARDRAIL_PASS_DETAIL,
    LINKED_POLICY_LABEL,
    LINKED_POLICY_TEXT,
    SCOPE_STRIPS,
    SCORE_BAR_CEILING,
    SCORE_LABEL,
    SNIPPET_CHARS,
    SOURCES_HEADER,
    _REPORTER_FAILURE_REASONS,
    _UNSPENT_REPORTER_REASONS,
)

def _score_band(score: float) -> str:
    """Map a similarity score onto its calibrated display band.

    Display colouring only -- routing decisions come exclusively from the
    graph. The bands read the calibrated runtime thresholds rather than any
    UI-defined cut-off, so the colour can never disagree with the route.
    """
    if score >= config.DIRECT_ANSWER_THRESHOLD:
        return "green"
    if score >= config.REWRITE_FLOOR:
        return "amber"
    return "red"


def _similarity_html(score: float) -> str:
    """Render one score with its band colour and the target glyph."""
    return (
        f'<span class="araya-score--{_score_band(score)}" '
        f'title="{SCORE_LABEL}">'
        '<span class="material-symbols-outlined">track_changes</span> '
        f"{score:.4f}</span>"
    )


def _cite_score_html(document: RetrievedDocument) -> str:
    """Render how one evidence document entered this request.

    The bar is a picture of the number printed beside it, scaled by a
    display-only ceiling and coloured by the calibrated band. Showing the
    placeholder 0.0000 of a canonically linked policy would read as a very
    poor retrieval match rather than as a metadata link, so that case
    names itself instead of drawing an empty bar.
    """
    if document.matched_query_type is None:
        return (
            f'<span class="araya-cite-value" title="{LINKED_POLICY_LABEL}">'
            f"{LINKED_POLICY_TEXT}</span>"
        )
    width = min(document.score / SCORE_BAR_CEILING, 1.0) * 100
    return (
        '<span class="araya-cite-score">'
        '<span class="araya-cite-bar">'
        f'<span class="araya-cite-fill araya-cite-fill--'
        f'{_score_band(document.score)}" style="width:{width:.1f}%">'
        "</span></span>"
        f'<span class="araya-cite-value" title="{SCORE_LABEL}">'
        f"{document.score:.4f}</span></span>"
    )


def _evidence_pill_html(score: float) -> str:
    """Render the evidence meter shown beside the answer's verdict.

    Three bars and one word, both driven by ``_score_band``, so the meter
    reads the same calibrated thresholds the route did and cannot claim
    strong evidence for a score that barely cleared the floor.
    """
    band = _score_band(score)
    lit = {"green": 3, "amber": 2, "red": 1}[band]
    bars = "".join(
        f'<span class="{"is-on" if position <= lit else ""}"></span>'
        for position in (1, 2, 3)
    )
    return (
        f'<span class="araya-evidence araya-evidence--{band}" '
        f'title="{SCORE_LABEL} {score:.4f}">{EVIDENCE_LABEL}'
        f'<span class="araya-evidence-bars">{bars}</span>'
        f'<span class="araya-evidence-word">'
        f"{EVIDENCE_STRENGTH_WORDS[band]}</span></span>"
    )


def _gating_score(state: PipelineState) -> float | None:
    """Return the similarity score that admitted the answer, if any."""
    expanded_score = state.get("expanded_retrieval_score")
    if expanded_score is not None:
        return expanded_score
    return state.get("raw_retrieval_score")


def _display_route(state: PipelineState) -> tuple[str, str, str]:
    """Derive the console route label, icon, and css class from state."""
    route = state.get("route")
    if route == "blocked":
        return "Blocked", "shield", "blocked"
    if route == "fallback":
        return "Fallback", "alt_route", "fallback"
    if state.get("rewritten_queries") is not None:
        return "Rewrite", "edit_note", "rewrite"
    return "Direct", "straight", "direct"


def _turn_clock(timestamp: str) -> str:
    """Format one turn's timestamp as the HH:MM the transcript labels use."""
    try:
        return datetime.fromisoformat(timestamp).strftime("%H:%M")
    except ValueError:
        return timestamp


def _citation_numbers(state: PipelineState) -> dict[str, int]:
    """Number the validated citations in the order the source list shows.

    The numbering is presentation bookkeeping: it indexes the list under
    the answer so an inline chip can be traced by eye. The ids themselves
    come from ``valid_citations`` and are never invented here.
    """
    return {
        source_id: number
        for number, source_id in enumerate(
            state.get("valid_citations", []), start=1
        )
    }


def _answer_html(state: PipelineState) -> str:
    """Render the answer text with its ``[ID]`` markers as numbered chips.

    ``src.answer_renderer`` writes those markers from validated ids; this
    only restyles them in place, so a chip can never name a source the
    citation validator did not accept, and an id with no marker in the
    text simply keeps its place in the list below.
    """
    text = html.escape(state.get("answer", ""))
    for source_id, number in _citation_numbers(state).items():
        text = text.replace(
            f"[{source_id}]",
            f'<span class="araya-cite">{number} {html.escape(source_id)}'
            "</span>",
        )
    return f'<div class="araya-answer">{text}</div>'


def _source_list_html(state: PipelineState) -> str:
    """Build one expandable row per validated citation.

    The first row opens by default: on a two-source answer it shows the
    evidence without a click, and the rest stay one click away instead of
    pushing the composer off the screen.
    """
    documents_by_id = {
        document.source_id: document
        for document in state.get("answer_evidence", [])
    }
    rows: list[str] = []
    for source_id, number in _citation_numbers(state).items():
        document = documents_by_id.get(source_id)
        head = (
            f'<span class="araya-cite-badge">{number}</span>'
            '<span class="araya-cite-head">'
            f'<span class="araya-cite-title">'
            f"{html.escape(document.title) if document else ''}</span>"
            f'<span class="araya-cite-meta">{html.escape(source_id)}'
            f"{f' · {document.source_type}' if document else ''}</span></span>"
        )
        if document is None:
            # The id passed validation but its document is not in the
            # selected evidence: name it rather than dropping a citation.
            rows.append(
                '<div class="araya-cite-card">'
                f'<div class="araya-cite-row">{head}</div></div>'
            )
            continue
        snippet = document.content[:SNIPPET_CHARS]
        if len(document.content) > SNIPPET_CHARS:
            snippet += "…"
        rows.append(
            f'<details class="araya-cite-card"{" open" if number == 1 else ""}>'
            f"<summary>{head}{_cite_score_html(document)}"
            '<span class="material-symbols-outlined araya-chevron">'
            "expand_more</span></summary>"
            f'<div class="araya-cite-body">{html.escape(snippet)}</div>'
            "</details>"
        )
    return "".join(rows)


def _answer_markdown(record: dict) -> str:
    """Build the .md export of one answered request.

    Carries what a case forwarded to HR or Finance needs: the question,
    the answer with its citation markers intact, the documents behind it,
    and the request id that ties the file back to this session.
    """
    state: PipelineState = record["state"]
    documents_by_id = {
        document.source_id: document
        for document in state.get("answer_evidence", [])
    }
    numbers = _citation_numbers(state)
    lines = [
        f"# {record['query']}",
        "",
        state.get("answer", ""),
        "",
        f"## {SOURCES_HEADER.format(count=len(numbers))}",
    ]
    for source_id, number in numbers.items():
        document = documents_by_id.get(source_id)
        title = document.title if document is not None else ""
        lines.append(f"{number}. [{source_id}] {title}".rstrip())
    lines += ["", f"Request ID: {record['request_id']}"]
    return "\n".join(lines)


def _scope_strips_html() -> str:
    """Build the three-outcome legend shown on the opening screen."""
    strips = "".join(
        f'<div class="araya-scope araya-scope--{tone}">'
        f'<div class="araya-scope-label">{html.escape(label)}</div>'
        f'<div class="araya-scope-text">{html.escape(text)}</div></div>'
        for label, text, tone in SCOPE_STRIPS
    )
    return f'<div class="araya-scope-row">{strips}</div>'


def _kpi_html(label: str, value: str, unit: str = "") -> str:
    """Render one KPI card in the console's stat style.

    Every field is escaped. The card is emitted into an
    ``unsafe_allow_html=True`` block, and its neighbour
    ``_stat_card_html`` already escapes -- so the next caller that feeds
    this one corpus-derived or query-derived text would have injected raw
    markup where the identical text through the other helper is safe.

    The ``icon``/``tone`` badge parameters are gone with the badge: the
    single call site passes neither, so the branch and its five CSS rules
    were unreachable.

    Args:
        label: Caps label describing the metric and its scope.
        value: Already-formatted metric value.
        unit: Optional unit or qualifier rendered next to the value.

    Returns:
        The card markup.
    """
    unit_html = (
        f'<span class="araya-kpi-unit">{html.escape(unit)}</span>'
        if unit
        else ""
    )
    return (
        '<div class="araya-kpi"><div class="araya-kpi-head">'
        f'<span class="araya-kpi-label">{html.escape(label)}</span></div>'
        f'<div class="araya-kpi-body"><span class="araya-kpi-value">'
        f"{html.escape(value)}</span>{unit_html}</div></div>"
    )


def _brand_html() -> str:
    """Sidebar brand lockup shared by both rails.

    The mark is inline SVG rather than an icon-font glyph so it keeps its
    two-tone form at 30px, and the wordmark drops the "Thai RAG System"
    line: the rail is read on every screen, and the descriptor earned no
    second look after the first session.
    """
    name, _, accent = BRAND_NAME.partition(" ")
    return (
        '<div class="araya-brand">'
        '<svg class="araya-brand-mark" width="30" height="30" '
        'viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="9" fill="#0052CC"></rect>'
        '<path d="M10.5 22 L16 9.5 L21.5 22" stroke="#ffffff" '
        'stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" '
        'fill="none"></path>'
        '<path d="M13.2 17.6 H18.8" stroke="#8FBCFF" stroke-width="2.3" '
        'stroke-linecap="round"></path>'
        '</svg>'
        f'<span class="araya-brand-name">{name}'
        f'<span class="araya-brand-accent"> {accent}</span></span></div>'
    )


def _matches_filters(
    record: dict, query_filter: str, routes: tuple[str, ...]
) -> bool:
    """Test one session record against the console filters.

    Presentation filtering only: routing itself already happened in the
    graph, and this never changes what was logged.

    Args:
        record: One session history entry.
        query_filter: Free-text needle matched against query and reason.
        routes: Selected route labels; empty means every route.

    Returns:
        True when the record should stay visible.
    """
    state: PipelineState = record["state"]
    label, _, _ = _display_route(state)
    if routes and label not in routes:
        return False
    if not query_filter:
        return True
    needle = query_filter.casefold()
    reason = (
        state.get("fallback_reason") or state.get("guardrail_reason") or ""
    )
    return needle in record["query"].casefold() or needle in reason.casefold()


def _format_score(score: float | None) -> str:
    """Format one similarity score for a text field, dash when unset."""
    return "–" if score is None else f"{score:.4f}"


def _stat_card_html(
    label: str,
    value: str,
    unit: str = "",
    share: float | None = None,
    tone: str = "green",
    note: str = "",
) -> str:
    """Render one console stat block.

    Args:
        label: Metric name.
        value: Already-formatted value.
        unit: Qualifier printed next to the value, e.g. the scope or the
            dominant reason code.
        share: Fraction of the session's requests this count represents,
            drawn as a bar. ``None`` renders no bar.
        tone: Bar tint token: ``green``, ``amber`` or ``red``.
        note: Optional footnote, used where a number has a caveat.

    Returns:
        The card markup.
    """
    unit_html = (
        f'<span class="araya-stat-unit">{html.escape(unit)}</span>'
        if unit
        else ""
    )
    bar_html = (
        '<div class="araya-stat-track">'
        f'<span class="araya-stat-fill araya-stat-fill--{tone}" '
        f'style="width:{share * 100:.1f}%"></span></div>'
        if share is not None
        else ""
    )
    note_html = (
        f'<div class="araya-stat-note">{html.escape(note)}</div>'
        if note
        else ""
    )
    return (
        '<div class="araya-stat">'
        f'<div class="araya-stat-label">{html.escape(label)}</div>'
        f'<div class="araya-stat-body">'
        f'<span class="araya-stat-value araya-mono-face">{value}</span>'
        f"{unit_html}</div>{bar_html}{note_html}</div>"
    )


def _axis_position(score: float) -> float:
    """Map a similarity score onto the threshold strip, as a percentage.

    Display geometry only: the strip is drawn against the same ceiling the
    answer-side bars use, so a position here and a bar there mean the same
    thing. The cut-offs it draws come from config.
    """
    return min(max(score, 0.0) / SCORE_BAR_CEILING, 1.0) * 100


def _band_word(score: float) -> str:
    """Name the band a score falls into, for the trace's node details."""
    return {"green": "high", "amber": "medium", "red": "low"}[
        _score_band(score)
    ]


def _trace_rows(state: PipelineState) -> list[tuple[str, str, str, str]]:
    """Describe every graph node's outcome for one request.

    Returns:
        ``(node, badge tone, glyph, detail)`` in graph order. Nodes the
        request never reached are returned too, dimmed by the caller, so
        the panel shows the shape of the pipeline and not only the path
        taken. Per-node timing is deliberately absent: ``PipelineState``
        carries none, and inventing it here would be a fiction.
    """
    route = state.get("route")
    blocked = route == "blocked"
    raw = state.get("raw_retrieval_score")
    expanded = state.get("expanded_retrieval_score")
    rewrites = state.get("rewritten_queries")
    evidence = state.get("answer_evidence") or []
    citations = state.get("valid_citations") or []
    skipped = ("skip", "remove", "ไม่ถูกเรียกใน request นี้")

    rows: list[tuple[str, str, str, str]] = [
        (
            "input_guardrail",
            *(
                ("stop", "shield", f"blocked · {state.get('guardrail_reason')}")
                if blocked
                else ("done", "check", GUARDRAIL_PASS_DETAIL)
            ),
        )
    ]
    # A blocked request traverses input_guardrail -> refuse -> END. The
    # node that writes its telemetry had no row at all, and the fallback
    # row below claimed the write instead.
    rows.append(
        (
            "refuse",
            "done",
            "block",
            f"telemetry_logged="
            f"{str(state.get('telemetry_logged', True)).lower()}",
        )
        if blocked
        else ("refuse", *skipped)
    )
    rows.append(
        ("retrieve_original", *skipped)
        if raw is None
        else (
            "retrieve_original",
            "done",
            "check",
            f"top1 {raw:.4f} → {_band_word(raw)} band",
        )
    )
    scope_score = state.get("scope_score")
    scope_topics = state.get("scope_topics") or []
    if state.get("scope_reason") or scope_topics:
        detail = (
            f"supported · {', '.join(scope_topics)}"
            if scope_topics
            else f"unsupported · {state.get('scope_reason')}"
        )
        if scope_score is not None:
            detail += (
                f" · alias {scope_score:.4f}"
                f" / {config.SCOPE_MATCH_THRESHOLD:.2f}"
            )
        rows.append(
            (
                "validate_scope",
                "done" if scope_topics else "stop",
                "check" if scope_topics else "block",
                detail,
            )
        )
    else:
        rows.append(("validate_scope", *skipped))
    if rewrites is None:
        rows.append(("rewrite", *skipped))
    else:
        rows.append(
            (
                "rewrite",
                "llm",
                "edit_note",
                f"LLM call · {len(rewrites)} queries"
                + (
                    f" · {state['rewrite_failure_reason']}"
                    if state.get("rewrite_failure_reason")
                    else ""
                )
                + (
                    " · rewrite_rejected"
                    if state.get("rewrite_rejected")
                    else ""
                ),
            )
        )
    rows.append(
        ("retrieve_expanded", *skipped)
        if expanded is None
        else (
            "retrieve_expanded",
            "done",
            "check",
            f"max-pool {expanded:.4f} / final "
            f"{config.FINAL_ANSWER_THRESHOLD:.2f}",
        )
    )
    rows.append(
        ("select_evidence", *skipped)
        if not evidence
        else (
            "select_evidence",
            "done",
            "check",
            " · ".join(document.source_id for document in evidence),
        )
    )
    # The reporter ran whenever evidence selection handed it evidence.
    # Gating this row on route == "answered" hid a billed LLM call on
    # exactly the routes an auditor opens the panel for: every reason
    # code after the reporter (fabricated_citation, reporter_failure,
    # insufficient_reporter_evidence, ...) showed "was not called".
    # select_evidence writes answer_evidence only when it succeeded, and
    # the graph always follows that with the report node, so the presence
    # of evidence is exactly the condition "the reporter ran".
    reporter_ran = bool(evidence)
    # Reaching the node is not the same as spending a call: a request
    # that arrives here with no deadline left, or with no credential, is
    # refused before the provider is touched.
    reporter_spent_a_call = (
        reporter_ran
        and state.get("fallback_reason") not in _UNSPENT_REPORTER_REASONS
    )
    rows.append(
        (
            "report",
            "llm" if reporter_spent_a_call else "stop",
            "smart_toy" if reporter_spent_a_call else "block",
            "LLM call · grounded answer"
            if route == "answered"
            else (
                f"LLM call · {state.get('fallback_reason')}"
                if reporter_spent_a_call
                else f"not called · {state.get('fallback_reason')}"
            ),
        )
        if reporter_ran
        else ("report", *skipped)
    )
    # Likewise: the validator is the node that REJECTED the answer on the
    # contract routes, so an empty citation list is not evidence that it
    # never ran.
    validator_ran = reporter_ran and state.get(
        "fallback_reason"
    ) not in _REPORTER_FAILURE_REASONS
    rows.append(
        (
            "validate_citations",
            "done" if citations else "stop",
            "verified" if citations else "block",
            f"valid · {', '.join(citations)}"
            if citations
            else f"rejected · {state.get('fallback_reason')}",
        )
        if validator_ran
        else ("validate_citations", *skipped)
    )
    # A blocked request is accounted for by the refuse row above.
    degraded = route == "fallback"
    rows.append(
        (
            "fallback",
            "done",
            "alt_route",
            f"{state.get('fallback_reason') or state.get('guardrail_reason')}"
            f" · telemetry_logged="
            f"{str(state.get('telemetry_logged', True)).lower()}",
        )
        if degraded
        else ("fallback", *skipped)
    )
    return rows


def _clock(timestamp: str) -> str:
    """Format an ISO timestamp as the console's HH:MM:SS clock column."""
    try:
        return datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
    except ValueError:
        return timestamp


def _log_table_html(records: list[dict]) -> str:
    """Render JSONL telemetry rows in the console table style.

    Every field is escaped: the sink stores raw user queries, so log text
    reaches this table exactly as it was typed.

    Args:
        records: Decoded JSONL objects, oldest first.

    Returns:
        One HTML table, newest record first.
    """
    rows: list[str] = []
    for record in reversed(records):
        raw_score = record.get("raw_retrieval_score")
        expanded_score = record.get("expanded_retrieval_score")
        sources = ", ".join(record.get("top_sources") or []) or "–"
        query = str(record.get("query", ""))
        rows.append(
            "<tr>"
            f'<td>{html.escape(_clock(str(record.get("timestamp", ""))))}</td>'
            f'<td class="araya-cell-clip" title="{html.escape(query)}">'
            f"{html.escape(query)}</td>"
            f'<td>{html.escape(str(record.get("reason", "–")))}</td>'
            "<td>"
            f'{"–" if raw_score is None else _similarity_html(raw_score)}'
            "</td><td>"
            f'{"–" if expanded_score is None else _similarity_html(expanded_score)}'
            "</td>"
            f"<td>{html.escape(sources)}</td>"
            "</tr>"
        )
    return (
        '<table class="araya-table"><thead><tr>'
        "<th>Time</th><th>Query</th><th>Reason</th><th>Raw</th>"
        "<th>Expanded</th><th>Top sources</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _log_matches(record: dict, query_filter: str) -> bool:
    """Test one JSONL record against the console's free-text filter."""
    if not query_filter:
        return True
    needle = query_filter.casefold()
    haystack = (
        f"{record.get('query', '')} {record.get('reason', '')} "
        f"{' '.join(record.get('top_sources') or [])}"
    )
    return needle in haystack.casefold()


def _metric_rows_html(metrics: list[tuple[str, int, int, str]]) -> str:
    """Render one group's metrics as name, proportion bar, and fraction.

    The bar is one neutral colour on purpose. Some of these rates are good
    at zero -- ``Invalid Candidate Leakage Rate: 0/17`` is the result the
    contract wants -- so colouring by proportion would tell the reader the
    opposite of what the baseline means. The proportion is the picture;
    the meaning stays in BASELINE.md.
    """
    rows: list[str] = []
    for name, passed, total, remark in metrics:
        width = (passed / total * 100) if total else 0.0
        remark_html = (
            f'<span class="araya-panel-meta">{html.escape(remark)}</span>'
            if remark
            else ""
        )
        rows.append(
            '<div class="araya-metric">'
            f'<span class="araya-metric-name">{html.escape(name)}</span>'
            f"{remark_html}"
            '<span class="araya-metric-bar">'
            f'<span class="araya-metric-fill" style="width:{width:.1f}%;'
            'background:#0052CC"></span></span>'
            f'<span class="araya-metric-value araya-mono-face">'
            f"{passed}/{total}</span></div>"
        )
    return "".join(rows)
