"""The operations and audit console page.

Everything technical about a request lives here: the route it took, the
scores against their thresholds, the node trace, the JSONL fallback log,
the loaded corpus, the evaluation artefacts, and the frozen runtime
configuration. The separation from the assistant page is information
architecture for the demo, NOT a security boundary -- no authentication
and no RBAC exist.

The persistent JSONL sink holds raw employee questions from every past
session, so it stays hidden unless ``ENABLE_OPS_VIEW`` is turned on
deliberately, and is then read through the bounded tail reader rather
than loaded whole (remediation plan Finding 7).
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from src import config
from src.fallback import ReasonFamily, reason_family
from src.logging_utils import read_persistent_events
from src.schemas import (
    LogReadResult,
    PipelineState,
)

from ui.labels import (
    BASELINE_READER_LABEL,
    BASELINE_RUNS_NOTE,
    BASELINE_RUNS_TITLE,
    BASELINE_UNPARSED,
    BASELINE_UNREADABLE,
    CALIBRATION_BADGE,
    CONSOLE_SEARCH_PLACEHOLDER,
    CONSOLE_SECTIONS,
    EMPLOYEE_VIEW,
    EVAL_DIR,
    EVAL_PAGE_NOTE,
    EXPORT_EMPTY_NOTE,
    EXPORT_FORMATS,
    EXPORT_LABEL,
    EXPORT_SCOPE_NOTE,
    HELDOUT_BADGE,
    MAX_LOG_ROWS,
    MISSING_KEY_WARNING,
    OPS_LOG_DISABLED_NOTE,
    OPS_VIEW,
    PERSISTENT_LOG_TITLE,
    SESSION_LOG_EMPTY,
    SESSION_LOG_NOTE,
    SESSION_LOG_TITLE,
    ROUTE_FILTER_OPTIONS,
    THRESHOLD_PANEL_NOTE,
    THRESHOLD_PANEL_TITLE,
    TRACE_PANEL_NOTE,
    TRIAGE_PANEL_NOTE,
    TRIAGE_PANEL_TITLE,
    VIEW_SEPARATION_NOTE,
    VIEW_SWITCH_LABELS,
    _BAND_COLOURS,
    _EVAL_GROUP_LINE,
    _EVAL_METRIC_LINE,
    _EVAL_REMARK_LINE,
    _ROUTE_BADGE_COLOURS,
)
from ui.styles import (
    _CONSOLE_DARK_CSS,
    _CONSOLE_LAYOUT_CSS,
)
from ui.formatting import (
    _axis_position,
    _baseline_run_html,
    _baseline_run_lines,
    _brand_html,
    _clock,
    _degraded_session_rows,
    _display_route,
    _escape_markdown,
    _format_score,
    _gating_score,
    _kpi_html,
    _log_matches,
    _log_table_html,
    _matches_filters,
    _metric_rows_html,
    _corpus_export_rows,
    _evaluation_export_rows,
    _overview_export_rows,
    _overview_stats,
    _parse_baseline_metrics,
    _request_key,
    _rows_to_csv,
    _rows_to_jsonl,
    _rows_to_markdown,
    _runtime_export_rows,
    _score_band,
    _session_export_rows,
    _stat_card_html,
    _trace_rows,
    _triage_row_html,
)
from ui.runtime import (
    _PAGE_REFS,
    _all_requests,
    _sessions,
    _corpus,
    _graph_and_retriever,
)

def _threshold_axis_html() -> str:
    """Draw the calibrated bands, their cut-offs, and this session's scores.

    Every marker is a config value and every dot is a real request's
    gating score, so the panel answers "why did this one fall back" by
    position instead of by comparing two numbers in a table.
    """
    floor = _axis_position(config.REWRITE_FLOOR)
    direct = _axis_position(config.DIRECT_ANSWER_THRESHOLD)
    final = _axis_position(config.FINAL_ANSWER_THRESHOLD)
    bands = (
        f'<span class="araya-axis-band--red" style="width:{floor:.1f}%">'
        "</span>"
        f'<span class="araya-axis-band--amber" '
        f'style="width:{direct - floor:.1f}%"></span>'
        f'<span class="araya-axis-band--yellow" '
        f'style="width:{final - direct:.1f}%"></span>'
        f'<span class="araya-axis-band--green" '
        f'style="width:{100 - final:.1f}%"></span>'
    )
    marks = "".join(
        f'<div class="araya-axis-mark" style="left:{position:.1f}%;'
        f'background:{colour}"></div>'
        for position, colour in (
            (floor, "#B76E00"),
            (direct, "#00714b"),
            (final, "#0052CC"),
        )
    )
    dots: list[str] = []
    legend: list[str] = []
    for row in _all_requests():
        score = _gating_score(row["state"])
        if score is None:
            continue
        band = _score_band(score)
        label, _, _ = _display_route(row["state"])
        dots.append(
            f'<div class="araya-axis-dot araya-axis-dot--{band}" '
            f'style="left:{_axis_position(score):.1f}%" '
            f'title="{row["request_id"]} · {score:.4f}"></div>'
        )
        legend.append(
            f'<span><i style="background:{_BAND_COLOURS[band]}"></i>'
            f'{row["request_id"]} {score:.4f} → {label.lower()}</span>'
        )
    labels = "".join(
        f'<span class="araya-axis-label" style="left:{position:.1f}%">'
        f"{value:.2f}<br><span>{name}</span></span>"
        for position, value, name in (
            (floor, config.REWRITE_FLOOR, "rewrite floor"),
            (direct, config.DIRECT_ANSWER_THRESHOLD, "direct"),
            (final, config.FINAL_ANSWER_THRESHOLD, "final"),
        )
    )
    legend_html = (
        f'<div class="araya-axis-legend">{"".join(legend)}</div>'
        if legend
        else ""
    )
    return (
        '<div class="araya-axis">'
        f'<div class="araya-axis-track">{bands}</div>{marks}'
        f'{"".join(dots)}</div>'
        f'<div class="araya-axis-labels araya-mono-face">{labels}</div>'
        f"{legend_html}"
    )


def _triage_groups() -> list[tuple[str, int, str, ReasonFamily | None]]:
    """Group this session's degraded requests by reason code.

    Returns:
        ``(reason, count, newest query, family)`` per reason, most
        frequent first. The family comes from ``src.fallback`` rather
        than from which state key carried the code: a request refused
        for being empty is not an attack, and only a knowledge gap is a
        reason to go and write a document.

    Scope is this session on purpose: the JSONL sink holds raw questions
    from every past session and stays behind ``ENABLE_OPS_VIEW``
    (Finding 7), so it is not aggregated here.
    """
    groups: dict[str, list[dict]] = {}
    for row in _all_requests():
        state: PipelineState = row["state"]
        reason = state.get("fallback_reason") or state.get("guardrail_reason")
        if reason is None:
            continue
        groups.setdefault(str(reason), []).append(row)
    ranked = sorted(
        groups.items(), key=lambda item: (-len(item[1]), item[0])
    )
    return [
        (reason, len(rows), rows[-1]["query"], reason_family(reason))
        for reason, rows in ranked
    ]


def _render_console_overview() -> None:
    """Overview: route split, the threshold axis, and fallback triage.

    Scope is the browser tab, not one transcript: the counts cover every
    session the rail lists, which is what makes them comparable with the
    request list in Query Logs.
    """
    requests = _all_requests()
    sessions = _sessions()
    _, _, built_at = _graph_and_retriever()
    st.markdown(
        '<p class="araya-headline">Overview</p>', unsafe_allow_html=True
    )
    st.caption(
        f"{len(sessions)} sessions · {len(requests)} requests · "
        f"index build {built_at.strftime('%H:%M:%S')}"
    )
    stats = _overview_stats(requests)
    total = stats["requests"] or 1
    latency = (
        "–"
        if stats["avg_latency_seconds"] is None
        else f"{stats['avg_latency_seconds']:.3f}"
    )
    reasons = _triage_groups()
    # The input family is exactly the guardrail's own codes, so it is
    # also what separates a blocked request from a degraded one here.
    fallback_reason = next(
        (
            reason
            for reason, _, _, family in reasons
            if family is not ReasonFamily.SECURITY_OR_INVALID_INPUT
        ),
        "–",
    )
    blocked_reason = next(
        (
            reason
            for reason, _, _, family in reasons
            if family is ReasonFamily.SECURITY_OR_INVALID_INPUT
        ),
        "–",
    )
    # One grid rather than four columns: st.columns leaves each card at
    # its own content height, and the only card carrying a footnote was
    # then taller than the row and overlapped the panels below it.
    cards = (
        _stat_card_html(
            "Answered",
            str(stats["answered"]),
            f"/ {stats['requests']} requests",
            stats["answered"] / total,
            "green",
        ),
        _stat_card_html(
            "Fallback", str(stats["fallback"]), fallback_reason,
            stats["fallback"] / total, "amber",
        ),
        _stat_card_html(
            "Blocked", str(stats["blocked"]), blocked_reason,
            stats["blocked"] / total, "red",
        ),
        _stat_card_html(
            "Avg latency", latency, "s · this tab",
            note=f"{stats['llm_calls']} LLM call ในแท็บนี้",
        ),
    )
    st.markdown(
        f'<div class="araya-card-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 1], gap="medium")
    with left:
        st.markdown(
            '<div class="araya-panel"><div class="araya-panel-head">'
            f'<span class="araya-panel-title">{THRESHOLD_PANEL_TITLE}</span>'
            '<span class="araya-panel-meta">ค่าจาก src/config.py</span></div>'
            f"{_threshold_axis_html()}"
            f'<div class="araya-panel-note">{THRESHOLD_PANEL_NOTE}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    with right:
        rows = "".join(
            _triage_row_html(reason, count, query, family)
            for reason, count, query, family in reasons
        )
        empty = (
            '<div class="araya-panel-note">'
            "ยังไม่มี request ที่ถูก block หรือ fallback ในเซสชันนี้</div>"
        )
        st.markdown(
            '<div class="araya-panel"><div class="araya-panel-head">'
            f'<span class="araya-panel-title">{TRIAGE_PANEL_TITLE}</span>'
            '<span class="araya-panel-meta">session นี้</span></div>'
            f"{rows or empty}"
            f'<div class="araya-panel-note">{TRIAGE_PANEL_NOTE}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    st.divider()
    _render_kb_stats()


def _render_trace_panel(record: dict) -> None:
    """Render the node-by-node trace for the selected request."""
    state: PipelineState = record["state"]
    label, _, _ = _display_route(state)
    rewrites = state.get("rewritten_queries")
    rows_html: list[str] = []
    # Timings are measured per request by the streaming run; a request
    # recorded before that existed simply shows none.
    timings = record.get("node_seconds") or {}
    for node, tone, glyph, detail, duration in _trace_rows(state, timings):
        rows_html.append(
            f'<div class="araya-trace-row'
            f'{" araya-trace-row--skipped" if tone == "skip" else ""}">'
            f'<span class="araya-trace-badge araya-trace-badge--{tone}">'
            f'<span class="material-symbols-outlined">{glyph}</span></span>'
            f'<span class="araya-trace-node araya-mono-face">{node}</span>'
            f'<span class="araya-trace-detail">{html.escape(detail)}</span>'
            f'<span class="araya-trace-time araya-mono-face">'
            f"{html.escape(duration)}</span>"
            "</div>"
        )
        if node == "rewrite" and rewrites:
            # The rewrite is the node operators most often distrust, so the
            # original and its generated queries are shown side by side.
            rows_html.append(
                '<div class="araya-trace-aside">'
                f"<span>เดิม “{html.escape(record['query'])}”</span>"
                f"<span>→ {html.escape(' · '.join(rewrites))}</span>"
                f"<span>rewrite_failure_reason = "
                f"{html.escape(str(state.get('rewrite_failure_reason')))}"
                f" · rewrite_rejected = "
                f"{str(state.get('rewrite_rejected', False)).lower()}</span>"
                "</div>"
            )
    st.markdown(
        '<div class="araya-panel">'
        '<div class="araya-panel-head">'
        f'<span class="araya-panel-title araya-mono-face">'
        f"{record['request_id']} · route = {label.lower()}</span>"
        f'<span class="araya-panel-meta araya-mono-face">'
        f"{record['latency_seconds']:.3f}s</span></div>"
        f'<div class="araya-panel-note" style="margin:6px 0 12px">'
        f"{html.escape(record['query'])}</div>"
        f"{''.join(rows_html)}"
        f'<div class="araya-panel-note">{TRACE_PANEL_NOTE}</div></div>',
        unsafe_allow_html=True,
    )


def _render_console_logs(query_filter: str, log: LogReadResult) -> None:
    """Query Logs: every request of this tab beside one request's trace.

    The list spans all sessions, not the one the assistant is currently
    showing: an operator reading the console is asking what the app did.
    Requests are keyed on session and id together, because the request id
    restarts at ``Q-001`` in every session.
    """
    requests = _all_requests()
    st.markdown(
        '<p class="araya-headline">Requests</p>', unsafe_allow_html=True
    )
    if not requests:
        st.info("No requests in this tab yet.")
        st.divider()
        _render_log_section(log, query_filter)
        return
    counts = {label: 0 for label in ROUTE_FILTER_OPTIONS}
    for row in requests:
        counts[_display_route(row["state"])[0]] += 1
    options = [f"ทั้งหมด {len(requests)}"] + [
        f"{label} {counts[label]}" for label in ROUTE_FILTER_OPTIONS
    ]
    selected = st.segmented_control(
        "Route filter",
        options=options,
        default=options[0],
        key="console_route_chip",
        label_visibility="collapsed",
    )
    routes: tuple[str, ...] = ()
    if selected and not selected.startswith("ทั้งหมด"):
        routes = (selected.rsplit(" ", 1)[0],)
    visible = [
        row
        for row in reversed(requests)
        if _matches_filters(row, query_filter, routes)
    ]
    if not visible:
        st.info("No request in this tab matches the current filters.")
        st.divider()
        _render_log_section(log, query_filter)
        return

    selected_key = st.session_state.get("console_request")
    if selected_key not in {_request_key(row) for row in visible}:
        selected_key = _request_key(visible[0])
        st.session_state["console_request"] = selected_key

    left, right = st.columns([1, 1.2], gap="medium")
    with left:
        for row in visible:
            state: PipelineState = row["state"]
            label, _, css = _display_route(state)
            raw = state.get("raw_retrieval_score")
            expanded = state.get("expanded_retrieval_score")
            score = _format_score(raw)
            if expanded is not None:
                score += f" → {expanded:.4f}"
            reason = state.get("fallback_reason") or state.get(
                "guardrail_reason"
            )
            meta = " · ".join(
                part
                for part in (
                    row["session_id"],
                    _clock(row["timestamp"]),
                    score if raw is not None else reason,
                    f"{row['latency_seconds']:.3f}s",
                )
                if part
            )
            key = _request_key(row)
            if st.button(
                # The badge is this module's own markup; the question is
                # employee text on the same line, so it is neutralised --
                # unescaped, a query spelling `:green-badge[Direct]`
                # draws a second, forged verdict on a triage row.
                f":{_ROUTE_BADGE_COLOURS[css]}-badge[{label}] "
                f"**{row['request_id']}**  \n"
                f"{_escape_markdown(row['query'])}  \n`{meta}`",
                key=f"araya_req_{key}",
                type="primary" if key == selected_key else "secondary",
                use_container_width=True,
                wrap=True,
            ):
                st.session_state["console_request"] = key
                st.rerun()
    with right:
        record = next(
            row for row in visible if _request_key(row) == selected_key
        )
        _render_trace_panel(record)
    st.divider()
    _render_log_section(log, query_filter)


def _render_session_log(query_filter: str) -> None:
    """Blocked and fallback events of the current session.

    This table exists because the panel was empty by default: the sink is
    read only while ``ENABLE_OPS_VIEW`` is on, and with the flag off the
    section had nothing to show even when the session had just produced a
    refusal. These rows come from session state instead, so they are the
    same requests the list above already renders -- no new reader of the
    file, and no way around its gate.

    Args:
        query_filter: Free-text filter from the console header.
    """
    st.markdown(
        f'<p class="araya-panel-title">{SESSION_LOG_TITLE}</p>',
        unsafe_allow_html=True,
    )
    st.caption(SESSION_LOG_NOTE)
    rows = _degraded_session_rows(_session_export_rows(_all_requests()))
    if not rows:
        st.info(SESSION_LOG_EMPTY)
        return
    visible = [row for row in rows if _log_matches(row, query_filter)]
    if not visible:
        st.info("No blocked or fallback event in this session matches "
                "the current search.")
        return
    st.markdown(_log_table_html(visible), unsafe_allow_html=True)


def _render_persistent_log(log: LogReadResult, query_filter: str) -> None:
    """The JSONL sink itself, when the demo flag admits it.

    Args:
        log: Bounded read result from the sink. It is empty whenever
            ``ENABLE_OPS_VIEW`` is off, because the gate lives in
            ``read_persistent_events`` rather than here.
        query_filter: Free-text filter from the console header.
    """
    st.markdown(
        f'<p class="araya-panel-title">{PERSISTENT_LOG_TITLE}</p>',
        unsafe_allow_html=True,
    )
    if not config.ENABLE_OPS_VIEW:
        st.info(OPS_LOG_DISABLED_NOTE)
        return
    log_path = Path(config.FALLBACK_LOG_PATH)
    st.caption(
        "Scope: JSONL records appended by every run, across sessions. "
        f"File: {log_path.name}. The route filter applies to this session's "
        "requests above; this table follows the search box only."
    )
    records = list(log.records)
    if not records:
        st.info("No blocked or fallback events logged yet.")
        return
    visible = [row for row in records if _log_matches(row, query_filter)]
    if not visible:
        st.info("No logged events match the current search.")
        return
    st.markdown(_log_table_html(visible), unsafe_allow_html=True)
    if log.skipped_lines:
        # The count is reportable; the malformed text is raw query data
        # and is never rendered.
        st.caption(f"Skipped {log.skipped_lines} malformed line(s).")
    if len(records) == MAX_LOG_ROWS:
        st.caption(f"Showing the newest {MAX_LOG_ROWS} records.")


def _render_log_section(log: LogReadResult, query_filter: str = "") -> None:
    """Blocked and fallback telemetry, this session and on disk.

    The two scopes stay separate tables rather than one merged view: they
    have different reach and different privacy rules, and a reader has to
    be able to tell which is which.

    Args:
        log: Bounded read result from the sink.
        query_filter: Free-text filter from the console header.
    """
    st.markdown('<p class="araya-section">Blocked / Fallback Log</p>',
                unsafe_allow_html=True)
    _render_session_log(query_filter)
    _render_persistent_log(log, query_filter)


def _render_kb_stats() -> None:
    """Index-health strip: real counts and this process's build time.

    Drawn on the same card grid the rest of the console uses, which is
    what keeps the three cards the same height and leaves a measured gap
    before the document list rather than letting it butt against them.
    """
    documents = _corpus()
    _, _, built_at = _graph_and_retriever()
    cells = (
        ("Indexed", f"{len(documents)} / {len(documents)}", ""),
        ("Duplicate IDs", "0", "enforced at load"),
        ("Index built", built_at.strftime("%H:%M:%S"), "this process"),
    )
    cards = "".join(
        _kpi_html(label, value, unit) for label, value, unit in cells
    )
    st.markdown(
        f'<div class="araya-card-grid araya-card-grid--three">{cards}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _baseline_text() -> str:
    """Read ``eval/BASELINE.md`` once per process.

    Returns:
        The file contents, or an empty string when it cannot be read.
        The panel treats both the same way: it reports what the file
        recorded and never computes a number of its own.
    """
    try:
        return (EVAL_DIR / "BASELINE.md").read_text(encoding="utf-8")
    except OSError:
        return ""


def _baseline_snapshot() -> list[dict]:
    """Return the newest measured-results block, parsed.

    The parsing itself lives in ``ui.formatting`` so a test can hold it
    against the real file; this only supplies the text.

    Returns:
        One dict per metric group, as ``_parse_baseline_metrics``
        describes, or an empty list when the block cannot be read.
    """
    return _parse_baseline_metrics(_baseline_text())


@st.cache_data(show_spinner=False)
def _eval_fixture_counts() -> list[tuple[str, int, str]]:
    """Count the cases in each evaluation fixture on disk.

    These are counts of what the repository actually holds, read straight
    from the JSON, so they can be checked against the baseline's own case
    counts without trusting either one.

    Returns:
        ``(file name, case count, breakdown)`` per fixture file.
    """
    counts: list[tuple[str, int, str]] = []
    for path in sorted(EVAL_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        kinds: dict[str, int] = {}
        for case in payload:
            kind = case.get("type") or case.get("category") if isinstance(
                case, dict
            ) else None
            if kind:
                kinds[str(kind)] = kinds.get(str(kind), 0) + 1
        breakdown = " · ".join(
            f"{kind} {count}" for kind, count in sorted(kinds.items())
        )
        counts.append((path.name, len(payload), breakdown))
    return counts


def _render_evaluation() -> None:
    """Evaluation page: the frozen baseline and the fixtures behind it."""
    st.markdown(
        '<p class="araya-headline">Evaluation</p>', unsafe_allow_html=True
    )
    st.caption(EVAL_PAGE_NOTE)
    text = _baseline_text()
    if not text:
        st.warning(BASELINE_UNREADABLE)
        return
    runs = _baseline_run_lines(text)
    if runs:
        st.markdown(
            '<div class="araya-panel"><div class="araya-panel-head">'
            f'<span class="araya-panel-title">{BASELINE_RUNS_TITLE}</span>'
            f'<span class="araya-panel-meta">{BASELINE_RUNS_NOTE}</span>'
            f'</div><div style="margin-top:10px">'
            f"{_baseline_run_html(runs)}</div></div>",
            unsafe_allow_html=True,
        )
    groups = _baseline_snapshot()
    if not groups:
        # The file was readable and its newest block still yielded no
        # measurement, which is a shape this parser does not know rather
        # than a result. Saying so is the honest option; inventing a
        # number from an older block would report the wrong phase.
        st.warning(BASELINE_UNPARSED)
    # One grid for every group card: the previous two-column layout left
    # cards in a row at different heights and needed a spacer element
    # between rows, so the spacing between panels was never the same twice.
    cards: list[str] = []
    for group in groups:
        name = group["name"]
        badge = ""
        if name.lower().startswith("calibration"):
            badge = (
                f'<span class="araya-badge araya-badge--tuning">'
                f"{CALIBRATION_BADGE}</span>"
            )
        elif name.lower().startswith("held"):
            badge = (
                f'<span class="araya-badge araya-badge--reporting">'
                f"{HELDOUT_BADGE}</span>"
            )
        cases = (
            f'<span class="araya-panel-meta araya-mono-face">'
            f'{group["cases"]} cases</span>'
            if group["cases"] is not None
            else ""
        )
        # Lines the block carried but did not measure -- "every other
        # metric unchanged from Phase 8" -- are shown as written: they
        # are why a group can legitimately list one metric.
        remarks = "".join(
            f'<div class="araya-panel-note">{html.escape(line)}</div>'
            for line in group["remarks"]
        )
        cards.append(
            '<div class="araya-panel">'
            '<div class="araya-panel-head">'
            f'<span class="araya-panel-title">{html.escape(name)}'
            f"</span>{badge}{cases}</div>"
            f'<div style="margin-top:12px">'
            f'{_metric_rows_html(group["metrics"])}</div>'
            f"{remarks}</div>"
        )
    if cards:
        st.markdown(
            '<div class="araya-card-grid araya-card-grid--panels">'
            f'{"".join(cards)}</div>',
            unsafe_allow_html=True,
        )

    fixtures = _eval_fixture_counts()
    if fixtures:
        rows = "".join(
            '<div class="araya-metric">'
            f'<span class="araya-metric-name araya-mono-face">'
            f"{html.escape(name)}</span>"
            f'<span class="araya-panel-meta">{html.escape(breakdown)}</span>'
            f'<span class="araya-metric-value araya-mono-face">{count}</span>'
            "</div>"
            for name, count, breakdown in fixtures
        )
        st.markdown(
            '<div class="araya-panel"><div class="araya-panel-head">'
            '<span class="araya-panel-title">Fixtures on disk</span>'
            '<span class="araya-panel-meta">นับจาก eval/*.json โดยตรง</span>'
            f"</div><div style=\"margin-top:12px\">{rows}</div>"
            '<div class="araya-panel-note">'
            "ใช้ตรวจว่าจำนวน case ในไฟล์ตรงกับที่ BASELINE.md รายงานไว้"
            "</div></div>",
            unsafe_allow_html=True,
        )
    with st.expander(BASELINE_READER_LABEL):
        # The file is a long report whose own headings are page-sized;
        # rendered straight into the page they dwarf the console's type
        # scale, so it is read inside a fixed-height block that the
        # stylesheet scales down.
        with st.container(key="araya_baseline_doc", height=460):
            st.markdown(text)


def _render_kb_cards() -> None:
    """The index on the left, the selected document's text on the right.

    A card used to be markup only, which meant the console could say a
    document was indexed but never show what had been indexed. The rows
    are buttons now and the body is rendered as plain text rather than as
    Markdown: corpus text is evidence, and evidence that can restyle the
    page it is being audited on is evidence nobody can read plainly.
    """
    documents = _corpus()
    if not documents:
        st.info("The corpus is empty.")
        return
    selected_id = st.session_state.get("console_document")
    if selected_id not in {document.source_id for document in documents}:
        selected_id = documents[0].source_id
        st.session_state["console_document"] = selected_id

    left, right = st.columns([1, 1.3], gap="medium")
    with left:
        for document in documents:
            # The glyph tells policy from chat at a glance. Streamlit
            # places it inline before the first line, so the stylesheet
            # lifts it out of the text flow and keeps all three lines on
            # one left edge.
            icon = (
                "description" if document.source_type == "policy" else "forum"
            )
            if st.button(
                # The title is free-form frontmatter, so it is
                # neutralised here for the same reason the body is drawn
                # with ``st.code`` below; ``source_type`` and ``status``
                # are allowlisted by the loader and need no guard.
                f"**{document.source_id}**  \n"
                f"{_escape_markdown(document.title)}  \n"
                f"`{document.source_type} · {document.status}`",
                icon=f":material/{icon}:",
                # The type is in the key so the stylesheet can tint the
                # glyph by stratum: policy is authoritative, chat is not.
                key=f"araya_doc_{document.source_type}_{document.source_id}",
                type=(
                    "primary"
                    if document.source_id == selected_id
                    else "secondary"
                ),
                use_container_width=True,
                wrap=True,
            ):
                st.session_state["console_document"] = document.source_id
                st.rerun()
    with right:
        document = next(
            doc for doc in documents if doc.source_id == selected_id
        )
        topics = ", ".join(document.topics) or "–"
        canonical = ", ".join(document.canonical_source_ids) or "–"
        st.markdown(
            '<div class="araya-panel"><div class="araya-panel-head">'
            f'<span class="araya-panel-title araya-mono-face">'
            f"{html.escape(document.source_id)}</span>"
            f'<span class="araya-panel-meta">{html.escape(document.title)}'
            "</span></div>"
            f'<div class="araya-panel-note" style="margin:8px 0 0">'
            f"type {html.escape(document.source_type)} · "
            f"authority {html.escape(document.authority)} · "
            f"status {html.escape(document.status)}<br>"
            f"topics {html.escape(topics)} · "
            f"canonical {html.escape(canonical)} · "
            f"quarantined lines {document.quarantined_line_count}"
            "</div></div>",
            unsafe_allow_html=True,
        )
        st.code(document.content, language=None, wrap_lines=True)


def _render_knowledge_base() -> None:
    """The corpus exactly as the loader validated it."""
    st.markdown('<p class="araya-section">Knowledge Base Index</p>',
                unsafe_allow_html=True)
    st.caption(
        f"{len(_corpus())} mock documents · doc-level character TF-IDF · "
        "no chunking, embeddings, or vector database."
    )
    _render_kb_stats()
    _render_kb_cards()


def _runtime_snapshot() -> dict:
    """Return the frozen runtime configuration the console reports.

    Split from the renderer so the panel and its export read the same
    dict: the credential appears here as presence, never as a value, and
    that stays true of anything built from it.
    """
    _, retriever, _ = _graph_and_retriever()
    return {
        "retriever": {
            "type": "character_tfidf",
            "analyzer": "char",
            "ngram_range": list(retriever.ngram_range),
            "retrieval_unit": "document",
            "documents": len(_corpus()),
            "top_k": config.TOP_K,
        },
        "routing": {
            "mode": "calibrated_3_band",
            "rewrite_floor": config.REWRITE_FLOOR,
            "direct_answer_threshold": config.DIRECT_ANSWER_THRESHOLD,
            "final_answer_threshold": config.FINAL_ANSWER_THRESHOLD,
            "provenance": "calibrated 2026-08-20 on eval/retrieval_calibration.json",
        },
        "guardrail": {
            "type": "deterministic_regex",
            "stage": "pre_retrieval",
            "enabled": True,
        },
        "citation_validator": {
            "mode": "source_id_provenance",
            "enabled": True,
        },
        "logging": {
            "sink": "jsonl",
            "file": Path(config.FALLBACK_LOG_PATH).name,
            "events": ["blocked", "fallback"],
            "persistent_view_enabled": config.ENABLE_OPS_VIEW,
            "max_rows_read": MAX_LOG_ROWS,
        },
        "llm": {
            "model_name": config.MODEL_NAME,
            "api_key": "configured"
            if config.has_llm_credential()
            else "missing",
            "usage": ["medium_band_rewrite", "grounded_reporter"],
            "request_deadline_seconds": config.REQUEST_DEADLINE_SECONDS,
            "reporter_timeout_seconds": config.LLM_TIMEOUT_SECONDS,
            "rewrite_timeout_seconds": config.LLM_REWRITE_TIMEOUT_SECONDS,
            "max_query_chars": config.MAX_QUERY_CHARS,
        },
    }


def _render_runtime_section() -> None:
    """Frozen configuration and real pipeline components, nothing more."""
    runtime = _runtime_snapshot()
    st.markdown(
        '<details class="araya-details" open><summary>'
        '<span class="material-symbols-outlined">terminal</span>'
        "<span>Runtime Configuration &amp; Health</span>"
        '<span class="material-symbols-outlined araya-chevron">'
        "expand_more</span></summary>"
        f'<div class="araya-runtime">{json.dumps(runtime, indent=2)}</div>'
        "</details>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Score colour bands in this console follow the calibrated "
        "thresholds above; the UI defines no thresholds of its own."
    )


def _console_rail_badge(label: str) -> str:
    """Return the count shown beside one rail entry, empty when it has none.

    Both numbers are real: requests made in this session, and documents in
    the loaded index. A section with nothing to count carries no badge
    rather than a zero that looks like a failure.
    """
    if label == "Query Logs":
        return str(len(_all_requests()))
    if label == "Knowledge Base":
        return str(len(_corpus()))
    return ""


def _render_console_sidebar() -> str:
    """Console rail acting as the section switcher.

    The link back to the assistant lives in the app bar with the view
    switch, as it does on the other page, so the rail holds sections only.

    Returns:
        The label of the section the operator selected.
    """
    active = st.session_state.setdefault(
        "console_section", CONSOLE_SECTIONS[0][0]
    )
    with st.sidebar:
        st.markdown(_brand_html(), unsafe_allow_html=True)
        for label, icon in CONSOLE_SECTIONS:
            badge = _console_rail_badge(label)
            if st.button(
                f"{label} `{badge}`" if badge else label,
                icon=f":material/{icon}:",
                use_container_width=True,
                type="primary" if label == active else "tertiary",
                key=f"console_nav_{icon}",
            ):
                st.session_state["console_section"] = label
                st.rerun()
        st.markdown(
            f'<div class="araya-rail-note">{VIEW_SEPARATION_NOTE}</div>',
            unsafe_allow_html=True,
        )
    return st.session_state["console_section"]


def _leave_for_assistant() -> None:
    """Arm the assistant jump when the console's switch moves off Ops.

    Mirrors ``_leave_for_console``: the control is sent home inside the
    callback so it cannot bounce the operator back out on return.
    """
    if st.session_state.get("araya_console_switch") == EMPLOYEE_VIEW:
        st.session_state["araya_console_switch"] = OPS_VIEW
        st.session_state["araya_goto_assistant"] = True


def _export_payload(
    section: str, log: LogReadResult, query_filter: str
) -> tuple[str, list[dict], int]:
    """Collect what the section on screen would export.

    Every branch reuses the builder the panel itself renders from, so an
    export cannot show a number the page does not, and the persistent
    sink still only appears when ``read_persistent_events`` let it be
    read at all.

    Args:
        section: Rail section currently selected.
        log: Bounded sink read for this run.
        query_filter: Free-text filter from the app bar.

    Returns:
        The export title, its rows, and how many of those rows came from
        the persistent sink.
    """
    if section == "Overview":
        requests = _all_requests()
        return (
            "Overview",
            _overview_export_rows(
                _overview_stats(requests), _triage_groups(), len(_sessions())
            ),
            0,
        )
    if section == "Query Logs":
        rows = [
            row
            for row in _session_export_rows(_all_requests())
            if _log_matches(row, query_filter)
        ]
        persistent = [
            dict(record, scope="persistent")
            for record in log.records
            if _log_matches(record, query_filter)
        ]
        return "Query Logs", rows + persistent, len(persistent)
    if section == "Evaluation":
        return (
            "Evaluation",
            _evaluation_export_rows(
                _baseline_snapshot(), _eval_fixture_counts()
            ),
            0,
        )
    if section == "Knowledge Base":
        return "Knowledge Base", _corpus_export_rows(_corpus()), 0
    return "Runtime", _runtime_export_rows(_runtime_snapshot()), 0


def _render_export_popover(
    section: str, log: LogReadResult, query_filter: str
) -> None:
    """One button that offers the page's own data in three formats.

    The formats share one row shape, so JSONL, CSV and Markdown are three
    serialisations of the same rows rather than three exports. The
    popover states the page and the row count and nothing else: what the
    rows cover is what the page itself already shows.
    """
    with st.popover(
        EXPORT_LABEL, icon=":material/download:", use_container_width=True
    ):
        title, rows, _ = _export_payload(section, log, query_filter)
        st.caption(
            EXPORT_SCOPE_NOTE.format(section=title, rows=len(rows))
            if rows
            else EXPORT_EMPTY_NOTE
        )
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
        base = f"araya_{title.lower().replace(' ', '_')}_{stamp}"
        payloads = {
            "jsonl": _rows_to_jsonl(rows),
            "csv": _rows_to_csv(rows),
            "md": _rows_to_markdown(title, rows),
        }
        for label, extension, mime in EXPORT_FORMATS:
            st.download_button(
                label,
                data=payloads[extension],
                file_name=f"{base}.{extension}",
                mime=mime,
                key=f"araya_export_{extension}",
                use_container_width=True,
                # A download is not a state change, and a rerun here
                # would rebuild the bar while the browser is still
                # taking the file.
                on_click="ignore",
                disabled=not rows,
            )


def _render_console_header(log: LogReadResult, section: str) -> str:
    """Console app bar: view switch, search, and the page's own actions.

    Search and export live here rather than in the page body so they stay
    in the same place across sections. The two actions share one column so
    they read as a group pinned to the right of the bar instead of two
    controls floating in the middle of it.

    Args:
        log: Bounded sink read backing the export.
        section: Rail section on screen, which is what the export follows.

    Returns:
        The free-text filter typed into the bar.
    """
    with st.container(key="araya_console_header"):
        switch, search, actions = st.columns(
            [3, 4, 3], vertical_alignment="center"
        )
        with switch:
            st.session_state.setdefault("araya_console_switch", OPS_VIEW)
            st.segmented_control(
                "View",
                options=(EMPLOYEE_VIEW, OPS_VIEW),
                format_func=VIEW_SWITCH_LABELS.get,
                key="araya_console_switch",
                on_change=_leave_for_assistant,
                label_visibility="collapsed",
            )
        with search:
            query_filter = st.text_input(
                "Search",
                placeholder=CONSOLE_SEARCH_PLACEHOLDER,
                label_visibility="collapsed",
                icon=":material/search:",
            )
        with actions:
            with st.container(key="araya_console_actions"):
                theme, export = st.columns(
                    [1, 1], vertical_alignment="center"
                )
                with theme:
                    # A native toggle rather than the mockup's icon
                    # button: the state has to be readable at a glance,
                    # and Streamlit gives no icon-only control that also
                    # shows whether it is on.
                    st.toggle("Dark", key="araya_dark")
                with export:
                    _render_export_popover(section, log, query_filter)
    if st.session_state.pop("araya_goto_assistant", False):
        st.switch_page(_PAGE_REFS["assistant"])
    return query_filter


def _render_console_section(
    section: str, query_filter: str, log: LogReadResult
) -> None:
    """Render the console section selected in the rail."""
    if section == "Overview":
        _render_console_overview()
        return
    if section == "Query Logs":
        _render_console_logs(query_filter, log)
        return
    if section == "Evaluation":
        _render_evaluation()
        return
    if section == "Knowledge Base":
        _render_knowledge_base()
        return
    _render_runtime_section()


def _console_page() -> None:
    """Operations and audit console: telemetry, index, runtime health."""
    st.markdown(_CONSOLE_LAYOUT_CSS, unsafe_allow_html=True)
    # The dark sheet is injected after the light tokens so it overrides
    # them; it covers the panels this console draws, while Streamlit's own
    # widgets keep following the base theme in .streamlit/config.toml.
    if st.session_state.get("araya_dark"):
        st.markdown(_CONSOLE_DARK_CSS, unsafe_allow_html=True)
    log = read_persistent_events(MAX_LOG_ROWS)
    section = _render_console_sidebar()
    query_filter = _render_console_header(log, section)
    if not config.has_llm_credential():
        st.warning(MISSING_KEY_WARNING)
    _render_console_section(section, query_filter, log)
