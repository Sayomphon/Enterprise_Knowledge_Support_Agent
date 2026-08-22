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
    CALIBRATION_BADGE,
    CONSOLE_SEARCH_PLACEHOLDER,
    CONSOLE_SECTIONS,
    EMPLOYEE_VIEW,
    EVAL_DIR,
    EVAL_PAGE_NOTE,
    HELDOUT_BADGE,
    MAX_LOG_ROWS,
    MISSING_KEY_WARNING,
    OPS_LOG_DISABLED_NOTE,
    OPS_VIEW,
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
    _brand_html,
    _clock,
    _display_route,
    _format_score,
    _gating_score,
    _kpi_html,
    _log_matches,
    _log_table_html,
    _matches_filters,
    _metric_rows_html,
    _score_band,
    _stat_card_html,
    _trace_rows,
    _triage_row_html,
)
from ui.runtime import (
    _PAGE_REFS,
    _corpus,
    _graph_and_retriever,
    _session_id,
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
    for row in st.session_state.get("history", []):
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
    for row in st.session_state.get("history", []):
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
    """Overview: route split, the threshold axis, and fallback triage."""
    history = st.session_state.get("history", [])
    _, _, built_at = _graph_and_retriever()
    st.markdown(
        '<p class="araya-headline">Overview</p>', unsafe_allow_html=True
    )
    st.caption(
        f"session {_session_id()} · {len(history)} requests · "
        f"index build {built_at.strftime('%H:%M:%S')}"
    )
    labels = [_display_route(row["state"])[0] for row in history]
    total = len(history) or 1
    answered = sum(1 for label in labels if label in {"Direct", "Rewrite"})
    fallback = labels.count("Fallback")
    blocked = labels.count("Blocked")
    latency = (
        f"{sum(row['latency_seconds'] for row in history) / len(history):.3f}"
        if history
        else "–"
    )
    # Counted from the state rather than inferred from the route: a
    # medium-band request the alias catalog settled spends one call, and
    # one whose deadline ran out spends none.
    llm_calls = sum(
        int(row["state"].get("llm_calls", 0)) for row in history
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
    cells = (
        _stat_card_html(
            "Answered", str(answered), f"/ {len(history)} requests",
            answered / total, "green",
        ),
        _stat_card_html(
            "Fallback", str(fallback), fallback_reason,
            fallback / total, "amber",
        ),
        _stat_card_html(
            "Blocked", str(blocked), blocked_reason, blocked / total, "red",
        ),
        _stat_card_html(
            "Avg latency", latency, "s · this session",
            note=f"{llm_calls} LLM call ในเซสชันนี้",
        ),
    )
    for column, card in zip(st.columns(4), cells):
        with column:
            st.markdown(card, unsafe_allow_html=True)

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
    for node, tone, glyph, detail in _trace_rows(state):
        rows_html.append(
            f'<div class="araya-trace-row'
            f'{" araya-trace-row--skipped" if tone == "skip" else ""}">'
            f'<span class="araya-trace-badge araya-trace-badge--{tone}">'
            f'<span class="material-symbols-outlined">{glyph}</span></span>'
            f'<span class="araya-trace-node araya-mono-face">{node}</span>'
            f'<span class="araya-trace-detail">{html.escape(detail)}</span>'
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
    """Query Logs: the session's requests beside one request's trace."""
    history = st.session_state.get("history", [])
    st.markdown(
        '<p class="araya-headline">Requests</p>', unsafe_allow_html=True
    )
    if not history:
        st.info("No requests in this session yet.")
        st.divider()
        _render_log_section(log, query_filter)
        return
    counts = {label: 0 for label in ROUTE_FILTER_OPTIONS}
    for row in history:
        counts[_display_route(row["state"])[0]] += 1
    options = [f"ทั้งหมด {len(history)}"] + [
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
        for row in reversed(history)
        if _matches_filters(row, query_filter, routes)
    ]
    if not visible:
        st.info("No requests in this session match the current filters.")
        st.divider()
        _render_log_section(log, query_filter)
        return

    selected_id = st.session_state.get("console_request")
    if selected_id not in {row["request_id"] for row in visible}:
        selected_id = visible[0]["request_id"]
        st.session_state["console_request"] = selected_id

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
                    _clock(row["timestamp"]),
                    score if raw is not None else reason,
                    f"{row['latency_seconds']:.3f}s",
                )
                if part
            )
            if st.button(
                f":{_ROUTE_BADGE_COLOURS[css]}-badge[{label}] "
                f"**{row['request_id']}**  \n{row['query']}  \n`{meta}`",
                key=f"araya_req_{row['request_id']}",
                type=(
                    "primary"
                    if row["request_id"] == selected_id
                    else "secondary"
                ),
                use_container_width=True,
                wrap=True,
            ):
                st.session_state["console_request"] = row["request_id"]
                st.rerun()
    with right:
        record = next(
            row for row in visible if row["request_id"] == selected_id
        )
        _render_trace_panel(record)
    st.divider()
    _render_log_section(log, query_filter)


def _render_log_section(log: LogReadResult, query_filter: str = "") -> None:
    """Blocked/fallback telemetry from the real JSONL sink.

    Args:
        log: Bounded read result from the sink.
        query_filter: Free-text filter from the console header.
    """
    st.markdown('<p class="araya-section">Blocked / Fallback Log</p>',
                unsafe_allow_html=True)
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


def _render_kb_stats() -> None:
    """Index-health strip: real counts and this process's build time."""
    documents = _corpus()
    _, _, built_at = _graph_and_retriever()
    stats = st.columns(3)
    cells = (
        ("Indexed", f"{len(documents)} / {len(documents)}", ""),
        ("Duplicate IDs", "0", "enforced at load"),
        ("Index built", built_at.strftime("%H:%M:%S"), "this process"),
    )
    for column, (label, value, unit) in zip(stats, cells):
        with column:
            st.markdown(_kpi_html(label, value, unit), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _baseline_snapshot() -> list[dict]:
    """Read the newest measured-results block from ``eval/BASELINE.md``.

    The baseline file is the record of what was actually run, and it holds
    one block per phase; this returns the last one. Nothing is recomputed
    here, so the console cannot report a number the baseline does not
    contain, and a file that no longer matches the expected shape yields
    an empty list rather than a guess.

    Returns:
        One dict per metric group, each with ``name``, ``cases`` (or
        ``None``), a list of ``(metric, passed, total, remark)`` tuples,
        and any group-level remarks the block carried.
    """
    path = EVAL_DIR / "BASELINE.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    marker = text.rfind("## Measured results")
    if marker < 0:
        return []
    block = text[marker:].split("```")
    if len(block) < 2:
        return []
    groups: list[dict] = []
    for line in block[1].splitlines():
        metric = _EVAL_METRIC_LINE.match(line)
        if metric is not None and groups:
            groups[-1]["metrics"].append(
                (
                    metric["name"].strip(),
                    int(metric["passed"]),
                    int(metric["total"]),
                    (metric["note"] or "").strip(" <-"),
                )
            )
            continue
        group = _EVAL_GROUP_LINE.match(line)
        if group is not None:
            groups.append(
                {
                    "name": group["name"].strip(),
                    "cases": (
                        int(group["cases"]) if group["cases"] else None
                    ),
                    "metrics": [],
                    "remarks": [],
                }
            )
            continue
        remark = _EVAL_REMARK_LINE.match(line)
        if remark is not None and groups:
            groups[-1]["remarks"].append(remark["text"])
    return [group for group in groups if group["metrics"]]


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
    groups = _baseline_snapshot()
    if not groups:
        st.warning(
            "Could not read a measured-results block from "
            f"{(EVAL_DIR / 'BASELINE.md').name}. The file is the source of "
            "these numbers, so nothing is shown rather than an estimate."
        )
    for index in range(0, len(groups), 2):
        for column, group in zip(
            st.columns(2, gap="medium"), groups[index:index + 2]
        ):
            name = group["name"]
            badge = ""
            if name.lower().startswith("calibration"):
                badge = (
                    f'<span class="araya-badge araya-badge--tuning">'
                    f"{CALIBRATION_BADGE}</span>"
                )
            elif name.lower().startswith("held-out"):
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
            with column:
                st.markdown(
                    '<div class="araya-panel">'
                    '<div class="araya-panel-head">'
                    f'<span class="araya-panel-title">{html.escape(name)}'
                    f"</span>{badge}{cases}</div>"
                    f'<div style="margin-top:12px">'
                    f'{_metric_rows_html(group["metrics"])}</div>'
                    f"{remarks}</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("")

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
    with st.expander("อ่าน eval/BASELINE.md ฉบับเต็ม"):
        try:
            st.markdown((EVAL_DIR / "BASELINE.md").read_text(encoding="utf-8"))
        except OSError as error:
            st.info(f"Baseline file unavailable: {type(error).__name__}")


def _render_kb_cards() -> None:
    """One card per validated corpus document, in two columns."""
    documents = _corpus()
    cards: list[str] = []
    for document in documents:
        icon_css = (
            "policy" if document.source_type == "policy" else "chat"
        )
        icon = "description" if icon_css == "policy" else "forum"
        cards.append(
            '<div class="araya-doc-card">'
            f'<div class="araya-doc-icon araya-doc-icon--{icon_css}">'
            f'<span class="material-symbols-outlined">{icon}</span></div>'
            '<div style="flex:1;min-width:0">'
            f'<span class="araya-source-id">{document.source_id}</span>'
            '<span class="araya-indexed">Indexed</span>'
            f'<div class="araya-source-title">{html.escape(document.title)}'
            f" · {document.source_type}</div>"
            "</div></div>"
        )
    left, right = st.columns(2)
    half = (len(cards) + 1) // 2
    with left:
        st.markdown("".join(cards[:half]), unsafe_allow_html=True)
    with right:
        st.markdown("".join(cards[half:]), unsafe_allow_html=True)


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


def _render_runtime_section() -> None:
    """Frozen configuration and real pipeline components, nothing more."""
    _, retriever, _ = _graph_and_retriever()
    runtime = {
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
        return str(len(st.session_state.get("history", [])))
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


def _render_console_header(log: LogReadResult) -> str:
    """Console app bar: view switch, search, theme, and export.

    Search and export move out of the page body and into the bar, where
    they stay in the same place across every section.

    Args:
        log: Bounded sink read backing the export button. It is empty
            while ``ENABLE_OPS_VIEW`` is off, which disables the export
            with no separate rule to keep in sync.

    Returns:
        The free-text filter typed into the bar.
    """
    with st.container(key="araya_console_header"):
        switch, search, theme, export = st.columns(
            [3, 4, 2, 2], vertical_alignment="center"
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
        with theme:
            # A native toggle rather than the mockup's icon button: the
            # state has to be readable at a glance, and Streamlit gives no
            # icon-only control that also shows whether it is on.
            st.toggle("Dark", key="console_dark")
        with export:
            filtered = [
                record
                for record in log.records
                if _log_matches(record, query_filter)
            ]
            payload = "\n".join(
                json.dumps(record, ensure_ascii=False) for record in filtered
            )
            st.download_button(
                "Export JSONL",
                data=payload,
                file_name="fallback_queries_export.jsonl",
                mime="application/x-ndjson",
                icon=":material/download:",
                use_container_width=True,
                disabled=not filtered,
                help="Download the JSONL rows matching the current search.",
            )
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
    if st.session_state.get("console_dark"):
        st.markdown(_CONSOLE_DARK_CSS, unsafe_allow_html=True)
    log = read_persistent_events(MAX_LOG_ROWS)
    section = _render_console_sidebar()
    query_filter = _render_console_header(log)
    if not config.has_llm_credential():
        st.warning(MISSING_KEY_WARNING)
    _render_console_section(section, query_filter, log)
