"""The employee assistant page: answers, citations, safe messaging.

Operational telemetry -- reason codes, scores, latency, credential state
-- never appears on this surface; it belongs to the console. What the
employee reads is either a validated answer with its sources or one of
the fixed texts selected by ``src.fallback``, never model output that
the validator refused.
"""

from __future__ import annotations

import html
from datetime import datetime

import streamlit as st

from src import config
from src.fallback import (
    is_ambiguous_topic,
    is_service_failure,
    response_text_for_state,
)
from src.schemas import PipelineState

from ui.labels import (
    AMBIGUOUS_TOPIC_CHIP,
    ANSWERED_CHIP,
    AUDIT_ONLY_NOTE,
    BLOCKED_CHIP,
    CHAT_PLACEHOLDER,
    COPY_ANSWER_LABEL,
    DOWNLOAD_ANSWER_LABEL,
    EMPLOYEE_VIEW,
    EMPTY_STATE_HEADLINE,
    EMPTY_STATE_SUBTITLE,
    FALLBACK_CHIP,
    HELP_TEXT,
    NEW_SESSION_LABEL,
    NEXT_STEP_PROMPTS,
    OPS_VIEW,
    SESSION_LIST_LABEL,
    SESSION_LIST_NOTE,
    SERVICE_UNAVAILABLE_CHIP,
    SIMILARITY_FOOTNOTE,
    SOURCES_HEADER,
    SUGGESTED_QUESTIONS,
    THINKING_LABEL,
    USER_TURN_LABEL,
    VIEW_SEPARATION_NOTE,
    VIEW_SWITCH_LABELS,
)
from ui.styles import _CHAT_DARK_CSS, _CHAT_LAYOUT_CSS
from ui.formatting import (
    _answer_html,
    _answer_markdown,
    _brand_html,
    _evidence_pill_html,
    _gating_score,
    _scope_strips_html,
    _session_rail_label,
    _source_list_html,
    _turn_clock,
)
from ui.runtime import (
    _PAGE_REFS,
    _active_session,
    _corpus,
    _invoke_graph,
    _record_request,
    _session_id,
    _sessions,
    _start_new_session,
    _switch_session,
)

def _render_user_bubble(query: str, timestamp: str) -> None:
    """Render one employee turn: its clock label above the question."""
    st.markdown(
        '<div class="araya-turn">'
        f'<span class="araya-turn-label">{USER_TURN_LABEL} · '
        f"{_turn_clock(timestamp)}</span>"
        f'<div class="araya-bubble-user">{html.escape(query)}</div></div>',
        unsafe_allow_html=True,
    )


def _render_answer_card(record: dict) -> None:
    """Render one grounded answer with its evidence meter and citations.

    The card is a native container rather than a single markdown block so
    the two head controls can be real widgets: "copy" is a popover holding
    ``st.code``, whose own copy button reaches the clipboard without a
    custom component, and "save" is a download button.
    """
    state: PipelineState = record["state"]
    request_id = record["request_id"]
    citations = state.get("valid_citations", [])
    score = _gating_score(state)
    with st.container(key=f"araya_answer_{request_id}"):
        verdict, evidence, copy, save = st.columns(
            [4, 3, 2, 2], vertical_alignment="center"
        )
        with verdict:
            st.markdown(
                '<span class="araya-chip araya-chip--ok">'
                '<span class="material-symbols-outlined">verified</span>'
                f"{ANSWERED_CHIP}</span>",
                unsafe_allow_html=True,
            )
        with evidence:
            if score is not None:
                st.markdown(
                    _evidence_pill_html(score), unsafe_allow_html=True
                )
        with copy:
            with st.popover(
                COPY_ANSWER_LABEL,
                icon=":material/content_copy:",
                use_container_width=True,
            ):
                st.code(
                    state.get("answer", ""), language=None, wrap_lines=True
                )
        with save:
            st.download_button(
                DOWNLOAD_ANSWER_LABEL,
                data=_answer_markdown(record),
                file_name=f"{request_id}.md",
                mime="text/markdown",
                icon=":material/download:",
                key=f"araya_save_{request_id}",
                on_click="ignore",
                use_container_width=True,
            )
        st.markdown(
            '<div class="araya-card-body">'
            f"{_answer_html(state)}"
            '<div class="araya-sources">'
            '<div class="araya-sources-head">'
            f"{SOURCES_HEADER.format(count=len(citations))}</div>"
            f"{_source_list_html(state)}"
            f'<div class="araya-footnote">{SIMILARITY_FOOTNOTE} · '
            f'<span class="araya-request-id">Request {request_id}</span>'
            "</div></div></div>",
            unsafe_allow_html=True,
        )


def _render_notice_card(
    record: dict,
    tone: str,
    icon: str,
    title: str,
    text: str,
    next_steps: bool,
    note: str,
) -> None:
    """Render one degraded turn: what happened, and where to go next.

    Args:
        record: The session history entry being rendered.
        tone: ``fallback`` or ``blocked``; selects the card's colour.
        icon: Material Symbols glyph shown beside the title.
        title: Card heading, in the employee's language.
        text: The fixed response text from ``src.fallback``. The UI never
            writes its own wording for a degraded route.
        next_steps: Whether to offer the two starter questions. An
            evidence fallback earns them because the corpus can answer
            something nearby; the other cases do not.
        note: Footnote shown when no ways forward are offered.
    """
    request_id = record["request_id"]
    with st.container(key=f"araya_notice_{tone}_{request_id}"):
        st.markdown(
            f'<div class="araya-notice araya-notice--{tone}">'
            '<span class="material-symbols-outlined araya-notice-icon">'
            f"{icon}</span><div>"
            f'<div class="araya-notice-title">{title}</div>'
            f'<div class="araya-notice-text">{html.escape(text)}</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
        if not next_steps:
            st.markdown(
                f'<div class="araya-footnote">{note} · '
                f'<span class="araya-request-id">Request {request_id}</span>'
                "</div>",
                unsafe_allow_html=True,
            )
            return
        columns = st.columns(len(NEXT_STEP_PROMPTS) + 1)
        for index, (label, question) in enumerate(NEXT_STEP_PROMPTS):
            with columns[index]:
                if st.button(
                    label,
                    key=f"araya_next_{index}_{request_id}",
                    help=question,
                    use_container_width=True,
                ):
                    # Same door as the composer: the next turn is drawn
                    # at the end of the transcript, not inside the card
                    # that offered it.
                    st.session_state["araya_pending"] = question
                    st.rerun()
        with columns[-1]:
            st.markdown(
                f'<span class="araya-request-id">{request_id}</span>',
                unsafe_allow_html=True,
            )


def _render_agent_card(record: dict) -> None:
    """Render one agent response according to the route the graph chose."""
    state: PipelineState = record["state"]
    route = state.get("route")
    fixed_text = response_text_for_state(
        route,
        state.get("guardrail_reason"),
        state.get("fallback_reason"),
        telemetry_logged=state.get("telemetry_logged", True),
    )

    if route == "blocked":
        _render_notice_card(
            record,
            tone="blocked",
            icon="gpp_bad",
            title=BLOCKED_CHIP,
            text=fixed_text or "",
            next_steps=False,
            note=AUDIT_ONLY_NOTE,
        )
        return

    if route == "fallback":
        # Asked of the shared selector rather than compared here: the
        # notice card must describe the same failure the fixed text
        # above already states, and a second copy of that boundary in
        # the UI would drift from it.
        reason = state.get("fallback_reason")
        service_unavailable = is_service_failure(reason)
        under_specified = is_ambiguous_topic(reason)
        if service_unavailable:
            icon, title = "cloud_off", SERVICE_UNAVAILABLE_CHIP
        elif under_specified:
            icon, title = "help", AMBIGUOUS_TOPIC_CHIP
        else:
            icon, title = "help_center", FALLBACK_CHIP
        _render_notice_card(
            record,
            tone="fallback",
            icon=icon,
            title=title,
            text=fixed_text or "",
            # An unconfigured answer service is not a gap in the corpus,
            # so pointing the employee at other topics would misdescribe
            # the failure: the ways forward stay for the two outcomes
            # that really are about what the corpus can answer, and an
            # under-specified question is the one they help most.
            next_steps=not service_unavailable,
            note=AUDIT_ONLY_NOTE,
        )
        return

    if route != "answered":
        return

    _render_answer_card(record)


def _ask(query: str) -> None:
    """Run one question through the pipeline and store its outcome.

    The question is drawn before the run starts, with the status under
    it: a request can take half a minute, and until now the screen sat
    unchanged for all of it and then produced the question and its answer
    together. Both the composer and the starter questions arrive here
    through ``araya_pending``, so the turn is always appended at the end
    of the transcript rather than wherever the widget happened to be.
    """
    _render_user_bubble(
        query, datetime.now().astimezone().isoformat(timespec="seconds")
    )
    with st.container(key="araya_thinking"):
        with st.spinner(THINKING_LABEL):
            state, latency, node_seconds = _invoke_graph(query)
    _record_request(query, state, latency, node_seconds)
    st.rerun()


def _render_empty_state() -> None:
    """Opening screen for a session that has not asked anything yet.

    The first run used to be a blank page above a text box, which says
    nothing about what the assistant knows or how it behaves when it does
    not know. This screen answers both before the employee types: the
    subject and the size of the corpus, four questions the corpus can
    actually answer, and the three outcomes a question can have.
    """
    st.markdown(
        f'<p class="araya-hero-title">{EMPTY_STATE_HEADLINE}</p>'
        '<p class="araya-hero-sub">'
        f"{EMPTY_STATE_SUBTITLE.format(count=len(_corpus()))}</p>",
        unsafe_allow_html=True,
    )
    # Two per row, as in the mockup. Each button is one starter question,
    # styled as a card so the whole card is the click target; its leading
    # bold run is the department tag the stylesheet lifts out.
    for row_start in range(0, len(SUGGESTED_QUESTIONS), 2):
        pairs = SUGGESTED_QUESTIONS[row_start:row_start + 2]
        for offset, (column, (tag, question)) in enumerate(
            zip(st.columns(len(pairs)), pairs)
        ):
            with column:
                if st.button(
                    f"**{tag}**  \n{question}",
                    key=f"araya_suggest_{row_start + offset}",
                    use_container_width=True,
                    wrap=True,
                ):
                    st.session_state["araya_pending"] = question
                    st.rerun()
    st.markdown(_scope_strips_html(), unsafe_allow_html=True)


def _render_employee_view() -> None:
    """Chat-style employee flow: ask, then read a grounded, cited answer."""
    st.markdown(_CHAT_LAYOUT_CSS, unsafe_allow_html=True)
    # Injected after the light tokens so it overrides them, exactly as
    # the console does with its own sheet.
    if st.session_state.get("araya_dark"):
        st.markdown(_CHAT_DARK_CSS, unsafe_allow_html=True)
    history = st.session_state.get("history", [])
    pending = st.session_state.get("araya_pending")
    # The opening screen gives way as soon as a question is in flight:
    # the starter cards would otherwise sit above the question they just
    # sent. The block is always rendered, empty or not -- Streamlit keeps
    # the previous run's elements on screen, faded, until something takes
    # their place, and an omitted block leaves those ghosts behind for as
    # long as the request takes.
    with st.container(key="araya_opening"):
        if not history and not pending:
            _render_empty_state()
    # Each turn states its own clock, so the transcript no longer opens
    # with a single date divider that scrolls away after two questions.
    for record in history:
        _render_user_bubble(record["query"], record["timestamp"])
        _render_agent_card(record)
    if pending:
        st.session_state.pop("araya_pending", None)
        _ask(pending)

    # The composer enforces the same length ceiling the guardrail applies,
    # so an over-length question is stopped at the keyboard instead of
    # being sent and rejected.
    query = st.chat_input(CHAT_PLACEHOLDER, max_chars=config.MAX_QUERY_CHARS)
    if query and query.strip():
        # Handed to the next run rather than answered here: that run
        # draws the question at the end of the transcript first, which is
        # where a reader is looking.
        st.session_state["araya_pending"] = query.strip()
        st.rerun()


def _render_chat_sidebar() -> None:
    """Employee rail: brand, the session list, and the standing caveats.

    "New Session" used to discard the transcript outright, which made the
    rail's own list impossible; it now archives the current session and
    opens a new one, and every session of this browser tab is listed under
    the button. Rail entries with no real data source behind them
    (settings, support) stay omitted rather than being faked, and the
    console link lives in the top bar with the view switch.
    """
    with st.sidebar:
        st.markdown(_brand_html(), unsafe_allow_html=True)
        if st.button(
            NEW_SESSION_LABEL,
            icon=":material/add:",
            key="araya_new_session",
            use_container_width=True,
        ):
            _start_new_session()
            st.rerun()
        sessions = _sessions()
        active = _active_session()["session_id"]
        st.markdown(
            f'<div class="araya-rail-label">{SESSION_LIST_LABEL}</div>',
            unsafe_allow_html=True,
        )
        # Newest first: the session being read is normally the newest one,
        # and an older session should never push it below the fold.
        for session in reversed(sessions):
            session_id = session["session_id"]
            if st.button(
                _session_rail_label(session),
                key=f"araya_session_{session_id}",
                type="primary" if session_id == active else "tertiary",
                use_container_width=True,
                wrap=True,
            ):
                _switch_session(session_id)
                st.rerun()
        st.markdown(
            f'<div class="araya-rail-note">{SESSION_LIST_NOTE}<br><br>'
            f"{VIEW_SEPARATION_NOTE}</div>",
            unsafe_allow_html=True,
        )


def _leave_for_console() -> None:
    """Arm the console jump when the top-bar switch moves off the assistant.

    The switch keeps its own widget state, so it is reset here, inside the
    change callback, before the rerun reads it. Without the reset the
    control would still say "console" on the way back and bounce the
    employee straight out of the assistant again.
    """
    if st.session_state.get("araya_view_switch") == OPS_VIEW:
        st.session_state["araya_view_switch"] = EMPLOYEE_VIEW
        st.session_state["araya_goto_console"] = True


def _render_chat_topbar() -> None:
    """View switch, session identity, and help, as the mockup's app bar.

    The switch is the assistant's only route to the console, replacing the
    rail link so that both pages offer the same control in the same place.
    The mockup's notification bell is omitted: this prototype has no event
    stream behind it, and the language toggle is omitted because no
    translated copy exists to switch to.
    """
    with st.container(key="araya_topbar"):
        # The switch column carries its own floor in the stylesheet, the
        # identity takes what is left, and the two utilities share the
        # last cell so they read as one group at the end of the bar.
        switch, identity, action = st.columns(
            [3, 4, 3], vertical_alignment="center"
        )
        with switch:
            # Seeded through session state rather than `default=`: the
            # change callback writes the same key to send the control
            # home, and Streamlit warns when a widget carries both.
            st.session_state.setdefault("araya_view_switch", EMPLOYEE_VIEW)
            st.segmented_control(
                "View",
                options=(EMPLOYEE_VIEW, OPS_VIEW),
                format_func=VIEW_SWITCH_LABELS.get,
                key="araya_view_switch",
                on_change=_leave_for_console,
                label_visibility="collapsed",
            )
        with identity:
            st.markdown(
                '<span class="araya-mono araya-session-id">'
                '<span class="material-symbols-outlined">tag</span>'
                f"Session {_session_id()}</span>",
                unsafe_allow_html=True,
            )
        with action:
            with st.container(key="araya_chat_actions"):
                theme, help_action = st.columns(
                    [1, 1], vertical_alignment="center"
                )
                with theme:
                    # The same switch the console carries: the theme is a
                    # property of the reader, not of one of the two pages.
                    st.toggle("Dark", key="araya_dark")
                with help_action:
                    with st.popover(
                        "Help",
                        icon=":material/help:",
                        use_container_width=False,
                    ):
                        st.markdown(HELP_TEXT)
    # Navigating from inside the column would abandon the layout block
    # half-built, so the armed jump is taken once the bar is complete.
    if st.session_state.pop("araya_goto_console", False):
        st.switch_page(_PAGE_REFS["console"])


def _employee_page() -> None:
    """Employee assistant: grounded answers, citations, safe messaging.

    Operational telemetry (reason codes, scores, latency, key state) never
    appears here; it belongs to the console page.
    """
    _render_chat_sidebar()
    _render_chat_topbar()
    _render_employee_view()
