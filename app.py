"""Streamlit UI for the enterprise support pipeline.

Presentation and integration only (AGENTS.md section 3): this module
invokes the compiled graph and renders ``PipelineState``, session
telemetry, the JSONL fallback log, the loaded corpus, and the frozen
runtime configuration. No routing, threshold, guardrail, rewrite, or
citation logic is duplicated here.

Visual language follows the "Enterprise Logic" design system, whose
single definition is the ``_DESIGN_SYSTEM_CSS`` block below: Enterprise
Blue, Be Vietnam Pro with a Noto Sans Thai fallback for Thai glyphs,
JetBrains Mono for machine data, tonal cards with subtle borders. Mockup elements without a real data source
(24h KPIs, recent-session lists, human avatars, notification badges) are
intentionally omitted, and score colours are bound to the calibrated
runtime thresholds as the design system itself requires.

The app serves two pages: the employee assistant on ``/`` and the audit
console on ``/console``. Each owns its rail, app bar, and layout measure
so they read as separate products, and the console holds every technical
detail (reason codes, scores, latency, runtime state). That separation is
information architecture for the demo only; it is NOT a security boundary
and no authentication or RBAC is implemented.

Because it is not a security boundary, the persistent JSONL sink -- which
holds raw employee questions from every past session -- stays hidden
unless ``ENABLE_OPS_VIEW`` is turned on deliberately, and is then read
through the bounded tail reader rather than loaded whole (remediation
plan Finding 7). Only the current session's own requests are shown by
default.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import html
import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src import config
from src.fallback import ReasonCode, response_text_for_state
from src.graph import build_graph
from src.ingestion.loader import load_documents
from src.logging_utils import read_persistent_events
from src.retrievers.local_tfidf import LocalTfidfRetriever
from src.schemas import (
    Document,
    LogReadResult,
    PipelineState,
    RetrievedDocument,
)

# The only permitted framing for cosine-similarity numbers in the UI
# (AGENTS.md section 4, invariant 6): a similarity heuristic, never an
# answer-correctness probability.
SCORE_LABEL = "Retrieval Similarity (heuristic)"
# A policy that entered the evidence through a chat document's canonical
# link was never ranked by retrieval, so it carries no similarity score.
LINKED_POLICY_LABEL = (
    "Authoritative policy linked from a chat source, not ranked by retrieval"
)
LINKED_POLICY_TEXT = "Linked policy"

APP_TITLE = "Araya Enterprise AI"
BRAND_NAME = "Araya AI"
APP_SUBTITLE = "Thai RAG System"
EMPLOYEE_VIEW = "Employee Assistant"
OPS_VIEW = "AI Operations / Audit"

# Console rail: section label paired with its Material Symbols glyph.
CONSOLE_SECTIONS: tuple[tuple[str, str], ...] = (
    ("Overview", "dashboard"),
    ("Query Logs", "terminal"),
    ("Knowledge Base", "library_books"),
    ("Runtime", "monitor_heart"),
)
ROUTE_FILTER_OPTIONS: tuple[str, ...] = (
    "Direct",
    "Rewrite",
    "Fallback",
    "Blocked",
)
OVERVIEW_QUERY_ROWS = 5

# Page objects for the current script run so each rail can link to the other
# view. Streamlit rebuilds them on every rerun, so this is a cache of the
# current run only, never cross-session state.
_PAGE_REFS: dict[str, object] = {}

# Thai user-facing strings, kept as named constants per repo convention.
CHAT_PLACEHOLDER = (
    "พิมพ์คำถามเกี่ยวกับการเบิกค่าใช้จ่าย การลา หรือขั้นตอนภายในองค์กร "
    "(Shift + Enter เพื่อขึ้นบรรทัดใหม่)"
)
BLOCKED_CHIP = "Blocked by Guardrail"
FALLBACK_CHIP = "Insufficient Evidence · Fallback"
# A missing credential produces the same route but a different cause, so
# the chip must not tell the employee the evidence was thin when the
# answer service was simply not configured (remediation plan Finding 8).
SERVICE_UNAVAILABLE_CHIP = "Answer Service Unavailable · Fallback"
ANSWERED_CHIP = "Grounded Answer"
SOURCES_HEADER = "Sources"
NEW_SESSION_LABEL = "New Session"
MISSING_KEY_WARNING = (
    "ยังไม่ได้ตั้งค่า OPENAI_API_KEY: คำถามที่ต้องเรียก LLM "
    "จะจบด้วยเหตุผล llm_not_configured และแสดงข้อความว่าบริการยังไม่พร้อม "
    "(เส้นทาง blocked และ out-of-domain ทำงานได้ปกติโดยไม่ต้องใช้คีย์)"
)
INPUT_DISCLAIMER = (
    "Araya Enterprise AI may produce inaccurate information. "
    "Always verify critical facts."
)
SIMILARITY_FOOTNOTE = (
    "Retrieval similarity is a heuristic for evidence matching, "
    "not a probability that the answer is correct."
)
# Employee-facing wording never names the console: an end user only needs to
# know the diagnostics exist and who holds them. Naming an internal surface
# in employee copy invites requests for access that the flag cannot grant.
AUDIT_ONLY_NOTE = (
    "Technical guardrail diagnostics are available to authorized "
    "administrators only."
)
HELP_TEXT = (
    "ผู้ช่วยนี้ตอบจากเอกสารภายในองค์กรที่จัดเตรียมไว้เท่านั้น "
    "ทุกคำตอบจะอ้างอิงรหัสเอกสารที่ตรวจสอบย้อนกลับได้ "
    "หากหลักฐานไม่เพียงพอ ระบบจะแจ้งให้ทราบแทนการเดา "
    "หากต้องการความช่วยเหลือเพิ่มเติม โปรดติดต่อฝ่าย IT Support "
    "พร้อมแจ้ง Request ID ที่แสดงในคำตอบ"
)
VIEW_SEPARATION_NOTE = (
    "View separation is an information-architecture demo for the prototype, "
    "not a security boundary; no authentication or RBAC is implemented."
)

MAX_LOG_ROWS = 50
# Shown in place of the persistent table while the demo flag is off. It
# names the flag but never the file path: a log destination is deployment
# detail, not console content.
OPS_LOG_DISABLED_NOTE = (
    "Persistent query history is hidden. The JSONL sink stores raw "
    "questions from every past session, so it is off unless "
    "ENABLE_OPS_VIEW is enabled for a local walkthrough. This flag is a "
    "demo switch, not authentication or RBAC. Requests made in this "
    "session are still shown above."
)
SNIPPET_CHARS = 180

# "Enterprise Logic" design tokens, injected once per page. This block is
# the design system: no other module may define a colour or a typeface.
_DESIGN_SYSTEM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

/* Keep Streamlit's own icon spans and our Material spans on their icon
   fonts; everything else moves to the design-system typeface stack. */
html, body,
[data-testid="stAppViewContainer"]
  *:not(.material-symbols-outlined):not([data-testid="stIconMaterial"]) {
    font-family: 'Be Vietnam Pro', 'Noto Sans Thai', sans-serif;
}
.material-symbols-outlined {
    font-family: 'Material Symbols Outlined' !important;
    font-size: 16px; vertical-align: -3px;
    font-variation-settings: 'FILL' 0, 'wght' 400;
}
/* Machine-data typeface. JetBrains Mono separates machine output (source
   ids, scores, logs, JSON) from human prose, so this list is the single
   definition of what counts as machine data. The reset above
   resolves at specificity (0,3,0) because of its two :not() arguments, which
   no single-class rule can outrank -- !important is the same escape hatch the
   icon font already needs. Thai has no mono glyphs, so it falls through to
   Noto Sans Thai instead of an arbitrary system face. */
.araya-mono-face,
.araya-mono,
.araya-brand-sub,
.araya-source-id,
.araya-source-body,
.araya-audit-value,
.araya-runtime,
.araya-score--green, .araya-score--amber, .araya-score--red,
table.araya-table td,
.araya-mono-face *:not(.material-symbols-outlined),
.araya-runtime *:not(.material-symbols-outlined),
table.araya-table td *:not(.material-symbols-outlined) {
    font-family: 'JetBrains Mono', 'Noto Sans Thai', monospace !important;
}
#MainMenu, footer { visibility: hidden; }

.araya-brand { display: flex; gap: 12px; align-items: center; }
.araya-brand-mark {
    width: 40px; height: 40px; border-radius: 8px; background: #0052CC;
    color: #fff; display: flex; align-items: center; justify-content: center;
}
.araya-brand-name { color: #003d9b; font-weight: 700; font-size: 20px; line-height: 1.2; }
.araya-brand-sub {
    font-size: 11px; color: #434654;
}

/* Scoped to the markdown container on purpose: Streamlit styles every <p>
   inside it at 1rem with specificity (0,1,1), which silently flattened the
   headline token to body size when these rules were plain classes. */
[data-testid="stMarkdownContainer"] p.araya-headline {
    font-size: 30px; line-height: 38px; font-weight: 700;
    letter-spacing: -0.02em; color: #191c1e; margin: 0;
}
[data-testid="stMarkdownContainer"] p.araya-subtitle {
    font-size: 16px; line-height: 24px; color: #434654; margin: 4px 0 0 0;
}
[data-testid="stMarkdownContainer"] p.araya-section {
    font-size: 22px; line-height: 28px; font-weight: 600; color: #191c1e;
    margin: 0;
}

.araya-bubble-user {
    background: #dae2ff; color: #001848; border: 1px solid #DFE1E6;
    border-radius: 16px 16px 2px 16px; padding: 16px 24px;
    max-width: 85%; margin-left: auto; width: fit-content;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
/* Agent turns are an avatar plus the card, as in the mockup: the brand mark
   anchors the column so answers read as one speaker across long threads. */
.araya-agent-row { display: flex; gap: 16px; align-items: flex-start; }
.araya-agent-avatar {
    width: 40px; height: 40px; border-radius: 9999px; background: #0052CC;
    color: #ffffff; flex-shrink: 0; display: flex;
    align-items: center; justify-content: center;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.araya-agent-avatar .material-symbols-outlined { font-size: 20px; }
.araya-card {
    background: #FFFFFF; border: 1px solid #DFE1E6;
    border-radius: 2px 16px 16px 16px; padding: 16px 24px;
    max-width: 90%; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.araya-card--blocked { background: #ffdad6; border-color: #ba1a1a; }
/* Level 1 surface with an amber outline, not an amber fill: colour stays an
   accent so the fallback state cannot outshout a real block. */
.araya-card--fallback { background: #FFFFFF; border-color: #FFAB00; }
.araya-date-divider {
    display: flex; justify-content: center; margin-bottom: 8px;
}
.araya-date-divider span {
    background: #edeef0; color: #434654; border-radius: 9999px;
    padding: 4px 16px; font-size: 12px; font-weight: 700;
    letter-spacing: 0.05em;
}

.araya-card-head {
    display: flex; justify-content: space-between; align-items: center; gap: 8px;
    border-bottom: 1px solid #DFE1E6; padding-bottom: 6px; margin-bottom: 10px;
    flex-wrap: wrap;
}
.araya-chip {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 12px; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase;
}
.araya-chip--ok { color: #00875A; }
.araya-chip--blocked { color: #ba1a1a; }
.araya-chip--fallback { color: #624000; }
/* body-md, and the 24px line box is the minimum that clears Thai tone marks
   above and vowel signs below. */
.araya-answer, .araya-blocked-text, .araya-fallback-text {
    white-space: pre-wrap; font-size: 16px; line-height: 24px;
}
.araya-answer { color: #191c1e; }
.araya-blocked-text { color: #93000a; }
.araya-fallback-text { color: #191c1e; }

.araya-mono { font-size: 13px; }
.araya-score--green { color: #00875A; }
.araya-score--amber { color: #b97900; }
.araya-score--red { color: #DE350B; }

.araya-meta-box {
    border: 1px solid #c3c6d6; border-radius: 6px; background: #FFFFFF;
    padding: 8px 10px; margin-top: 10px; display: flex; flex-wrap: wrap;
    justify-content: space-between; align-items: center; gap: 8px;
}
.araya-pill {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; padding: 2px 10px; border-radius: 9999px;
}
.araya-pill--blocked { color: #ba1a1a; background: #ffdad6; }
.araya-footnote { font-size: 11px; color: #737685; margin-top: 6px; }

.araya-sources { border-top: 1px solid #DFE1E6; margin-top: 10px; padding-top: 8px; }
/* Two columns, as in the mockup: citations are scanned as a set, and a
   single column pushes the second source below the fold on short answers. */
.araya-source-grid {
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px; margin-top: 8px;
}
@media (max-width: 640px) {
    .araya-source-grid { grid-template-columns: minmax(0, 1fr); }
}
.araya-source-card {
    border: 1px solid #c3c6d6; border-radius: 8px; background: #ffffff;
    overflow: hidden;
}
.araya-source-card summary {
    display: flex; align-items: center; gap: 8px; padding: 8px 10px;
    cursor: pointer; list-style: none;
}
.araya-source-card summary::-webkit-details-marker { display: none; }
.araya-source-icon { font-size: 24px; color: #0052CC; }
.araya-source-head {
    display: flex; flex-direction: column; flex: 1; min-width: 0;
}
.araya-source-id { font-size: 13px; color: #0052CC; font-weight: 700; }
.araya-source-title {
    font-size: 12px; font-weight: 700; letter-spacing: 0.05em; color: #434654;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.araya-source-meta {
    display: flex; justify-content: space-between; align-items: center;
    gap: 8px; padding: 6px 10px; border-top: 1px solid #c3c6d6;
    font-size: 12px; font-weight: 700; letter-spacing: 0.05em; color: #434654;
}
.araya-source-body {
    background: #f8f9fb; padding: 8px 10px;
    font-size: 12px; color: #42526E;
    line-height: 1.6; white-space: pre-wrap;
}

.araya-kpi {
    background: #FFFFFF; border: 1px solid #DFE1E6; border-radius: 8px;
    padding: 16px; height: 100%;
}
.araya-kpi-head {
    display: flex; align-items: center; gap: 12px; margin-bottom: 8px;
}
.araya-kpi-badge {
    width: 32px; height: 32px; border-radius: 9999px; flex-shrink: 0;
    display: inline-flex; align-items: center; justify-content: center;
}
.araya-kpi-badge--primary { background: #dae2ff; color: #003d9b; }
.araya-kpi-badge--error { background: #ffdad6; color: #ba1a1a; }
.araya-kpi-badge--secondary { background: #8af5be; color: #006c47; }
.araya-kpi-badge--neutral { background: #edeef0; color: #434654; }
.araya-kpi-label {
    font-size: 12px; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase; color: #434654;
}
.araya-kpi-body { display: flex; align-items: baseline; gap: 8px; }
.araya-kpi-value {
    font-size: 30px; line-height: 38px; font-weight: 700; color: #191c1e;
}
.araya-kpi-unit { font-size: 13px; color: #737685; }

table.araya-table { width: 100%; border-collapse: collapse; background: #fff;
    border: 1px solid #DFE1E6; border-radius: 12px; overflow: hidden; }
table.araya-table th {
    text-align: left; font-size: 12px; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase; color: #434654; background: #f3f4f6;
    border-bottom: 1px solid #DFE1E6; padding: 12px 16px;
}
table.araya-table td {
    font-size: 13px; color: #191c1e;
    border-bottom: 1px solid #DFE1E6; padding: 12px 16px;
    vertical-align: middle;
}
table.araya-table .material-symbols-outlined { font-size: 14px; }
table.araya-table tbody tr:hover { background: #f8f9fb; }
/* Blocked requests are the rows an auditor scans for first, so they carry a
   standing tint rather than relying on the route chip alone. */
table.araya-table tbody tr.araya-row--blocked { background: rgba(255,218,214,0.35); }
/* Long Thai queries must not turn one row into three lines; the untruncated
   text stays reachable through the cell's title attribute. */
.araya-cell-clip {
    max-width: 260px; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap;
}
.araya-route {
    display: inline-flex; align-items: center; gap: 4px; font-size: 12px;
    font-weight: 600; padding: 4px 8px; border-radius: 9999px; white-space: nowrap;
}
.araya-route--direct { background: #8af5be; color: #00714b; }
.araya-route--rewrite { background: #dae2ff; color: #001848; }
.araya-route--fallback { background: #edeef0; color: #434654; }
.araya-route--blocked { background: #ba1a1a; color: #ffffff; }

.araya-doc-card {
    display: flex; gap: 10px; align-items: flex-start; border: 1px solid #DFE1E6;
    border-radius: 8px; background: #f8f9fb; padding: 10px; margin-top: 6px;
}
.araya-doc-icon {
    width: 32px; height: 32px; border-radius: 4px; display: flex;
    align-items: center; justify-content: center; flex-shrink: 0;
}
.araya-doc-icon--policy { background: #8af5be; color: #006c47; }
.araya-doc-icon--chat { background: #dae2ff; color: #003d9b; }
.araya-indexed {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; color: #00875A; background: rgba(113,219,166,0.3);
    padding: 1px 8px; border-radius: 9999px; float: right;
}
.araya-runtime {
    font-size: 12.5px; color: #42526E;
    background: #ffffff; border: 1px solid #DFE1E6; border-radius: 8px;
    padding: 12px; white-space: pre; overflow-x: auto;
    /* The mockup caps the console at 260px so the JSON payload cannot push
       the rest of the audit surface below the fold. */
    max-height: 260px; overflow-y: auto; margin: 0 12px 12px;
}
.araya-details {
    background: #f3f4f6; border: 1px solid #DFE1E6; border-radius: 8px;
}
.araya-details > summary {
    display: flex; align-items: center; gap: 8px; padding: 10px 12px;
    cursor: pointer; list-style: none; color: #434654;
    font-size: 13px; font-weight: 700;
}
.araya-details > summary::-webkit-details-marker { display: none; }
.araya-chevron {
    margin-left: auto; color: #737685; transition: transform 0.15s ease;
}
details[open] > summary .araya-chevron { transform: rotate(180deg); }
.araya-audit { margin-top: 8px; }
.araya-audit-grid {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px; padding: 0 12px 12px;
}
@media (max-width: 720px) {
    .araya-audit-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.araya-audit-key { display: block; font-size: 11px; color: #737685; }
.araya-audit-value { font-size: 13px; color: #42526E; word-break: break-word; }

/* The design system docks a 280px rail; Streamlit ships 300px by default and
   sets it inline, so the token only lands with !important. */
[data-testid="stSidebar"] {
    width: 280px !important; min-width: 280px !important;
}
/* Both mockups pin their app bar. st.container(key=...) renders a
   .st-key-<key> wrapper, which is the only stable hook Streamlit exposes for
   styling one layout block. */
.st-key-araya_topbar, .st-key-araya_console_header {
    position: sticky; top: 0; z-index: 30; background: #FFFFFF;
    border-bottom: 1px solid #DFE1E6;
    padding: 8px 0 12px; margin-bottom: 16px;
}
.araya-session-id {
    display: inline-flex; align-items: center; gap: 6px; color: #434654;
}
/* Rail entries read as a navigation list, not as centred buttons. Streamlit
   centres the label in an inner wrapper, so both levels need the override. */
[data-testid="stSidebar"] [data-testid^="stBaseButton-"] {
    justify-content: flex-start; border-radius: 8px;
}
[data-testid="stSidebar"] [data-testid^="stBaseButton-"] > div {
    justify-content: flex-start; width: 100%;
}
</style>
"""

# Per-view layout tokens. The two mockups use different measures -- an 800px
# reading column for the chat, the 1200px content grid for the console -- so
# the width is injected by whichever view is rendering instead of once for
# the whole app. Both add the 24px gutter on top of the measure.
_CHAT_LAYOUT_CSS = f"""
<style>
[data-testid="stMainBlockContainer"] {{
    max-width: 848px; padding-left: 24px; padding-right: 24px;
}}
[data-testid="stBottomBlockContainer"] {{
    max-width: 848px; padding-left: 24px; padding-right: 24px;
}}
/* The composer disclaimer sits under the input in the mockup, not at the top
   of the message flow where a Streamlit caption lands. Streamlit already
   reserves 56px of padding below the composer, so it is rendered there as
   generated content instead of as a stray element in the transcript. */
[data-testid="stBottomBlockContainer"]::after {{
    content: {json.dumps(INPUT_DISCLAIMER)};
    display: block; text-align: center; margin-top: 10px;
    font-size: 12px; font-weight: 700; letter-spacing: 0.05em; color: #737685;
}}
/* Circular enterprise-blue send affordance from the mockup composer. */
[data-testid="stChatInputSubmitButton"] {{
    width: 40px; height: 40px; border-radius: 9999px;
    background: #0052CC !important;
}}
[data-testid="stChatInputSubmitButton"]:hover:enabled {{
    background: #003d9b !important;
}}
[data-testid="stChatInputSubmitButton"]:disabled {{
    background: #c3c6d6 !important;
}}
[data-testid="stChatInputSubmitButton"] span,
[data-testid="stChatInputSubmitButton"] svg {{
    color: #ffffff !important; fill: #ffffff !important;
}}
</style>
"""

_CONSOLE_LAYOUT_CSS = """
<style>
[data-testid="stMainBlockContainer"] {
    max-width: 1248px; padding-left: 24px; padding-right: 24px;
}
</style>
"""


@st.cache_resource
def _corpus() -> list[Document]:
    """Load the validated corpus once per Streamlit process."""
    return load_documents()


@st.cache_resource
def _graph_and_retriever():
    """Build the TF-IDF index and compile the graph once per process.

    Streamlit reruns the script on every interaction; caching keeps the
    expensive index build and graph compilation out of that loop. The
    build timestamp is real and feeds the Ops "last index build" stat.
    """
    retriever = LocalTfidfRetriever(_corpus())
    built_at = datetime.now().astimezone()
    return build_graph(retriever=retriever), retriever, built_at


def _invoke_graph(query: str) -> tuple[PipelineState, float]:
    """Run one query through the real pipeline, measuring wall latency."""
    graph, _, _ = _graph_and_retriever()
    started = time.perf_counter()
    state: PipelineState = graph.invoke({"query": query})
    return state, time.perf_counter() - started


def _record_request(query: str, state: PipelineState, latency: float) -> None:
    """Append one real request outcome to the session history.

    The request id is session-scoped presentation bookkeeping (it indexes
    this history list); every other stored field comes from the graph or
    is measured here. History is telemetry for the audit view, never
    conversation memory for the pipeline.
    """
    history = st.session_state.setdefault("history", [])
    history.append(
        {
            "request_id": f"Q-{len(history) + 1:03d}",
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "query": query,
            "state": state,
            "latency_seconds": round(latency, 3),
        }
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


def _evidence_provenance_html(document: RetrievedDocument) -> str:
    """Render how one evidence document entered this request.

    Showing the placeholder 0.0000 of a canonically linked policy would
    read as a very poor retrieval match rather than as a metadata link.
    """
    if document.matched_query_type is None:
        return (
            f'<span title="{LINKED_POLICY_LABEL}">{LINKED_POLICY_TEXT}</span>'
        )
    return _similarity_html(document.score)


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


def _render_user_bubble(query: str) -> None:
    st.markdown(
        f'<div class="araya-bubble-user">{html.escape(query)}</div>',
        unsafe_allow_html=True,
    )


def _source_cards_html(state: PipelineState) -> str:
    """Build expandable source cards for every validated citation."""
    documents_by_id = {
        document.source_id: document
        for document in state.get("answer_evidence", [])
    }
    cards: list[str] = []
    for source_id in state.get("valid_citations", []):
        document = documents_by_id.get(source_id)
        if document is None:
            cards.append(
                f'<div class="araya-source-card"><summary>'
                f'<span class="araya-source-id">{html.escape(source_id)}'
                "</span></summary></div>"
            )
            continue
        icon = "description" if document.source_type == "policy" else "forum"
        snippet = document.content[:SNIPPET_CHARS]
        if len(document.content) > SNIPPET_CHARS:
            snippet += "…"
        cards.append(
            '<details class="araya-source-card"><summary>'
            '<span class="material-symbols-outlined araya-source-icon">'
            f"{icon}</span>"
            '<span class="araya-source-head">'
            f'<span class="araya-source-id">{html.escape(source_id)}</span>'
            f'<span class="araya-source-title">{html.escape(document.title)}'
            "</span></span>"
            '<span class="material-symbols-outlined araya-chevron">'
            "expand_more</span></summary>"
            '<div class="araya-source-meta">'
            f"<span>{html.escape(document.source_type.capitalize())} source"
            "</span>"
            f"<span>{_evidence_provenance_html(document)}</span></div>"
            f'<div class="araya-source-body">{html.escape(snippet)}</div>'
            "</details>"
        )
    return f'<div class="araya-source-grid">{"".join(cards)}</div>'


def _agent_row(card_html: str) -> str:
    """Wrap one agent card in the avatar column used by the mockup."""
    return (
        '<div class="araya-agent-row">'
        '<div class="araya-agent-avatar">'
        '<span class="material-symbols-outlined">memory</span></div>'
        f"{card_html}</div>"
    )


def _render_date_divider(timestamp: str) -> None:
    """Open the transcript with the day and clock of the first request."""
    stamp = datetime.fromisoformat(timestamp)
    day = (
        "Today"
        if stamp.date() == datetime.now().astimezone().date()
        else stamp.strftime("%d %b %Y")
    )
    st.markdown(
        f'<div class="araya-date-divider"><span>{day}, '
        f'{stamp.strftime("%H:%M")}</span></div>',
        unsafe_allow_html=True,
    )


def _render_agent_card(record: dict) -> None:
    """Render one agent response card according to its real route."""
    state: PipelineState = record["state"]
    route = state.get("route")
    fixed_text = response_text_for_state(
        route,
        state.get("guardrail_reason"),
        state.get("fallback_reason"),
        telemetry_logged=state.get("telemetry_logged", True),
    )

    if route == "blocked":
        st.markdown(
            _agent_row(
                '<div class="araya-card araya-card--blocked">'
                '<div class="araya-chip araya-chip--blocked">'
                '<span class="material-symbols-outlined">gpp_bad</span> '
                f"{BLOCKED_CHIP}</div>"
                '<div class="araya-blocked-text">'
                f'{html.escape(fixed_text or "")}</div>'
                '<div class="araya-meta-box">'
                '<span class="araya-mono" style="color:#434654">'
                '<span class="material-symbols-outlined">receipt_long</span> '
                f'Request ID {record["request_id"]}</span>'
                '<span class="araya-pill araya-pill--blocked">'
                "Blocked before LLM</span></div>"
                f'<div class="araya-footnote">{AUDIT_ONLY_NOTE}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    if route == "fallback":
        service_unavailable = (
            state.get("fallback_reason") == ReasonCode.LLM_NOT_CONFIGURED
        )
        chip = (
            SERVICE_UNAVAILABLE_CHIP
            if service_unavailable
            else FALLBACK_CHIP
        )
        icon = "cloud_off" if service_unavailable else "report"
        st.markdown(
            _agent_row(
                '<div class="araya-card araya-card--fallback">'
                '<div class="araya-chip araya-chip--fallback">'
                f'<span class="material-symbols-outlined">{icon}</span> '
                f"{chip}</div>"
                '<div class="araya-fallback-text">'
                f'{html.escape(fixed_text or "")}</div>'
                '<div class="araya-meta-box">'
                '<span class="araya-mono" style="color:#434654">'
                '<span class="material-symbols-outlined">receipt_long</span> '
                f'Request ID {record["request_id"]}</span></div>'
                f'<div class="araya-footnote">{AUDIT_ONLY_NOTE}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    if route != "answered":
        return

    score = _gating_score(state)
    score_html = (
        '<span class="araya-chip" style="color:#434654">Retrieval '
        f"{_similarity_html(score)}</span>"
        if score is not None
        else ""
    )
    st.markdown(
        _agent_row(
            '<div class="araya-card">'
            '<div class="araya-card-head">'
            '<span class="araya-chip araya-chip--ok">'
            '<span class="material-symbols-outlined">verified</span> '
            f"{ANSWERED_CHIP}</span>"
            f"{score_html}</div>"
            '<div class="araya-answer">'
            f'{html.escape(state.get("answer", ""))}</div>'
            '<div class="araya-sources">'
            '<span class="araya-chip" style="color:#434654">'
            f'{SOURCES_HEADER} ({len(state.get("valid_citations", []))})'
            "</span>"
            f"{_source_cards_html(state)}"
            f'<div class="araya-footnote">{SIMILARITY_FOOTNOTE}</div>'
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def _render_employee_view() -> None:
    """Chat-style employee flow: ask, then read a grounded, cited answer."""
    st.markdown(_CHAT_LAYOUT_CSS, unsafe_allow_html=True)
    history = st.session_state.get("history", [])
    if history:
        _render_date_divider(history[0]["timestamp"])
    for record in history:
        _render_user_bubble(record["query"])
        _render_agent_card(record)

    query = st.chat_input(CHAT_PLACEHOLDER)
    if query and query.strip():
        with st.spinner("กำลังค้นหาจากฐานความรู้..."):
            state, latency = _invoke_graph(query.strip())
        _record_request(query.strip(), state, latency)
        st.rerun()


def _kpi_html(
    label: str,
    value: str,
    unit: str = "",
    icon: str = "",
    tone: str = "neutral",
) -> str:
    """Render one KPI card in the console's stat style.

    Args:
        label: Caps label describing the metric and its scope.
        value: Already-formatted metric value.
        unit: Optional unit or qualifier rendered next to the value.
        icon: Material Symbols glyph for the tinted badge; omit for the
            plain stat strips that the mockup renders without icons.
        tone: Badge tint token: ``primary``, ``error``, ``secondary`` or
            ``neutral``.

    Returns:
        The card markup.
    """
    unit_html = f'<span class="araya-kpi-unit">{unit}</span>' if unit else ""
    badge_html = (
        f'<span class="araya-kpi-badge araya-kpi-badge--{tone}">'
        f'<span class="material-symbols-outlined">{icon}</span></span>'
        if icon
        else ""
    )
    return (
        '<div class="araya-kpi"><div class="araya-kpi-head">'
        f'{badge_html}<span class="araya-kpi-label">{label}</span></div>'
        f'<div class="araya-kpi-body"><span class="araya-kpi-value">{value}'
        f"</span>{unit_html}</div></div>"
    )


def _brand_html() -> str:
    """Sidebar brand lockup shared by both rails."""
    return (
        '<div class="araya-brand">'
        '<div class="araya-brand-mark">'
        '<span class="material-symbols-outlined">memory</span></div>'
        f'<div><div class="araya-brand-name">{BRAND_NAME}</div>'
        f'<div class="araya-brand-sub">{APP_SUBTITLE}</div></div></div>'
    )


def _session_id() -> str:
    """Return this browser session's identifier, creating it on first use.

    Presentation bookkeeping only: it labels the session in the chat top bar
    so a user can quote it to support, and it shares the clock with the
    request ids the console lists for the same session.
    """
    session_id = st.session_state.get("session_id")
    if session_id is None:
        started = datetime.now().astimezone()
        session_id = f"RAG-{started.strftime('%Y%m%d-%H%M')}"
        st.session_state["session_id"] = session_id
    return session_id


def _render_kpis() -> None:
    """Session- and corpus-scoped stats only; no invented 24h KPIs."""
    history = st.session_state.get("history", [])
    degraded = sum(
        1
        for row in history
        if row["state"].get("route") in {"blocked", "fallback"}
    )
    latency_value = "–"
    if history:
        latencies = [row["latency_seconds"] for row in history]
        latency_value = f"{sum(latencies) / len(latencies):.3f}"
    columns = st.columns(4)
    # Scope stays visible in the unit slot: every number here is session- or
    # corpus-scoped, and the mockup's 24h aggregates have no data source.
    cells = (
        ("Requests", str(len(history)), "this session", "history", "neutral"),
        ("Blocked / fallback", str(degraded), "this session", "shield",
         "error"),
        ("Avg latency", latency_value, "s · this session", "speed",
         "primary"),
        ("Knowledge docs", str(len(_corpus())), "doc-level TF-IDF",
         "library_books", "secondary"),
    )
    for column, (label, value, unit, icon, tone) in zip(columns, cells):
        with column:
            st.markdown(
                _kpi_html(label, value, unit, icon, tone),
                unsafe_allow_html=True,
            )


def _matches_filters(
    record: dict, query_filter: str, routes: tuple[str, ...]
) -> bool:
    """Test one session record against the console header filters.

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


def _render_session_queries(
    query_filter: str = "",
    routes: tuple[str, ...] = (),
    limit: int | None = None,
) -> None:
    """Current-session request log styled as the console table.

    Args:
        query_filter: Free-text filter from the console header.
        routes: Route labels selected in the console header.
        limit: Optional cap on rendered rows, newest first.
    """
    st.markdown('<p class="araya-section">Recent Queries</p>',
                unsafe_allow_html=True)
    st.caption(
        "Scope: current Streamlit session only. "
        f"Similarity columns are {SCORE_LABEL}."
    )
    history = st.session_state.get("history", [])
    if not history:
        st.info("No requests in this session yet.")
        return
    visible = [
        row
        for row in reversed(history)
        if _matches_filters(row, query_filter, routes)
    ]
    if not visible:
        st.info("No requests in this session match the current filters.")
        return
    truncated = limit is not None and len(visible) > limit
    if limit is not None:
        visible = visible[:limit]

    rows: list[str] = []
    for row in visible:
        state: PipelineState = row["state"]
        label, icon, css = _display_route(state)
        raw_score = state.get("raw_retrieval_score")
        expanded_score = state.get("expanded_retrieval_score")
        similarity = "–" if raw_score is None else _similarity_html(raw_score)
        if expanded_score is not None:
            similarity += f" → {_similarity_html(expanded_score)}"
        reason = state.get("fallback_reason") or state.get(
            "guardrail_reason"
        )
        row_css = (
            ' class="araya-row--blocked"'
            if state.get("route") == "blocked"
            else ""
        )
        rows.append(
            f"<tr{row_css}>"
            f'<td>{row["request_id"]}</td>'
            f'<td>{html.escape(_clock(row["timestamp"]))}</td>'
            f'<td class="araya-cell-clip" title="{html.escape(row["query"])}">'
            f'{html.escape(row["query"])}</td>'
            f'<td><span class="araya-route araya-route--{css}">'
            f'<span class="material-symbols-outlined">{icon}</span>{label}'
            "</span></td>"
            f"<td>{similarity}</td>"
            f'<td>{html.escape(reason) if reason else "–"}</td>'
            f'<td>{row["latency_seconds"]:.3f}s</td>'
            "</tr>"
        )
    st.markdown(
        '<table class="araya-table"><thead><tr>'
        "<th>Request</th><th>Time</th><th>User Query</th><th>Route</th>"
        "<th>Similarity</th><th>Reason</th><th>Latency</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>",
        unsafe_allow_html=True,
    )
    if truncated:
        st.caption(
            f"Showing the {limit} newest matching requests; open Query Logs "
            "for the full session."
        )


def _format_score(score: float | None) -> str:
    """Format one similarity score for a text field, dash when unset."""
    return "–" if score is None else f"{score:.4f}"


def _audit_details_html(title: str, icon: str, fields: dict[str, str]) -> str:
    """Render one collapsible audit strip with a key/value grid.

    Args:
        title: Strip heading, already scoped to a request id.
        icon: Material Symbols glyph for the heading.
        fields: Ordered field name to displayed value.

    Returns:
        The details markup used by the console's audit strips.
    """
    cells = "".join(
        f'<div><span class="araya-audit-key">{html.escape(key)}</span>'
        f'<span class="araya-audit-value">{html.escape(value)}</span></div>'
        for key, value in fields.items()
    )
    return (
        '<details class="araya-details araya-audit"><summary>'
        f'<span class="material-symbols-outlined">{icon}</span>'
        f"<span>{html.escape(title)}</span>"
        '<span class="material-symbols-outlined araya-chevron">'
        "expand_more</span></summary>"
        f'<div class="araya-audit-grid">{cells}</div></details>'
    )


def _render_inspections(
    query_filter: str = "", routes: tuple[str, ...] = ()
) -> None:
    """Guardrail and rewrite audit strips for the filtered session rows.

    Admin-only detail (AGENTS.md section 4): reason codes, rewrite output,
    and the model-call count never appear in the employee view.
    """
    history = st.session_state.get("history", [])
    strips: list[str] = []
    for row in reversed(history):
        if not _matches_filters(row, query_filter, routes):
            continue
        state: PipelineState = row["state"]
        if state.get("route") == "blocked":
            strips.append(
                _audit_details_html(
                    f"Guardrail audit · {row['request_id']}",
                    "policy",
                    {
                        "reason_code": str(
                            state.get("guardrail_reason") or "–"
                        ),
                        "stage": "pre_retrieval",
                        "llm_calls": "0",
                        "latency": f"{row['latency_seconds']:.3f}s",
                    },
                )
            )
        if state.get("rewritten_queries") is not None:
            strips.append(
                _audit_details_html(
                    f"Rewrite inspection · {row['request_id']}",
                    "edit_note",
                    {
                        "raw_retrieval_score": _format_score(
                            state.get("raw_retrieval_score")
                        ),
                        "rewritten_queries": " | ".join(
                            state.get("rewritten_queries") or []
                        )
                        or "–",
                        "rewrite_failed": str(
                            state.get("rewrite_failed", False)
                        ),
                        "expanded_retrieval_score": _format_score(
                            state.get("expanded_retrieval_score")
                        ),
                    },
                )
            )
    if strips:
        st.markdown("".join(strips), unsafe_allow_html=True)


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


def _render_chat_sidebar() -> None:
    """Employee rail: brand, session reset, and the link to the console.

    Mockup rail entries without a real data source (recent sessions,
    settings, support) stay omitted rather than being faked.
    """
    with st.sidebar:
        st.markdown(_brand_html(), unsafe_allow_html=True)
        st.markdown("")
        if st.button(
            NEW_SESSION_LABEL,
            icon=":material/add:",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.pop("history", None)
            st.session_state.pop("session_id", None)
            st.rerun()
        st.page_link(
            _PAGE_REFS["console"],
            label=OPS_VIEW,
            icon=":material/monitoring:",
        )
        st.caption(VIEW_SEPARATION_NOTE)


def _render_chat_topbar() -> None:
    """Session identity and help, matching the mockup's top app bar.

    The mockup's notification bell is omitted: this prototype has no event
    stream behind it.
    """
    with st.container(key="araya_topbar"):
        identity, action = st.columns([4, 1], vertical_alignment="center")
        with identity:
            st.markdown(
                '<span class="araya-mono araya-session-id">'
                '<span class="material-symbols-outlined">tag</span>'
                f"Session {_session_id()}</span>",
                unsafe_allow_html=True,
            )
        with action:
            with st.popover(
                "Help", icon=":material/help:", use_container_width=True
            ):
                st.markdown(HELP_TEXT)


def _render_console_sidebar() -> str:
    """Console rail acting as the section switcher.

    Returns:
        The label of the section the operator selected.
    """
    active = st.session_state.setdefault(
        "console_section", CONSOLE_SECTIONS[0][0]
    )
    with st.sidebar:
        st.markdown(_brand_html(), unsafe_allow_html=True)
        st.markdown("")
        for label, icon in CONSOLE_SECTIONS:
            if st.button(
                label,
                icon=f":material/{icon}:",
                use_container_width=True,
                type="primary" if label == active else "tertiary",
                key=f"console_nav_{icon}",
            ):
                st.session_state["console_section"] = label
                st.rerun()
        st.page_link(
            _PAGE_REFS["assistant"],
            label=EMPLOYEE_VIEW,
            icon=":material/chat:",
        )
        st.caption(VIEW_SEPARATION_NOTE)
    return st.session_state["console_section"]


def _render_console_header(
    log: LogReadResult,
) -> tuple[str, tuple[str, ...]]:
    """Console identity plus the three header actions, all data-bound.

    Args:
        log: Bounded sink read backing the export button. It is empty
            while ``ENABLE_OPS_VIEW`` is off, which disables the export
            with no separate rule to keep in sync.

    Returns:
        The free-text filter and the selected route labels.
    """
    with st.container(key="araya_console_header"):
        identity, search, routes, export = st.columns(
            [8, 3, 3, 2], vertical_alignment="bottom"
        )
        with identity:
            st.markdown(
                '<p class="araya-headline">AI Operations &amp; Audit '
                "Console</p>"
                '<p class="araya-subtitle">Decision telemetry, knowledge '
                "index status, and runtime health for the enterprise RAG "
                "prototype.</p>",
                unsafe_allow_html=True,
            )
        with search:
            query_filter = st.text_input(
                "Search",
                placeholder="Search queries and reasons",
                label_visibility="collapsed",
                icon=":material/search:",
            )
        with routes:
            selected = st.multiselect(
                "Routes",
                ROUTE_FILTER_OPTIONS,
                label_visibility="collapsed",
                placeholder="All routes",
            )
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
                "Export",
                data=payload,
                file_name="fallback_queries_export.jsonl",
                mime="application/x-ndjson",
                icon=":material/download:",
                use_container_width=True,
                disabled=not filtered,
                help="Download the JSONL rows matching the current search.",
            )
    st.caption(VIEW_SEPARATION_NOTE)
    return query_filter, tuple(selected)


def _render_console_section(
    section: str,
    query_filter: str,
    routes: tuple[str, ...],
    log: LogReadResult,
) -> None:
    """Render the console section selected in the rail."""
    if section == "Overview":
        _render_kpis()
        st.divider()
        _render_session_queries(query_filter, routes, OVERVIEW_QUERY_ROWS)
        st.divider()
        _render_kb_stats()
        return
    if section == "Query Logs":
        _render_session_queries(query_filter, routes)
        _render_inspections(query_filter, routes)
        st.divider()
        _render_log_section(log, query_filter)
        return
    if section == "Knowledge Base":
        _render_knowledge_base()
        return
    _render_runtime_section()


def _employee_page() -> None:
    """Employee assistant: grounded answers, citations, safe messaging.

    Operational telemetry (reason codes, scores, latency, key state) never
    appears here; it belongs to the console page.
    """
    _render_chat_sidebar()
    _render_chat_topbar()
    _render_employee_view()


def _console_page() -> None:
    """Operations and audit console: telemetry, index, runtime health."""
    st.markdown(_CONSOLE_LAYOUT_CSS, unsafe_allow_html=True)
    log = read_persistent_events(MAX_LOG_ROWS)
    section = _render_console_sidebar()
    query_filter, routes = _render_console_header(log)
    if not config.has_llm_credential():
        st.warning(MISSING_KEY_WARNING)
    _render_console_section(section, query_filter, routes, log)


def main() -> None:
    """Compose the two separated pages and run the selected one.

    The assistant answers on ``/`` and the console on ``/console``. Each
    page carries its own rail, header, and layout tokens so the two read as
    different products. This is information architecture, not access
    control: no authentication or RBAC exists, and both rails say so.
    """
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    load_dotenv()
    st.markdown(_DESIGN_SYSTEM_CSS, unsafe_allow_html=True)
    _PAGE_REFS.update(
        {
            # The default page always answers on the root URL, so it takes
            # no url_path of its own: "/" is the assistant, "/console" is
            # the audit console.
            "assistant": st.Page(
                _employee_page,
                title=EMPLOYEE_VIEW,
                default=True,
            ),
            "console": st.Page(
                _console_page, title=OPS_VIEW, url_path="console"
            ),
        }
    )
    st.navigation(list(_PAGE_REFS.values()), position="hidden").run()


main()
