"""Streamlit UI for the enterprise support pipeline.

Presentation and integration only (AGENTS.md section 3): this module
invokes the compiled graph and renders ``PipelineState``, session
telemetry, the JSONL fallback log, the loaded corpus, and the frozen
runtime configuration. No routing, threshold, guardrail, rewrite, or
citation logic is duplicated here.

Visual language follows the "Enterprise Logic" design system, whose
single definition is the ``_DESIGN_SYSTEM_CSS`` block below: Enterprise
Blue, Be Vietnam Pro with a Noto Sans Thai fallback for Thai glyphs,
JetBrains Mono for machine data, tonal cards with subtle borders. Mockup
elements without a real data source
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
import re
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langgraph.graph.state import CompiledStateGraph

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
EMPLOYEE_VIEW = "Employee Assistant"
OPS_VIEW = "AI Operations / Audit"
# Short forms for the top-bar switch. The page titles are what the browser
# tab and the console rail carry; the switch itself has room for a glyph
# and one or two words.
VIEW_SWITCH_LABELS: dict[str, str] = {
    EMPLOYEE_VIEW: ":material/chat: Assistant",
    OPS_VIEW: ":material/monitoring: Ops Console",
}

# Console rail: section label paired with its Material Symbols glyph.
CONSOLE_SECTIONS: tuple[tuple[str, str], ...] = (
    ("Overview", "dashboard"),
    ("Query Logs", "terminal"),
    ("Evaluation", "rule"),
    ("Knowledge Base", "library_books"),
    ("Runtime", "monitor_heart"),
)
ROUTE_FILTER_OPTIONS: tuple[str, ...] = (
    "Direct",
    "Rewrite",
    "Fallback",
    "Blocked",
)

# Page objects for the current script run so each rail can link to the other
# view. Streamlit rebuilds them on every rerun, so this is a cache of the
# current run only, never cross-session state.
_PAGE_REFS: dict[str, object] = {}

# Thai user-facing strings, kept as named constants per repo convention.
CHAT_PLACEHOLDER = "พิมพ์คำถามของคุณ…"

# Opening screen shown while this session has no history. It replaces the
# blank first run with the two things a first-time employee needs: what the
# assistant can be asked, and where its answers come from. The document
# count is read from the loaded corpus, never written down here.
EMPTY_STATE_HEADLINE = "ถามเรื่องการเบิกค่าใช้จ่ายและการลาได้เลย"
EMPTY_STATE_SUBTITLE = (
    "ผู้ช่วยตอบจากเอกสารภายในองค์กร {count} ฉบับ "
    "และอ้างอิงรหัสเอกสารทุกคำตอบ ถ้าหลักฐานไม่พอจะบอกตรง ๆ แทนการเดา"
)
# Starter questions, each one a topic the corpus actually documents:
# FIN-001 / FIN-002 for the finance pair, HR-001 / HR-002 for the leave
# pair. A suggestion whose answer would fall back teaches the wrong thing
# about the system on first contact.
SUGGESTED_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("Finance", "ขั้นตอนการเบิกค่าแท็กซี่หลังทำ OT ต้องทำอย่างไร"),
    ("Finance", "ใบเสร็จหาย ยังเคลมค่าที่จอดรถได้ไหม"),
    ("HR", "ลาพักร้อนต้องแจ้งล่วงหน้ากี่วัน และกดที่เมนูไหน"),
    ("HR", "ลาป่วยกี่วันต้องมีใบรับรองแพทย์"),
)
# Stating the three outcomes up front is what keeps a fallback from being
# read as a broken system: refusing without evidence is designed behaviour,
# so it is advertised before the first question, not explained after it.
SCOPE_STRIPS: tuple[tuple[str, str, str], ...] = (
    ("ตอบได้", "นโยบาย HR / Finance ที่มีเอกสารรองรับ", "ok"),
    ("บอกว่าไม่รู้", "เรื่องที่ไม่มีเอกสารรองรับ เช่น เงินเดือน", "unknown"),
    ("ปฏิเสธ", "คำขอที่พยายามเปลี่ยนคำสั่งของระบบ", "refused"),
)
# Turn headings and card titles are read by employees, so they are Thai:
# the English chips above the answer were the last piece of interface
# language a Thai-first reader had to decode mid-sentence.
BLOCKED_CHIP = "คำขอนี้ถูกปฏิเสธ"
FALLBACK_CHIP = "ยังไม่มีเอกสารรองรับคำถามนี้"
# A missing credential produces the same route but a different cause, so
# the chip must not tell the employee the evidence was thin when the
# answer service was simply not configured (remediation plan Finding 8).
SERVICE_UNAVAILABLE_CHIP = "บริการตอบคำถามยังไม่พร้อมใช้งาน"
ANSWERED_CHIP = "ตอบจากเอกสารอ้างอิง"
SOURCES_HEADER = "เอกสารอ้างอิง {count} ฉบับ"
USER_TURN_LABEL = "คุณถาม"
EVIDENCE_LABEL = "หลักฐาน"
# Words for the evidence meter, one per calibrated band. The band itself
# comes from _score_band, so the word can never disagree with the colour
# or with the route the graph chose.
EVIDENCE_STRENGTH_WORDS: dict[str, str] = {
    "green": "แข็งแรง",
    "amber": "พอใช้",
    "red": "อ่อน",
}
COPY_ANSWER_LABEL = "คัดลอก"
DOWNLOAD_ANSWER_LABEL = "บันทึก .md"
# Ways forward offered on a fallback card. Each one asks a question the
# corpus documents, so the employee's next click cannot fall back again;
# the button says the subject and its tooltip says the exact question.
NEXT_STEP_PROMPTS: tuple[tuple[str, str], ...] = (
    ("ถามเรื่องการเบิกค่าใช้จ่าย", SUGGESTED_QUESTIONS[0][1]),
    ("ถามเรื่องการลา", SUGGESTED_QUESTIONS[2][1]),
)
# Display ceiling for the similarity bars, from the revised design: a
# score of 0.5 fills the bar. It scales a picture and decides nothing --
# every routing cut-off still lives in config.py (invariant 5). The
# console's threshold strip uses the same ceiling so a bar in an answer
# and a marker in the console describe the same axis.
SCORE_BAR_CEILING = 0.5

# Console copy. The operator's page keeps English labels for the machine
# surfaces (routes, reason codes, node names) and Thai for the sentences
# that explain what a panel means.
CONSOLE_SEARCH_PLACEHOLDER = "ค้นหาคำถาม / reason code"
THRESHOLD_PANEL_TITLE = "คะแนน retrieval เทียบ threshold"
THRESHOLD_PANEL_NOTE = (
    "จุดคือคะแนนของ request จริงในเซสชันนี้ · เส้นคือ threshold จาก "
    "src/config.py ซึ่ง UI ไม่ได้นิยามเอง"
)
TRIAGE_PANEL_TITLE = "คำถามที่ยังตอบไม่ได้ (ไปปรับ KB)"
TRIAGE_PANEL_NOTE = (
    "จัดกลุ่มตาม reason code แล้วเรียงตามจำนวน เพื่อชี้ว่าควรเพิ่มเอกสาร"
    "เรื่องใดก่อน"
)
TRACE_PANEL_NOTE = (
    "ทุกค่าอ่านจาก PipelineState ของ request นั้น ไม่มีการคำนวณใหม่ในชั้น UI "
    "· เวลาต่อ node ยังไม่มีใน state จึงไม่แสดง"
)
EVAL_PAGE_NOTE = (
    "อ่านจาก eval/BASELINE.md และ eval/*.json ตามที่บันทึกไว้ "
    "ไม่ได้รันใหม่ในหน้านี้"
)
CALIBRATION_BADGE = "tuning only"
HELDOUT_BADGE = "reporting only"

# Evaluation artefacts live beside this module; the console reads them, it
# never runs an evaluation itself.
EVAL_DIR = Path(__file__).resolve().parent / "eval"
# Shapes inside a BASELINE.md measured-results block. Group headers appear
# as "Held-out (14 cases), strict exit 1:" and metric lines as an indented
# name followed by "6/7", with or without a colon and with an optional
# trailing remark -- both spellings occur across the file's snapshots.
# Every quantifier is bounded so a long prose line cannot backtrack here.
_EVAL_GROUP_LINE = re.compile(
    r"^(?P<name>[A-Za-z][^:(]{0,40})"
    r"(?: \((?P<cases>\d{1,4}) cases\))?"
    r"(?:(?:,| --)\s?[^:]{0,60})?:\s*$"
)
_EVAL_METRIC_LINE = re.compile(
    r"^ {2,8}(?P<name>\S.{0,58}?)(?::\s{1,40}|\s{2,40})"
    r"(?P<passed>\d{1,5})/(?P<total>\d{1,5})"
    r"(?:\s{1,8}(?P<note>\S.{0,60}?))?\s*$"
)
# An indented line inside a group that carries no fraction is a remark
# about the group, e.g. "every other metric unchanged from Phase 8".
_EVAL_REMARK_LINE = re.compile(r"^ {2,8}(?P<text>\S.{0,120}?)\s*$")

# Band colour for the console's score dots, matching the answer-side bars.
_BAND_COLOURS: dict[str, str] = {
    "green": "#00875A",
    "amber": "#B76E00",
    "red": "#DE350B",
}
# Streamlit's own badge colours, mapped onto the four display routes so the
# request list uses native markdown instead of hand-built chips.
_ROUTE_BADGE_COLOURS: dict[str, str] = {
    "direct": "green",
    "rewrite": "blue",
    "fallback": "orange",
    "blocked": "red",
}
NEW_SESSION_LABEL = "New Session"

# Detail line for a query the injection screen let through.
GUARDRAIL_PASS_DETAIL = "pass · ไม่พบ pattern injection"

# Reason codes the report node itself produces. Past them the request
# never reached the citation validator, so the trace must not claim it
# did; every other post-evidence reason comes from the validator.
_REPORTER_FAILURE_REASONS = frozenset(
    {
        ReasonCode.LLM_NOT_CONFIGURED.value,
        ReasonCode.REPORTER_FAILURE.value,
    }
)
MISSING_KEY_WARNING = (
    "ยังไม่ได้ตั้งค่า OPENAI_API_KEY: คำถามที่ต้องเรียก LLM "
    "จะจบด้วยเหตุผล llm_not_configured และแสดงข้อความว่าบริการยังไม่พร้อม "
    "(เส้นทาง blocked และ out-of-domain ทำงานได้ปกติโดยไม่ต้องใช้คีย์)"
)
# Under the composer, in the reader's own language and naming who to check
# with: a generic English caveat is skipped by the people it is meant for.
INPUT_DISCLAIMER = (
    "คำตอบอ้างอิงเอกสารภายในเท่านั้น "
    "โปรดตรวจสอบข้อมูลสำคัญกับ HR/Finance ก่อนใช้ตัดสินใจ"
)
# Sits under the source list, next to the bars it explains. It keeps
# invariant 6 in the employee's own language: the bar measures how well
# the evidence matched, never how likely the answer is to be right. The
# English name of the measure stays in the tooltip (SCORE_LABEL).
SIMILARITY_FOOTNOTE = (
    "แถบด้านขวาคือความคล้ายของข้อความ (retrieval similarity) "
    "ใช้บอกว่าเจอหลักฐานตรงแค่ไหน ไม่ใช่ความน่าจะเป็นที่คำตอบถูก"
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
.araya-source-id,
.araya-cite,
.araya-cite-badge,
.araya-cite-meta,
.araya-cite-value,
.araya-request-id,
.araya-runtime,
.araya-score--green, .araya-score--amber, .araya-score--red,
table.araya-table td,
.araya-mono-face *:not(.material-symbols-outlined),
.araya-runtime *:not(.material-symbols-outlined),
table.araya-table td *:not(.material-symbols-outlined) {
    font-family: 'JetBrains Mono', 'Noto Sans Thai', monospace !important;
}
#MainMenu, footer { visibility: hidden; }

/* The rail lockup is a mark plus a wordmark on one line: at 30px the mark
   reads as an app icon, and the wordmark no longer competes with the rail's
   own action for the first line of attention. */
.araya-brand { display: flex; gap: 10px; align-items: center; padding: 0 8px 18px; }
.araya-brand-mark { display: block; flex: 0 0 30px; }
.araya-brand-name {
    font-size: 16px; font-weight: 700; letter-spacing: -0.01em; color: #191c1e;
}
.araya-brand-name .araya-brand-accent { color: #0052CC; }

/* Scoped to the markdown container on purpose: Streamlit styles every <p>
   inside it at 1rem with specificity (0,1,1), which silently flattened the
   headline token to body size when these rules were plain classes. */
/* Console page title. The section heading below it is the sub-level, and
   both are scoped to the markdown container for the same reason: Streamlit
   styles every <p> inside it at 1rem with specificity (0,1,1). */
[data-testid="stMarkdownContainer"] p.araya-headline {
    font-size: 26px; line-height: 34px; font-weight: 700;
    letter-spacing: -0.01em; color: #191c1e; margin: 0;
}
[data-testid="stMarkdownContainer"] p.araya-section {
    font-size: 22px; line-height: 28px; font-weight: 600; color: #191c1e;
    margin: 0;
}

/* A turn is a right-aligned label above the question. The label carries the
   clock the transcript used to state once at the top, so a long thread stays
   locatable without a date divider between the days. */
.araya-turn {
    display: flex; flex-direction: column; align-items: flex-end; gap: 4px;
}
.araya-turn-label {
    font-size: 11px; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; color: #9aa1ae;
}
.araya-bubble-user {
    background: #EEF2FF; color: #001848;
    border-radius: 14px 14px 4px 14px; padding: 14px 18px;
    max-width: 78%; width: fit-content; font-size: 17px; line-height: 30px;
}
/* A degraded turn is an explanation with a way forward, not a warning box:
   the left rule carries the colour and the surface stays close to paper, so
   a fallback cannot read louder than the answers around it. The card itself
   is a native container (see the st-key rules below) because the ways
   forward inside it are real buttons. */
.araya-notice { display: flex; gap: 14px; align-items: flex-start; }
.araya-notice .araya-notice-icon { font-size: 22px; }
.araya-notice--fallback .araya-notice-icon { color: #B76E00; }
.araya-notice--blocked .araya-notice-icon { color: #BA1A1A; }
.araya-notice-title { font-size: 15px; font-weight: 700; }
.araya-notice--fallback .araya-notice-title { color: #7a4b00; }
.araya-notice--blocked .araya-notice-title { color: #93000a; }
.araya-notice-text {
    white-space: pre-wrap; font-size: 16px; line-height: 28px;
    color: #42526E; margin-top: 6px;
}

.araya-chip {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 13px; font-weight: 700;
}
.araya-chip .material-symbols-outlined { font-size: 18px; }
.araya-chip--ok { color: #00875A; }
/* Evidence meter: three bars and one word, both driven by the calibrated
   band, so the picture cannot disagree with the score printed beside it. */
.araya-evidence {
    display: inline-flex; align-items: center; gap: 6px; font-size: 12px;
    color: #5A5D6B; background: #F5F6F8; border-radius: 9999px;
    padding: 4px 10px;
}
.araya-evidence-bars {
    display: inline-flex; gap: 2px; align-items: flex-end;
}
.araya-evidence-bars span {
    width: 4px; border-radius: 1px; background: #EDEEF0;
}
.araya-evidence-bars span:nth-child(1) { height: 8px; }
.araya-evidence-bars span:nth-child(2) { height: 11px; }
.araya-evidence-bars span:nth-child(3) { height: 14px; }
.araya-evidence-word { font-weight: 600; }
.araya-evidence--green .araya-evidence-bars span.is-on { background: #00875A; }
.araya-evidence--green .araya-evidence-word { color: #00714b; }
.araya-evidence--amber .araya-evidence-bars span.is-on { background: #B76E00; }
.araya-evidence--amber .araya-evidence-word { color: #7a4b00; }
.araya-evidence--red .araya-evidence-bars span.is-on { background: #DE350B; }
.araya-evidence--red .araya-evidence-word { color: #93000a; }
/* body-lg for answers: 17/32 gives Thai marks room across a paragraph long
   enough to be read as a document rather than as a chat message. */
.araya-answer {
    white-space: pre-wrap; font-size: 17px; line-height: 32px; color: #191c1e;
}
/* Inline citation. The number points at the source list under the answer and
   the id is the auditable half; both are rendered from validated citations,
   never from anything the model wrote. */
.araya-cite {
    font-size: 12px; font-weight: 500; color: #0052CC; background: #EEF2FF;
    border-radius: 6px; padding: 2px 7px; vertical-align: 2px;
    white-space: nowrap;
}

.araya-mono { font-size: 13px; }
.araya-score--green { color: #00875A; }
.araya-score--amber { color: #b97900; }
.araya-score--red { color: #DE350B; }

.araya-request-id { font-size: 12px; color: #8e909c; }
.araya-footnote {
    font-size: 12px; line-height: 20px; color: #8e909c; margin-top: 10px;
}

/* Source list under an answer: one numbered row per citation, so a chip in
   the text can be traced by eye, with the similarity bars stacked in one
   column where they can be compared against each other. */
.araya-sources {
    border-top: 1px solid #F0F1F4; margin-top: 20px; padding-top: 14px;
}
.araya-sources-head {
    font-size: 12px; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; color: #737685; margin-bottom: 10px;
}
.araya-cite-card {
    border: 1px solid #E4E6EB; border-radius: 10px; overflow: hidden;
    margin-bottom: 8px;
}
.araya-cite-card summary, .araya-cite-row {
    display: flex; align-items: center; gap: 12px; padding: 12px 14px;
}
.araya-cite-card summary { cursor: pointer; list-style: none; }
.araya-cite-card summary::-webkit-details-marker { display: none; }
.araya-cite-badge {
    width: 24px; height: 24px; border-radius: 6px; background: #EEF2FF;
    color: #0052CC; display: inline-flex; align-items: center;
    justify-content: center; font-size: 12px; font-weight: 700; flex: 0 0 24px;
}
.araya-cite-head { flex: 1; min-width: 0; }
.araya-cite-title {
    display: block; font-size: 15px; font-weight: 600; color: #191c1e;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.araya-cite-meta { display: block; font-size: 12px; color: #737685; }
.araya-cite-score { display: inline-flex; align-items: center; gap: 8px; }
.araya-cite-bar {
    width: 64px; height: 6px; border-radius: 9999px; background: #EDEEF0;
    display: inline-block; position: relative; overflow: hidden; flex: 0 0 64px;
}
.araya-cite-fill {
    position: absolute; left: 0; top: 0; bottom: 0; border-radius: 9999px;
}
.araya-cite-fill--green { background: #00875A; }
.araya-cite-fill--amber { background: #B76E00; }
.araya-cite-fill--red { background: #DE350B; }
.araya-cite-value { font-size: 12px; color: #5A5D6B; }
.araya-cite-body {
    background: #FAFBFC; border-top: 1px solid #E4E6EB; padding: 12px 14px;
    font-size: 15px; line-height: 28px; color: #42526E; white-space: pre-wrap;
}
.araya-source-id { font-size: 13px; color: #0052CC; font-weight: 700; }
.araya-source-title {
    font-size: 12px; font-weight: 700; letter-spacing: 0.05em; color: #434654;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.araya-kpi {
    background: #FFFFFF; border: 1px solid #DFE1E6; border-radius: 8px;
    padding: 16px; height: 100%;
}
.araya-kpi-head {
    display: flex; align-items: center; gap: 12px; margin-bottom: 8px;
}
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
/* Long Thai queries must not turn one row into three lines; the untruncated
   text stays reachable through the cell's title attribute. */
.araya-cell-clip {
    max-width: 260px; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap;
}
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
/* The design system docks a 260px rail; Streamlit ships 300px by default and
   sets it inline, so the token only lands with !important. The narrower rail
   buys the reading column back the width Thai paragraphs need. */
[data-testid="stSidebar"] {
    width: 260px !important; min-width: 260px !important;
    background: #FAFAFB; border-right: 1px solid #E4E6EB;
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
/* The rail's own action is an outline button: a filled blue here competed
   with the send affordance for the eye's first stop on an empty screen. */
.st-key-araya_new_session [data-testid^="stBaseButton-"] {
    background: #FFFFFF; border: 1px solid #C9CCD4; border-radius: 9px;
    padding: 9px 12px; color: #191c1e; font-size: 14px; font-weight: 600;
    box-shadow: 0 1px 2px rgba(25,28,30,0.04);
}
.st-key-araya_new_session [data-testid^="stBaseButton-"]:hover {
    border-color: #0052CC; color: #191c1e;
}
.st-key-araya_new_session [data-testid="stIconMaterial"] {
    color: #0052CC; font-size: 18px;
}
/* The rail's standing caveat belongs at the foot of the rail, under its own
   rule, not wedged between the actions. Streamlit stacks sidebar blocks in
   stSidebarUserContent, so the column has to own the full height before the
   note can be pushed down. */
[data-testid="stSidebarUserContent"] {
    display: flex; flex-direction: column; min-height: 100%;
}
.araya-rail-note {
    margin-top: auto; border-top: 1px solid #EDEEF0; padding: 12px 10px 0;
    font-size: 11.5px; line-height: 18px; color: #8e909c;
}

/* The assistant bar carries the view switch, so it runs taller than the
   console header and takes the lighter rule that matches the rail. */
.st-key-araya_topbar { border-bottom-color: #E4E6EB; padding: 14px 0 12px; }
.st-key-araya_topbar .araya-session-id {
    display: flex; justify-content: flex-end; font-size: 12px; color: #737685;
}
/* Segmented control as the mockup's pill switch. Streamlit renders it as a
   button group; only the group wrapper is a stable hook, so the tray is
   styled here and the selected item keeps the widget's own affordance. */
.st-key-araya_view_switch [data-testid="stButtonGroup"] {
    background: #F1F2F5; border-radius: 9999px; padding: 3px; gap: 2px;
}

/* Opening screen. The type scale is the reading scale of an answer, not a
   marketing hero: one headline, one line of provenance, then the starter
   questions and the three outcomes the assistant can produce. */
[data-testid="stMarkdownContainer"] p.araya-hero-title {
    font-size: 30px; line-height: 44px; font-weight: 700;
    letter-spacing: -0.01em; color: #191c1e; margin: 64px 0 0 0;
}
[data-testid="stMarkdownContainer"] p.araya-hero-sub {
    font-size: 17px; line-height: 30px; color: #5A5D6B; margin: 10px 0 0 0;
}
/* Each starter question is one native button styled as a card so the whole
   card is the click target. Streamlit renders the label as a paragraph, and
   its leading <strong> carries the department tag. */
[class*="st-key-araya_suggest_"] [data-testid^="stBaseButton-"] {
    width: 100%; text-align: left; background: #FFFFFF;
    border: 1px solid #E4E6EB; border-radius: 12px; padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(25,28,30,0.04);
}
[class*="st-key-araya_suggest_"] [data-testid^="stBaseButton-"]:hover {
    border-color: #0052CC;
}
[class*="st-key-araya_suggest_"] [data-testid^="stBaseButton-"] > div {
    justify-content: flex-start; width: 100%;
}
[class*="st-key-araya_suggest_"] [data-testid="stMarkdownContainer"] p {
    font-size: 16px; line-height: 26px; color: #191c1e; margin: 0;
    white-space: normal;
}
[class*="st-key-araya_suggest_"] [data-testid="stMarkdownContainer"] strong {
    display: block; font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #0052CC; margin-bottom: 6px;
}
/* Naming the three outcomes before the first question is what keeps a
   fallback from reading as a failure: colour marks the band, the left rule
   keeps it a label rather than a warning box. */
.araya-scope-row { display: flex; gap: 12px; margin-top: 28px; }
.araya-scope { flex: 1; border-left: 3px solid; padding: 2px 0 2px 12px; }
.araya-scope-label {
    font-size: 12px; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase;
}
.araya-scope-text {
    font-size: 14px; line-height: 24px; color: #5A5D6B; margin-top: 2px;
}
.araya-scope--ok { border-color: #00875A; }
.araya-scope--ok .araya-scope-label { color: #00875A; }
.araya-scope--unknown { border-color: #B76E00; }
.araya-scope--unknown .araya-scope-label { color: #B76E00; }
.araya-scope--refused { border-color: #BA1A1A; }
.araya-scope--refused .araya-scope-label { color: #BA1A1A; }
@media (max-width: 640px) {
    .araya-scope-row { flex-direction: column; }
}

/* The answer card is a native container, not a markdown block, because the
   copy and save controls in its head have to be real widgets. The container
   draws the card, its single horizontal block is the head, and the markdown
   that follows is the body. */
[class*="st-key-araya_answer_"] {
    border: 1px solid #E4E6EB; border-radius: 16px; background: #FFFFFF;
    box-shadow: 0 2px 10px rgba(25,28,30,0.05); overflow: hidden;
}
[class*="st-key-araya_answer_"] [data-testid="stVerticalBlock"] { gap: 0; }
[class*="st-key-araya_answer_"] [data-testid="stHorizontalBlock"] {
    padding: 14px 20px; border-bottom: 1px solid #F0F1F4;
}
.araya-card-body { padding: 18px 20px 20px; }
/* Head actions read at the card's own scale: they belong beside the verdict,
   never above the answer, so they stay outlined text buttons. */
[class*="st-key-araya_answer_"] [data-testid^="stBaseButton-"] {
    background: #FFFFFF; border: 1px solid #E4E6EB; border-radius: 8px;
    padding: 5px 10px; color: #5A5D6B; font-size: 13px; font-weight: 400;
}
[class*="st-key-araya_answer_"] [data-testid^="stBaseButton-"]:hover {
    border-color: #0052CC; color: #0052CC;
}
[class*="st-key-araya_answer_"] [data-testid="stIconMaterial"] {
    font-size: 16px;
}
/* Fallback and blocked turns are containers for the same reason: the ways
   forward inside them are buttons. The tone lives in the key so the two
   surfaces stay one rule apart. */
[class*="st-key-araya_notice_fallback_"],
[class*="st-key-araya_notice_blocked_"] {
    border: 1px solid #E4E6EB; border-radius: 12px; padding: 16px 18px;
}
[class*="st-key-araya_notice_fallback_"] {
    border-left: 4px solid #B76E00; background: #FFFCF5;
}
[class*="st-key-araya_notice_blocked_"] {
    border-left: 4px solid #BA1A1A; background: #FFF7F6;
}
/* Ways forward on a fallback card: quiet blue outlines, sized to sit under
   the explanation rather than to compete with the composer. */
[class*="st-key-araya_next_"] [data-testid^="stBaseButton-"] {
    background: #FFFFFF; border: 1px solid #C7D6F5; border-radius: 8px;
    padding: 6px 12px; color: #0052CC; font-size: 14px; font-weight: 400;
}
[class*="st-key-araya_next_"] [data-testid^="stBaseButton-"]:hover {
    border-color: #0052CC; background: #EEF2FF;
}
</style>
"""

# Per-view layout tokens. The two mockups use different measures -- a 760px
# reading column for the chat, the 1200px content grid for the console -- so
# the width is injected by whichever view is rendering instead of once for
# the whole app. Both add the 24px gutter on top of the measure. The chat
# measure narrows from 800px to the 760px the revised design specifies for
# its 17/30 Thai body, and the rail gives back the 20px it needs.
_CHAT_LAYOUT_CSS = f"""
<style>
[data-testid="stMainBlockContainer"] {{
    max-width: 808px; padding-left: 24px; padding-right: 24px;
}}
[data-testid="stBottomBlockContainer"] {{
    max-width: 808px; padding-left: 24px; padding-right: 24px;
}}
/* The assistant reads as a document surface, so its page is white while the
   rail keeps the app's neutral. The console keeps the base theme background
   set in .streamlit/config.toml. */
[data-testid="stMain"], [data-testid="stBottom"] {{ background: #FFFFFF; }}
/* The composer disclaimer sits under the input in the mockup, not at the top
   of the message flow where a Streamlit caption lands. Streamlit already
   reserves 56px of padding below the composer, so it is rendered there as
   generated content instead of as a stray element in the transcript. */
[data-testid="stBottomBlockContainer"]::after {{
    /* ensure_ascii=False is required: the default escapes every Thai
       character to a JSON unicode escape, which CSS then decodes as a
       literal letter followed by hex digits instead of as text. */
    content: {json.dumps(INPUT_DISCLAIMER, ensure_ascii=False)};
    display: block; text-align: center; margin-top: 10px;
    font-size: 12px; color: #8e909c;
}}
/* Composer shell from the mockup: a white field on the white page, held by
   its border and a low shadow rather than by a filled grey. The design's
   14/18 padding is applied as a small outer inset because the widget's own
   textarea and send button already carry most of that space; the value is
   the one part of this block not verified against a running browser. */
[data-testid="stChatInput"] {{
    background: #FFFFFF; border: 1px solid #C9CCD4; border-radius: 14px;
    padding: 4px 4px 4px 6px; box-shadow: 0 2px 10px rgba(25,28,30,0.06);
}}
/* Same 17/30 as the answer body: the question and its answer are read in
   one type size, and the line box clears Thai tone marks and vowel signs. */
[data-testid="stChatInputTextArea"] {{
    font-size: 17px; line-height: 30px;
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
/* The console rail is narrower than the assistant's and sits on white: it
   is a section switcher, not a place to read. */
[data-testid="stSidebar"] {
    width: 240px !important; min-width: 240px !important; background: #FFFFFF;
}
[data-testid="stMain"] { background: #F7F8FA; }
/* Rail entries: the active one is tinted rather than filled, so a selected
   section does not read as a primary action. */
[class*="st-key-console_nav_"] [data-testid="stBaseButton-primary"] {
    background: #EEF2FF; color: #003d9b; border-color: #EEF2FF;
    font-weight: 600;
}
[class*="st-key-console_nav_"] [data-testid="stBaseButton-tertiary"] {
    color: #5A5D6B;
}
[class*="st-key-console_nav_"] [data-testid^="stBaseButton-"] {
    border-radius: 8px; padding: 9px 12px; font-size: 14px;
}

/* Stat card: the console's own number block, wider than the KPI strip and
   carrying a share bar so three counts read as parts of one total. */
.araya-stat {
    background: #FFFFFF; border: 1px solid #E4E6EB; border-radius: 12px;
    padding: 16px; height: 100%;
}
.araya-stat-label { font-size: 12px; font-weight: 600; color: #737685; }
.araya-stat-body {
    display: flex; align-items: baseline; gap: 6px; margin-top: 6px;
}
.araya-stat-value { font-size: 28px; font-weight: 700; color: #191c1e; }
.araya-stat-unit { font-size: 12px; color: #8e909c; }
.araya-stat-track {
    height: 6px; border-radius: 9999px; background: #EDEEF0;
    margin-top: 10px; overflow: hidden;
}
.araya-stat-fill { display: block; height: 6px; }
.araya-stat-fill--green { background: #00875A; }
.araya-stat-fill--amber { background: #B76E00; }
.araya-stat-fill--red { background: #BA1A1A; }
.araya-stat-note { font-size: 12px; line-height: 18px; color: #8e909c; margin-top: 10px; }

.araya-panel {
    background: #FFFFFF; border: 1px solid #E4E6EB; border-radius: 12px;
    padding: 18px; height: 100%;
}
.araya-panel-head {
    display: flex; align-items: baseline; justify-content: space-between;
    gap: 12px;
}
.araya-panel-title { font-size: 15px; font-weight: 700; color: #191c1e; }
.araya-panel-meta { font-size: 12px; color: #8e909c; }
.araya-panel-note {
    font-size: 12px; line-height: 20px; color: #8e909c; margin-top: 12px;
}

/* Threshold strip: the calibrated cut-offs drawn as one axis with this
   session's scores plotted on it, so "why did this fall back" is answered
   by position rather than by comparing two numbers in a table. */
.araya-axis { position: relative; height: 44px; margin-top: 22px; }
.araya-axis-track {
    position: absolute; left: 0; right: 0; top: 14px; height: 10px;
    border-radius: 9999px; overflow: hidden; display: flex;
}
.araya-axis-band--red { background: #FFDAD6; }
.araya-axis-band--amber { background: #FFEBC7; }
.araya-axis-band--yellow { background: #FFF3D6; }
.araya-axis-band--green { background: #C8F2DA; }
.araya-axis-mark { position: absolute; top: 0; bottom: -18px; width: 1px; }
.araya-axis-dot {
    position: absolute; top: 8px; width: 22px; height: 22px;
    border-radius: 9999px; border: 3px solid #FFFFFF;
    box-shadow: 0 1px 4px rgba(0,0,0,.25); transform: translateX(-50%);
}
.araya-axis-dot--green { background: #00875A; }
.araya-axis-dot--amber { background: #B76E00; }
.araya-axis-dot--red { background: #DE350B; }
.araya-axis-labels {
    position: relative; height: 36px; font-size: 11px; color: #5A5D6B;
}
.araya-axis-label {
    position: absolute; transform: translateX(-50%); text-align: center;
    line-height: 14px; white-space: nowrap;
}
.araya-axis-label span { color: #8e909c; }
.araya-axis-legend {
    display: flex; flex-wrap: wrap; gap: 16px; margin-top: 10px;
    font-size: 12px; color: #5A5D6B;
}
.araya-axis-legend i {
    display: inline-block; width: 8px; height: 8px; border-radius: 9999px;
    margin-right: 6px; font-style: normal;
}

/* Triage rows: reason code first, count first in the eye's path, because
   the panel exists to say which document to write next. */
.araya-triage {
    display: flex; align-items: center; gap: 12px; border: 1px solid #F0F1F4;
    border-radius: 10px; padding: 10px 12px; margin-top: 10px;
}
.araya-triage-count {
    font-size: 18px; font-weight: 700; width: 24px; flex: 0 0 24px;
}
.araya-triage-count--amber { color: #B76E00; }
.araya-triage-count--red { color: #BA1A1A; }
.araya-triage-body { flex: 1; min-width: 0; }
.araya-triage-query {
    display: block; font-size: 14px; color: #191c1e; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
}
.araya-triage-meta { display: block; font-size: 11px; color: #8e909c; }
.araya-triage-tag {
    font-size: 12px; border-radius: 6px; padding: 3px 8px;
    border: 1px solid; white-space: nowrap;
}
.araya-triage-tag--scope { color: #0052CC; border-color: #C7D6F5; }
.araya-triage-tag--attack { color: #BA1A1A; border-color: #F5C7C7; }

/* Request list: one button per row so a Thai question can wrap instead of
   being clipped into a table cell. */
[class*="st-key-araya_req_"] [data-testid^="stBaseButton-"] {
    width: 100%; text-align: left; background: #FFFFFF;
    border: 1px solid #E4E6EB; border-radius: 10px; padding: 12px 14px;
    color: #191c1e; font-weight: 400;
}
[class*="st-key-araya_req_"] [data-testid^="stBaseButton-"] > div {
    justify-content: flex-start; width: 100%;
}
[class*="st-key-araya_req_"] [data-testid="stBaseButton-primary"] {
    background: #EEF2FF; border-color: #0052CC;
}
[class*="st-key-araya_req_"] [data-testid="stMarkdownContainer"] p {
    font-size: 14px; line-height: 22px; margin: 0; white-space: normal;
}
[class*="st-key-araya_req_"] [data-testid="stMarkdownContainer"] code {
    font-size: 11px; color: #737685; background: none; padding: 0;
}

/* Node trace: one row per graph node, in graph order. A node the request
   never reached stays visible but dimmed -- the shape of the pipeline is
   part of what the panel explains. */
.araya-trace-row {
    display: flex; align-items: center; gap: 12px; padding: 9px 0;
    border-top: 1px solid #F0F1F4;
}
.araya-trace-row--skipped { opacity: 0.5; }
.araya-trace-badge {
    width: 22px; height: 22px; border-radius: 9999px; flex: 0 0 22px;
    display: inline-flex; align-items: center; justify-content: center;
}
.araya-trace-badge .material-symbols-outlined { font-size: 14px; }
.araya-trace-badge--done { background: #C8F2DA; color: #00714b; }
.araya-trace-badge--llm { background: #DAE2FF; color: #001848; }
.araya-trace-badge--stop { background: #FFDAD6; color: #ba1a1a; }
.araya-trace-badge--skip { background: #EDEEF0; color: #737685; }
.araya-trace-node {
    font-size: 12.5px; color: #191c1e; width: 150px; flex: 0 0 150px;
}
.araya-trace-detail {
    flex: 1; min-width: 0; font-size: 13px; line-height: 20px; color: #5A5D6B;
}
.araya-trace-aside {
    margin: 2px 0 6px 34px; border-left: 2px solid #DAE2FF;
    padding: 6px 0 6px 14px; display: flex; flex-direction: column; gap: 4px;
    font-size: 13px; line-height: 22px; color: #42526E;
}
.araya-trace-aside span:nth-child(2) { color: #0052CC; }
.araya-trace-aside span:nth-child(3) { font-size: 12px; color: #8e909c; }

/* Evaluation: one row per metric, bar and fraction side by side, so a
   4/4 and a 1/2 cannot be skimmed as the same result. */
.araya-metric {
    display: flex; align-items: center; gap: 12px; padding: 9px 0;
    border-bottom: 1px solid #F0F1F4;
}
.araya-metric:last-child { border-bottom: none; }
.araya-metric-name { flex: 1; min-width: 0; font-size: 14px; color: #42526E; }
.araya-metric-bar {
    width: 120px; flex: 0 0 120px; height: 6px; border-radius: 9999px;
    background: #EDEEF0; overflow: hidden;
}
.araya-metric-fill { display: block; height: 6px; }
.araya-metric-value {
    font-size: 13px; width: 56px; flex: 0 0 56px; text-align: right;
    color: #191c1e;
}
.araya-badge {
    font-size: 11px; font-weight: 600; border-radius: 9999px; padding: 3px 9px;
}
.araya-badge--tuning { color: #7a4b00; background: #FFF3D6; }
.araya-badge--reporting { color: #003d9b; background: #DAE2FF; }
</style>
"""

# Dark surfaces for the console only, injected over the light tokens when the
# operator asks for them. Streamlit's own widgets keep following the base
# theme in .streamlit/config.toml -- a pure-Streamlit build cannot swap that
# at runtime -- so this covers the panels the console draws itself and leaves
# inputs and buttons on their light chrome.
_CONSOLE_DARK_CSS = """
<style>
[data-testid="stMain"] { background: #0E1116; }
[data-testid="stSidebar"] { background: #141922; border-right-color: #262C36; }
[data-testid="stMain"] p, [data-testid="stMain"] span,
[data-testid="stMain"] div, [data-testid="stMain"] li { color: #C3C9D4; }
[data-testid="stMarkdownContainer"] p.araya-headline,
[data-testid="stMarkdownContainer"] p.araya-section,
.araya-panel-title, .araya-stat-value, .araya-metric-value {
    color: #E6E8EC;
}
.araya-panel-meta, .araya-panel-note, .araya-stat-label, .araya-stat-note,
.araya-triage-meta, .araya-axis-labels, .araya-trace-detail {
    color: #7B8494;
}
.araya-stat, .araya-panel {
    background: #161A21; border-color: #262C36;
}
.araya-stat-track, .araya-metric-bar { background: #222835; }
.araya-triage, .araya-metric, .araya-trace-row {
    border-color: #1F2530;
}
.araya-trace-node, .araya-triage-query { color: #C3C9D4; }
table.araya-table { background: #161A21; border-color: #262C36; }
table.araya-table th {
    background: #1A2029; color: #9AA1AE; border-bottom-color: #262C36;
}
table.araya-table td { color: #C3C9D4; border-bottom-color: #1F2530; }
table.araya-table tbody tr:hover { background: #1A2029; }
.araya-details { background: #161A21; border-color: #262C36; }
.araya-details > summary { color: #9AA1AE; }
.araya-runtime {
    background: #0E1116; border-color: #262C36; color: #9AA1AE;
}
.araya-doc-card { background: #161A21; border-color: #262C36; }
.araya-kpi { background: #161A21; border-color: #262C36; }
.araya-kpi-value { color: #E6E8EC; }
.araya-badge--tuning { color: #FFD37A; background: #3A2E12; }
.araya-badge--reporting { color: #8FBCFF; background: #16283F; }
</style>
"""


@st.cache_resource
def _corpus() -> list[Document]:
    """Load the validated corpus once per Streamlit process."""
    return load_documents()


@st.cache_resource
def _graph_and_retriever() -> tuple[
    CompiledStateGraph, LocalTfidfRetriever, datetime
]:
    """Build the TF-IDF index and compile the graph once per process.

    Streamlit reruns the script on every interaction; caching keeps the
    expensive index build and graph compilation out of that loop. The
    build timestamp is real and feeds the Ops "last index build" stat.
    """
    retriever = LocalTfidfRetriever(_corpus())
    built_at = datetime.now().astimezone()
    return (
        build_graph(retriever=retriever, documents=_corpus()),
        retriever,
        built_at,
    )


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


def _render_user_bubble(query: str, timestamp: str) -> None:
    """Render one employee turn: its clock label above the question."""
    st.markdown(
        '<div class="araya-turn">'
        f'<span class="araya-turn-label">{USER_TURN_LABEL} · '
        f"{_turn_clock(timestamp)}</span>"
        f'<div class="araya-bubble-user">{html.escape(query)}</div></div>',
        unsafe_allow_html=True,
    )


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
                    _ask(question)
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
        service_unavailable = (
            state.get("fallback_reason") == ReasonCode.LLM_NOT_CONFIGURED
        )
        _render_notice_card(
            record,
            tone="fallback",
            icon="cloud_off" if service_unavailable else "help_center",
            title=(
                SERVICE_UNAVAILABLE_CHIP
                if service_unavailable
                else FALLBACK_CHIP
            ),
            text=fixed_text or "",
            # An unconfigured answer service is not a gap in the corpus,
            # so pointing the employee at other topics would misdescribe
            # the failure: the ways forward stay for evidence fallbacks.
            next_steps=not service_unavailable,
            note=AUDIT_ONLY_NOTE,
        )
        return

    if route != "answered":
        return

    _render_answer_card(record)


def _ask(query: str) -> None:
    """Run one question through the pipeline and store its outcome.

    Shared by the composer and the starter questions so that both enter
    the graph by the same door; the view adds nothing to the query but
    the surrounding whitespace it strips.
    """
    with st.spinner("กำลังค้นหาจากฐานความรู้..."):
        state, latency = _invoke_graph(query)
    _record_request(query, state, latency)
    st.rerun()


def _scope_strips_html() -> str:
    """Build the three-outcome legend shown on the opening screen."""
    strips = "".join(
        f'<div class="araya-scope araya-scope--{tone}">'
        f'<div class="araya-scope-label">{html.escape(label)}</div>'
        f'<div class="araya-scope-text">{html.escape(text)}</div></div>'
        for label, text, tone in SCOPE_STRIPS
    )
    return f'<div class="araya-scope-row">{strips}</div>'


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
                    _ask(question)
    st.markdown(_scope_strips_html(), unsafe_allow_html=True)


def _render_employee_view() -> None:
    """Chat-style employee flow: ask, then read a grounded, cited answer."""
    st.markdown(_CHAT_LAYOUT_CSS, unsafe_allow_html=True)
    history = st.session_state.get("history", [])
    if not history:
        _render_empty_state()
    # Each turn states its own clock, so the transcript no longer opens
    # with a single date divider that scrolls away after two questions.
    for record in history:
        _render_user_bubble(record["query"], record["timestamp"])
        _render_agent_card(record)

    # The composer enforces the same length ceiling the guardrail applies,
    # so an over-length question is stopped at the keyboard instead of
    # being sent and rejected.
    query = st.chat_input(CHAT_PLACEHOLDER, max_chars=config.MAX_QUERY_CHARS)
    if query and query.strip():
        _ask(query.strip())


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


def _triage_groups() -> list[tuple[str, int, str, str]]:
    """Group this session's degraded requests by reason code.

    Returns:
        ``(reason, count, newest query, tag)`` per reason, most frequent
        first. The tag separates an attack from a gap in the corpus,
        because only one of the two is a reason to write a document.

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
        (
            reason,
            len(rows),
            rows[-1]["query"],
            "attack" if rows[-1]["state"].get("guardrail_reason") else "scope",
        )
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
    reasons = _triage_groups()
    fallback_reason = next(
        (reason for reason, _, _, tag in reasons if tag == "scope"), "–"
    )
    blocked_reason = next(
        (reason for reason, _, _, tag in reasons if tag == "attack"), "–"
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
            note=(
                "จำนวน LLM call ต่อ request ยังไม่มีใน PipelineState "
                "จึงยังรายงานไม่ได้"
            ),
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
            '<div class="araya-triage">'
            f'<span class="araya-triage-count araya-mono-face '
            f'araya-triage-count--{"red" if tag == "attack" else "amber"}">'
            f"{count}</span>"
            '<span class="araya-triage-body">'
            f'<span class="araya-triage-query">{html.escape(query)}</span>'
            f'<span class="araya-triage-meta araya-mono-face">'
            f"{html.escape(reason)}</span></span>"
            f'<span class="araya-triage-tag araya-triage-tag--{tag}">'
            f'{"attack" if tag == "attack" else "out of scope"}</span></div>'
            for reason, count, query, tag in reasons
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
    rows.append(
        (
            "report",
            "llm",
            "smart_toy",
            "LLM call · grounded answer"
            if route == "answered"
            else f"LLM call · {state.get('fallback_reason')}",
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


def _metric_rows_html(metrics: list[tuple[str, int, int, str]]) -> str:
    """Render one group's metrics as name, proportion bar, and fraction.

    The bar is one neutral colour on purpose. Some of these rates are good
    at zero -- ``Invalid Candidate Leakage Rate: 0/11`` is the result the
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
    """Employee rail: brand, session reset, and the standing caveat.

    Mockup rail entries without a real data source (a list of past
    sessions, settings, support) stay omitted rather than being faked.
    The console link lives in the top bar with the view switch, so the
    rail holds one action and does not compete with it.
    """
    with st.sidebar:
        st.markdown(_brand_html(), unsafe_allow_html=True)
        if st.button(
            NEW_SESSION_LABEL,
            icon=":material/add:",
            key="araya_new_session",
            use_container_width=True,
        ):
            st.session_state.pop("history", None)
            st.session_state.pop("session_id", None)
            st.rerun()
        st.markdown(
            f'<div class="araya-rail-note">{VIEW_SEPARATION_NOTE}</div>',
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
        switch, identity, action = st.columns(
            [3, 3, 1], vertical_alignment="center"
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
            with st.popover(
                "Help", icon=":material/help:", use_container_width=True
            ):
                st.markdown(HELP_TEXT)
    # Navigating from inside the column would abandon the layout block
    # half-built, so the armed jump is taken once the bar is complete.
    if st.session_state.pop("araya_goto_console", False):
        st.switch_page(_PAGE_REFS["console"])


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
