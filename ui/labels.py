"""User-facing strings and display constants of the Streamlit UI.

Thai text is allowed here because these are user-facing message
constants (AGENTS.md section 6.1); keeping them in one module is what
lets the render functions stay free of inline copy, and lets a reviewer
read everything the app can say without opening a layout file.

Nothing here decides behaviour: the pipeline's own wording lives in
``src/fallback.py``, and the numbers in this file are display bounds
(how many log rows to show, where a score bar ends), never thresholds.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.fallback import ReasonCode

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

# Evaluation artefacts live at the project root, one level above this
# package; the console reads them, it never runs an evaluation itself.
EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
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
        ReasonCode.REQUEST_DEADLINE_EXCEEDED.value,
    }
)
# The subset of those in which no provider call was made at all. The
# trace labels the reporter row as a billed call, and an exhausted
# deadline or a missing credential is exactly the case where it was not.
_UNSPENT_REPORTER_REASONS = frozenset(
    {
        ReasonCode.LLM_NOT_CONFIGURED.value,
        ReasonCode.REQUEST_DEADLINE_EXCEEDED.value,
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
