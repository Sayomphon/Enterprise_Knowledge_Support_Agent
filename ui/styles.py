"""The design-system stylesheets injected into the two pages.

One definition of the "Enterprise Logic" visual language, split by the
surface that needs it: the base tokens every page loads, the assistant's
chat layout, and the console's layout with its optional dark sheet. They
are string constants rather than a CSS file because Streamlit injects
styles through ``st.markdown``, and because the composer disclaimer has
to be JSON-encoded into generated content.

Score colours are bound to the calibrated runtime thresholds by the
formatting module, not here: this file holds the palette, not the rule
that picks from it.
"""

from __future__ import annotations

import json

from ui.labels import INPUT_DISCLAIMER

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
/* Streamlit reserves a 60px band at the top of the page for its own
   toolbar. Both app bars are sticky at top 0, so they were sliding under
   that band and the switch lost its upper half behind it. The band is
   collapsed and its toolbar floated instead: the Deploy and menu buttons
   stay reachable in the corner, and the page starts where the app bar
   does. */
[data-testid="stHeader"] {
    height: 0 !important; min-height: 0 !important; background: transparent;
}
[data-testid="stToolbar"] {
    position: fixed !important; top: 6px; right: 12px;
    height: auto !important;
}
/* The rail starts level with the app bar rather than a band below it, so
   "New Session" and the view switch read as one row of controls. The rail
   has a 60px band of its own for the collapse control; it is collapsed the
   same way the page header is, and the control floats in the corner so it
   stays reachable. */
[data-testid="stSidebarHeader"] {
    height: 0 !important; min-height: 0 !important; padding: 0 !important;
}
[data-testid="stSidebarCollapseButton"] {
    position: absolute !important; top: 8px; right: 8px; z-index: 5;
}
[data-testid="stSidebarUserContent"] { padding-top: 12px; }
.araya-brand { padding-bottom: 10px !important; }

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
/* The rail's session list. Each entry is a button so the whole row is the
   target, but it reads as a list item: no fill until it is the session
   being read, and the question sits above its clock and count. */
.araya-rail-label {
    margin: 18px 4px 6px; font-size: 11px; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase; color: #8e909c;
}
[class*="st-key-araya_session_"] [data-testid^="stBaseButton-"] {
    background: transparent; border: 1px solid transparent;
    border-radius: 9px; padding: 8px 10px; margin-bottom: 2px;
}
[class*="st-key-araya_session_"] [data-testid^="stBaseButton-"]:hover {
    background: #F1F2F5;
}
[class*="st-key-araya_session_"] [data-testid="stBaseButton-primary"] {
    background: #EEF2FF; border-color: #DDE4FF;
}
[class*="st-key-araya_session_"] [data-testid="stMarkdownContainer"] p {
    font-size: 13px; line-height: 20px; color: #191c1e; margin: 0;
    white-space: normal; text-align: left;
}
[class*="st-key-araya_session_"] [data-testid="stMarkdownContainer"] code {
    font-size: 11px; color: #8e909c; background: transparent; padding: 0;
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
    display: flex; font-size: 12px; color: #737685;
}
/* Help is a small utility, so it is sized by its icon and label instead
   of stretching to whatever the column happens to be -- and it is pushed
   to the end of that column so the bar ends on the control, not on the
   empty half of a cell. */
.st-key-araya_topbar [data-testid="stColumn"]:last-child
[data-testid="stLayoutWrapper"] {
    /* Streamlit's column is a block and its wrapper shrinks to the
       control, so the control is pushed rather than aligned. */
    margin-left: auto;
}
.st-key-araya_topbar [data-testid="stPopover"] button {
    width: auto; padding: 7px 12px; min-height: 38px; white-space: nowrap;
}
/* Segmented control as the mockup's pill switch. Streamlit renders it as a
   button group; only the group wrapper is a stable hook, so the tray is
   styled here and the selected item keeps the widget's own affordance. */
.st-key-araya_view_switch [data-testid="stButtonGroup"],
.st-key-araya_console_switch [data-testid="stButtonGroup"] {
    background: #F1F2F5; border-radius: 9999px; padding: 3px; gap: 2px;
    /* The tray must never be what overflows: it wraps with its own group
       and each pill keeps its label on one line, so a narrow column
       produces two stacked pills instead of two pills sitting on top of
       the control beside them. */
    flex-wrap: wrap; width: fit-content; max-width: 100%;
}
.st-key-araya_view_switch [data-testid="stButtonGroup"] button,
.st-key-araya_console_switch [data-testid="stButtonGroup"] button {
    white-space: nowrap;
}
/* Both app bars lay their controls out with st.columns, and a Streamlit
   column is `flex: 1 1 <percent>` with min-width auto -- it shrinks past its
   content rather than wrapping. The switch column's share is 27%, which is
   less than its two pills need even on a 1280px window (259px against 275px),
   so the pills wrapped inside their tray at every width. The floor below is
   what the two pills actually measure, and the columns wrap to a second row
   once the bar cannot seat them all. */
.st-key-araya_topbar [data-testid="stColumn"],
.st-key-araya_console_header [data-testid="stColumn"] {
    /* Sized from their contents rather than from a percentage of the whole
       row: a percentage basis plus a control with a floor adds up to more
       than the row, and the surplus pushed one control onto a line of its
       own where it then stretched across the page. Nothing grows here
       except the field in the middle. */
    flex: 0 0 auto; min-width: 150px;
}
.st-key-araya_topbar [data-testid="stColumn"]:first-child,
.st-key-araya_console_header [data-testid="stColumn"]:first-child {
    flex: 0 0 300px; min-width: 300px;
}
/* Second column -- the console's search, the assistant's session id -- is
   the elastic one, so the bar breathes without the buttons moving. */
.st-key-araya_topbar [data-testid="stColumn"]:nth-child(2),
.st-key-araya_console_header [data-testid="stColumn"]:nth-child(2) {
    flex: 1 1 200px; min-width: 180px;
}
@media (max-width: 1100px) {
    /* Once the bar wraps, the session id is no longer opposite the switch,
       so right-aligning it would leave it floating mid-row. */
    .st-key-araya_topbar .araya-session-id { justify-content: flex-start; }
}
/* The bar's two actions are one group pinned to the right, not two
   controls adrift in the middle of the row. The nested columns must undo
   the bar's own column sizing, which would otherwise give each action the
   150px floor meant for the top-level cells. */
.st-key-araya_console_actions [data-testid="stHorizontalBlock"],
.st-key-araya_chat_actions [data-testid="stHorizontalBlock"] {
    justify-content: flex-end; gap: 12px; flex-wrap: nowrap;
}
.st-key-araya_console_header
.st-key-araya_console_actions [data-testid="stColumn"],
.st-key-araya_topbar
.st-key-araya_chat_actions [data-testid="stColumn"] {
    flex: 0 0 auto; min-width: 0;
}
/* One control height across the bar: a text field, a toggle and a button
   each carry their own default, and the mismatch is what made the row
   read as three unrelated things. */
.st-key-araya_console_header [data-testid="stTextInput"] input,
.st-key-araya_console_header [data-testid="stPopover"] button {
    min-height: 38px;
}
.st-key-araya_console_actions [data-testid="stPopover"] button,
.st-key-araya_chat_actions [data-testid="stPopover"] button {
    white-space: nowrap;
}
/* The export sheet is a menu, not a panel: it holds a caption and three
   buttons, so it is sized to them instead of to Streamlit's default
   popover measure. */
[data-testid="stPopoverBody"]:has([class*="st-key-araya_export_"]) {
    width: 188px; min-width: 0; padding: 10px 10px 12px;
}
[data-testid="stPopoverBody"] [class*="st-key-araya_export_"] button {
    border-radius: 8px; font-weight: 600; padding: 5px 8px;
    min-height: 32px; font-size: 13px;
}
/* The caption is one line of context, not a paragraph. */
[data-testid="stPopoverBody"]:has([class*="st-key-araya_export_"])
[data-testid="stCaptionContainer"] p {
    font-size: 11.5px; margin-bottom: 2px;
}
[data-testid="stPopoverBody"]:has([class*="st-key-araya_export_"])
[data-testid="stElementContainer"] {
    margin-bottom: -6px;
}
/* One colour per format, so the three are told apart at a glance rather
   than read one by one. */
.st-key-araya_export_jsonl button {
    border-color: #B6C6F2; color: #0052CC; background: #F2F6FF;
}
.st-key-araya_export_jsonl button:hover { border-color: #0052CC; }
.st-key-araya_export_csv button {
    border-color: #A9DCC4; color: #00714b; background: #F0FAF5;
}
.st-key-araya_export_csv button:hover { border-color: #00875A; }
.st-key-araya_export_md button {
    border-color: #D5C9F0; color: #5E4DB2; background: #F7F4FF;
}
.st-key-araya_export_md button:hover { border-color: #6554C0; }
/* Phone width: the switch takes the full row and splits it, because a
   fit-content tray leaves the two pills too small a target. */
@media (max-width: 640px) {
    /* The floors above are desktop measures; on a phone every control
       takes the full row instead. The selectors mirror the ones they
       override, because a plain column rule would lose to them. */
    .st-key-araya_topbar [data-testid="stColumn"],
    .st-key-araya_console_header [data-testid="stColumn"],
    .st-key-araya_topbar [data-testid="stColumn"]:first-child,
    .st-key-araya_console_header [data-testid="stColumn"]:first-child,
    .st-key-araya_topbar [data-testid="stColumn"]:nth-child(2),
    .st-key-araya_console_header [data-testid="stColumn"]:nth-child(2) {
        flex: 1 1 100%; min-width: 0;
    }
    .st-key-araya_view_switch [data-testid="stButtonGroup"],
    .st-key-araya_console_switch [data-testid="stButtonGroup"] {
        width: 100%;
    }
    .st-key-araya_view_switch [data-testid="stButtonGroup"] button,
    .st-key-araya_console_switch [data-testid="stButtonGroup"] button {
        flex: 1 1 50%;
    }
}
/* The rails are pinned to a fixed width for the desktop measure. Streamlit
   turns the rail into an overlay on a narrow viewport, and a 240-260px
   floor then covers most of a phone screen, so the pin is relaxed to a
   share of the viewport rather than dropped. */
@media (max-width: 768px) {
    /* The attribute is repeated to outweigh the page stylesheets, which
       pin their own rail width with !important and are injected after
       this one; specificity is the only lever left once both sides are
       important. */
    [data-testid="stSidebar"][data-testid="stSidebar"] {
        width: min(260px, 84vw) !important; min-width: 0 !important;
    }
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
    /* Streamlit reserves 96px above the first block for its own toolbar;
       this app draws its bar there instead, so the gap between the brand
       and the app bar was empty screen. */
    padding-top: 12px;
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
/* The running status sits at the right end of the reading column, just
   above the composer -- the corner the employee is already looking at
   after pressing send. The block itself carries no surface: while the run
   blocks, Streamlit keeps the previous run's elements on screen in a
   faded state, and a bordered block drew those ghosts inside itself. Only
   the spinner is a pill. */
.st-key-araya_thinking {{
    /* Streamlit's block is a flex column, so the pill is pushed to the
       right with align-items -- justify-content moves it along the
       vertical axis and would leave it on the left. */
    display: flex; align-items: flex-end;
    margin: 4px 0 10px; background: none; border: none;
}}
.st-key-araya_thinking [data-testid="stSpinner"] {{
    margin: 0; display: inline-flex; align-items: center;
    padding: 6px 14px 6px 10px; border-radius: 9999px;
    background: #FFFFFF; border: 1px solid #E4E6EB;
    box-shadow: 0 1px 2px rgba(25,28,30,0.06);
}}
/* Streamlit stacks the ring and its caption in a block, so the caption
   sat below the ring's centre line. They are centred against each other
   here, and the caption's own line box is tightened so the two read as
   one row. */
.st-key-araya_thinking [data-testid="stSpinner"] > div {{
    display: flex; align-items: center; gap: 9px; margin: 0;
}}
.st-key-araya_thinking [data-testid="stSpinner"] i,
.st-key-araya_thinking [data-testid="stSpinner"] svg {{
    flex: 0 0 auto; align-self: center; margin: 0;
}}
.st-key-araya_thinking p {{
    font-size: 13px; line-height: 18px; color: #5A5D6B; margin: 0;
    white-space: nowrap; align-self: center;
}}
</style>
"""

_CHAT_DARK_CSS = """
<style>
/* The assistant's dark sheet. It mirrors the console's: the same ground,
   the same panel and border tokens, injected after the light ones so it
   overrides them. Only this page's own surfaces are restyled -- Streamlit
   widgets keep following the base theme in .streamlit/config.toml. */
[data-testid="stMain"], [data-testid="stBottom"] { background: #0E1116; }
[data-testid="stSidebar"] { background: #141922; border-right-color: #262C36; }
[data-testid="stMain"] p, [data-testid="stMain"] span,
[data-testid="stMain"] div, [data-testid="stMain"] li { color: #C3C9D4; }
[data-testid="stMarkdownContainer"] p.araya-hero-title { color: #E6E8EC; }
[data-testid="stMarkdownContainer"] p.araya-hero-sub { color: #98A2B3; }
.araya-brand-name { color: #E6E8EC; }
.araya-rail-note, .araya-rail-label { color: #7B8494; }
.araya-rail-note { border-top-color: #262C36; }
.st-key-araya_new_session [data-testid^="stBaseButton-"] {
    background: #161A21; border-color: #2C3442; color: #E6E8EC;
}
[class*="st-key-araya_session_"] [data-testid^="stBaseButton-"]:hover {
    background: #1A2029;
}
[class*="st-key-araya_session_"] [data-testid="stBaseButton-primary"] {
    background: #1B2437; border-color: #2A3550;
}
[class*="st-key-araya_session_"] [data-testid="stMarkdownContainer"] p {
    color: #E6E8EC;
}
.araya-turn-label { color: #7B8494; }
.araya-bubble-user { background: #1B2437; color: #E6E8EC; }
[class*="st-key-araya_answer_"] {
    background: #161A21; border-color: #262C36;
}
[class*="st-key-araya_answer_"] [data-testid="stHorizontalBlock"] {
    border-bottom-color: #1F2530;
}
[class*="st-key-araya_answer_"] [data-testid^="stBaseButton-"] {
    background: #161A21; border-color: #2C3442; color: #98A2B3;
}
.araya-answer, .araya-cite-title { color: #E6E8EC; }
.araya-footnote, .araya-sources-head, .araya-cite-meta { color: #7B8494; }
.araya-cite-card { background: #131820; border-color: #262C36; }
[class*="st-key-araya_suggest_"] [data-testid^="stBaseButton-"] {
    background: #161A21; border-color: #262C36;
}
[class*="st-key-araya_suggest_"] [data-testid="stMarkdownContainer"] p {
    color: #E6E8EC;
}
.araya-scope-text { color: #98A2B3; }
[data-testid="stChatInput"] {
    background: #161A21; border-color: #2C3442;
}
[data-testid="stBottomBlockContainer"]::after { color: #7B8494; }
.st-key-araya_thinking {
    background: #161A21; border-color: #262C36;
}
.st-key-araya_thinking p { color: #98A2B3; }
</style>
"""

_CONSOLE_LAYOUT_CSS = """
<style>
[data-testid="stMainBlockContainer"] {
    max-width: 1248px; padding-left: 24px; padding-right: 24px;
    padding-top: 12px;
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

/* Knowledge Base rows are buttons so a document can be opened and read.
   The selected one is tinted rather than filled: it marks a position in a
   list, it is not the page's primary action. */
[class*="st-key-araya_doc_"] [data-testid^="stBaseButton-"] {
    background: #FFFFFF; border: 1px solid #E4E6EB; border-radius: 10px;
    padding: 10px 12px;
}
/* Streamlit centres a button's icon-and-label block, and renders the icon
   inline before the first line -- which in a list puts the id on a
   different left edge from the two lines under it. The row is laid out as
   a grid instead: the glyph gets its own column, and all three lines
   start at the same edge of the second one. */
[class*="st-key-araya_doc_"] [data-testid^="stBaseButton-"] > div {
    justify-content: flex-start; align-items: flex-start; width: 100%;
}
[class*="st-key-araya_doc_"] [data-testid="stMarkdownContainer"] {
    display: grid; grid-template-columns: 30px 1fr; align-items: start;
    column-gap: 4px; width: 100%;
}
[class*="st-key-araya_doc_"] [data-testid="stMarkdownContainer"] p {
    grid-column: 2;
}
[class*="st-key-araya_doc_"] [data-testid="stIconMaterial"] {
    grid-row: 1 / span 2; grid-column: 1; align-self: center;
    width: 28px !important; height: 28px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px !important;
}
/* Policy is the authoritative stratum and chat is the noisy one, so the
   two carry the colours the answer card already uses for that split. */
[class*="st-key-araya_doc_policy_"] [data-testid="stIconMaterial"] {
    color: #00714b; background: #E3F5EC;
}
[class*="st-key-araya_doc_chat_"] [data-testid="stIconMaterial"] {
    color: #0052CC; background: #EEF2FF;
}
/* Rows sit closer together than Streamlit's default element gap, while
   still reading as separate cards. */
[class*="st-key-araya_doc_"] { margin-bottom: -8px; }
[class*="st-key-araya_doc_"]:last-of-type { margin-bottom: 0; }
[class*="st-key-araya_doc_"] [data-testid="stBaseButton-primary"] {
    background: #EEF2FF; border-color: #C7D4FF;
}
[class*="st-key-araya_doc_"] [data-testid="stMarkdownContainer"] p {
    font-size: 13.5px; line-height: 20px; color: #191c1e; margin: 0;
    text-align: left; white-space: normal;
}
[class*="st-key-araya_doc_"] [data-testid="stMarkdownContainer"] code {
    font-size: 11px; color: #737685; background: transparent; padding: 0;
}

/* Stat card: the console's own number block, wider than the KPI strip and
   carrying a share bar so three counts read as parts of one total. */
/* Card rows are a grid rather than st.columns: a Streamlit column sizes
   itself to its own content, so a card carrying one extra line -- the LLM
   call footnote under Avg latency -- grew past its row and overlapped the
   panels beneath it. A grid row is as tall as its tallest card and every
   card fills it, and auto-fit reflows the row on a narrow window without a
   media query. */
.araya-card-grid {
    display: grid; gap: 16px; margin-bottom: 20px; align-items: stretch;
    /* The stat row is always four cards, so the counts are stepped
       explicitly: auto-fit would leave a 3 + 1 row on a mid-width window,
       which reads as a card that failed to load. */
    grid-template-columns: repeat(4, minmax(0, 1fr));
}
@media (max-width: 1200px) {
    .araya-card-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
    .araya-card-grid { grid-template-columns: minmax(0, 1fr); }
}
/* Panels vary in number, so they stay on auto-fit -- wider than the stat
   grid, because a panel holds metric rows and a narrower column forces
   the fraction onto its own line. The declaration repeats inside the two
   media queries above it, which would otherwise win on the panels too. */
.araya-card-grid--panels,
.araya-card-grid--panels[class] {
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
}
/* Three-card strips (the index health row) keep their own count for the
   same reason the four-card row does. */
.araya-card-grid--three,
.araya-card-grid--three[class] {
    grid-template-columns: repeat(3, minmax(0, 1fr));
    margin-bottom: 24px;
}
@media (max-width: 900px) {
    .araya-card-grid--three,
    .araya-card-grid--three[class] {
        grid-template-columns: minmax(0, 1fr);
    }
}
.araya-stat {
    background: #FFFFFF; border: 1px solid #E4E6EB; border-radius: 12px;
    padding: 16px; height: 100%;
    display: flex; flex-direction: column;
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
/* Pushed to the foot of its card so a footnote lines up across the row
   instead of sitting right under a shorter value. */
.araya-stat-note {
    font-size: 12px; line-height: 18px; color: #8e909c;
    margin-top: auto; padding-top: 10px;
}

.araya-panel {
    background: #FFFFFF; border: 1px solid #E4E6EB; border-radius: 12px;
    padding: 18px; height: 100%;
    display: flex; flex-direction: column;
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
.araya-triage-tag--knowledge { color: #0052CC; border-color: #C7D6F5; }
.araya-triage-tag--input { color: #BA1A1A; border-color: #F5C7C7; }
.araya-triage-tag--service { color: #B76E00; border-color: #F0D9B5; }
.araya-triage-tag--validation { color: #5E4DB2; border-color: #D7D0F0; }
.araya-triage-tag--unknown { color: #6b6d76; border-color: #DCDDE3; }

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
/* The measured duration of one node, on the same monospace ramp as the
   node name so the column reads as data rather than prose. */
.araya-trace-time {
    flex: 0 0 auto; font-size: 11.5px; color: #737685; text-align: right;
    min-width: 54px;
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
/* Recorded run: one line per command the baseline says was executed, with
   the outcome it recorded. The colour comes from that text, never from a
   run this page performed -- the console evaluates nothing. */
.araya-run {
    display: flex; align-items: center; gap: 12px; padding: 8px 0;
    border-bottom: 1px solid #F0F1F4;
}
.araya-run:last-child { border-bottom: none; }
.araya-run-cmd {
    flex: 1; min-width: 0; font-size: 12.5px; color: #42526E;
    overflow-wrap: anywhere;
}
.araya-run-result {
    flex: none; font-size: 12px; font-weight: 600; border-radius: 999px;
    padding: 3px 10px; white-space: nowrap;
}
.araya-run-result--ok { background: #E3F5EC; color: #00714b; }
.araya-run-result--fail { background: #FCE9E6; color: #B3261E; }
.araya-run-result--neutral { background: #F1F2F5; color: #5A5D6B; }
/* The baseline file is a long report with page-sized headings of its own.
   Inside the reader block they are scaled to the console's type ramp, so
   opening it no longer pushes every panel off the screen. */
.st-key-araya_baseline_doc { border: 1px solid #E4E6EB; border-radius: 12px;
    background: #FFFFFF; padding: 4px 18px;
}
.st-key-araya_baseline_doc h1 { font-size: 19px; line-height: 30px; }
.st-key-araya_baseline_doc h2 { font-size: 16px; line-height: 26px; }
.st-key-araya_baseline_doc h3 { font-size: 14px; line-height: 24px; }
.st-key-araya_baseline_doc p,
.st-key-araya_baseline_doc li,
.st-key-araya_baseline_doc td,
.st-key-araya_baseline_doc th {
    font-size: 13px; line-height: 22px;
}
.araya-metric {
    display: flex; align-items: center; gap: 12px; padding: 9px 0;
    border-bottom: 1px solid #F0F1F4;
    /* Two panels sit side by side, so a metric row can be as narrow as
       ~215px. With a fixed bar and value the name was left a few pixels
       and broke one letter per line; wrapping the row keeps the name
       readable and drops the bar underneath it instead. */
    flex-wrap: wrap;
}
.araya-metric:last-child { border-bottom: none; }
.araya-metric-name {
    flex: 1 1 130px; min-width: 110px; font-size: 14px; color: #42526E;
}
/* A metric can carry a remark -- a second rate the file states in the same
   cell. It reads as a second line of the name, so the row keeps its three
   columns and the name is never squeezed into a column of single letters. */
.araya-metric-remark {
    display: block; font-size: 11.5px; line-height: 18px; color: #8e909c;
}
.araya-metric-bar {
    flex: 1 1 60px; max-width: 110px; min-width: 44px; height: 6px;
    border-radius: 9999px; background: #EDEEF0; overflow: hidden;
}
.araya-metric-fill { display: block; height: 6px; }
.araya-metric-value {
    font-size: 13px; flex: 0 0 52px; text-align: right; color: #191c1e;
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
.araya-triage-meta, .araya-axis-labels, .araya-trace-detail,
.araya-trace-time, .araya-metric-remark {
    color: #7B8494;
}
.araya-stat, .araya-panel {
    background: #161A21; border-color: #262C36;
}
.araya-stat-track, .araya-metric-bar { background: #222835; }
.araya-triage, .araya-metric, .araya-trace-row, .araya-run {
    border-color: #1F2530;
}
.araya-run-cmd { color: #98A2B3; }
[class*="st-key-araya_doc_"] [data-testid^="stBaseButton-"] {
    background: #161A21; border-color: #262C36;
}
[class*="st-key-araya_doc_policy_"] [data-testid="stIconMaterial"] {
    color: #5FC79B; background: #10281F;
}
[class*="st-key-araya_doc_chat_"] [data-testid="stIconMaterial"] {
    color: #8FB2FF; background: #141C2E;
}
[class*="st-key-araya_doc_"] [data-testid="stBaseButton-primary"] {
    background: #1B2437; border-color: #2A3550;
}
[class*="st-key-araya_doc_"] [data-testid="stMarkdownContainer"] p {
    color: #E6E8EC;
}
/* The format buttons keep their identity on the dark sheet, at the
   contrast the dark ground needs. */
.st-key-araya_export_jsonl button {
    border-color: #24365C; color: #8FB2FF; background: #141C2E;
}
.st-key-araya_export_csv button {
    border-color: #1D3B2E; color: #6FCFA3; background: #101F19;
}
.st-key-araya_export_md button {
    border-color: #322A52; color: #B5A6F0; background: #171430;
}
/* The outcome chips carry their own light-theme fills, which turn into
   pale blocks on the dark sheet; these are the same two verdicts read
   against the console's dark ground. */
/* Prefixed with the page container: the blanket text colour at the top of
   this sheet is more specific than a bare class, and would otherwise grey
   out the one colour these chips exist to carry. */
[data-testid="stMain"] .araya-run-result--ok {
    background: #10281F; color: #5FC79B;
}
[data-testid="stMain"] .araya-run-result--fail {
    background: #2A1614; color: #F1938C;
}
[data-testid="stMain"] .araya-run-result--neutral {
    background: #222835; color: #98A2B3;
}
.st-key-araya_baseline_doc {
    background: #161A21; border-color: #262C36;
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
