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
