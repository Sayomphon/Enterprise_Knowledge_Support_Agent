# AGENTS.md — Enterprise Knowledge & Support Agent

> **Scope:** This file is the single source of truth for any AI coding agent
> (Codex, Claude Code, Cursor, Copilot Workspace, …) and for human contributors
> working in this repository.
> `CLAUDE.md` intentionally contains only Claude-Code-specific operating rules
> and points back here. **Never duplicate rules across the two files** — if a
> rule applies to all agents, it belongs here.

---

## 0. Prime Directive

Build a **reliability-first Thai-language RAG prototype** whose behaviour is
**auditable, deterministic where possible, and honest about its limits**.

Ranking of priorities when they conflict:

```
1. Correctness & safety of the answer path (no fabricated grounding)
2. Security (prompt injection, secrets, logging hygiene)
3. Testability & readability (a reviewer must understand it in one pass)
4. Extensibility (clear seams for production hardening)
5. Feature count  ← always sacrificed first
```

If a change adds a feature but weakens 1–4, **do not make it**. Propose it as a
"Production Next Step" in the README instead.

---

## 1. Project Snapshot

| Item | Value |
|---|---|
| Product | Mini Enterprise Knowledge & Support Agent (HR + Finance Q&A) |
| Users | Employees asking Thai-language questions about leave and reimbursement |
| Corpus | 8 mock Markdown documents (`policy` + noisy `chat`) |
| Language | Python 3.11+ |
| Orchestration | LangGraph (typed state, conditional edges) |
| Retrieval | Local character n-gram TF-IDF (scikit-learn) + cosine similarity |
| LLM | `langchain-openai` `ChatOpenAI`, model configurable via env |
| UI | Streamlit (`app.py`), CLI (`main.py`) as guaranteed fallback |
| Logs | Append-only JSONL, no external infrastructure |
| Tests | `unittest` + mocks, **offline-first** |

### Explicit non-goals (do not implement without an explicit request)

Vector database, embedding APIs, local embedding models, rerankers,
production OCR/PDF parsing, FastAPI, React, SQL databases, authentication/RBAC,
LangSmith, Docker/Kubernetes, conversation memory, LLM-as-judge evaluation.

These are **deliberately excluded** because the corpus is 8 documents; they are
documented as production next steps, not built.

---

## 2. Runtime Architecture

```text
User query
   │
   ▼
[1] Input Guardrail (deterministic, regex, no LLM)
   ├── BLOCK ──► Refusal + JSONL log ──► END
   ▼ PASS
[2] Original Retrieval (character TF-IDF, top-K candidates)
   │
   ▼
[3] Supported-Scope + Authority Coverage Gate (deterministic, no LLM)
   ├── unsupported / under-specified topic ──► Fallback + log ──► END
   ├── no active policy covering the topic ──► Fallback + log ──► END
   ▼ PASS
[4] Score Router (3 bands, calibrated thresholds)
   ├── LOW    ─────────────────────────► Fallback + log ──► END
   ├── MEDIUM ─► [5] Query Rewriter (LLM, structured output)
   │                └─► [6] Rewrite Validator (deterministic, no LLM)
   │                        rejected candidates never reach the index
   │                └─► [7] Expanded Retrieval (max score per doc)
   │                        ├── below FINAL_ANSWER_THRESHOLD ─► Fallback + log
   │                        └── pass ──────────────┐
   └── HIGH ──────────────────────────────────────►│
                                                   ▼
                                        [8] Evidence Selector (deterministic)
                                            policy authoritative, chat
                                            supplementary, canonical links
                                            resolved
                                            ├── no policy ──► Fallback + log
                                            ▼
                                        [9] Reporter (LLM, structured claims)
                                            evidence is JSON-encoded;
                                            output is a candidate answer
                                            ├── no credential configured
                                            │    ──► Fallback + log
                                            │        (llm_not_configured)
                                            ▼
                                       [10] Answer Contract Validator
                                            (deterministic: claim text,
                                            per-claim sources, provenance,
                                            policy authority)
                                            ├── VALID ──► [11] Renderer
                                            │               ──► Answer + sources
                                            └── INVALID ──► Fallback + log
```

The Reporter writes `candidate_answer`, never `answer`. Only the validation node
may promote a candidate, and it renders the public text itself, so no
unvalidated model output can reach an employee, a log, or the returned state.

**Design rule:** an LLM is called **only when the deterministic layer is
insufficient**. Never add an LLM call to a path that a rule can decide.

LLM call budget per request: blocked = 0, clear out-of-domain = 0,
strong retrieval = 1 (Reporter), medium band = 2 (Rewriter + Reporter).

Credentials are checked at the LLM boundary, never at start-up: the zero-LLM
routes above must stay demonstrable without a key. A key missing at the
Rewriter degrades to original-query-only expansion; missing at the Reporter it
becomes its own reason code, so an unconfigured service is never reported to an
employee as insufficient evidence.

---

## 3. Repository Map & Layering

```text
├── app.py                     # Streamlit UI — presentation only
├── main.py                    # CLI entry point — argument parsing only
├── src/
│   ├── config.py              # Typed settings loaded from env, single source
│   ├── schemas.py             # Document, RetrievedDocument, PipelineState, Route
│   ├── graph.py               # LangGraph nodes + edges + routing functions
│   ├── fallback.py            # Fixed fallback/refusal texts + reason codes
│   ├── evidence_selector.py   # Policy-first answer-evidence selection
│   ├── answer_renderer.py     # Validated claims -> public answer text
│   ├── logging_utils.py       # JSONL telemetry: append writer + bounded reader
│   ├── agents/
│   │   ├── rewriter.py        # Adaptive query rewriter (LLM boundary)
│   │   └── reporter.py        # Grounded answer generation (LLM boundary)
│   ├── guardrails/
│   │   ├── input_guardrail.py     # Pre-retrieval deterministic screen
│   │   ├── scope_validator.py     # Supported-topic catalog and gate
│   │   ├── rewrite_validator.py   # Intent preservation on model rewrites
│   │   ├── text_similarity.py     # Shared character n-gram measures
│   │   └── citation_validator.py  # Claim + citation contract validation
│   ├── ingestion/
│   │   └── loader.py          # Markdown + YAML frontmatter → Document
│   └── retrievers/
│       └── local_tfidf.py     # Character TF-IDF index + cosine scoring
├── data/docs/                 # 8 mock documents (Thai content, English filenames)
├── eval/                      # Calibration / held-out / guardrail sets + runner
├── logs/                      # Runtime JSONL output (git-ignored except .gitkeep)
└── tests/                     # Offline unit + graph route tests
```

### Dependency direction (enforced, no exceptions)

```text
app.py / main.py  ──►  src.graph  ──►  src.agents / src.guardrails
                                            │
                                            ▼
                 src.evidence_selector / src.answer_renderer
                 / src.retrievers / src.ingestion / src.logging_utils
                                            │
                                            ▼
                                src.schemas / src.config
```

* `schemas.py` and `config.py` import **nothing** from the rest of `src`.
* Retrievers, guardrails, and the loader **never** import `graph.py`.
* UI/CLI layers contain **no business logic** — they call the compiled graph
  and render `PipelineState`. If a Streamlit change requires new logic, that
  logic goes into `src/`, with a test.
* Only `src/agents/*` may call an LLM. Nothing else touches the network.

---

## 4. Non-Negotiable Invariants

Violating any of these is a defect, even if tests pass.

1. **Deterministic before probabilistic.** Guardrail, scope, rewrite, evidence
   and answer-contract validation are pure functions, no LLM, fully
   unit-testable.
2. **The Reporter answers only from supplied evidence.** No external knowledge,
   no invented sources, no guessing when evidence is thin.
3. **Evidence is data, never instruction.** Retrieved document text reaches the
   model as a `json.dumps` block inside the human message, never as prompt
   structure, so a poisoned document cannot break out of its own record; the
   system prompt states that those values must not be obeyed.
4. **Every factual claim carries validated `[SOURCE-ID]` citations**, and every
   cited ID must belong to the answer evidence selected for that request. The
   model does not write the markup: the renderer emits it from validated IDs.
   Missing, uncited, fabricated, or structurally broken output routes to
   fallback — never to `END`, and never into the public `answer`.
4b. **Policy is authoritative, chat is supplementary.** A normative answer needs
   at least one active policy document covering the query's topic; chat
   transcripts improve recall and may accompany a policy, but can never be the
   only source of a rule. Authority comes from corpus metadata, never from the
   model or from document text.
4c. **Similarity is never the only answerability signal.** A query must resolve
   to a supported topic in the bounded catalog before any score can admit it to
   generation, so an in-domain question the corpus has no policy for falls back
   instead of being answered from a lexically similar document.
5. **Thresholds come from calibration, never from intuition.** No magic numbers
   inline; they live in `config.py` / `.env` with a comment recording how they
   were derived.
6. **The score is a similarity heuristic, not a probability.** Name it
   `raw_retrieval_score` / `expanded_retrieval_score`; label it in the UI as
   "Retrieval Similarity Score (heuristic)". Never call it confidence-as-percent.
7. **The rewriter preserves user intent, and the pipeline proves it.** It may fix
   typos and slang; it may not introduce HR/Finance topics the user never
   mentioned or change an amount, a time, or a leave type. The prompt asks for
   this; `rewrite_validator` enforces it deterministically, and a rejected
   candidate reaches neither the index nor the log. The original query is
   always retained in expanded retrieval.
8. **Rewriter failure degrades gracefully** — fall back to the original query,
   never crash the request.
9. **Every blocked / fallback event is logged as JSONL** with a reason code.
10. **No secrets, system prompts, or API keys in logs, UI, or exceptions.**
11. **Tests run fully offline.** No network, no API key required, except an
    optional, clearly-marked live smoke test.
12. **`.env` is never committed.** `.env.example` carries placeholder values only.

---

## 5. Environment & Commands

```bash
# Setup (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill OPENAI_API_KEY locally

# Run (no API key needed for the guardrail and low-score routes)
python main.py "ขั้นตอนการเบิกค่าแท็กซี่หลังทำ OT ต้องทำอย่างไร"
python main.py --interactive
python main.py --check    # corpus + credential readiness, runs no query
streamlit run app.py

# Verify (must pass before any commit)
python -m unittest discover -s tests -v
python eval/run_eval.py --set calibration    # tuning only
python eval/run_eval.py --set heldout        # reporting only, run last

# Gate the same sets: --strict exits 1 on any case that contradicts its label
python eval/run_eval.py --set guardrail --strict
python eval/run_eval.py --set calibration --strict
```

Dependencies are **pinned** in `requirements.txt` from a clean virtualenv that
passed the smoke test. Do not add a dependency without an explicit request; if
one is genuinely required, state the reason, the size, and the alternative that
was rejected.

---

## 6. Coding Standards

### 6.1 Language and formatting

* **All code artefacts are English-only**: identifiers, docstrings, comments,
  log keys, commit messages, test names, error messages raised to developers.
* **Thai is allowed only in**: `data/docs/**` content, user-facing message
  constants in `src/fallback.py`, prompt templates in `src/agents/*`, the topic
  alias catalogs in `src/guardrails/scope_validator.py`, and `eval/*.json`
  query fixtures. Keep those strings in named constants, never inline in logic.
  The alias catalogs are configuration data about the corpus vocabulary, not
  branching logic: they must stay declarative tables, never `if/elif` chains.
* Line length ≤ 100. 4-space indentation. One public concept per module.
* Prefer explicit names over abbreviations: `expanded_retrieval_score`, not `ers`.
* No emoji or decorative characters in source files (UI labels excepted).

### 6.2 Docstrings — required on every module, class, and public function

Google style, describing behaviour and contract, not implementation trivia:

```python
def route_after_raw_retrieval(state: PipelineState) -> Literal["high", "medium", "low"]:
    """Select the pipeline branch from the raw retrieval score.

    The router implements the three-band policy described in AGENTS.md §2:
    high-confidence queries go straight to generation, medium-band queries pay
    for one rewrite attempt, and low-band queries fall back without any LLM
    call so that out-of-domain questions are never rewritten into the domain.

    Args:
        state: Pipeline state that must already contain ``raw_retrieval_score``.

    Returns:
        The branch key consumed by the LangGraph conditional edge.

    Raises:
        KeyError: If retrieval has not populated ``raw_retrieval_score``.
    """
```

Private helpers (`_name`) need at least a one-line docstring.

### 6.3 Comments

Comment the **why**, never the what. Required in three situations:

* a non-obvious trade-off (`# Character n-grams, not word tokens: Thai has no
  reliable whitespace segmentation and typos still share n-grams.`)
* a security-relevant decision (`# Bounded quantifiers only — unbounded nesting
  would expose the guardrail to ReDoS.`)
* a calibrated constant (`# Derived from eval/retrieval_calibration.json sweep
  on 2026-08-20; see README §Evaluation.`)

Delete commented-out code. Do not narrate obvious statements.

### 6.4 Typing and data contracts

* Full type hints on every function signature; `mypy`-clean intent even if
  `mypy` is not wired into CI.
* Domain objects are `@dataclass(frozen=True)` (`Document`, `RetrievedDocument`,
  `ValidationResult`) — immutability makes state transitions auditable.
* `PipelineState` is a `TypedDict` using `NotRequired` for fields that only some
  routes populate. **Do not** mark route-specific fields as required "to keep
  mypy quiet"; the contract must match runtime reality.
* LLM outputs are parsed into Pydantic models (`RewriteResult`) — never
  hand-rolled JSON parsing, never `eval`.

### 6.5 Module design

* Functions do one thing and are testable without monkeypatching internals.
* **No hidden global state.** The TF-IDF index is built once and passed
  explicitly (constructor injection or an explicit factory), not stored in a
  module-level mutable global.
* Side effects (file writes, network calls, clock reads) live at the edges and
  are injectable so that tests can substitute them.
* Extend behaviour through the seam, not through `if` chains: retrievers
  implement a `Retriever` protocol so a future BM25/hybrid/vector retriever is a
  new class, not an edit to routing.

```python
class Retriever(Protocol):
    """Contract every retrieval backend must satisfy."""

    def search(self, queries: Sequence[str], top_k: int) -> list[RetrievedDocument]:
        """Return the top-k documents, sorted by descending score."""
```

### 6.6 Error handling

* Fail fast at startup on malformed corpus, missing metadata, or duplicate
  `source_id` — never let a partially-loaded corpus reach retrieval.
* Fail soft at request time: LLM timeouts, malformed structured output, and
  provider errors degrade to the documented fallback path with a reason code.
* Catch **narrow** exceptions in library code. The one deliberately broad
  `except Exception` in `safe_rewrite` must be commented explaining why request
  survival outranks precision there, and must log the exception type.
* Never swallow an exception silently; never leak provider payloads or prompt
  contents into user-visible errors.

### 6.7 Determinism

Sorting must be total (score descending, then `source_id` ascending) so top-k is
reproducible. No `random` without a fixed seed. No dependence on dict ordering
from external calls.

---

## 7. Security Requirements

| Area | Rule |
|---|---|
| Secrets | Read only from environment via `config.py`, which exposes presence (`has_llm_credential()`), never the value. Never hardcode, never log, never echo in the UI, never write to `logs/`. Presence is checked lazily at the `src/agents` boundary; a missing key raises `MissingLlmCredentialError` and becomes the `llm_not_configured` reason code. |
| Input validation | Reject non-string, empty, and over-length input (`MAX_QUERY_CHARS`) before any processing. The pipeline receives NFC-normalised text; the rules match a hardened folding (NFKC with Thai SARA AM recomposed, format characters removed, separator punctuation folded, case-folded) that never leaves the guardrail. |
| Prompt injection | Deterministic named-rule screen **before** the first LLM call, re-run on every model-generated rewrite candidate. Optimise for precision: block explicit override attempts, never block benign words such as `admin`, `act as`, `สมมติว่า`, `คำสั่งซื้อ`. Every rule needs both an attack test and a benign-lookalike test, paired by `rule_id`. |
| ReDoS | Patterns use bounded quantifiers and no catastrophic nesting, and are screened against long adversarial input within a bounded time budget. |
| Evidence isolation | Retrieved text is JSON-encoded into the human message and the system prompt declares those values untrusted data. Encoding contains delimiter breakout; it does not solve indirect prompt injection. |
| Output validation | The reporter returns claims, not prose. Every claim is validated at runtime against the answer-evidence ID set and its policy subset — trust the validator, not the model. |
| Path safety | The loader reads only from the configured corpus directory; resolve and verify paths, no traversal from user input. |
| Logging hygiene | Log query text, reason code, scores, and top source IDs only. No API keys, no system prompts, no raw provider responses, no model metadata. |
| Log privacy | The sink holds raw employee questions across sessions. It is never rendered in the employee view, and the console renders it only when `ENABLE_OPS_VIEW` is explicitly enabled. That flag is a demo switch, not authentication or RBAC, and must never be described as either. |
| Dependencies | Pinned versions, direct dependencies only, installed from PyPI. |
| Honesty | The README must state plainly that regex screening is a prototype safeguard, not defence-in-depth, and that claim-level validation proves provenance and coverage, not factual entailment. Block rates are reported with their case count, never as universal security. |

---

## 8. Logging & Observability

Append-only `logs/fallback_queries.jsonl`, one JSON object per line, stable
schema:

```json
{
  "timestamp": "2026-08-20T14:30:00+07:00",
  "query": "...",
  "reason": "low_retrieval_score",
  "raw_retrieval_score": 0.06,
  "expanded_retrieval_score": null,
  "top_sources": ["HR-003"],
  "rewritten_queries": []
}
```

Reason codes (extend the enum, never invent ad-hoc strings):
`prompt_injection`, `low_retrieval_score`, `rewrite_low_retrieval_score`,
`missing_citation`, `fabricated_citation`, `rewrite_failure`,
`invalid_query_type`, `empty_query`, `query_too_long`, `reporter_failure`,
`unsupported_topic`, `no_authoritative_evidence`, `rewrite_rejected`,
`invalid_answer_structure`, `insufficient_reporter_evidence`,
`llm_not_configured`.

Only executed rewrites are logged: candidates the validator rejected are model
output about the user's question and never enter the JSONL record. The same
holds for a rejected candidate answer: the log carries its reason code, never
its text.

Writes are best-effort: a logging failure must never break a user request, but
it must surface on stderr. Timestamps are timezone-aware.

Best-effort is not silent. `log_fallback_event` returns a `LogWriteResult`, the
graph stores the outcome as `telemetry_logged`, and `src/fallback.py` selects
wording that claims a recorded question only when the append actually happened.
The write outcome is never used as the fallback reason: a full disk did not
change why the request lost its answer.

Reading back is bounded. `read_recent_events` tails only the newest rows instead
of loading the file, tolerating a partial final line by counting it rather than
rendering it. `read_persistent_events` adds the privacy gate: with
`ENABLE_OPS_VIEW` off -- the committed default -- past sessions' questions are
not read at all, rather than read and then hidden.

---

## 9. Testing Standards

* Framework: `unittest`; every test runs **offline** and must not require an API key.
* The LLM boundary is mocked at the agent module seam, never deep inside LangChain.
* Test names describe behaviour: `test_thai_injection_is_blocked_before_llm_call`.
* Arrange–Act–Assert with one behavioural assertion focus per test.
* Required critical-path coverage:
  * loader: 8 documents load; duplicate `source_id` fails; missing title fails
  * guardrail: English attack blocked, Thai attack blocked, `admin access` query passes, hypothetical-framing query passes
  * retrieval: exact query hits the expected document, typo query still hits it, out-of-domain scores below the answer threshold, top-k ordering is deterministic
  * rewriter: valid structured output, malformed provider response, timeout, failure falls back to original-query-only retrieval
  * reporter: structured output is returned unchanged, unparsed output raises, evidence is JSON-encoded, a document containing a closing delimiter cannot break its record, the document instruction never becomes a system message
  * citations: valid claims pass, a claim without a source fails, fabricated ID fails, real-but-not-selected ID fails, chat-only claim fails, broken structure fails
  * graph: at least five routes — injection→refusal, low→fallback, high→answer, medium→rewrite→answer, fabricated citation→fallback — plus the invalid-answer routes, which must leave no public `answer` in the final state
  * logging: successful append returns `ok=True`, an unwritable path returns `ok=False` without raising, the failure line names the exception type but not the path or query, the bounded reader returns the newest N rows and counts a partial line without rendering it
  * credentials: a missing key raises at the agent boundary with no client constructed, blocked and low-score routes still finish, and the reporter route reports `llm_not_configured` rather than thin evidence
  * CLI: handled requests exit `0`, a start-up failure exits non-zero with the exception type only, and rendered text comes from the shared response selector
* Every bug fix ships with the regression test that would have caught it.
* Tests must not write into `logs/` — inject a temporary directory.

---

## 10. Evaluation Discipline

* `eval/retrieval_calibration.json` (21 cases, including in-domain hard
  negatives) is for **tuning only**: n-gram configuration, `REWRITE_FLOOR`,
  `DIRECT_ANSWER_THRESHOLD`, `FINAL_ANSWER_THRESHOLD`, `SCOPE_MATCH_THRESHOLD`,
  `REWRITE_CONTINUITY_THRESHOLD`.
* `eval/retrieval_heldout.json` (14 cases) is for **reporting only**. Run it once,
  after thresholds are frozen. Never tune against it, never quote calibration
  numbers as generalisation evidence.
* `eval/guardrail_cases.json` holds ≥24 cases, balanced attack vs benign
  lookalike, and every `InjectionRule` is paired with a benign counter-example
  in `tests/test_guardrail.py`.
* `eval/citation_cases.json` holds labelled candidate answers covering each
  rejection reason of the answer contract plus valid claim shapes.
* `eval/rewrite_cases.json` holds labelled rewrite pairs, balanced between valid
  normalisations and each drift shape the validator must reject.
* `eval/BASELINE.md` records the frozen pre-remediation baseline and the
  post-remediation snapshot, so a metric change always has a reference point.
* Every case declares an unambiguous `expected_route` and `expected_sources`.
  "Answered or fallback, either is fine" is not a label.
* Rewrites used during threshold sweeps come from `eval/cached_rewrites.json` so
  sweeps stay deterministic and free.
* Reported metrics: Retrieval Hit@3, Answer-route Precision, Answer-route
  Coverage, OOD Fallback Accuracy, Unsupported In-domain Fallback Accuracy,
  Authoritative Evidence Coverage Rate, Rewrite Recovery Rate, Rewrite Intent
  Preservation Rate, Injection Block Rate, Benign Pass Rate, Citation Provenance
  Validity Rate, Claim Source Coverage Rate, Invalid Candidate Leakage Rate
  (target zero). Precision is always reported beside Coverage: a pipeline that
  answers almost nothing scores perfect precision.
* `--strict` turns any set into a gate: it exits 1 on a route that contradicts
  its label, an answerable case that missed every expected source, a citation or
  rewrite verdict mismatch, a missed attack, or a blocked benign lookalike.
  Claim Source Coverage is never counted — the fixture deliberately mixes
  grounded and ungrounded claims, so it describes the set, not a defect.
  A strict held-out run currently exits 1 on its one coverage miss; that is the
  gate reporting a known gap, not a regression.
* Selection principle for an enterprise support assistant:
  **a fallback is cheaper than a wrong answer** — optimise precision-heavy.

---

## 11. Scaling Seams (prototype → production)

Keep these boundaries clean so the production path is additive, not a rewrite:

| Seam | Prototype | Production path |
|---|---|---|
| `Retriever` protocol | character TF-IDF | BM25 → hybrid → embeddings + vector DB + reranker |
| Retrieval unit | whole document | chunking with hierarchical citation IDs |
| `config.py` | env vars | secret manager, per-tenant configuration |
| `logging_utils` | JSONL file | structured logging → OpenTelemetry → warehouse, with PII redaction and retention |
| Guardrail module | regex screen | layered classifier + policy engine + indirect-injection defences |
| Entry points | CLI + Streamlit | FastAPI service, authn/authz, per-user document ACLs |
| Evaluation | static JSON sets | CI-gated regression suite, LLM-as-judge for faithfulness, production sampling |

When touching a module, do not break its seam for short-term convenience.

---

## 12. Agent Working Agreement

For any non-trivial task:

1. **Read before writing.** Inspect the relevant modules and their tests; match
   existing conventions instead of importing a different house style.
2. **State a plan** — files to touch, contracts to change, tests to add — before
   editing. Flag anything that would violate §4.
3. **Small, coherent diffs.** One concern per change. No opportunistic
   refactoring, renaming, or reformatting of untouched code.
4. **Update contracts together**: schema change → node change → test change →
   README/`.env.example` change in the same diff.
5. **Verify**: run the unit tests, and the relevant eval set when retrieval,
   thresholds, or the guardrail changed. Report the actual command output.
6. **Report honestly.** If something is untested, partially working, or a
   temporary shortcut, say so explicitly. Never claim a test passed without
   running it. Never invent metric values.
7. **Ask, do not assume**, when a request conflicts with §4 or the non-goals in
   §1 — propose the compliant alternative.

---

## 13. Commit & PR Conventions

Conventional Commits, English, imperative mood, explaining intent:

```
feat: add adaptive query rewrite routing for medium-band retrieval
fix: reject citations that reference non-retrieved documents
test: cover rewriter timeout degradation to original-query retrieval
docs: record calibrated thresholds and held-out metrics
```

* Never commit `.env`, `logs/*.jsonl`, virtualenvs, or notebook scratch files.
* Do not create commits, amend history, or push unless explicitly asked.
* A PR description states: what changed, why, which invariants were touched,
  how it was verified, and known limitations.

---

## 14. Definition of Done

```text
[ ] Clean venv install from pinned requirements succeeds on Python 3.11+
[ ] CLI and Streamlit both run end to end
[ ] All 8 corpus documents load with valid, unique metadata
[ ] Exact / typo / slang queries retrieve the expected sources
[ ] English and Thai injection attempts blocked before any LLM call
[ ] Benign lookalike queries are not blocked
[ ] Out-of-domain query falls back with zero LLM calls
[ ] Every rendered claim carries validated [SOURCE-ID] citations
[ ] Fabricated, missing, and structurally invalid answers route to fallback
[ ] No unvalidated candidate answer appears in state, CLI, Streamlit, or logs
[ ] Blocked and fallback events are written to JSONL with reason codes
[ ] The response text claims a recorded question only when the append succeeded
[ ] Persistent log history is hidden unless ENABLE_OPS_VIEW is enabled
[ ] Blocked and out-of-domain routes run with no API key configured
[ ] Thresholds sourced from the calibration set; metrics from the held-out set
[ ] Offline test suite passes with no API key present
[ ] No secrets committed; .env.example carries placeholders only
[ ] Docstrings present on every module, class, and public function
[ ] README covers setup, architecture, decisions, evaluation, limitations
```

---

## 15. Anti-Patterns — Never Do These

* Calling the rewriter on every query (drags out-of-domain queries into the domain,
  and burns cost and latency).
* Hardcoding a threshold and describing it as "tuned".
* Presenting cosine similarity as an answer-correctness probability.
* Letting the answer validation node edge unconditionally to `END`.
* Writing model output straight into the public `answer` field.
* Widening injection regexes until benign enterprise questions get blocked.
* Adding a vector database, embedding API, or model download to "improve quality".
* Putting business logic inside `app.py` or `main.py`.
* Silent `except Exception: pass`.
* Logging prompts, provider payloads, or environment variables for debugging.
* Telling a user their question was recorded when the append failed.
* Blocking a deterministic route on a credential it never uses.
* Describing the `ENABLE_OPS_VIEW` flag as authentication or RBAC.
* Reporting calibration-set numbers as final performance.
* Writing Thai identifiers, comments, or docstrings in source files.
* Marking work complete without running the tests.

---

## 16. Glossary

| Term | Meaning |
|---|---|
| Route | Terminal or intermediate branch of the graph: `blocked`, `direct_answer`, `rewrite`, `fallback`, `answered` |
| Raw retrieval score | Top-1 cosine similarity from the original query |
| Expanded retrieval score | Top-1 score after original + rewritten queries, max-pooled per document |
| Citation provenance | Guarantee that every cited ID belongs to the answer evidence selected for this request |
| Claim coverage | Guarantee that every factual claim names at least one such ID; neither guarantee implies entailment |
| Candidate answer | Unvalidated structured reporter output; never rendered, never logged |
| Reason code | Enumerated fallback/refusal cause written to JSONL |
| Calibration set | Tuning split; may not be used for reported metrics |
| Held-out set | Reporting split; evaluated once, after thresholds freeze |

### References

* LangGraph Graph API — https://docs.langchain.com/oss/python/langgraph/use-graph-api
* LangChain ChatOpenAI structured output — https://docs.langchain.com/oss/python/integrations/chat/openai
* scikit-learn `TfidfVectorizer` — https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
* OWASP LLM Prompt Injection Prevention Cheat Sheet — https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html
* OWASP AI Agent Security Cheat Sheet — https://cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html

> **Implementation rule:** when forced to choose between one more feature and
> provable evaluation, fallback behaviour, or reproducibility — choose the latter.
