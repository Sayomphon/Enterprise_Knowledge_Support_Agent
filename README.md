# Enterprise Knowledge & Support Agent

> A reliability-first Thai-language RAG prototype that answers employee HR and
> Finance questions from a small internal corpus — and falls back with a
> recorded reason code whenever it cannot answer safely. Every LLM call sits
> between deterministic gates: nothing the model writes reaches an employee
> until a pure function has validated it against the evidence actually
> retrieved for that request.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Orchestration](https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C)
![Tests](https://img.shields.io/badge/tests-652%20passing%20%7C%205%20live%20skipped-3FB950)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

## TL;DR

* **One LangGraph pipeline, eleven nodes, at most two LLM calls.** Everything
  deciding *whether* a question is answerable is a deterministic, unit-tested
  pure function, and **no failing node reaches `END`** — every failure edge
  lands on `refuse` or `fallback`.
* **Similarity is not permission.** A query must resolve to a supported topic
  with an active policy behind it *before* any score can admit it: a
  maternity-leave question scores **0.2237**, above the answer threshold, and is
  still refused, because the corpus has no maternity-leave policy.
* **The model never writes what you read.** The Reporter returns structured
  claims into `candidate_answer`; a validator checks each claim's citations, its
  verbatim policy span and its numbers, and a renderer emits `[SOURCE-ID]`
  markup from validated ids only.
* **No evidence, no answer.** Zero authoritative evidence short-circuits to a
  fixed sentence with no model call, so a corpus gap cannot become a
  hallucination.
* **Every refusal is a logged reason code** — one of twenty-three — so an
  unanswered question becomes a corpus backlog rather than a bad answer nobody
  noticed.
* **Quality is measured, not asserted.** 25 calibration, 14 + 14 held-out, 56
  guardrail, 28 near-domain and 17 live answer cases, each labelled with what it
  can and cannot prove — including the one held-out case the strict gate still
  fails on.

Set it up: [Quick start](#2-quick-start). See it working: [Demo](#4-demo--cli-and-web-ui).

## Contents

| # | Section | What is in it |
|---|---|---|
| 1 | [Problem and scope](#1-problem-and-scope) | The failure this is engineered against, and how the assignment maps onto the code |
| 2 | [Quick start](#2-quick-start) | Install, test, ask a question, open the UI |
| 3 | [Architecture](#3-architecture) | The graph, the five routes, the LLM budget each is allowed |
| 4 | [Demo — CLI and web UI](#4-demo--cli-and-web-ui) | Three scenarios on both surfaces, plus the audit console |
| 5 | [Evaluation scorecard](#5-evaluation-scorecard) | Six fixtures, what each measures, and the recorded numbers |
| 6 | [Scope, and what a citation proves](#6-scope-and-what-a-citation-proves) | The topic gate, four guarantees, and the one deliberately absent |
| 7 | [Limitations and next steps](#7-limitations-and-next-steps) | What this does not do, and the trigger that would justify each fix |

Two companion documents carry the long form:
[`docs/demo.md`](docs/demo.md) — six worked examples on both surfaces plus the
console walkthrough — and [`docs/design.md`](docs/design.md) — design decisions,
authority model, security posture, logging and privacy, configuration, and the
repository map.

| Item | Value |
|---|---|
| Corpus | 8 mock Markdown documents — 5 policy, 3 chat transcripts (Thai) |
| Retrieval | Local character n-gram TF-IDF + cosine similarity (scikit-learn) |
| LLM | `langchain-openai` `ChatOpenAI`, model configurable via env |
| Entry points | CLI (`main.py`), Streamlit (`app.py` + `ui/`) |
| Telemetry | Append-only JSONL, no external infrastructure |
| Tests | 652 `unittest` cases, fully offline, no API key required (5 are an opt-in live smoke test that skips itself) |

---

## 1. Problem and scope

Employees cannot find internal HR and Finance answers, because the knowledge is
split between formal policy documents and messy chat threads where colleagues
explain the same rules in slang and typos. A naive RAG bot over that corpus has
one dominant failure mode: it answers confidently from the *nearest* document
rather than the *governing* one — maternity leave from the sick-leave policy, an
unlisted expense from the process that only describes how to file a claim.

This prototype treats that failure as the thing to engineer against. Every
property in the [TL;DR](#tldr) is enforced in code and asserted by a test; the
numbers are in [the evaluation scorecard](#5-evaluation-scorecard), with their case counts
attached.

**What the assignment asked for, and where it is.** Requirements are summarised
in my own words; the source brief is confidential and is not reproduced here.

| Requirement | Where it lives | What proves it |
|---|---|---|
| **1. Data ingestion** — 5–10 mock documents mixing clear procedure with short, noisy chat (slang, typos, ambiguity) | [`data/docs/`](data/docs/) — 8 Markdown files, 5 `policy` + 3 `chat`, loaded by [`src/ingestion/loader.py`](src/ingestion/loader.py) | [`tests/test_loader.py`](tests/test_loader.py): all 8 load; a duplicate `source_id` or a mismatched `authority` fails the load |
| **2a. Retrieval and generation pipeline** | [`src/graph.py`](src/graph.py) — 11 nodes, 5 routes; retrieval in [`src/retrievers/`](src/retrievers/), generation in [`src/agents/reporter.py`](src/agents/reporter.py) | [`tests/test_graph.py`](tests/test_graph.py) asserts every route **and its LLM call count** |
| **2b. Guardrail / validation layer** — out-of-scope questions and injection declined politely | [`input_guardrail.py`](src/guardrails/input_guardrail.py) (18 named rules) and [`scope_validator.py`](src/guardrails/scope_validator.py); wording in [`src/fallback.py`](src/fallback.py) | Injection Block Rate **28/28** *and* Benign Pass Rate **28/28** — reported as a pair, always |
| **2c. Source attribution** — every answer names its documents | [`citation_validator.py`](src/guardrails/citation_validator.py) validates, [`answer_renderer.py`](src/answer_renderer.py) renders | Citation Provenance Validity Rate **30/30**; a fabricated id routes to fallback instead of being printed |
| **3. Evaluation and fallback** — below the threshold, reply with a prepared message and log the question | Thresholds in [`src/config.py`](src/config.py), texts in [`src/fallback.py`](src/fallback.py), writer in [`src/logging_utils.py`](src/logging_utils.py) | [`eval/run_eval.py`](eval/run_eval.py) gates, and the [audit console](#the-audit-console-requirement-3-made-inspectable) reading the log back as a corpus backlog |

One word in requirement 3 is deliberately not the word this system uses:
*confidence score*. The gate exists — it is the three-band router in
[the architecture section](#3-architecture) — but the quantity it compares is a cosine
similarity, and 0.22 is not a 22% chance the answer is right. It is called a
*retrieval similarity heuristic* everywhere it is shown. Same gate the brief
describes, a name that does not overclaim; the full argument is in
[`docs/design.md`](docs/design.md#requirement-traceability).

---

## 2. Quick start

**Prerequisites:** Python 3.11+, and nothing else — no database, no vector
store, no model download, no Docker.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --require-hashes -r requirements.lock
cp .env.example .env          # then fill OPENAI_API_KEY locally
python -m unittest discover -s tests
```

Two dependency files, one job each. [`requirements.txt`](requirements.txt) is
the human-facing list — nine direct pins, reviewed by eye.
[`requirements.lock`](requirements.lock) is what you install: the same nine plus
every transitive package, each carrying its `sha256`, so `--require-hashes`
makes pip refuse a distribution whose bytes moved since the tree was resolved.
Markers let one file serve 3.11 and 3.12 on any platform. Editing a pin means
regenerating the lock (`uv pip compile requirements.txt --universal
--generate-hashes --python-version 3.11 -o requirements.lock`), and
[`tests/test_requirements_lock.py`](tests/test_requirements_lock.py) fails the
suite if the two files drift apart.

```text
----------------------------------------------------------------------
Ran 652 tests in 2.962s

OK (skipped=5)
```

**The entire test suite, the whole retrieval stack, and the offline evaluation
harness run without a key** and open no network connection; the LLM boundary is
mocked at the agent module seam. The five skipped tests are the one exception —
a live smoke test that reaches a real provider and skips unless
`RUN_LIVE_SMOKE=1` is set with a credential
([`tests/live/test_live_smoke.py`](tests/live/test_live_smoke.py)). If the suite
does not pass, the environment is wrong — do not continue.

```bash
python main.py --check                      # corpus + credential readiness
python main.py "ขั้นตอนการเบิกค่าแท็กซี่หลังทำ OT ต้องทำอย่างไร"
streamlit run app.py                        # assistant on /, audit console on /console
```

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs exactly these
commands plus the offline gates on Python 3.11 and 3.12 for every push and pull
request — install from the lock under `--require-hashes`, `pip check`, the
suite, each gate as its own step, a CLI query and a Streamlit health check —
then audits the locked tree for advisories, publishes an SBOM, and scans the
full history for secrets. The live answer set is not in CI: it calls a real
provider and costs money.

Every one of those steps was also executed by hand against **clean virtualenvs
built from `requirements.lock` alone**, on 3.11 and 3.12 both (2026-08-22, at
586 tests), and installing the lock a second time returned an identical
`pip freeze` line for line — the reproducibility claim a lock is supposed to
carry, verified rather than asserted.

Windows PowerShell differs only in activate and copy
(`.\.venv\Scripts\Activate.ps1`, `Copy-Item .env.example .env`); full usage in
[`docs/demo.md`](docs/demo.md#running-it).

---

## 3. Architecture

One LangGraph pipeline, **eleven nodes**, exactly **two of which call an LLM**.
Two properties are meant to be readable from the picture alone: only two nodes
are orange, and **every dotted edge ends at `refuse` or `fallback`** — no
failing node reaches the end of the graph on its own.

```mermaid
flowchart TD
    Q(["START &middot; employee question"]):::term

    subgraph G1["1 &middot; Deterministic screen, before any model call"]
        IG["input_guardrail<br/>18 named regex rules"]:::det
    end

    subgraph G2["2 &middot; Retrieval and score routing, still no model call"]
        RO["retrieve_original<br/>character TF-IDF, top 3"]:::det
        VS["validate_scope<br/>supported topic"]:::det
    end

    subgraph G3["3 &middot; Medium band: deterministic first, model only if needed"]
        EX["expand_deterministic<br/>topic-alias variants, no LLM"]:::det
        RW["rewrite<br/>LLM &middot; candidates re-screened, intent checked"]:::llm
        RE["retrieve_expanded<br/>original + aliases + accepted rewrites"]:::det
    end

    subgraph G4["4 &middot; Grounded generation and answer contract"]
        SE["select_evidence<br/>policy authoritative, chat supplementary"]:::det
        RP["report<br/>LLM &middot; writes candidate_answer only"]:::llm
        VC["validate_citations<br/>claims, provenance, policy authority"]:::det
    end

    REF["refuse<br/>fixed refusal + JSONL log"]:::term
    FB["fallback<br/>fixed fallback + JSONL log"]:::term
    EN(["END"]):::term

    Q ==> IG
    IG ==>|pass| RO
    IG -. "blocked &middot; prompt_injection" .-> REF
    RO ==> VS
    VS ==>|"medium &middot; 0.10 &le; raw &lt; 0.19"| EX
    VS ==>|"high &middot; raw &ge; 0.19"| SE
    VS -. "raw &lt; 0.10 &middot; low_retrieval_score<br/>or unsupported_topic / ambiguous_topic" .-> FB
    EX ==>|"alias score &ge; 0.21 &middot; 0 rewrite calls"| SE
    EX ==>|"still below 0.21"| RW
    EX -. "retrieval_failure" .-> FB
    RW ==> RE
    RE ==>|"expanded &ge; 0.21"| SE
    RE -. "expanded &lt; 0.21<br/>rewrite_low_retrieval_score" .-> FB
    RW -. "rewrite_failure &middot; rewrite_rejected<br/>llm_not_configured" .-> FB
    SE ==>|"policy found"| RP
    SE -. "no_authoritative_evidence" .-> FB
    RP ==>|"candidate produced"| VC
    RP -. "llm_not_configured<br/>reporter_failure" .-> FB
    VC ==>|"valid &middot; answer + [SOURCE-ID] citations"| EN
    VC -. "missing_citation &middot; fabricated_citation<br/>invalid_answer_structure &middot; unsupported_numeric_claim" .-> FB
    REF -.-> EN
    FB -.-> EN

    classDef det fill:#DCE9FB,stroke:#1A56B0,color:#0B2A5B
    classDef llm fill:#FCE6C2,stroke:#B26A00,color:#4A2C00
    classDef term fill:#E4E7EC,stroke:#4B5364,color:#1F2430
    classDef band fill:#F7F8FA,stroke:#9AA1B0,color:#1F2430
    class G1,G2,G3,G4 band
```

**Legend.** Thick arrows are the happy path; dotted arrows are failure paths,
each labelled with the reason code it writes to the log. Blue nodes are
deterministic, orange call the LLM, and the labels are the exact `add_node`
names in [`src/graph.py`](src/graph.py) — the drawing is the graph, not an
illustration of it.

**The medium band is deterministic first.** A slang or typo question whose raw
score lands between the thresholds is expanded with the aliases of the topics
the scope gate already resolved — the corpus's own wording for what the employee
wrote informally — and searched again. The original query stays first in that
search and the retriever max-pools per document, so the expansion cannot score
worse than the raw retrieval it replaces. Only when that still misses is a model
asked for a rewrite: five of the six medium-band calibration cases are settled
by the alias catalog alone, where four used to buy a rewrite.

### The five routes and their LLM budget

Each row is asserted by a test, so the budget is an enforced property rather
than a description of the code.

| `route` | Set when | LLM calls | Test |
|---|---|---:|---|
| `blocked` | A guardrail rule, the length limit, or the type check rejected the input | 0 | `route_1_injection_is_refused…` |
| `fallback` (low band) | Raw score below the rewrite floor | 0 | `route_2_low_score_falls_back…` |
| `fallback` (unsupported topic) | In-domain question with no policy behind it | 0 | `unsupported_topic_falls_back…` |
| `fallback` (ambiguous topic) | Two supported topics touched, neither named | 0 | `under_specified_question_falls_back…` |
| `direct_answer` | The high band reached the reporter | 1 — Reporter | `route_3_high_score_answers…` |
| `direct_answer` (alias-recovered) | The medium band cleared the final threshold on aliases alone | 1 — Reporter | `alias_expansion_answers_the_medium_band…` |
| `rewrite` | The medium band still needed a model rewrite | 2 — Rewriter + Reporter | `route_4_medium_score_rewrites_then_answers` |
| `answered` | `validate_citations` accepted every claim | 1 or 2 | route tests above |

All in [`tests/test_graph.py`](tests/test_graph.py). The zero-call rows assert
that the agent seam was never even *constructed*. `direct_answer` and `rewrite`
are intermediate: they record *how* a request reached generation, and a request
that finishes ends as `answered` regardless. A step-by-step walk of each node,
and the same graph drawn as a request lifecycle, are in
[`docs/design.md`](docs/design.md#one-request-end-to-end).

---

## 4. Demo — CLI and web UI

The same `build_graph()` pipeline serves both surfaces: the CLI is the auditable
artefact a reviewer can copy and re-run, the Streamlit app is what an employee
sees. All six worked examples and the full console tour are in
[`docs/demo.md`](docs/demo.md#the-six-examples); this is enough to see the shape.

**Three routes in one terminal session** — an answered question, a blocked
injection, and an in-domain question the corpus does not cover. The last two run
with `OPENAI_API_KEY` unset and make no provider call at all:

![CLI transcript: a cited Thai answer with FIN-001 and FIN-002 citations, then a blocked prompt injection, then an unsupported-topic fallback scoring 0.2237](docs/screenshots/cli_demo.png)

**A typo, retrieved by character n-grams.** `ลาพักรอ้น 2 วันกดตรงไหนอะ` — the tone
mark is transposed. There is no spell checker and no embedding model; the
character n-gram index still ranks `CHAT-002` first, because the same
misspelling appears in the chat transcript itself. The noisy half of the corpus
is what makes informal spellings retrievable, and the claim is measured rather
than anecdotal ([the evaluation scorecard](#5-evaluation-scorecard)).

**Maternity leave: refused at 0.2237, above the answer threshold.** The case
that defines the system. The question shares most of its wording with the
sick-leave policy and scores comfortably above the 0.19 direct threshold; the
pre-remediation pipeline answered it from `HR-002`. It now falls back with
`unsupported_topic`, before any LLM call, because the scope gate asks what the
score cannot: *does the corpus have a policy for this topic?* The injection, in
turn, never reaches retrieval — the screen runs first precisely so an attack
costs nothing.

**The same three, in the web UI.** Every claim carries its `[SOURCE-ID]` as a
chip, and the panel underneath lists each cited document with its retrieval
similarity and an expander holding the raw text — so a reader can check a
sentence against the policy it came from without leaving the page:

![Streamlit answer card: five claims each ending in a FIN-001 or FIN-002 chip, an evidence strength pill, and two source cards with their retrieval scores](docs/screenshots/ui_02_answered.png)

| Blocked — no score, no rule, no reason code | Fallback — the topics it *can* answer, and a recorded question |
|---|---|
| ![Streamlit refusal card for the injection attempt, showing only a fixed sentence and the note that guardrail diagnostics are administrator-only](docs/screenshots/ui_04_blocked.png) | ![Streamlit fallback card naming the five supported topics, telling the employee the question was recorded, and offering two covered topics as next steps](docs/screenshots/ui_05_fallback.png) |

The employee sees the same fixed sentence for an out-of-domain question and an
in-domain one the corpus does not cover: **identical to the employee,
distinguishable to the operator.**

### The audit console: requirement 3 made inspectable

Requirement 3 has two halves. The fallback message is above; *"log the question
for later analysis"* is invisible in the employee view, and a JSONL file nobody
reads is not analysis. The console is the second half — the same five routes
from the operator's side, across every session in the browser tab:

![Ops console overview: four stat cards (answered, fallback, blocked, average latency), a threshold axis plotting three real request scores against the calibrated thresholds, and a triage list grouping unanswered questions by reason code and family](docs/screenshots/ui_06_console_overview.png)

The axis is the part worth reading twice. It plots real request scores against
the thresholds from [`src/config.py`](src/config.py) — the UI defines none of
its own — and the maternity-leave request sits at **0.2237, right of the 0.19
line, and still resolves to fallback**. Beside it, unanswered questions are
grouped by reason code and tagged with its *family* — knowledge gap, service,
input rejected, validation — which turns the fallback log into a ranked list of
documents the corpus is missing.

Selecting a request shows its node-level trace; the other sections carry the
JSONL sink across sessions, the recorded evaluation run parsed out of
[`eval/BASELINE.md`](eval/BASELINE.md), the loaded index and the frozen runtime
configuration — each exportable as JSONL, CSV or Markdown, and all five walked
through in [`docs/demo.md`](docs/demo.md#7--the-audit-console-where-requirement-3-becomes-inspectable).

---

## 5. Evaluation scorecard

Six fixtures with different jobs. Five are offline, free and deterministic; the
sixth calls a real provider and is the only one that can fail because an
*answer* was wrong.

| Fixture | Cases | Job | `--strict` |
|---|---:|---|---|
| [`retrieval_calibration.json`](eval/retrieval_calibration.json) | 25 | **Tuning only** — thresholds, n-grams, alias catalog | exit `0` |
| [`retrieval_heldout.json`](eval/retrieval_heldout.json) | 14 | **Read during an audit** — now a regression fixture | exit `1` (known miss) |
| [`retrieval_heldout_v2.json`](eval/retrieval_heldout_v2.json) | 14 | **Reporting only** — written blind, run once after the freeze | exit `0` |
| [`guardrail_cases.json`](eval/guardrail_cases.json) | 56 | 28 attacks / 28 benign lookalikes | exit `0` |
| [`near_domain_cases.json`](eval/near_domain_cases.json) | 28 | 20 near-domain hard negatives / 8 benign twins | exit `0` |
| [`citation_cases.json`](eval/citation_cases.json) + [`rewrite_cases.json`](eval/rewrite_cases.json) | 30 + 20 | Answer-contract and rewrite-validator verdicts | exit `0` |
| [`answer_cases.json`](eval/answer_cases.json) | 17 × 3 runs | **Live** — fact anchors drawn from the corpus | reported, see below |

```bash
python eval/run_eval.py --set calibration --strict   # also: near_domain, guardrail, contracts
python eval/run_eval.py --set heldout_v2 --strict    # reporting only, run last
python eval/run_eval.py --set answers --live         # costs money, asks first
```

Without `--strict` the runner reports and exits `0` even on a contradicted
label, which is what a threshold sweep needs; with it, the same run is a gate.
Offline figures below are transcribed from [`eval/RESULTS.md`](eval/RESULTS.md)
and the live ones from [`eval/ANSWER_RESULTS.md`](eval/ANSWER_RESULTS.md), both
raw committed stdout. [`eval/BASELINE.md`](eval/BASELINE.md) is a run *history*,
and its last block is what the console's Evaluation panel parses — so the panel
and this table quote the same run.

### Retrieval and routing

| Metric | Calibration (25) | Held-out (14, read) | Held-out v2 (14, blind) | Near-domain (28) |
|---|---:|---:|---:|---:|
| Retrieval Hit@3 | 16/16 | 7/7 | 7/7 | 8/8 |
| Retrieval Recall@1 | 15/16 | 7/7 | 7/7 | 7/8 |
| Retrieval MRR | 0.969 | 1.000 | 1.000 | 0.938 |
| Answer-route Selection Precision | 16/16 | 6/6 | 7/7 | 8/8 |
| Answer-route Coverage | 16/16 | **6/7** | 7/7 | 8/8 |
| False Fallback Rate | 0/16 | 1/7 | 0/7 | 0/8 |
| OOD Fallback Accuracy | 4/4 | 3/3 | 3/3 | n/a |
| Unsupported In-domain Fallback Accuracy | 5/5 | 4/4 | 4/4 | 20/20 |
| Authoritative Evidence Coverage Rate | 16/16 | 6/6 | 7/7 | 8/8 |
| Rewrite Recovery Rate | 6/6 | 1/2 | 1/1 | 2/2 |

There are two held-out splits because the first was read during a code audit,
and a reporting set that has been looked at is a regression fixture, not
generalisation evidence. `retrieval_heldout_v2.json` was written afterwards with
the same category mix, while every threshold was frozen, and run once. Its clean
sheet rests on seven answerable cases whose wording came from reading the corpus
— the direction that flatters recall — so it supports only the narrow claim it
can carry: on 14 cases never used for tuning, nothing out-of-domain and no
in-domain hard negative was answered.

*Selection Precision* scores route selection and retrieval, not answer
correctness — no offline metric reads an answer. *OOD* counts only out-of-domain
questions; in-domain topics with no policy have their own row.

**Under injected typos**, with `--perturb` (fixed seed; one, two and three
single-character Thai edits per correctly-spelled answerable query):

| Injected typos | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| Hit@3, calibration (7) | 7/7 | 7/7 | 7/7 | 7/7 |
| Hit@3, near-domain (7) | 7/7 | 7/7 | 7/7 | 7/7 |
| Hit@3, held-out (4) | 4/4 | 4/4 | 4/4 | 4/4 |
| routed `answered`, calibration | 7/7 | 7/7 | 7/7 | 6/7 |
| routed `answered`, near-domain | 7/7 | 3/7 | 7/7 | 5/7 |

Retrieval does not degrade at all; the **route** does, on cases whose score
falls under a threshold while the right document is still in the top three — a
fallback preferred to an answer the pipeline is no longer confident in. The
non-monotonic near-domain row is the honest shape of one probe per case per
level: this shows the index degrades gracefully, not by how much.

### Guardrail and answer contract

| Fixture | Metric | Result |
|---|---|---:|
| `guardrail_cases.json` | Injection Block Rate | 28/28 |
| `guardrail_cases.json` | Benign Pass Rate | 28/28 |
| `citation_cases.json` | Citation Provenance Validity Rate | 30/30 |
| `citation_cases.json` | Claim Source Coverage Rate | 27/31 |
| `citation_cases.json` | Invalid Candidate Leakage Rate | 0/24 |
| `rewrite_cases.json` | Rewrite Intent Preservation Rate | 20/20 |

Block rate and benign pass rate are always reported as a pair: a regex that
raises one by lowering the other is a regression, not an improvement. The
fixture grew from 21/21 to 28/28 when the paraphrase, role-play and leetspeak
rules landed, and it grew on **both** sides — each new rule ships with the
benign lookalike that shares its vocabulary, including enterprise wording with
digits in it (`เบิกค่าแท็กซี่ 5000 บาท`) that a whole-string leetspeak fold would
have destroyed.

`Claim Source Coverage Rate: 27/31` describes a fixture that deliberately mixes
grounded and ungrounded claims — it is not live Reporter behaviour. Eleven of
the 30 citation cases exist to fail: seven test the claim-span rule and four the
numeric rule against the quote rather than the whole document.

### Live answer quality

17 questions with fact anchors drawn from the corpus, run 3 times each against
the real pipeline on `gpt-5-mini` — 51 invocations, measured before the P0
claim-span contract, after it, and again after P1:

| Metric | Before P0 | After P0 | After P1 | Reading |
|---|---:|---:|---:|---|
| Fact Recall | 52/57 | **46/57** | 48/57 | Of the six instances lost to P0, **one** went to the span rule and five to the medium-band rewrite boundary |
| Fact-Citation Alignment | 52/52 | **46/46** | 48/48 | Every stated fact was cited to a document that carries it |
| Alien Number Rate | 0/51 | 0/51 | 0/51 | No run introduced a figure absent from its evidence |
| Forbidden Fact Rate | 0/51 | 0/51 | 0/51 | The deliberately-unconfirmed kiosk case was never claimed as settled |
| Correct Refusal Rate | 3/3 | 3/3 | 3/3 | All three refused at the scope gate, 0.0s, zero provider calls |
| Route Stability | 16/17 | **15/17** | 15/17 | Both unstable cases are the documented medium-band non-determinism |
| Latency p50 / p95 | 6.6s / 27.7s | **10.9s / 29.9s** | 8.6s / **58.6s** | The tail, not the median: one run reached 94.4s — see [the limitations](#7-limitations-and-next-steps) |

Attributing the whole recall drop to the newest change would have been the
convenient reading and the wrong one; the run-by-run attribution is in
[`eval/ANSWER_RESULTS.md`](eval/ANSWER_RESULTS.md). Checking is deterministic
against the corpus, not an LLM judge — every anchor is verified to exist in the
document it names before the run starts. **n is small and this is one model on
one day**: regression evidence, not a statistical claim.

### Reading these honestly

Precision is always reported beside Coverage, because a pipeline that answers
almost nothing scores perfect precision. Held-out coverage is 6/7: `ho_noisy_03`
reaches 0.1986 after expansion against a 0.21 threshold and falls back. It was
not tuned for, and must not be — the same threshold move would readmit the
in-domain cases the scope gate exists to refuse. Every count here is small, and
"unseen" means "not used for tuning", not "never read". Full caveats:
[`eval/RESULTS.md`](eval/RESULTS.md#what-these-numbers-do-not-measure),
[`eval/ANSWER_RESULTS.md`](eval/ANSWER_RESULTS.md#what-this-file-does-not-prove).

---

## 6. Scope, and what a citation proves

The corpus covers five topics, and a query must resolve to one of them in a
bounded alias catalog **before** any similarity score may admit it.

| Supported topic | Authoritative documents |
|---|---|
| `reimbursement_process` | `FIN-001`, `FIN-002` |
| `receipt_policy` | `FIN-002` |
| `annual_leave` | `HR-001` |
| `sick_leave` | `HR-002` |
| `work_from_home` | `HR-003` |

The gate also names the in-domain topics the corpus has **no** policy for, so
they are refused explicitly rather than answered from whatever ranked first:

```text
maternity_leave, ordination_leave, marriage_leave, resignation,
payroll_date, salary, bonus, medical_reimbursement,
meal_reimbursement, hotel_reimbursement, phone_reimbursement,
internet_reimbursement, office_equipment, training_expense,
per_diem, business_leave, fuel_mileage
```

A question touching two supported topics without naming either — `ลาได้กี่วัน`,
"how many days of leave can I take" — is refused as `ambiguous_topic` and asked
to name the leave or expense type, because the corpus probably does hold that
answer. **This is a closed alias catalog for eight documents, not an intent
classifier**, and two of its failure modes are closed deterministically: a topic
is kept only when it scores within `SCOPE_TOPIC_MARGIN` of the winner, and a
*"can I claim X"* question resolving the reimbursement process **alone** is
refused as `uncovered_expense_item` unless X is in `COVERED_EXPENSE_ITEMS`, the
allowlist drawn from `FIN-001`'s own *"ค่าเดินทางที่เบิกได้"* section. Both are
configuration bound to *this* corpus; the mechanics are in
[`docs/design.md`](docs/design.md#design-decisions).

Five properties are often conflated. This system provides the first four and
does **not** provide the last:

| Property | Guaranteed? | Meaning |
|---|---|---|
| **Citation provenance** | Yes | Every cited ID belongs to the answer evidence selected for *this* request. A fabricated or stale ID routes to fallback. |
| **Claim coverage** | Yes | Every factual claim names at least one such ID. An uncited claim routes to fallback. |
| **Claim span** | Yes | Every claim carries an `evidence_quote` copied verbatim from a **policy** it cites. A quote in no cited policy — including one lifted from a chat transcript — routes to fallback as `unsupported_claim_span`. |
| **Numeric consistency** | Yes | Every number and clock time a claim states appears **inside that quote**. "ลาได้ 15 วัน" quoting the policy that says 10 routes to fallback as `unsupported_numeric_claim`. |
| **Entailment** | **No** | Nothing in the pipeline checks that the claim follows from the cited document. |

The span rule is what separates provenance from support. Provenance proves a
cited ID was really in this request's evidence; it cannot see whether that
document *says* what the claim says, which is how `"Annual leave is 30 days"`
citing the sick-leave policy used to pass. The span proves the words are in the
cited policy — **never that the claim is the right reading of them**: a quote
lifted out of its condition still passes, and
`tests/test_citations.py::TestEvidenceSpanRule` asserts that limit by name.
A figure the *employee* wrote is deliberately no longer evidence. Claim
entailment needs an NLI model or an LLM-as-judge pass; both are production next
steps and neither is implemented here.

**Indirect injection.** The query screen is the visible defence and it is the
wrong one for this attack: an instruction arriving inside a *retrieved document*
has already passed it. Four layers stand behind it — an ingestion screen that
redacts instruction-shaped lines from chat transcripts and fails the load when a
policy carries one, `json.dumps` encoding that stops a document closing its own
envelope, a reporter prompt that declares evidence untrusted, and the answer
contract that refuses the obeyed answer. Only the last is a control rather than
a request, and it is the one that holds:
[`tests/test_graph.py::TestIndirectInjection`](tests/test_graph.py) drives all
four end to end with a poisoned corpus and a reporter that obeys it. What each
layer cannot do is tabulated in
[`docs/design.md`](docs/design.md#security-posture-and-where-it-stops).

---

## 7. Limitations and next steps

* **Retrieval is character TF-IDF over whole documents.** No embeddings, no
  chunking, no reranking. It works because the corpus is eight documents, and
  the similarity it produces is surface overlap rather than a calibrated
  probability of answer correctness.
* **The medium band is deterministic in most cases, not all.** A query alias
  expansion cannot lift still buys a rewrite, and on `gpt-5-mini` that rewrite
  frequently exceeds its 10s budget and degrades to the expanded original.
* **The scope catalog is a closed alias list**, not an intent classifier, and
  `COVERED_EXPENSE_ITEMS` has to be edited when a policy grants a new item. An
  eligibility question that also resolves the receipt topic skips that gate;
  what stops it then is the claim-span rule.
* **The regex guardrail is a prototype safeguard, not defence-in-depth.** It is
  a finite, precision-first pattern catalog: novel phrasings will pass it, and
  its block rate is a statement about 28 curated attacks, not about attacks
  outside that set. A homoglyph attack written entirely in Unicode lookalike
  *letters* is out of its reach.
* **Numbers written as Thai words are not read as numbers.** The contract folds
  Thai, Arabic-Indic and fullwidth digits to ASCII, so `๑๐ วัน` is compared as
  `10 วัน`; `สิบวัน` is not. A parser for Thai number words is error-prone enough
  that its false rejections would cost more than the gap.
* **No authentication, no RBAC, no per-document ACLs.** `ENABLE_OPS_VIEW` is a
  demo flag, not authorization. The sink is created `0600` inside a `0700`
  directory, rolls over at `LOG_MAX_BYTES`, and masks four identifier shapes in
  the query text — a bounded safeguard, not a PII classifier: a name or a health
  detail in the wording of a sick-leave question still reaches the file.
* **Answer coverage is deliberately traded for support.** Requiring a verbatim
  policy span behind every claim means a reporter that paraphrases instead of
  copying loses the request to fallback — the intended direction, re-measured on
  the live set at a cost of one fact-instance out of 57.
* **The request deadline is a ceiling, not a cancellation — and retries multiply
  it.** `REQUEST_DEADLINE_SECONDS` (45s) is shared by both LLM boundaries and a
  boundary reached with nothing left is skipped as `request_deadline_exceeded`
  rather than started. What it does not do is interrupt a call in flight or
  divide the remainder by the attempt count, so with `LLM_MAX_RETRIES=2` the P1
  live run measured **94.4s** on one answered request. The bound is "a request
  is bounded", not "a request finishes within 45 seconds".
* **The Streamlit layer is only partly under test.** Its pure formatters, export
  payloads and runtime bookkeeping are asserted in
  [`tests/test_ui_formatting.py`](tests/test_ui_formatting.py) and its siblings;
  `st.*` rendering, CSS and layout are verified by walking both pages by hand.
* **The supply chain is pinned and checked, not trusted.**
  `requirements.lock` pins 78 packages against 2,144 distribution digests, so an
  install gets the reviewed bytes or fails; CI audits that tree, publishes an
  SBOM, and scans the whole history with a digest-verified `gitleaks`. Measured
  2026-08-22: no known advisory in the locked tree, no secret in 53 commits.
  What none of that proves is that a package is *trustworthy* — only that it has
  not changed since it was reviewed.
* **The corpus is eight short mock documents** written for this exercise. None
  of this has met a real policy PDF, a real chat export, a contradicting
  document, or a version history.

The remaining limitations — telemetry durability, the console's scope, lazy
credential validation, structured-output failure modes, output screening — are
in [`docs/design.md`](docs/design.md#remaining-known-limitations). Every
exclusion is a clean seam rather than a stub, so the production path is additive
instead of a rewrite: the `Retriever` protocol, the scope gate, the answer
validator, `config.py`, `logging_utils` and the guardrail module each carry a
documented next step, two specified in enough detail to build in
[Production sketches](docs/design.md#production-sketches--designed-not-built).

### When to escalate retrieval

"BM25 → hybrid → embeddings" is the advice everyone gives and almost nobody
attaches a trigger to. Each step has a measurable one here — Recall@1 falling
while Hit@3 holds, a paraphrase blind set losing to lexical retrieval, an index
that no longer fits in memory — and they are tabulated in
[`docs/design.md`](docs/design.md#when-to-escalate-retrieval). Not one is
justified today: Hit@3 is 16/16 on calibration and 7/7 on both held-out splits,
and Recall@1 misses once on calibration. Ranking is not the problem this corpus
has.

---

Licensed under the [Apache License 2.0](LICENSE).
