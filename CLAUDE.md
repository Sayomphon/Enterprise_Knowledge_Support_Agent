# CLAUDE.md — Claude Code Operating Guide

> **Read `AGENTS.md` first and treat it as authoritative.** It defines the
> architecture, invariants, coding standards, security rules, testing policy and
> Definition of Done for this repository, and applies to every agent equally.
>
> This file adds only what is specific to running Claude Code here. If a rule is
> in both files, `AGENTS.md` wins — and the duplicate should be deleted from here.

---

## 1. Orientation — read these first

Load in this order before touching code; stop once you have what the task needs.

```text
AGENTS.md                 # contract: invariants, standards, DoD
src/schemas.py            # domain types + PipelineState contract
src/graph.py              # nodes, edges, routing functions
src/config.py             # calibrated thresholds and runtime settings
tests/test_graph.py       # executable specification of the routes
```

Do **not** pull into context: `logs/*.jsonl`, `.env`, `.venv/`, or the full
`data/docs/` corpus. Read individual corpus files only when the task concerns
them, and never read `.env` — use `.env.example` to learn the variable names.

---

## 2. Working loop

**Explore → Plan → Implement → Verify → Report.**

* **Explore.** Search and read the affected modules and their tests before
  proposing anything. Prefer targeted search over reading whole directories.
* **Plan.** For any change touching more than one file, changing
  `PipelineState`, adding a graph node, or altering a threshold: present the
  plan and wait for approval. Use plan mode when available.
* **Track.** Keep a visible todo list for multi-step tasks and update it as you
  go, so the next step is always inspectable.
* **Implement.** Smallest coherent diff. Follow the surrounding style. Docstrings
  and English-only code are mandatory (`AGENTS.md` §6).
* **Verify.** Run `python -m unittest discover -s tests -v` and paste the real
  result. Re-run the relevant `eval/run_eval.py` set when retrieval, thresholds,
  or the guardrail changed.
* **Report.** What changed, why, which invariants it touches, what is still
  unverified. If you did not run something, say so.

**Delegate to subagents** for read-heavy exploration ("find everywhere the score
thresholds are referenced") to keep the main context clean. Keep implementation
and verification in the main thread so the diff and its test output stay together.

---

## 3. Permissions

| Allowed without asking | Ask first | Never |
|---|---|---|
| Read source, tests, `eval/*.json`, `.env.example` | Adding or upgrading a dependency | Reading or printing `.env` |
| Run the test suite and `eval/run_eval.py` | Changing `PipelineState` or a public contract | `git push`, `git commit --amend`, force operations, history rewrites |
| Edit files under `src/`, `tests/`, `eval/`, `data/docs/` | Adding a LangGraph node or edge | Committing unless explicitly asked |
| Run the CLI or Streamlit locally | Editing calibrated thresholds in `config.py` / `.env.example` | Deleting or rewriting `logs/*.jsonl`, or `rm -rf` anywhere |
| Create scratch files under a temp dir | Any live-LLM call that spends API budget | Weakening a rule in `AGENTS.md` §4 to make a test pass |

Live LLM calls cost money. Default to mocked tests; run a live smoke test only
when asked, and only for a single query.

---

## 4. Task recipes

**Add a graph node**
`schemas.py` (state fields as `NotRequired`) → node function with a Google-style
docstring in the right package → wire in `graph.py` (both the happy path *and*
its failure edge — no node may reach `END` on failure) → add a route test in
`tests/test_graph.py` → update the architecture diagrams in `AGENTS.md` §2 and
the README.

**Add an injection pattern**
Add the bounded-quantifier regex to `input_guardrail.py` with a comment naming
the attack shape → add one attack case *and* one benign-lookalike case to
`eval/guardrail_cases.json` and `tests/test_guardrail.py` → re-run the guardrail
eval and report block rate and benign pass rate together. A pattern that raises
block rate while lowering benign pass rate is a regression.

**Change a threshold**
Never edit the number alone. Re-run the calibration sweep with cached rewrites,
record the resulting distribution, update the constant plus its provenance
comment and `.env.example`, then run the held-out set once and report both.

**Add a retriever**
Implement the `Retriever` protocol as a new class. Do not add `if backend == …`
branches to the graph or the router.

**Fix a bug**
Write the failing test first, then the fix, then show both runs.

---

## 5. Output style in this repository

* Explanations to the user may be in Thai; **everything written into files is
  English** — code, docstrings, comments, tests, commit messages, README.
  Thai appears only in `data/docs/**`, user-facing message constants, prompt
  templates, and `eval/*.json` fixtures.
* Show diffs and the reasoning behind non-obvious choices, not a re-listing of
  code you just wrote.
* No emoji in source files. No decorative headers in code.
* When a request conflicts with `AGENTS.md` §4 (invariants) or §1 (non-goals) —
  for example "just add a vector DB" or "hardcode 0.25 and move on" — say so
  directly, explain the risk in one or two sentences, and offer the compliant
  alternative. Do not silently comply, and do not silently refuse.

---

## 6. Reality checks before you say "done"

```text
[ ] Tests were actually executed and the output is in the response
[ ] No invariant in AGENTS.md §4 was weakened
[ ] Docstrings and comments explain why, not what
[ ] No secrets, prompts, or provider payloads reach logs or the UI
[ ] Any new failure path routes to fallback with a reason code, not to END
[ ] Metrics quoted come from a real run — calibration and held-out kept separate
[ ] Unrelated files are untouched
```

Uncertainty is reported, not smoothed over. A precise "this path is untested" is
worth more here than a confident summary.
