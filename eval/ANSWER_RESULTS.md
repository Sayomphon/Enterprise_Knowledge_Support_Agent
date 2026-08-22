# Live answer-quality results

Raw stdout of `python eval/run_eval.py --set answers --live --runs 3`, captured
verbatim. This is the only artifact in this repository produced by a real
provider call, and therefore the only one that can fail because an **answer** was
wrong rather than because a route was.

Everything in [`eval/RESULTS.md`](RESULTS.md) scores routing, retrieval, or a
validator verdict through a stub reporter. That file states plainly that nothing
in it measures generation quality. This file exists to stop that being the whole
story.

## Run metadata

| Item | Value |
|---|---|
| Date run | 2026-08-21 |
| Model | `gpt-5-mini`, `TEMPERATURE=0` |
| Cases | 17, across four categories |
| Runs per case | 3 (`--runs 3`), so route stability is measured rather than assumed |
| Pipeline invocations | 51 |
| Provider calls | 42 — the 9 invocations of the three `correct_refusal` cases were stopped by the deterministic scope gate and cost nothing |
| Approximate cost | under USD 0.10 at `gpt-5-mini` list pricing for this corpus size |
| Thresholds | committed defaults, `FINAL_ANSWER_THRESHOLD=0.21` |
| Seams injected | **none.** This runs `build_graph()` exactly as the CLI and Streamlit do |
| Raw transcript | [`eval/answer_transcript.json`](answer_transcript.json) — every answer, claim, citation and latency, so the checks can be re-scored without paying again |

## How a case is scored

Deterministic fact anchors, not an LLM judge. Every anchor is a regular
expression whose match is verified **against the corpus document it names** at
fixture load time, so the set cannot demand a fact these eight documents do not
carry — a fixture bug would otherwise be reported as a model failure, and the
obvious "fix" would be to weaken the model's job.

| Metric | What it counts |
|---|---|
| Fact Recall | Required corpus facts that appeared in the rendered answer |
| Fact-Citation Alignment | Found facts stated in a claim citing a document that actually contains that fact. A right number under the wrong citation fails here |
| Alien Number Rate | Runs whose answer contained a number present in neither the evidence nor the question |
| Forbidden Fact Rate | Runs asserting something the corpus deliberately leaves open |
| Correct Refusal Rate | Cases the corpus cannot answer that were refused rather than answered |
| Route Stability | Cases whose route was identical across all three runs |

## Results

```text
[live] 17 cases x 3 run(s) = 51 pipeline invocations, at most 102 LLM calls on gpt-5-mini (cases refused by the deterministic gates cost none)
== answer set: 17 cases x 3 run(s), model=gpt-5-mini ==
[live] transcript written to /private/tmp/claude-502/-Users-sayomphon-kha-Downloads-AA-ZZ-Enterprise-Knowledge-Support-Agent/667fcff5-19ae-4214-989f-393ab9be7f60/scratchpad/answer_transcript.json
  ans_normal_01    normal          routes=answered               facts=3/3 aligned=3 forbidden=0 alien=0
      run 1: route=answered 14.3s cites=['HR-001']
      run 2: route=answered 15.0s cites=['HR-001']
      run 3: route=answered 20.3s cites=['HR-001']
  ans_normal_02    normal          routes=answered               facts=3/3 aligned=3 forbidden=0 alien=0
      run 1: route=answered 8.0s cites=['HR-001']
      run 2: route=answered 11.3s cites=['HR-001']
      run 3: route=answered 6.9s cites=['HR-001']
  ans_normal_03    normal          routes=answered               facts=3/3 aligned=3 forbidden=0 alien=0
      run 1: route=answered 5.3s cites=['HR-001']
      run 2: route=answered 6.2s cites=['CHAT-002', 'HR-001']
      run 3: route=answered 4.2s cites=['HR-001']
  ans_normal_04    normal          routes=answered               facts=6/6 aligned=6 forbidden=0 alien=0
      run 1: route=answered 8.2s cites=['HR-001']
      run 2: route=answered 8.9s cites=['HR-001']
      run 3: route=answered 5.8s cites=['HR-001']
  ans_normal_05    normal          routes=answered               facts=3/3 aligned=3 forbidden=0 alien=0
      run 1: route=answered 5.0s cites=['HR-002']
      run 2: route=answered 4.9s cites=['HR-002']
      run 3: route=answered 5.7s cites=['HR-002']
  ans_normal_06    normal          routes=answered               facts=3/3 aligned=3 forbidden=0 alien=0
      run 1: route=answered 5.7s cites=['HR-002']
      run 2: route=answered 3.9s cites=['HR-002']
      run 3: route=answered 4.6s cites=['HR-002']
  ans_normal_07    normal          routes=answered               facts=3/3 aligned=3 forbidden=0 alien=0
      run 1: route=answered 4.6s cites=['HR-003']
      run 2: route=answered 5.7s cites=['HR-003']
      run 3: route=answered 4.9s cites=['HR-003']
  ans_normal_08    normal          routes=answered               facts=3/6 aligned=3 forbidden=0 alien=0
      run 1: route=answered 4.9s cites=['FIN-001']
        MISSING FACT: claim channel
      run 2: route=answered 8.3s cites=['FIN-001']
        MISSING FACT: claim channel
      run 3: route=answered 6.1s cites=['FIN-001']
        MISSING FACT: claim channel
  ans_multi_01     multi_condition routes=answered               facts=6/6 aligned=6 forbidden=0 alien=0
      run 1: route=answered 17.4s cites=['FIN-001', 'FIN-002']
      run 2: route=answered 22.3s cites=['FIN-001', 'FIN-002']
      run 3: route=answered 36.8s cites=['CHAT-001', 'FIN-001', 'FIN-002']
  ans_multi_02     multi_condition routes=answered               facts=6/6 aligned=6 forbidden=0 alien=0
      run 1: route=answered 11.6s cites=['FIN-001', 'FIN-002']
      run 2: route=answered 9.6s cites=['FIN-001', 'FIN-002']
      run 3: route=answered 9.8s cites=['FIN-001', 'FIN-002']
  ans_multi_03     multi_condition routes=answered               facts=3/3 aligned=3 forbidden=0 alien=0
      run 1: route=answered 6.6s cites=['FIN-002']
      run 2: route=answered 6.1s cites=['FIN-002']
      run 3: route=answered 8.8s cites=['FIN-002']
  ans_multi_04     multi_condition routes=answered               facts=6/6 aligned=6 forbidden=0 alien=0
      run 1: route=answered 8.7s cites=['CHAT-001', 'FIN-001', 'FIN-002']
      run 2: route=answered 7.7s cites=['CHAT-001', 'FIN-002']
      run 3: route=answered 10.5s cites=['FIN-001', 'FIN-002']
  ans_chat_01      ambiguous_chat  routes=answered               facts=3/3 aligned=3 forbidden=0 alien=0
      run 1: route=answered 29.4s cites=['CHAT-003', 'FIN-001', 'FIN-002']
      run 2: route=answered 15.2s cites=['CHAT-003', 'FIN-001', 'FIN-002']
      run 3: route=answered 12.3s cites=['FIN-001', 'FIN-002']
  ans_chat_02      ambiguous_chat  routes=answered/fallback      facts=1/3 aligned=1 forbidden=0 alien=0 UNSTABLE ROUTE
      run 1: route=fallback 12.0s cites=[] reason=rewrite_low_retrieval_score
        MISSING FACT: filing deadline
      run 2: route=fallback 27.7s cites=[] reason=rewrite_low_retrieval_score
        MISSING FACT: filing deadline
      run 3: route=answered 27.1s cites=['CHAT-001', 'CHAT-003', 'FIN-001']
  ans_refusal_01   correct_refusal routes=fallback               facts=0/0 aligned=0 forbidden=0 alien=0
      run 1: route=fallback 0.0s cites=[] reason=unsupported_topic
      run 2: route=fallback 0.0s cites=[] reason=unsupported_topic
      run 3: route=fallback 0.0s cites=[] reason=unsupported_topic
  ans_refusal_02   correct_refusal routes=fallback               facts=0/0 aligned=0 forbidden=0 alien=0
      run 1: route=fallback 0.0s cites=[] reason=unsupported_topic
      run 2: route=fallback 0.0s cites=[] reason=unsupported_topic
      run 3: route=fallback 0.0s cites=[] reason=unsupported_topic
  ans_refusal_03   correct_refusal routes=fallback               facts=0/0 aligned=0 forbidden=0 alien=0
      run 1: route=fallback 0.0s cites=[] reason=unsupported_topic
      run 2: route=fallback 0.0s cites=[] reason=unsupported_topic
      run 3: route=fallback 0.0s cites=[] reason=unsupported_topic
-- metrics --
  Fact Recall:             0.912 (52/57)
  Fact-Citation Alignment: 1.000 (52/52)
  Alien Number Rate:       0.000 (0/51)
  Forbidden Fact Rate:     0.000 (0/51)
  Correct Refusal Rate:    1.000 (3/3)
  Route Stability:         0.941 (16/17)
  Latency p50/p95:         6.6s / 27.7s (n=51)
```

## Findings

**Alignment is 52/52 and Forbidden Fact Rate is 0/51.** Every fact the system
stated was cited to a document that carries it, and the one case built to tempt
an over-claim — the unattended parking kiosk that `CHAT-003` leaves explicitly
unconfirmed — was never claimed as settled in any of the three runs. Alien Number
Rate is 0/51: no run introduced a figure absent from both the evidence and the
question.

**Fact Recall is 52/57, and both misses are worth naming rather than rounding.**

* `ans_normal_08` misses "claim channel" in all three runs. The question asks
  how many days a claim may be filed within; the model answered exactly that —
  *"ต้องยื่นเบิกภายใน 30 วันนับจากวันที่เกิดค่าใช้จ่าย [FIN-001]"* — and did not
  volunteer that the channel is the Expense Portal. **This is a fixture defect,
  not a model defect:** the anchor requires a fact the question does not ask for,
  so the metric is penalising a correctly-scoped answer for not over-answering.
  It is recorded here and **not** removed. Deleting an anchor after seeing it
  fail is tuning against the measurement, which is the one thing this repository
  is built not to do. The honest repair is to reword the *question* so it asks
  for both facts, in a later iteration and re-measured from scratch.
* `ans_chat_02` misses "filing deadline" in two of three runs because those two
  runs **fell back** rather than answering. See below.

**Route Stability is 16/17, and the unstable case is the documented one.**
`ans_chat_02` — *"ยื่นเบิกช้าเกิน 30 วันแล้วระบบยังรับไหม"* — lands in the medium
band. Deterministic alias expansion does not lift it over 0.21, so it buys an
LLM rewrite, and whether the expanded score clears the gate then depends on what
the provider returned that minute: two runs fell back with
`rewrite_low_retrieval_score`, one answered with three citations. This is exactly
the non-determinism the README names as a limitation, now measured instead of
described. Alias expansion shrank this surface — on the calibration split it cut
medium-band rewrite calls from 4 to 1 — but it did not remove it.

**The medium band's non-determinism is a timeout, not a wording lottery.** A
separate three-run probe of the slang query *"เบิกตังค่า taxi ได้ปะ"* against the
same live pipeline produced an identical route all three times — `fallback`, at
an expanded score of 0.1831 — and stderr says why:

```text
safe_rewrite: degraded to original-query retrieval after OpenAITimeoutError
```

`LLM_REWRITE_TIMEOUT_SECONDS` is 10, chosen so the optional half of a
medium-band request cannot add three 30-second waits in front of an employee.
`gpt-5-mini` is a reasoning model and does not reliably return a rewrite inside
that budget: the rewrite boundary timed out on all three runs here, and
`ans_chat_02` above answered on exactly the one run of three where it did not.
Two consequences, both worth stating plainly:

* **The degradation path works as designed.** A rewriter that never returns falls
  back to the alias-expanded original query rather than crashing the request
  (AGENTS.md invariant 8), which is why these runs produced a *stable* route
  rather than an error.
* **Deterministic alias expansion is carrying more of the medium band than its
  calibration figures suggest.** Offline, the harness replays a rewrite cache and
  never times out, so the cache makes the rewrite path look more available than
  it is live. The honest reading is that on this model the medium band is
  alias-expansion-plus-a-coin-flip, not alias-expansion-plus-a-rewrite.

The fix is a per-request deadline shared across both LLM boundaries, so the
rewrite gets whatever budget is genuinely spare instead of a fixed 10 seconds.
`src/config.py` already records that as the real fix and as not implemented.
Nothing was retuned here on the strength of one probe: changing a calibrated
timeout because a live run was slow is exactly the reflex this repository logs
findings instead of acting on.

**All three refusals were free and instant.** `ans_refusal_01` to `_03` — phone
bill, fuel, maternity leave — resolved at 0.0s with `unsupported_topic` and no
provider call at all. Those refusals come from the deterministic scope gate, not
from the reporter deciding its evidence was thin, which is the cheaper and more
reliable of the two places to refuse.

**Latency p95 is 27.7s against a 30s reporter timeout.** The slowest run in this
set finished 2.3 seconds inside the budget. A multi-document answer on the medium
band pays for a rewrite and a report in sequence, and there is no per-request
deadline shared across the two. That is a real operational risk, and it is listed
as a limitation rather than presented as fine.

## What this file does not prove

* **n is small and this is one day, one model.** 17 cases, 3 runs, one provider
  on one afternoon. It is regression evidence for this build, not a statistical
  claim about answer quality.
* **Anchors are containment, not entailment.** Fact-Citation Alignment proves a
  stated fact appears in the cited document. It does not prove the claim is a
  correct *reading* of that document — a sentence can quote the right number and
  still describe the wrong condition around it.
* **The fixture is what it measures.** These 17 questions were written against
  this corpus by the same person who built the pipeline. They cover the anchors
  that matter for HR and Finance, not the space of questions employees ask.
* **A rerun will not reproduce these numbers exactly.** Route Stability exists
  precisely because the medium band is not deterministic. Re-scoring the
  *existing* answers is reproducible; getting the same answers again is not.

## Reproducing

```bash
python eval/run_eval.py --set answers --live --runs 3 \
    --transcript eval/answer_transcript.json
```

It prints the exact call count, names the model, and asks for confirmation before
spending anything; `--yes` skips the prompt for CI. Without `--live` the command
refuses to run, and without a credential it exits with a message rather than
quietly reporting zero failures over zero measurements.

## Correction, recorded rather than erased

The first live run of this set reported **Alien Number Rate 0.745 (38/51)**. Every
one of those 38 was the `001` inside the renderer's own `[HR-001]` citation
marker — text emitted deterministically from ids the validator had already
approved, which no model wrote. The checker, not the pipeline, was wrong.

It now strips validated citation markup before counting numbers, a regression
test pins the behaviour
(`test_a_rendered_citation_id_is_not_an_alien_number`), and the runner learned
`--transcript` so that finding a scoring defect never again requires paying for
every call a second time. The numbers above are from the corrected run. The
original figure is recorded here because a metric that moved from 0.745 to 0.000
between runs deserves an explanation attached to it, not a quiet replacement.


## Wall clock after the claim cap and the request deadline (2026-08-22)

The set above was measured before answers were capped at six claims and before a
single deadline covered both LLM boundaries. Both changes are about the wait an
employee sees, and no offline set can measure that, so three demo queries were
re-run live against the committed defaults -- `LLM_TIMEOUT_SECONDS=30`,
`REQUEST_DEADLINE_SECONDS=45`, `gpt-5-mini`, one run each, three provider calls
in total.

| Query | Route | Provider calls | Claims | Answer chars | Wall clock |
|---|---|---:|---:|---:|---:|
| ขั้นตอนการเบิกค่าแท็กซี่หลังทำ OT ต้องทำอย่างไร | `answered` | 1 | 4 | 872 | **22.3 s** |
| ใบเสร็จหายต้องทำอย่างไรถึงจะเบิกได้ | `answered` | 1 | 4 | 501 | **13.9 s** |
| เบิกตังค่า taxi ได้ปะ | `fallback` (`rewrite_low_retrieval_score`) | 1 | 0 | 0 | **7.2 s** |

The first query is the one that produced the nine-claim, 29.6-second answer
recorded in [`docs/demo.md`](../docs/demo.md): the same question now answers in
four claims and 22.3 seconds, against the same 30-second reporter timeout. The
claim count is the mechanism -- the cap travels into the JSON schema sent with
the request, so the model is told the bound instead of having its answer
truncated afterwards.

**What this does not show.** n = 1 per query on one model on one day, so the
seconds are an order of magnitude, not a measurement. Nothing here exercised the
deadline itself: no boundary ran out of budget, which is what the mocked-clock
route tests in
[`tests/test_graph.py::TestRequestDeadline`](../tests/test_graph.py) cover
instead. And the deadline is a ceiling on *starting* a call, not a cancellation
of one in flight -- a reporter that starts with 40 seconds of budget and a
30-second timeout can still finish at 70 seconds of wall clock.

The third row is the medium-band instability this file already documents. It
degraded differently this time: the rewrite ran and returned inside its budget
(one provider call, no timeout), and the expanded score still landed under
`FINAL_ANSWER_THRESHOLD`, so the request fell back as
`rewrite_low_retrieval_score` rather than as a rewrite failure. Same route, a
different reason code, and both are the pipeline reporting what actually
happened.

---

# Re-measurement after the P0 claim-span contract (2026-08-22)

P0-1 of `TASK1_REMEDIATION_PLAN.md` changed what the provider must return: every
claim now carries an `evidence_quote` that has to occur verbatim in a **policy**
document the claim cites, and a figure the employee wrote is no longer accepted
as support. That is a change to the generation contract, so the whole set was
re-run rather than reasoned about.

```bash
python eval/run_eval.py --set answers --live --runs 3 --yes \
  --transcript eval/answer_transcript_p0.json
```

17 cases x 3 runs = 51 invocations on `gpt-5-mini`, same set, same anchors.

| Metric | Before P0 | After P0 |
|---|---:|---:|
| Fact Recall | 0.912 (52/57) | **0.807 (46/57)** |
| Fact-Citation Alignment | 1.000 (52/52) | **1.000 (46/46)** |
| Alien Number Rate | 0.000 (0/51) | 0.000 (0/51) |
| Forbidden Fact Rate | 0.000 (0/51) | 0.000 (0/51) |
| Correct Refusal Rate | 1.000 (3/3) | 1.000 (3/3) |
| Route Stability | 0.941 (16/17) | **0.882 (15/17)** |
| Latency p50 / p95 | 6.6s / 27.7s | **10.9s / 29.9s** |

## Where the six lost facts actually went

Fact Recall fell by six fact-instances. Attributing them to "the span rule"
would be the convenient reading and it is wrong: the two transcripts were
compared run by run, and exactly three cases changed route.

| Case | Before | After | Reason | Fact-instances lost | Caused by P0? |
|---|---|---|---|---:|---|
| `ans_multi_01` | answered x3 | fallback, fallback, answered | `rewrite_low_retrieval_score` (20.5s), `rewrite_failure` (20.5s) | 4 | **No** — the rewrite exceeded its 10s budget, the documented medium-band instability |
| `ans_chat_01` | answered x3 | answered, fallback, answered | `unsupported_claim_span` (22.2s) | 1 | **Yes** — this is the new contract refusing a claim whose quote was not in a cited policy |
| `ans_chat_02` | fallback, fallback, answered | fallback x3 | `rewrite_low_retrieval_score` (8-9s) | 1 | **No** — the same non-determinism, in the other direction; this case was already unstable |

4 + 1 + 1 = 6, which is the whole drop. **The claim-span contract cost one fact
instance out of 57 (1.8%).** The other five are the medium-band rewrite
boundary behaving as this file has documented since the first run: two of the
three changed cases fell back for reasons that existed before P0 and are
unrelated to the answer contract.

Across all 51 runs the fallback reasons were `unsupported_topic` x9 (the three
refusal cases, working as designed), `rewrite_low_retrieval_score` x4,
`rewrite_failure` x1, and `unsupported_claim_span` x1.

## What improved, and what it cost

* **Fact-Citation Alignment stayed at 1.000**, now over a smaller base (46/46).
  Every fact that reached an employee was cited to a document that carries it.
* **Alien Number Rate and Forbidden Fact Rate stayed at zero**, which is the
  result the numeric rule was tightened to protect: figures are now checked
  against the quoted span rather than against the whole document or the
  question, and the rate did not move.
* **Latency rose** (p50 6.6s -> 10.9s). The reporter now has to locate and copy a
  span, which is more work per claim. p95 at 29.9s sits 0.1s inside the 30s
  reporter timeout — closer than before, and the reason `ans_multi_01`'s
  rewrite budget was exceeded twice in this run.
* **Route Stability fell to 15/17**, both unstable cases being the medium band.
  This is a measurement of provider variance, not of the contract.

## Reading this honestly

n is small, this is one model on one day, and the before/after columns are two
runs of a non-deterministic pipeline, not a controlled experiment. The
attribution table above is what the two transcripts support: it names the reason
code each lost run recorded, which is a fact, rather than assigning the drop to
the newest change, which would be a guess. A rerun will not reproduce these
numbers exactly.


---

# Re-measurement after P1, and what the wall clock showed (2026-08-22)

Run once with approval after the P1 round: 17 cases x 3 runs on `gpt-5-mini`,
transcript in [`answer_transcript_p1.json`](answer_transcript_p1.json).

| Metric | Before P0 | After P0 | After P1 |
|---|---:|---:|---:|
| Fact Recall | 0.912 (52/57) | 0.807 (46/57) | 0.842 (48/57) |
| Fact-Citation Alignment | 1.000 (52/52) | 1.000 (46/46) | 1.000 (48/48) |
| Alien Number Rate | 0.000 (0/51) | 0.000 (0/51) | 0.000 (0/51) |
| Forbidden Fact Rate | 0.000 (0/51) | 0.000 (0/51) | 0.000 (0/51) |
| Correct Refusal Rate | 1.000 (3/3) | 1.000 (3/3) | 1.000 (3/3) |
| Route Stability | 0.941 (16/17) | 0.882 (15/17) | 0.882 (15/17) |
| Latency p50 / p95 | 6.6s / 27.7s | 10.9s / 29.9s | 8.6s / 58.6s |

**Fact Recall did not improve because of P1.** The round changed no prompt, no
provider contract and no threshold — it changed telemetry accuracy, two
exception seams, and the log sink. Two fact-instances came back and one
`unsupported_claim_span` from the P0 run did not recur; both are provider
variance between two runs of a non-deterministic pipeline, and reporting them
as a gain would be reading noise as signal. What the run does support is the
absence of a regression: alignment, alien numbers, forbidden facts and correct
refusals are all unchanged, on the same fixture.

## The wall clock is the finding

p95 nearly doubled and the slowest run reached **94.4s** (`ans_multi_01`,
run 2, answered) — more than twice `REQUEST_DEADLINE_SECONDS` (45s). This is
not a new defect and not variance; it is the known shape of the deadline,
measured further than before. The deadline is a **ceiling checked before a
boundary starts**, and the per-call timeout is trimmed to what is left of it —
but neither cancels a call in flight, and neither divides by the retry count:

```text
rewrite   starts at t=0    budget 45s -> timeout min(10, 45) = 10s x 2 attempts = 20s
reporter  starts at t=20s  budget 25s -> timeout min(30, 25) = 25s x 3 attempts = 75s
                                                                       total  ~95s
```

`LLM_MAX_RETRIES=2` is the multiplier the arithmetic in `config.py` does not
account for. The same shape produced `ans_chat_01` run 3: 91.4s of wall clock
ending in `reporter_failure`.

Two honest consequences:

* The employee-facing claim to make is "a request is bounded", not "a request
  finishes within 45 seconds". The README says the deadline does not interrupt
  a call in flight; it now also has to say that retries multiply what is left.
* Fixing it properly means dividing the remaining budget by the attempt count,
  or cancelling at the provider boundary, which `graph.invoke` being
  synchronous does not offer. Both are changes to the deadline design, not to
  P1, so this round records the measurement rather than acting on it.

## Reading this honestly

Same caveats as the P0 re-measurement, and one more: p50 fell (10.9s -> 8.6s)
while p95 doubled, on 51 runs. That is a distribution with a long tail being
sampled twice, not a latency improvement.
