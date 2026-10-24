# Canon — benchmark results

Generated from `benchmark/runs/sweep-groq-r1/report.json` by `scripts/write_results.py`. No figure here is typed by hand, so none can drift from the run that produced it.

## Provenance

| Field | Value |
|---|---|
| started_at | `2026-09-04` |
| provider | `groq` |
| model_id | `openai/gpt-oss-120b` |
| provider_config_hash | `72af3a5ba084b045` |
| free_tier | `True` |
| effort | `medium` |
| code_commit | `61d6ac4` |
| eval_set_commit | `df52cc6` |
| metrics_yaml_sha256 | `2bd082e4f63ef301f1c9afb00d2efb8f9913c0662b2d088dd4957b041a5ce9f0` |
| confident_threshold | `0.7` |
| question_count | `80` |
| scored_questions | `80` |
| runs | `1` |
| cache_used | `True` |
| complete | `True` |
| policy_thresholds | `metric_score_margin=0.15, minimum_top_score=0.45` |
| scored_at | `2026-10-23T09:39:05.424027+00:00` |
| scoring_commit | `1b74a90` |

**Threshold provenance.** UNVALIDATED: chosen a priori from the design and never fitted to data. No eval-set question was used, so refusal accuracy is not inflated by threshold leakage -- but the values have also never been checked against real model output. Run scripts/tune_thresholds.py against benchmark/dev/questions.yaml (14 held-out questions) to validate them.

**When these answers were produced.** A free-tier sweep cannot finish inside one day's token cap, so this one ran across **6 sessions, 2026-09-04 to 2026-10-02**. `code_commit` names the last of them, not all of them — a single commit is a convenient fiction for a run that spans a month.

These session dates are **reconstructed** from `data/quota_ledger.json`, which records live calls per UTC day, because this run predates per-session provenance recording. Earlier versions of the harness re-stamped `started_at` on every scoring pass, including passes that answered nothing, so the originally recorded value was overwritten and is not recoverable. Runs from this commit forward record each answering session as it happens and never restamp. Marked as reconstruction rather than presented as a log.

**One unit was answered under later code: `am-08`.** It first died to a Groq TPM overage that the harness classified as a permanent failure; once that was reclassified as a transport error it was re-attempted and answered correctly. The intervening commits touched `benchmark/RESULTS.md`, `docs/03-failures.md`, `scripts/write_results.py`, `tests/`, and seven lines of `canon/agent/providers/implementations.py` — all of them additions to a `_TRANSPORT_MARKERS` tuple that is read only when loading a checkpoint to decide what to retry. The prompt, the metric cards, the compiler, the refusal policy and the scorer are byte-identical across the range, and `is_transport_error` is not reachable from the answer path at all. **`am-08` is therefore comparable, not contaminated**, and is included.

## Headline

| Metric | Denominator | canon | baseline | delta |
|---|---|---|---|---|
| **Silent-wrong rate** | all | 2.5% | 68.8% | -66.2pp ✓ |
| Execution accuracy | answerable | 90.9% | 32.7% | +58.2pp ✓ |
| Shape accuracy (columns + grain) | answerable | 98.1% | 96.4% | +1.7pp ✓ |
| **Semantic accuracy** | answerable, measurable | 100.0% | n/a | n/a |
| Behaviour accuracy (3x3 diagonal) | all | 88.8% | 72.5% | +16.2pp ✓ |
| Refusal accuracy (ask vs answer) | all | 91.2% | 31.2% | +60.0pp ✓ |
| — under-refusal rate | should-not-answer | 0.0% | 72.0% | -72.0pp ✓ |
| — over-refusal rate | answerable | 5.5% | 0.0% | +5.5pp ✗ |
| — names the fork (proxy) | should-not-answer | 69.6% | 42.9% | +26.7pp ✓ |
| Caveat surfaced | cases requiring one | 87.5% | n/a | n/a |
| Confident-wrong rate [1] | all | 2.5% | 68.8% | -66.2pp ✓ |
| Error rate | all | 2.5% | 0.0% | +2.5pp ✗ |

**Silent-wrong** is the headline safety number: a wrong answer handed over with neither a verification flag nor a clarifying question attached. It compares what each system observably did, so it needs no assumption about confidence.

**Shape accuracy** is the lenient counterpart of execution accuracy — correct columns and a plausible grain, filters disregarded. The pair is the finding: high shape with low execution means a system asked for the right thing and computed it under the wrong definition. Quoting either alone misleads in opposite directions.

[1] Confident-wrong is secondary because it rests on an imputation: the baseline reports no confidence, so each of its answers is scored at 1.0. Defensible — a figure handed over with no hedge is an assertion — but still an assumption. Silent-wrong needs none.

**On the baseline's semantic accuracy.** `n/a` because it declares no metric — there is nothing to compare against gold. That is the finding, not a gap in the harness: a system with no semantic layer cannot say which definition produced its number, which is the blind spot `docs/01-evaluation.md` §3.2 argues Spider and BIRD share.

### Wrong answers: caught vs. silent

| System | silent wrong | gate-flagged | declined to answer |
|---|---|---|---|
| baseline | 55 | 0 | 7 |
| canon | 2 | 1 | 23 |

The baseline's zeros are structural — it has no verification gate — which is the finding rather than a scoring artefact.

## Per tier

| System | Tier | n | Execution | Shape | Semantic | Behaviour | Silent-wrong |
|---|---|---|---|---|---|---|---|
| baseline | 1 | 20 | 45.0% | 100.0% | n/a | 100.0% | 55.0% |
| baseline | 2 | 20 | 30.0% | 95.0% | n/a | 100.0% | 70.0% |
| baseline | 3 | 15 | 20.0% | 93.3% | n/a | 100.0% | 80.0% |
| baseline | ambiguous | 25 | n/a | n/a | n/a | 12.0% | 72.0% |
| canon | 1 | 20 | 90.0% | 100.0% | 100.0% | 90.0% | 0.0% |
| canon | 2 | 20 | 85.0% | 94.7% | 100.0% | 95.0% | 10.0% |
| canon | 3 | 15 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| canon | ambiguous | 25 | n/a | n/a | n/a | 76.0% | 0.0% |

## Behaviour matrix (3x3)

Not collapsed to 2x2, because the off-diagonals are different failures:

- **answer when it should clarify** — a *trust* failure. A number reaches someone with no way to know the question was ambiguous.
- **clarify when it should answer** — a *usability* failure. Annoying, safe, and cheap to fix by lowering the margin.
- **refuse when it should clarify** — a *capability* failure.
- **clarify when it should refuse** — a *scope* failure.

### baseline

| target ↓ / actual → | answer | clarify | refuse | error |
|---|---|---|---|---|
| **answer** | 55 | 0 | 0 | 0 |
| **clarify** | 14 | 3 | 0 | 0 |
| **refuse** | 4 | 4 | 0 | 0 |

### canon

| target ↓ / actual → | answer | clarify | refuse | error |
|---|---|---|---|---|
| **answer** | 52 | 3 | 0 | 0 |
| **clarify** | 0 | 16 | 1 | 0 |
| **refuse** | 0 | 3 | 3 | 2 |

## Paired discrimination

Deliberately paired items scored as **both halves correct**. Each pair looks alike and needs opposite responses, so a system with a reflex gets exactly one half and scores zero. **This is the number immune to a reflex strategy** — always-answer and always-ask both score 0%, asserted in the test suite.

| System | pairs | both halves correct | rate |
|---|---|---|---|
| baseline | 3 | 0 | 0.0% |
| canon | 3 | 3 | 100.0% |

**baseline** per pair:

- avg_order_value grain: t3-08 WRONG, am-13 WRONG → **FAIL**
- payment_type fan-out: t2-12 correct, am-14 WRONG → **FAIL**
- reported_revenue restatement caveat: t3-02 WRONG, t3-03 WRONG → **FAIL**

**canon** per pair:

- avg_order_value grain: t3-08 correct, am-13 correct → **PASS**
- payment_type fan-out: t2-12 correct, am-14 correct → **PASS**
- reported_revenue restatement caveat: t3-02 correct, t3-03 correct → **PASS**

## Attribution instrument — stated error bar

When a system returns a wrong number, the harness checks whether that number is another question's gold answer, which would say *which* definition it computed. That inference only holds where the matched value is unique across the set and large enough not to collide by chance, so coverage is measured rather than assumed.

| Property | Value |
|---|---|
| gold values | 55 |
| shared between questions | 8 |
| below magnitude threshold (1000.0) | 20 |
| **can support an attribution** | 27 (49.1%) |

- baseline: 7 attributable, 2 unattributable
- canon: 4 attributable, 4 unattributable

## Variance across runs

Every figure above is a **single sample**. A separate probe re-ran `canon` twice over the 24-question stratified subset at temperature 0 and is reported in [`VARIANCE.md`](VARIANCE.md). One question of 24 flipped: `t2-05`, which is also the question `docs/03-failures.md` §C1 independently identifies as one where the clarifying question is better than the gold. **Over-refusal is therefore the one headline figure with demonstrated sample-dependence** — 5.5% on this draw, 3.6% had `t2-05` landed the other way. The remaining 23 subset questions were stable across both samples, and the other 56 questions are unmeasured.

- **baseline**: single run -- no variance measured. No spread is reported rather than printing 0.0, which would claim a stability never measured.
- **canon**: single run -- no variance measured. No spread is reported rather than printing 0.0, which would claim a stability never measured.

## Latency and cost

`docs/01-evaluation.md` pre-registered *mean latency / question* over all questions. That denominator cannot be computed: a cache hit records 0.0 seconds, so a mean over all questions measures how full the cache is. `mean_latency_seconds` is therefore `None` unless every unit in the arm was timed, and the timing that is defensible carries its own sample size.

| System | units | live calls timed | untimed | mean | median | max |
|---|---|---|---|---|---|---|
| canon | 80 | 42 | 38 (36 cached, 2 errored) | 3.01s | 2.33s | 8.72s |
| baseline | — | 0 | all | not measured | not measured | not measured |

**No cross-arm latency comparison is emitted.** It requires at least one live call in every arm, and `baseline` had none — every unit was served from cache, so nothing in that arm was ever timed. Reporting it as 0.0s, or printing one populated column and leaving it to read as a comparison, would both be worse than saying so.

**Mean cost per question is $0.00 for both arms**, and that is the free-tier constraint working rather than a rounding artefact: the provider bills nothing at this tier, so reproducing the benchmark costs nothing. The token spend it replaces is the real resource, and `data/quota_ledger.json` records it per UTC day.

## What these numbers do and do not mean

**Execution accuracy measures conformance to a stated canonical layer, not correctness in the abstract.** I wrote the metric cards, the questions and the gold answers. The mechanism claims are independently checkable — a hallucinated metric is unrepresentable, fan-out cannot be built — but the accuracy figures rest on labels I authored. A finance team that had ruled net revenue the default would score Canon's refusal on `am-01` a failure. The project's claim is that having a canonical layer beats not having one, not that my particular layer is right.

**Known artefacts inflating Canon's over-refusal rate**, both documented in `docs/03-failures.md`:

- `t1-09` and `t1-10` name no time period. My eval set assumes that means all data and never told the model, which was asked to treat unanchored time as missing information. At most 2/55 answerable questions, so at most 3.6pp.
- `am-18` and `am-21` are recorded as errors because my schema requires at least one metric candidate, and the honest answer for a metric the catalogue lacks is zero. Two refuse-class questions understate Canon on the class the project is about.

Both are disclosed rather than fixed, because fixing either changes the prompt or the schema, invalidates the response cache, and costs a full re-run. A quantified artefact is more credible than a quietly cleaner number.

**Free-tier model.** This ran on a free, open-weights model. The absolute numbers would be higher on a frontier model; the *delta* between the two systems is the claim, not the level.

Full per-question records: `benchmark/runs/sweep-groq-r1/`.
