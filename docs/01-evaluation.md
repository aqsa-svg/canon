# 01 — Evaluation

This document is written **before** the system it evaluates. That ordering is deliberate:
if the eval is designed after the agent, it tends to measure what the agent happens to do
well. Defining the contract first means the agent is built toward a target it cannot
quietly move.

Nothing here is a result. Every figure in this document is `TBD` until
`benchmark/harness` runs on this machine (see the number rule in `CLAUDE.md`).

---

## 1. Input

A single natural-language business question, as a string.

```python
question: str  # e.g. "What was net revenue by month in 2018?"
```

That is the entire input. Explicitly **not** part of the input:

- conversation history / follow-up turns ("and by state?") — single-turn only
- a user-supplied metric hint, date range, or dimension list — if the question doesn't
  contain it, the system must either derive it from the canonical layer or ask
- database schema in the request — the semantic layer *is* the schema, and it is fixed

The eval set stores one extra field per question that is **input to the scorer, never to
the agent**: whether the question is answerable or ambiguous. Leaking that flag to the
agent would make refusal accuracy meaningless.

---

## 2. Output schema

One response object. `canon/schemas/answer.py` is the authoritative version; this is the
contract it must satisfy.

| Field | Type | Meaning |
|---|---|---|
| `answer_value` | `AnswerValue` or `None` | The verified result. `None` iff the system is refusing. |
| `sql` | `str` or `None` | The exact SQL executed to produce `answer_value`, verbatim — not a reconstruction, not pretty-printed after the fact. `None` iff refusing. |
| `metric_definition_used` | `MetricCard` or `None` | The full canonical card as loaded from `metrics.yaml`: name, description, expression, grain, caveats, owner. Not just the metric's name — the reader must be able to see what was actually computed without opening another file. `None` iff refusing. |
| `caveats` | `list[Caveat]` | Every caveat that applies to this answer. May be non-empty even for a confident, correct answer. |
| `chart_spec` | `ChartSpec` or `None` | How to render the result. `None` when a chart would not aid interpretation (a single scalar). |
| `confidence` | `float` in `[0, 1]` | The system's self-reported confidence that the answer is correct. Required in both branches. |
| `clarifying_question` | `str` or `None` | Populated **instead of** an answer when the question is genuinely ambiguous. |

### The exclusivity invariant

Exactly one of these two states holds, enforced by a Pydantic model validator:

- **Answered**: `answer_value`, `sql`, and `metric_definition_used` are all non-null;
  `clarifying_question` is null.
- **Refused**: `clarifying_question` is non-null; `answer_value`, `sql`, and
  `metric_definition_used` are all null.

There is no third state. A partial answer with a hedge attached ("here's gross revenue,
though you may have meant net") is the exact behaviour this project argues against, and
the schema makes it unrepresentable rather than merely discouraged.

### Field notes

**`answer_value`** is a typed result, not a loose float. Business answers are as often a
series as a scalar, so:

```python
class AnswerValue(BaseModel):
    columns: tuple[ResultColumn, ...]  # name + dtype per column
    rows: tuple[tuple[Scalar, ...], ...]
    scalar: Scalar | None  # set iff exactly 1 row x 1 column
```

`scalar` is a convenience for the 1x1 case so the common path is not a nested index. The
scorer compares `columns` and `rows`; it uses `scalar` only for reporting.

**`caveats`** are typed, not prose. Each carries a `source` so a reader knows whether the
caveat came from the metric card, from the compiler, or from the data:

- `from_metric_card` — copied verbatim from the card's `caveats` list
- `from_compiler` — e.g. a fan-out-safe aggregation was applied, or a filter narrowed
  the grain
- `from_data` — e.g. the requested range extends past the last date present

Caveats are **not** a substitute for refusal. If a caveat would change *which number the
user should act on*, that question was ambiguous and the correct output was a clarifying
question.

**`chart_spec`** is a small typed object (chart type enum, x field, y field, optional
series field) — not a Vega/Plotly JSON blob. A blob is an untyped dict crossing a module
boundary, which `CLAUDE.md` forbids, and Streamlit needs only the four fields.

**`confidence`** is self-reported and therefore weakly trustworthy on its own. It earns
its place in the schema only because it makes **confident-wrong rate** measurable. The
harness reports a calibration table (confidence bucket vs. observed accuracy) so that a
system reporting 0.9 on everything is visibly uncalibrated rather than plausibly good.

**`clarifying_question`** must be answerable in one short reply and must name the actual
fork. "Could you clarify?" scores as a *failed* refusal even on an ambiguous question: a
refusal is only correct if it moves the conversation forward. Rubric in §3.3.

---

## 3. The three metrics

Each question in the eval set is labelled `answerable` or `ambiguous`. The three metrics
partition cleanly over those two classes, which is what keeps them independently
meaningful.

### 3.1 Execution accuracy — does the number match gold?

> Of the **answerable** questions, the fraction where the returned result equals the gold
> result.

```
execution_accuracy = correct_values / answerable_questions
```

The denominator is *all* answerable questions. Refusing an answerable question counts
against execution accuracy — otherwise "refuse everything" would score 100%.

Result comparison rules, fixed in advance so they cannot be tuned to flatter a run:

- **Numeric**: equal within relative tolerance `1e-6`, or absolute `1e-9` near zero.
  Money is compared as `DECIMAL`, never as accumulated float.
- **Multi-row**: compared as a multiset of rows, unless the question implies an order
  ("top 5", "ranked by"), in which case order is compared too.
- **Column names**: ignored. Aliasing is cosmetic.
- **Row count mismatch**: failure, including gold-empty vs. returned-non-empty.
- **`NULL` vs `0`**: distinct. A month with no orders is not a month with zero revenue,
  and conflating them is a real analytics bug.

**Gold values are produced by hand-written SQL, executed against `data/canon.duckdb`.**
Gold SQL is written by a human reading the metric card — never generated by the model
under test, and never by an LLM at all. Both the gold SQL and its executed value are
committed, so a gold error is reviewable and a data reload that changes a value shows up
in the diff.

### 3.2 Semantic accuracy — was the right canonical metric chosen?

> Of the **answerable** questions, the fraction where `metric_definition_used` is the
> canonical metric the question actually refers to, at the correct grain, with the
> correct default filters applied.

```
semantic_accuracy = correct_metric_choices / answerable_questions
```

**This is the metric that matters, and it is the one Spider and BIRD do not measure.**

Those benchmarks score *denotation*: execute the predicted SQL, execute the gold SQL,
compare results. Under that scheme:

- There is **no semantic layer**, so there is no such thing as choosing the wrong metric.
  The question "what does revenue mean here?" simply does not arise — the gold query
  defines it implicitly, once.
- A prediction can match gold **by coincidence**. On the Olist data, `gross_revenue` and
  `net_revenue` agree on any slice with no freight and no cancellations. A system that
  picked the wrong definition scores a win on execution accuracy and nobody learns
  anything.
- Conversely, a system can choose the **right** definition and still miss the value
  because of a compiler bug — a fixable engineering error, scored identically to a
  conceptual one. That is the wrong signal for where to spend the next hour.

Semantic accuracy is therefore scored **independently of value correctness**, which is the
entire point. The four cells of (right metric?) x (right number?) are separated in the
report:

| | right number | wrong number |
|---|---|---|
| **right metric** | correct | compiler bug — engineering fix |
| **wrong metric** | *coincidence* — the dangerous cell, invisible to Spider/BIRD | wrong for a conceptual reason |

Scoring a metric choice as correct requires all three of:

1. `metric_definition_used.name` equals the gold metric name;
2. the requested grain matches the gold grain (a question about monthly revenue answered
   at daily grain is wrong even if summing the rows would reconcile);
3. the card's `default_filters` were applied, and no filter was invented beyond those the
   question states.

The trap metrics in `metrics.yaml` (`gross_revenue` vs `net_revenue`; the metric whose
definition shifts partway through the data; the metric with a deliberately restricted
`valid_dimensions`) exist to make this metric hard to pass by luck.

### 3.3 Refusal accuracy — does it ask when it should ask, and answer when it should answer?

> Accuracy on the binary ask-vs-answer decision, over **all** questions.

```
refusal_accuracy = (correct_refusals + correct_attempts) / all_questions
```

A 2x2 confusion matrix over the whole eval set:

| | agent answered | agent asked |
|---|---|---|
| **labelled answerable** | correct attempt | **over-refusal** |
| **labelled ambiguous** | **under-refusal** (guessed) | correct refusal |

Both error types are reported separately, because a single accuracy number hides the
trade-off and each failure mode has a distinct cure:

- **under-refusal rate** = guessed / ambiguous. The safety failure. Drives
  confident-wrong.
- **over-refusal rate** = asked / answerable. The usefulness failure. A system that asks
  a clarifying question about everything is useless, and this rate is what stops the
  project from gaming refusal accuracy by hedging. **A hedge-everything agent scores ~0
  here**, which is the intended pressure.

Aggregate refusal accuracy is reported alongside both rates, never instead of them, since
class balance in the eval set would otherwise make it flattering.

A refusal on an ambiguous question is scored correct only if the `clarifying_question`
**names the actual fork** — the specific choice the system could not make (which of two
metrics, which date range, which grain). Graded by a fixed rubric, by hand:

- **2 (correct)** — names the fork and offers the alternatives
- **1 (weak)** — signals ambiguity but doesn't identify what is ambiguous ("Which revenue
  metric did you mean?", with no options, is a 1)
- **0 (failed)** — generic ("Could you clarify?"), or names the wrong fork

Only a 2 counts as a correct refusal in the headline number. The 1s are reported
separately as `weak_refusals` — they are the interesting bucket to improve on Day 3.

### 3.4 Confident-wrong rate — the number we most want near zero

> The fraction of **all** questions where the system answered, the answer was wrong, and
> it reported confidence at or above the threshold.

```
confident_wrong_rate = (answered AND wrong AND confidence >= 0.7) / all_questions
```

`0.7` is fixed in advance and not tuned per run. The harness also reports the curve across
thresholds, so the choice is inspectable rather than load-bearing.

"Wrong" includes both wrong-value and wrong-metric, and includes under-refusals — guessing
on an ambiguous question with high confidence is the canonical instance of this failure.

This is the headline safety number because it is the one that maps to real damage: a
number that appears in a deck, sourced to a system that said it was sure.

---

## 4. Report format

The harness emits exactly this table. Every cell is `TBD` until a real run fills it.

| Metric | Value | Denominator |
|---|---|---|
| Execution accuracy | TBD | answerable questions |
| **Semantic accuracy** | TBD | answerable questions |
| Refusal accuracy | TBD | all questions |
| — under-refusal rate | TBD | ambiguous questions |
| — over-refusal rate | TBD | answerable questions |
| — weak refusals (rubric 1) | TBD | ambiguous questions |
| **Confident-wrong rate** | TBD | all questions |
| Mean latency / question | TBD | all questions |
| Mean cost / question | TBD | all questions |

Every run records its provenance: UTC timestamp, git commit of the code, git commit of the
eval set, model ID, and the `metrics.yaml` hash. A number without that provenance gets
deleted, not trusted.

Per-question results are written as newline-delimited JSON, so a regression is diffable
question by question rather than merely visible as a moved aggregate.

---

## Pre-registration — the Gemini cross-model arm

**Written and committed before the first Gemini call.** Nothing in this section has been
informed by Gemini output; the git history is the evidence, and if a figure here ever
disagrees with `RESULTS.md`, this document is the one that was fixed in advance.

**Why this section exists.** The Groq run changed its primary endpoint after seeing
baseline data — defensible, disclosed above, and still in-sample. Declaring the *same*
endpoint for Gemini in advance makes the cross-model arm an **out-of-sample test of that
choice**. If silent-wrong behaves on Gemini as it did on Groq, the switch was a better
instrument rather than a lucky fit to one run's data. If it does not, that is a finding
about the metric and it gets reported as one. Either way the deviation above stops being a
liability and becomes evidence that I understood what it cost — but only because this was
written first.

### Primary endpoint, declared in advance

**Silent-wrong rate** — a wrong number handed over with neither a verification flag nor a
clarifying question attached. Denominator: all questions. It compares observable behaviour
and needs no assumption about either system's confidence.

**Confident-wrong rate is secondary**, carrying the same footnote it carries on Groq: the
baseline reports no confidence, so each of its answers is imputed at 1.0. Defensible — an
unhedged figure is an assertion — but an imputation, which is why it is not the headline.

### The full metric table, fixed now

| Metric | Denominator |
|---|---|
| **Silent-wrong rate** (primary) | all questions |
| Execution accuracy | answerable questions |
| Shape accuracy (columns + grain) | answerable questions |
| Semantic accuracy | answerable, measurable |
| Behaviour accuracy (3x3 diagonal) | all questions |
| Refusal accuracy | all questions |
| — under-refusal rate | should-not-answer questions |
| — over-refusal rate | answerable questions |
| — names the fork (proxy) | should-not-answer questions |
| Caveat surfaced | cases requiring one |
| Paired discrimination | 3 pairs, both halves |
| Confident-wrong rate (secondary) | all questions |
| Error rate | all questions |
| Latency over live calls | live calls only, with n |

`mean_latency_seconds` is reported only if every unit in an arm was timed, and a cross-arm
latency comparison only if every arm has at least one live call. On a resumed, cached run
both will usually be absent, and absent is the correct output.

### Rules fixed in advance

- **Complete arms only.** No Gemini figure is published until both arms are whole. A
  partial arm is not reported, not averaged, and not previewed with a caveat.
- **Interleaved by question.** Canon and the baseline answer each question before the run
  moves to the next, so any quota stop leaves a balanced, scoreable pair. The Groq run
  completed one arm then the other and left a whole baseline with nothing to compare it
  against for days.
- **No provider mixing inside a comparison.** Enforced by the harness guard, which refuses
  a checkpoint containing two providers. A Canon-on-Gemini versus baseline-on-Groq delta
  would measure the backend.
- **Groq is not re-run.** The Groq numbers stand as published; Gemini is a second arm, not
  a replacement, and the comparison of interest is whether the *direction* holds across
  models, not whether the levels match.
- **Quota limits are recorded as observed**, with the same discipline as the Groq
  `verified` flag: a limit is `verified` only when the provider itself refused and quoted
  it. A limit this project's own limiter enforced is config checked against itself.

### Declared in advance: what would falsify the claim

The project claims a canonical layer reduces silent-wrong answers. On Gemini that claim is
wrong if **the baseline's silent-wrong rate is not materially higher than Canon's**. With
80 questions one question is 1.25 points, so "materially" means a gap that survives being
a few questions wide — not a one-question edge. A null or reversed result is published in
`RESULTS.md` with the same prominence as a positive one.

---

## Deviations from this pre-registration

This document was written before the code it evaluates, and the table above still reads
`TBD` on purpose: it is the plan, not the result. Results live in
[`benchmark/RESULTS.md`](../benchmark/RESULTS.md).

A pre-registration is only worth something if the departures from it are listed, including
the ones that flatter the system. The run reported in `RESULTS.md` departs from the plan
above in four ways.

**1. The headline metric changed, and this is the significant one.** This document bolds
**confident-wrong rate** as a primary outcome. `RESULTS.md` demotes it to secondary, behind
a **silent-wrong rate** that did not exist when this was written. The reason is sound — the
baseline reports no confidence, so scoring every baseline answer at 1.0 confidence is an
*imputation*, and silent-wrong (a wrong answer handed over with neither a verification flag
nor a clarifying question) needs no such assumption. But the reason was discovered by
running the baseline, which means **the primary endpoint was changed after seeing data**.
That is exactly what pre-registration exists to surface, so it is recorded here as a
deviation rather than presented only as an improvement. Both numbers are reported, and on
this run they happen to coincide (2.5% / 68.8%), so nothing turns on the choice — but that
is luck, not method.

**2. "Weak refusals (rubric 1)" was never implemented.** It was replaced by *names the
fork*, a mechanical proxy that checks whether a clarifying question actually names the
competing metrics rather than hedging generically. The rubric would have needed a human
grader or an LLM judge; the proxy needs neither and is checkable. It measures less: a
question can name both metrics and still be a poor question. Scored as a proxy, labelled as
one.

**3. The latency denominator here is wrong.** "Mean latency / question, all questions"
cannot be computed: a cache hit records 0.0 seconds, so that mean falls as the cache fills
and times nothing. `RESULTS.md` reports latency over live calls only, with the cache-served
count beside it. In the published run the entire baseline arm was cache-served, so there is
**no cross-arm latency comparison at all** — a limitation this document did not anticipate.

**4. Eight measures were added that are not planned here**: shape accuracy, the 3x3
behaviour matrix, paired discrimination, caveat-surfaced rate, error rate, the attribution
instrument with its coverage bound, cross-run variance, and the two-stage ablation. Adding
measures after the fact is far less dangerous than changing a headline, but it is not
nothing: with enough added measures something will favour the system under test. The
defence is that each is defined mechanically in `benchmark/harness/score.py` and reported
for both arms on the same denominators, not selected after inspection — and that paired
discrimination, the one I most want quoted, was specified to be immune to the reflex
strategies a system could otherwise game.

**What did not change:** the question set, the tiers, the gold answers, the refusal
thresholds (`metric_score_margin=0.15`, `minimum_top_score=0.45`, both chosen a priori and
never fitted), and the rule that a confident wrong answer is worse than a refusal.

**What deviation 1 costs, and how the Gemini arm repays it.** Changing a primary endpoint
after seeing data makes that endpoint *in-sample*: silent-wrong was chosen partly because
of what the Groq baseline turned out to do, so its good behaviour on Groq is not
independent evidence that it is the better instrument. The section above fixes that by
declaring silent-wrong as the primary endpoint for the Gemini arm **in advance**, in a
commit that precedes the first Gemini call. That makes the cross-model arm an out-of-sample
test of the choice. If it holds, the switch was an improvement in instrumentation rather
than a fit to one run; if it does not, that is reported as a finding about the metric. This
is the whole reason the Gemini pre-registration was written before any Gemini quota was
spent rather than alongside the results.

---

## 5. Target composition of the eval set

**Design intent, not a measurement.** Actual counts appear here once the set exists.

- **Answerable, unambiguous** — one clearly-named metric, explicit or clearly-implied time
  range. The baseline competence case.
- **Answerable but trap-adjacent** — wording that leans toward the wrong card ("total
  sales" when both revenue metrics exist but one is the documented default; "revenue"
  scoped to a period where the shifted definition applies). Answerable because the
  canonical layer *does* resolve it — but the caveat must surface.
- **Ambiguous — competing metrics.** "How much money did we make last month?" with both
  `gross_revenue` and `net_revenue` live and no documented default. Correct output: ask.
- **Ambiguous — wrong grain.** A metric requested by a dimension its `valid_dimensions`
  excludes. The compiler raises `InvalidIntent` *before* execution; the agent must turn
  that typed error into a clarifying question rather than silently substituting a
  dimension that does happen to work.
- **Ambiguous — underspecified time.** "Recently", "this quarter", against data with a
  fixed end date.
- **Unanswerable from the warehouse at all** — a metric that does not exist in the
  canonical layer (margin, CAC, LTV). Scored in the ambiguous class: the correct output is
  a refusal saying the metric isn't defined, not a plausible-looking substitute. This is
  the class where a non-enum architecture hallucinates.

The set is balanced so that neither always-answer nor always-ask scores well, and the
answerable/ambiguous split is recorded with the results so a reader can judge the baseline
for themselves.

---

## 6. Not measured

Stated so their absence is a decision rather than an oversight:

- **SQL string similarity to gold.** Two correct queries can differ completely. Canon's
  SQL is compiler-generated, so string comparison would measure the compiler's formatting
  and nothing else. Denotation is what matters.
- **Answer fluency / phrasing.** The deliverable is a number, its SQL, and its definition.
  Prose quality is not the claim.
- **Robustness to typos and adversarial phrasing.** Out of scope (`CLAUDE.md`). The honest
  eval set is the deliverable; an adversarial set is a different project.
- **Multi-turn clarification resolution.** A refusal is scored on whether asking was
  correct and whether the question named the fork — not on what happens after the user
  replies. Single-turn only.
