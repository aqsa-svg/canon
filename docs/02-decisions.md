# 02 — Decisions

Architecture decisions, each with the alternative it was chosen over and what would
change my mind. Ordered roughly by how much they matter.

**On numbers in this document.** Accuracy figures live in
[`benchmark/RESULTS.md`](../benchmark/RESULTS.md) and are `TBD` here until a run fills
them. Token counts *are* quoted, because they were measured locally on this machine by
`scripts/dry_run.py` — exactly, using the GPT-OSS harmony tokenizer — and the artifact is
committed at [`benchmark/dry_run.md`](../benchmark/dry_run.md). Re-derive any of them with
`uv run python scripts/dry_run.py`.

---

## ADR-001 — The LLM fills a typed intent; it never writes SQL

**Decision.** The model's only output is a `QueryIntent`: a metric enum, dimension enums,
a half-open date range, typed filters, an optional comparison. A deterministic compiler
turns that plus the canonical metric definitions into SQL.

**The alternative** is text-to-SQL: put the schema in the prompt, get SQL back, execute
it. That is what the baseline does, and it is the industry default.

**Why intent compilation wins, precisely.** Not because models write bad SQL — the
baseline's SQL is syntactically fine and mostly runs. It wins because of *what can go
wrong*:

- A hallucinated metric is **unrepresentable**, not caught. `MetricName` is generated from
  `metrics.yaml` and appears inside the JSON schema sent to the provider, so a metric the
  semantic layer does not define cannot come back. Nothing validates it away afterwards;
  there is nothing to validate.
- Fan-out is structural. The compiler traverses only `many_to_one` join edges, and
  `orders → order_payments` is simply absent from the graph. Joining payments to items
  inflates merchandise revenue from 13,591,643.70 to 14,209,115.34 on this warehouse —
  measured, not hypothetical — and no prompt instruction makes that impossible the way a
  missing edge does.
- Wrong-grain slices are refused *before execution*, so "compiles but returns nonsense"
  never happens. `avg_order_value` by `product_category` would run and return plausible
  numbers; 727 orders contain items from more than one category, so it is excluded from
  that metric's `valid_dimensions` by hand.
- The compiler is pure. Same intent, same SQL, byte for byte, so a change in output is a
  change in the intent — which makes the failure taxonomy possible at all.

**What it costs.** Expressiveness. Anything the intent schema cannot say, Canon cannot
answer: no window functions, no self-joins, no top-N-per-group. The baseline can attempt
all of those. That is a real ceiling and the honest framing is that Canon trades
expressiveness for the guarantee that what it does answer, it answers under a stated
definition.

**What would change my mind.** A question distribution where most questions need SQL
shapes the intent schema cannot express. At that point the semantic layer is the wrong
abstraction and a reviewed-SQL workflow is better.

---

## ADR-002 — Refusal is deterministic Python over model scores, not a model judgement

**Decision.** The model scores every plausible metric. A function in
`canon/agent/policy.py` decides whether to answer, using two thresholds and four rules in
priority order: the compiler rejected the intent; the model declined to commit; the model
named missing information; the top two scores are within `metric_score_margin`.

**The alternative** is asking the model "is this ambiguous?" — either in the same call or
a second one.

**Why the split.** Three reasons, in order of how much they matter:

1. **It is tunable.** The margin is one number in one place. Changing it re-scores an
   entire cached run without another API call, because the scores are already on disk.
   Tuning a prompt instead means re-running everything and hoping.
2. **It is auditable.** Every decision is a row in `decisions.jsonl` with the rule that
   fired, both candidate scores, and the gap. When refusal accuracy moves, the reason is
   in the log rather than in a model's mood on the day.
3. **It separates two questions that get conflated.** "Which metric does this mean?" is a
   reading-comprehension task the model is good at. "Is the gap between my top two
   readings small enough to warrant asking?" is a calibration decision with a
   business-risk tradeoff behind it. Asking a model to do both in one breath makes the
   second invisible.

**A note on where the work actually happens.** In the smoke sample the model frequently
declined to produce an intent on its own, so `model_declined_to_commit` fired before the
close-scores rule got a chance. The threshold is a backstop more often than the primary
mechanism — worth knowing, and an argument for keeping both.

**What it costs.** The policy cannot see anything the scores do not encode. A question
ambiguous for a reason unrelated to metric choice — an unanchored date, say — is caught
only because the model puts it in `missing_information`, which is a model judgement after
all. The split is partial, not clean.

**What would change my mind.** If measured refusal accuracy were driven almost entirely
by `missing_information` and almost never by the score gap, the threshold would be
theatre and should be deleted rather than defended.

---

## ADR-003 — Comparison windows are calendar-aware and computed by the compiler

**Decision.** The model sets `comparison: {kind: previous_period | previous_year}` and a
bounded `time_range`. The compiler derives the earlier window.

**Why the compiler does the arithmetic.** Because it is arithmetic, and the model gets it
wrong in a way that is invisible in the output. "March compared with the month before"
under a naive day-count shift gives 29 January to 1 March: 31 days back from a 31-day
month. The number that comes out is a real number over a real window and nothing about it
looks wrong. Month-aligned ranges therefore shift by whole calendar months, unaligned
windows shift by their own length, and month-end and leap days clamp the way finance
systems do.

**Why both windows land in one query with a period label**, rather than two queries or a
SQL-level difference:

- **Two queries** would be two chances to apply the filters differently. One query cannot
  drift from itself.
- **A difference** is wrong for ratio metrics, and silently. For `cancellation_rate` the
  difference of the numerators over the difference of the denominators is not the change
  in the rate. Labelling the periods and grouping by the label means the card's expression
  is evaluated unchanged, once per period, so a ratio stays a ratio.

The delta is then arithmetic on two rows, done in Python where it can be typed, rather
than in SQL where it would have to be re-derived per metric shape.

**What it costs.** A `LIMIT` cannot be combined with a comparison — the limit applies
after grouping and truncated an entire period while the result still looked complete. That
combination is refused rather than mangled. Top-N-per-period needs a window function the
compiler does not emit.

---

## ADR-004 — The baseline was given permission to refuse

**Decision.** The naive baseline's prompt tells it to reply `CLARIFY: <question>` when a
question is genuinely ambiguous or unanswerable from the schema. It also gets the real DDL
generated from the warehouse, row counts, and the distinct values of the two
low-cardinality columns any competent text-to-SQL prompt would include.

**Why.** Because the alternative is a rigged comparison. Without permission to refuse, the
baseline's under-refusal rate on the ambiguous half is 100% *by construction*, and the
headline delta would measure my prompt rather than the architecture. A reviewer would spot
that in a minute and discount everything else in the report — correctly.

The same reasoning drives the smaller choices: markdown fences are stripped rather than
scored, because a fence is a formatting slip and failing a system for it is strawmanning;
the baseline gets the same data window, the same provider, the same model, the same effort
setting and the same cache.

**What the baseline does not get** is the thing under test: canonical definitions, a
cardinality-annotated join graph, validation before execution, and a result gate.

**The consequence worth stating plainly.** The baseline cannot populate
`metric_definition_used`, because it has no notion of a metric. That makes semantic
accuracy structurally unmeasurable for it — which is not a handicap I imposed but the
finding itself, and the same blind spot `docs/01-evaluation.md` §3.2 argues Spider and
BIRD share. For the baseline, a wrong metric choice can only surface as an
execution-accuracy failure.

---

## ADR-005 — No vector store, and here is the arithmetic

**Decision.** All 24 metric cards go in the prompt on every question. No embedding index,
no retrieval step.

This decision was re-examined when the project moved to free-tier providers, because token
cost stopped being a line item and became the binding constraint. It survived, but the
reasoning is now quantitative rather than asserted.

### What stuffing actually costs

Measured with the GPT-OSS harmony tokenizer via `scripts/dry_run.py`:

| Component | Tokens |
|---|---|
| Full 24-card system prompt | 5,196 |
| `AgentDecision` JSON schema | 1,684 |
| Question + message framing | ~60 |
| **Prompt per question** | **~6,940** |
| Assumed output | 700 |
| **Total per question** | **~7,615** |

Against Groq's 200,000 tokens/day free tier that is **3.5% of a day's budget per
question**, or about 26 questions per day. An 80-question sweep of Canon plus the baseline
is 695,410 tokens — **3.5 days**.

For the record, an earlier version of this plan assumed 23,000 tokens per question. That
was a character count mistaken for a token count, and it inflated the projected cost of
the sweep by a factor of four. It is called out here because it is the kind of error that
quietly reshapes a plan.

### Where retrieval would start to pay

Retrieval replaces "all N cards" with "top k cards". At roughly 216 tokens per full card
(5,196 ÷ 24), the prompt cost is approximately:

```
stuffing:   fixed_instructions + N x 216
retrieval:  fixed_instructions + k x 216 + embedding_call
```

With N = 24 and k = 5, that saves about 4,100 tokens per question — a little over half the
prompt. Real, and not obviously worth it: the saving is 4,100 tokens against a failure mode
that costs an entire answer. **Retrieving the wrong card is silent.** The model receives a
plausible catalogue, chooses the best metric *in it*, and reports high confidence, because
nothing in its context indicates that `net_revenue` was never offered. That is precisely
the confident-wrong failure Canon exists to argue against, reintroduced one layer down.

The crossover is a function of N, not of budget:

| N metrics | Stuffed prompt | % of a 200k/day budget per question | Verdict |
|---|---|---|---|
| 24 (today) | ~5,200 | 3.5% | stuff |
| 50 | ~11,000 | 6% | stuff |
| 100 | ~22,000 | 11% | uncomfortable |
| 200 | ~43,000 | 21% | retrieval, or a hierarchy |
| 500 | ~108,000 | 54% | retrieval is mandatory |

So the honest statement of the decision is: **stuffing is right below roughly 100 metrics,
and this project has 24.** Above that, the answer is probably not a vector store either
but a *hierarchical* catalogue — metric families in the prompt, full cards fetched by name
— which keeps the choice inside an enum and so keeps the wrong-card failure loud instead
of silent.

### The two-stage prompt: measured, not assumed

There is a middle option that does not involve embeddings, and it is implemented: stage A
shows short cards (name, one line, synonyms, legal breakdowns) and picks a metric; stage B
shows the full card for that one metric and fills the intent. Both fold into the same
`AgentDecision`, so the policy, compiler and scorer cannot tell which produced it.

| | Full stuffing | Two-stage |
|---|---|---|
| Calls per question | 1 | 1 or 2 |
| Prompt tokens, 80 questions | 553,225 | 329,979 |
| Total tokens, 80 questions | 609,225 | 424,479 |
| Days at 200k/day | 3.05 | 2.12 |
| Execution accuracy | TBD | TBD |
| Semantic accuracy | TBD | TBD |

A 30% token saving for 55 extra requests. Notably *less* than the character-level saving
suggests, because stage B's `IntentFill` schema costs 1,468 tokens and is paid on every
answerable question — the schema, not the prose, is the floor.

**Whether it costs accuracy is an ablation, not an assumption.** Results in RESULTS.md. If
two-stage loses accuracy, full stuffing keeps the headline and the ablation is reported as
a negative result, which is worth publishing either way.

---

## ADR-006 — Provider abstraction, and why provenance pins model + provider + config hash

**Decision.** One `LLMProvider` protocol; Groq (`openai/gpt-oss-120b`) as the free-tier
primary, Gemini as a second free provider for a cross-model check, Anthropic retained for
the paid path. Every run records provider, model id, and a `config_hash` covering model,
context limit, max output tokens, temperature and effort.

**Why pin all three.** Because a result is a property of a configuration, not of a system.
Two runs with different config hashes are not comparable, and the hash makes that
*checkable* rather than something a reader has to trust. The harness refuses to mix
providers inside one comparison for the same reason: if Canon answered on Groq and the
baseline on Gemini, the delta would measure the backend.

The hash deliberately excludes rate limits. Throttling changes how long a run takes, not
what it answers, and folding limits in would make two identical results look incomparable
because a quota was raised.

**Why the protocol earns its place** rather than being premature abstraction: it was
written *after* the constraint changed, and it immediately surfaced a real incompatibility.
Groq's tool-argument validator does not resolve JSON Schema `$ref`, and Pydantic factors
nested models into `$defs`. The consequence was not a clean error — the model never saw
`QueryIntent`'s fields at all, so it invented `breakdown` for what is called `dimensions`,
and the validator reported `/intent: expected null, but got object` because the null branch
was the only one it could interpret. `$defs` are now inlined in the provider layer, which
is where a backend-specific shim belongs: the schema itself is correct JSON Schema.

**What it costs.** Three code paths for structured output where one would do — tool-calling
for Groq, native `response_schema` for Gemini, `messages.parse` for Anthropic — and the
guarantees differ. Gemini's and Anthropic's are enforced server-side; Groq's depends on the
tool definition. The metric enum is inside the schema in all three, so the core claim holds
everywhere, but "structured output" is not one thing.

---

## ADR-007 — Reproducible for $0, documented as a constraint

**Decision.** The benchmark runs entirely on free tiers. Reproducing it costs nothing.

This is a constraint, not a compromise to hide, and it shapes the work visibly:

- **The daily token ledger is on disk**, keyed by UTC date. An 80-question sweep does not
  fit in one free-tier day, so a limiter that forgot across restarts would make a
  resumable sweep impossible — which is the whole point.
- **The sweep checkpoints after every question.** Quota exhaustion is the expected end of
  a session rather than a failure, and the runner says so and prints the resume command.
- **A truncated sweep is never averaged.** An arm missing any question is excluded from the
  report, provenance records `complete: false`, and the runner exits non-zero.
- **The dry-run contacts no provider.** Calling a `count_tokens` endpoint to estimate quota
  would spend the request budget the dry-run exists to protect. Counts are local, and
  labelled `exact` for Groq and `approximate` for the others where no local tokenizer
  exists.
- **Rate limits are labelled unverified** until checked on the machine, because free-tier
  limits change without notice and a wall-clock projection is only as good as its inputs.

**What it costs.** Two things worth naming. Free-tier models are weaker than frontier
models, so the absolute accuracy numbers are lower than a paid run would produce — the
*delta* between Canon and the baseline is the claim, not the level. And Google's free tier
may use submitted prompts to improve their products, which is why Gemini is the secondary
provider and why the README says so plainly. The warehouse is a public dataset and the
questions are synthetic, so nothing confidential is at stake here.

---

## What I cut for the 3-day scope

Named because a reader should be able to tell a decision from an omission:

- **Vector retrieval over metrics.** ADR-005. At 24 metrics it buys a token saving and
  costs a silent failure mode.
- **Driver analysis** ("why did revenue drop?"). This needs decomposition over dimensions
  and a story about attribution. It is a project, not a feature.
- **Forecasting.** Same.
- **Dual-query reconciliation** — computing each number two ways and comparing. Genuinely
  valuable for trust and squarely out of scope for three days; the single verification
  gate is the cheap version.
- **A CI eval gate.** The harness runs on demand. With a 3.5-day sweep on a free tier, a
  per-commit gate is not even coherent.
- **An adversarial eval set.** The honest set is the deliverable. An adversarial one
  measures robustness to phrasing, which is a different question.
- **Multi-turn clarification.** Refusal is scored on whether asking was correct and whether
  the question named the fork, not on what happens after the user replies.

## What I would build next, in order

1. **Fix the biggest bucket in `docs/03-failures.md`.** Whatever the taxonomy says is most
   common, not what I guessed would be.
2. **Hierarchical metric catalogue.** The scaling answer from ADR-005, and it keeps metric
   selection inside an enum. This matters before the 100-metric mark, not after.
3. **Caveat surfacing as a first-class output**, not prose attached to an answer. A caveat
   that changes which number you should act on deserves to be structured enough to alert
   on.
4. **Definition-change awareness across the whole layer.** One metric has a
   `definition_change` today. In a real warehouse most metrics have several, and the
   compiler should refuse to span one without being asked.
5. **Dual-query reconciliation**, the cut item most worth un-cutting: compile each metric
   two ways from the same card and compare. It would catch compiler bugs that the current
   gate cannot see.
