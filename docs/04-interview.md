# 04 — Interview questions on the design decisions

One question per ADR, each answered in a paragraph I could say out loud without notes.
Every answer ends with **the weakest point** — where a sharp follow-up would land. Those
lines are deliberately unflattering; a defence I can't puncture myself is a defence I
haven't tested.

Design questions come first, then the results questions, which were held back until there
were numbers to defend. There are now: see
[`benchmark/RESULTS.md`](../benchmark/RESULTS.md),
[`benchmark/ABLATION.md`](../benchmark/ABLATION.md) and
[`benchmark/VARIANCE.md`](../benchmark/VARIANCE.md).

---

## 1. Why does the LLM fill a typed intent instead of writing SQL?

Not because models write bad SQL — the baseline's SQL is syntactically clean and almost
always runs. It's because of *what can go wrong*. When the model fills a `QueryIntent`,
the metric name is an enum generated from `metrics.yaml`, and that enum goes inside the
JSON schema sent to the provider, so a metric the semantic layer doesn't define cannot
come back at all. It isn't validated away afterwards; there's nothing to validate. The
same trick applies structurally to joins: the compiler traverses only `many_to_one` edges,
and `orders → order_payments` simply isn't in the graph, so the query that inflates
merchandise revenue from 13,591,643.70 to 14,209,115.34 can't be constructed. And because
the compiler is pure, the same intent yields byte-identical SQL, which is what makes a
failure taxonomy possible at all — a change in output means a change in the intent.
Text-to-SQL gives you a string you have to audit; intent compilation gives you a decision
you can check against a definition.

**Weakest point.** It buys that guarantee with expressiveness, and I never measured the
price. Canon can't express window functions, self-joins, or top-N-per-group; the baseline
can attempt all three. I can tell you Canon's answers are governed by a stated definition,
but I *cannot* tell you what fraction of real analyst questions fall outside the intent
schema — so I can't tell you where the ceiling is, only that it's lower than the
baseline's. A sharp interviewer asks "what percentage of your users' questions would this
refuse for structural reasons?" and I have no number.

---

## 2. Why is the refusal policy deterministic Python over model scores rather than an LLM judgment call?

Three reasons, in order of how much they matter. It's *tunable*: the margin is one number
in one place, so changing it re-scores an entire cached run without another API call,
whereas tuning a prompt means re-running everything and hoping. It's *auditable*: every
decision is a row in `decisions.jsonl` with the rule that fired, both candidate scores and
the gap, so when refusal accuracy moves the reason is in the log rather than in a model's
mood that day. And it *separates two questions that get conflated* — "which metric does
this question mean?" is reading comprehension, which the model is good at, while "is the
gap between my top two readings small enough to warrant asking?" is a calibration decision
with a business-risk tradeoff behind it. Asking a model to do both in one breath makes the
second one invisible.

**Weakest point.** Measured, and it points against me: of the refusals on the completed
ambiguous class, **every single one fired on `model_declined_to_commit`** — the model left
`intent` null on its own. Zero came from my score-gap threshold. On current evidence the
"deterministic policy" is mostly a pass-through for the model's own hedging, and the
tuning story is closer to theatre than to mechanism. Two honest caveats: those questions
are the most obviously unanswerable in the set, and the threshold is supposed to matter on
*borderline* cases, of which the ambiguous class has few. But I flagged this risk in
ADR-002 before measuring it, and the measurement has not rescued it. The full decision log
over tiers 1–3 is what settles it, and it may not settle in my favour.

---

## 3. Why are comparison windows calendar-aware and computed by the compiler, and why do both windows land in one query with a period label instead of a difference?

The compiler does the arithmetic because it *is* arithmetic, and the model gets it wrong
in a way that's invisible in the output. "March compared with the month before" under a
naive day-count shift gives 29 January to 1 March — 31 days back from a 31-day month — and
the number that comes out is a real number over a real window with nothing about it that
looks wrong. So month-aligned ranges shift by whole calendar months, unaligned windows
shift by their own length, and month-ends and leap days clamp the way finance systems do.
As for one query with a label: two queries would be two chances to apply the filters
differently, and one query can't drift from itself. A SQL-level *difference* is worse than
that — for a ratio metric like `cancellation_rate`, the difference of the numerators over
the difference of the denominators is not the change in the rate. Labelling the periods and
grouping by the label means the card's expression runs unchanged, once per period, so a
ratio stays a ratio. The delta is then arithmetic on two rows, in Python, where it can be
typed.

**Weakest point.** The label convention leaked into my own scoring and I only caught it by
reading the baseline's SQL rather than its score. Gold rows are keyed `current`/`comparison`
because gold was written from the metric cards — so a system answering the same question
with two month-keyed rows was being marked wrong on a convention it had no way to know. I
patched that in two places (value comparison, then shape comparison) and each time I found
it by accident. That's a sign the convention is doing more work in the evaluation than I
intended, and I can't be confident there isn't a third place it still bites.

---

## 4. Why no vector store at 20-odd metrics, and where is the crossover?

Because at this size retrieval buys a token saving and costs a silent failure mode.
Measured with the actual GPT-OSS tokenizer, the full 24-card prompt is 5,196 tokens and a
whole call is about 7,615 — 3.5% of Groq's 200,000-token daily free budget per question.
Retrieval at k=5 would save roughly 4,100 tokens a question, at about 216 tokens per card.
That's real, and it isn't obviously worth it, because **retrieving the wrong card is
silent**: the model receives a plausible catalogue, picks the best metric *in it*, and
reports high confidence, because nothing in its context indicates that `net_revenue` was
never offered. That's precisely the confident-wrong failure this project exists to argue
against, reintroduced one layer down. The crossover is a function of catalogue size, not of
budget — stuffing stays comfortable to roughly 100 metrics (about 22,000 tokens, 11% of a
daily budget) and becomes untenable around 200–500. Above that I'd reach for a
*hierarchical* catalogue rather than embeddings: metric families in the prompt, full cards
fetched by name, which keeps selection inside an enum and so keeps the wrong-card failure
loud instead of silent.

**The ablation has since run, and it cuts both ways.** Two-stage — short cards to choose,
full card to fill — matched full stuffing on the 24-question subset: paired discrimination
3/3 against 3/3, no question answered worse. The token saving is real and, on that subset,
free. But it only matched *after* a fix, and the fix is the part worth repeating here.
Iteration 1 lost three questions because stage A's schema (`MetricSelection`) has no
`time_grain` field, so asked for revenue "by month" it looked for `month` among the
breakdown dimensions, failed to find it, and declined. **The general form is that stage A
can only reason about what its own schema lets it express** — the same hazard as
retrieval, one layer up. Notably it argues *for* the thesis rather than against it: the
failure was a loud over-refusal, not a confident wrong number, because the metric enum
still bounded what could be said. Narrowing what the model sees is survivable when the
narrowing is visible to it and dangerous when it isn't.

**Weakest point.** The 100-metric crossover is still arithmetic on prompt size, not a
measured accuracy curve. The ablation varied *prompt structure* at a fixed catalogue size;
it says nothing about whether selection accuracy holds at 50 or 100 cards, and it remains
plausible that discrimination degrades well before the token budget does. The ablation is
also 24 questions, where one question moves any rate by 4.2 points. Better supported than
it was, and still the thinnest empirical claim in the repo.

---

## 5. Why does the baseline get explicit permission to refuse?

Because the alternative is a rigged comparison. Without it, the baseline's under-refusal
rate on the 25 ambiguous questions would be 100% *by construction*, the headline delta
would measure my prompt rather than the architecture, and a reviewer would spot that in a
minute and correctly discount everything else. So its prompt tells it to reply
`CLARIFY: <question>` when a question is genuinely ambiguous or unanswerable, and it used
that option on 7 questions — so it's doing real work, not sitting there as an alibi. The
same reasoning drives the smaller choices: real DDL generated from the warehouse, row
counts, distinct values for the low-cardinality columns any competent text-to-SQL prompt
would include, the same data window, model, effort and cache, and markdown fences stripped
rather than scored, because a fence is a formatting slip and failing a system for it is
strawmanning.

**Weakest point.** Permission to refuse doesn't fix the deeper asymmetry, and a sharp
interviewer will go straight there: the baseline's dominant failure is missing *default
filters*, and it cannot know `net_revenue` excludes canceled orders because that rule
exists nowhere in the schema I gave it. So "execution accuracy" is really
conformance-to-my-canonical-layer, and I've defined correctness as matching the thing the
baseline lacks. My defence — that a human analyst is held to the same canonical
definitions — is a claim about how organisations work, not evidence. `shape_correct` is my
partial answer (baseline 53/55 on shape against 18/55 on value, which locates the failure
precisely rather than just scoring it), but it's a mitigation, not a refutation.

---

## 6. Why is gold checked mechanically against the metric card it was written from?

Because gold was already verified two ways and neither caught the error that mattered. The
SQL runs, and its executed value is committed — but neither of those notices gold that
*quietly disagrees with the card it was written from*. Gold that omits a card's default
filter is wrong in exactly the way the baseline is wrong, and it would mark correct answers
as failures, which is the worst kind of eval bug because it looks like a finding. The check
is mechanical: for every answerable question, does the gold SQL mention the card's default
filter columns, with the right values, and compute over the card's own columns? All 55 pass.
I only thought to write it after reading the baseline's completed arm and noticing that its
largest failure mode was one my own gold could have committed.

**Weakest point.** It's a substring check, so it proves *mention*, not *correct use*. Gold
that references `order_status` inside a comment, or filters on the right column with an
inverted predicate, passes. It also can't see a conceptual error the card itself contains —
if a card says `net_revenue` should exclude returns and the warehouse has no returns
concept, gold and card agree with each other and both are wrong about the business. And
the deepest version: I wrote the cards, the questions, and the gold, so no mechanical check
can establish that any of them match what a real finance team means.

---

## 7. Why are provider, model_id, and config hash pinned in run provenance?

Because a result is a property of a configuration, not of a system. Two runs with different
config hashes aren't comparable, and pinning the hash makes that *checkable* rather than
something a reader has to trust — it covers model id, context limit, max output tokens,
temperature and effort. The harness refuses to mix providers inside one comparison for the
same reason: if Canon answered on Groq and the baseline on Gemini, the delta would measure
the backend. That guard is now enforced at the checkpoint level, because the subtle version
of the mistake isn't running two providers at once, it's *resuming* a Groq tag with
`--provider gemini` and averaging across backends invisibly. The hash deliberately excludes
rate limits: throttling changes how long a run takes, not what it answers, and folding
limits in would make two identical results look incomparable because a quota was raised.

**Weakest point.** Pinning the identifier isn't pinning the model. `openai/gpt-oss-120b` on
Groq today and in three months may not be the same weights, the same serving stack, or the
same quantisation, and nothing in my provenance would show it — the hash would be
identical and the numbers might not be. On a free tier I have no version guarantee at all
and no way to detect silent drift, which means the reproducibility claim is really
"reproducible from the response cache", not "reproducible against the provider". Those are
very different promises, and the README currently blurs them.

---

## Results questions

### 8. Your baseline scores 32.7% execution against Canon's 90.9%. Is it a strawman?

The fair version of this challenge is specific: did I cripple it? The baseline gets the
real DDL, writes its own SQL, runs on the same model and provider, and — unusually for a
text-to-SQL baseline — gets **explicit permission to refuse**, which it uses on seven
questions. So it is not a strawman in the sense of being denied a capability Canon has.

What it is denied is the metric cards, and that is the entire experiment. The result that
makes the comparison interpretable is not the execution gap but the pair: **96.4% shape
accuracy against 32.7% execution.** It asks for the right columns at the right grain and
then computes them under the wrong definition — overwhelmingly a missing default filter,
like summing `price` without excluding canceled orders. It is not confused about the
question; it is confused about what "revenue" means, which is the one thing a canonical
layer supplies.

**Weakest point.** "Right definition" means *my* definition. A baseline scoring 32.7%
against someone else's metric cards is not evidence about the baseline's competence, it is
evidence that definitional conventions are not guessable — a weaker and more interesting
claim than the number suggests. And a stronger baseline (few-shot examples, or the DDL plus
a page of prose conventions) would close some of that gap; I did not build one, so I cannot
say how much.

### 9. Everything is n=1. How do you know these numbers are stable?

For the most part I don't, and `RESULTS.md` says so rather than printing a zero spread.
What I did measure is a targeted probe: `canon` re-run over the 24-question stratified
subset at temperature 0, reported in [`VARIANCE.md`](../benchmark/VARIANCE.md). **23 of 24
questions were identical across samples. One flipped — `t2-05`** — which makes over-refusal
the one headline figure with demonstrated sample-dependence: 5.5% on the published draw,
3.6% had it landed the other way.

The flip is the useful part. `t2-05` is the question I had already flagged, on separate
evidence, as one where Canon's clarifying question is better than my gold — "net revenue in
São Paulo state" does not say whether that means `customer_state` or `seller_state`, and
both exist. A question that lands either side of the policy margin across identical runs is
one the model is genuinely undecided about, which is what ambiguous means operationally.

**Weakest point.** Two samples, one arm, 24 of 80 questions, on a subset deliberately
chosen to be *sensitive* rather than representative. That is enough to catch a coin-flip
question and nowhere near enough to put an error bar on a published rate. The baseline has
no variance measurement at all, despite using its own `CLARIFY` option on seven questions,
so it is not obviously stable either.

---

## The one I'd least like to be asked

> *You changed your primary endpoint after seeing the baseline's results, said a
> cross-model arm would make that choice testable out-of-sample, and then never ran it.
> Why should I trust the headline?*

This one has no good answer, which is why it is first. The facts are as stated: the plan in
[`docs/01-evaluation.md`](01-evaluation.md) bolded **confident-wrong** as primary; the
published report leads on **silent-wrong**, a metric that did not exist when the plan was
written. The reason is sound — the baseline reports no confidence, so scoring its answers
at 1.0 is an imputation, and silent-wrong needs no such assumption — but I found that
reason by running the baseline. That makes silent-wrong **in-sample**. I pre-registered it
for a Gemini arm precisely so the choice could be tested out-of-sample, and that arm was
not run, so the debt is still outstanding.

What I can say: the argument for silent-wrong stands independently of the data, the
deviation is disclosed in the document that pre-registered the alternative rather than
discovered by a reader, and both numbers are reported. What I cannot say is that it has
been validated. The two metrics coinciding on this run (2.5% / 68.8%) is not reassurance
either — they coincide because no Canon answer was both wrong and confidently delivered,
which is a property of this result, not of the metrics. It is one key and one pre-registered
arm away from being settled, and until then a reader is entitled to discount it.

## The other one I'd least like to be asked

> *You wrote the metric cards, the questions, and the gold answers. What in this repo
> would tell me you're wrong about what "revenue" means?*

The concession is real but it is narrower than it first sounds, and conceding it wholesale
gives away things it does not actually touch. Three categories:

**Threatened — every accuracy number.** Execution accuracy, semantic accuracy, refusal
accuracy, the tier breakdowns, the paired-discrimination result. All of them measure
*conformance to a canonical layer I authored*. A finance team that had ruled net revenue
the default would score Canon's refusal on `am-01` as a failure rather than a success, and
nothing in this repo could argue back. Every integrity check I added — dev/test threshold
separation, gold-versus-card, attribution coverage, shape-versus-execution — tightens
internal consistency and none of them reaches outside my own judgment.

**Not threatened — the mechanism claims.** That a hallucinated metric is unrepresentable
because `MetricName` is generated from the YAML and sits inside the provider's wire schema;
that fan-out cannot be constructed because only `many_to_one` edges are traversable; that
the compiler is a pure function of `(QueryIntent, MetricCard[])`; that every emitted string
is parsed by sqlglot before execution. These are properties of the code. You verify them by
reading `canon/compiler/` and `canon/semantic/enums.py`, not by trusting my gold. They would
hold identically if someone replaced `metrics.yaml` wholesale with their own definitions —
which is the actual claim, since the layer is meant to be *theirs*, not mine.

**Not threatened — the determinism finding.** The baseline returned a different number for
the same question on 2 of 80 questions, with different SQL behind each. That comparison is
between two runs of the same system; it needs no reference answer at all, so my authorship
of the gold cannot contaminate it. It is reported in
[`RESULTS.md`](../benchmark/RESULTS.md) beside the headline for that reason, and it is the
single result I would stake most on.

So the honest framing is narrower and more defensible than "you wrote the gold, so nothing
counts": the accuracy numbers measure conformance to a stated layer and should be read that
way, while the architectural claims and the determinism result stand on their own.

**Weakest point.** The clean split above is itself doing some work for me. The mechanism
claims are *verifiable* but they are also the easy half — nobody disputes that an enum
constrains output; the interesting question is whether constraining output that way
produces better analytics, and answering that needs exactly the accuracy numbers my
authorship compromises. And the determinism finding, while gold-independent, is measured on
2 samples with an asymmetric design that favours Canon on sample size. Neither escape hatch
is as load-bearing as I would like.
