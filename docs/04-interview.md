# 04 — Interview questions on the design decisions

One question per ADR, each answered in a paragraph I could say out loud without notes.
Every answer ends with **the weakest point** — where a sharp follow-up would land. Those
lines are deliberately unflattering; a defence I can't puncture myself is a defence I
haven't tested.

Results questions are not here. They need numbers, and numbers wait for
[`benchmark/RESULTS.md`](../benchmark/RESULTS.md).

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

**Weakest point.** The 100-metric crossover is arithmetic on prompt size, not a measured
accuracy curve — I have no evidence that selection accuracy holds at 50 or 100 cards, and
it's entirely plausible that a model's ability to discriminate degrades well before the
token budget does. If that's true the crossover is *lower* than I claim and my argument
protects the wrong quantity. The two-stage ablation was meant to give a first data point
and it hasn't run yet, so this ADR is currently the most confidently-argued and least
empirically-supported thing in the repo.

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

## The one I'd least like to be asked

> *You wrote the metric cards, the questions, and the gold answers. What in this repo
> would tell me you're wrong about what "revenue" means?*

Nothing would. The mechanism claims are independently checkable — you can read the code and
confirm a hallucinated metric is unrepresentable and that fan-out can't be built. But the
*accuracy* claims rest on labels I authored, and a finance team that had ruled net revenue
the default would score Canon's refusal on `am-01` as a failure rather than a success.
Every integrity check I've added — dev/test threshold separation, gold-versus-card,
attribution coverage, shape-versus-execution — tightens internal consistency. None of them
reaches outside my own judgment. The honest framing, which belongs in RESULTS.md and not
just here, is that this measures **conformance to a stated canonical layer**, and the
project's actual claim is that having one beats not having one.
