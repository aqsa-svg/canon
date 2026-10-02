# 03 — Failure taxonomy

Every failure read individually, from the per-question records in
`benchmark/runs/*/checkpoint.jsonl` and `decisions.jsonl` — not summarised from scores.

**Status: COMPLETE.** Both arms are whole (80/80 each, 160/160 units) and every Canon
failure below was read individually from its own record — SQL re-executed against the
warehouse where the diagnosis depended on it. Seven Canon units fail in total. Individual
cases are verified against the checkpoint and cited by question id; aggregate rates live
only in
[`benchmark/RESULTS.md`](../benchmark/RESULTS.md), per the number rule in `CLAUDE.md`.

The most credible section of this document is the last one, and the second-most credible is
the first. Both are about my mistakes rather than the system's.

---

## A. Failures of the harness and the eval set — mine, not the system's

These are scored as Canon failures and are not Canon's fault. They are listed first
because a taxonomy that buries them is advocacy, not analysis.

### A1. The eval set assumed a convention the prompt never declared

**Count: 2 of 55 answerable questions** (`t1-09`, `t1-10`). Verified: both asked for a
time range and were scored as over-refusals.

`t1-09` — *"On average, how many days did delivery take?"* Canon scored
`avg_delivery_days` at 1.00 with nothing else close (gap 1.00), then declined to commit,
asking *"which period should the average be calculated over?"*. `t1-10` behaved
identically.

That is defensible behaviour. My prompt tells the model that relative references —
"recently", "last month" — are missing information because the warehouse has no today. It
never says what an **absent** period means. My eval set silently assumes "no period stated"
means "all data", and never told the model. The model generalised the stated rule to the
unstated case, which is the more conservative reading and arguably the better one:
`CLAUDE.md` explicitly forbids "defaulting a missing time grain to month because it seemed
reasonable", and defaulting a missing *range* to all-time is the same species of guess.

**Effect on the headline:** inflates Canon's over-refusal rate by at most 2/55 = 3.6
percentage points. Stated so a reader can adjust rather than being quietly corrected.

**The fix, deliberately not applied mid-sweep:** declare the convention in the prompt — a
question naming no period covers the whole warehouse — because that is exactly the kind of
thing a canonical layer should state rather than leave to inference. Applying it changes
the prompt, which changes the cache key, which costs a full re-run of both arms. A 1.25-day
re-run to remove a 3.6pp artifact I can quantify precisely is worse value than disclosing
it, and disclosure is the more credible of the two.

### A2. The harness's own schema rejected the correct answer

**Count: 2 of 8 refuse-class questions** (`am-18` customer acquisition cost, `am-21`
inventory level). Verified: both recorded as `errored`, with the provider's message
preserved in the checkpoint.

Asked for a metric the catalogue does not contain, the model returned an **empty**
candidate list and named the gap — *"customer acquisition cost metric is not defined in the
catalogue"*. That is the most honest possible answer: nothing in the catalogue fits, so
nothing should be scored as fitting. My `AgentDecision` schema requires
`candidates: minItems 1`, so Groq rejected the payload and the question errored out.

The schema encoded a false assumption: that at least one metric is always plausible. For
*"what's our inventory level?"* the correct candidate count is zero. This is the same class
of error as the baseline not knowing about default filters — a rule that exists nowhere the
model can see it.

**Effect:** two refuse-class questions score as errors rather than as correct refusals, on
the exact question class the project is about. Canon's ambiguous-class result is therefore
understated.

**The fix:** allow an empty candidate list, and add a policy rule that reads it as the
strongest available refusal signal — no metric fits, so say the metric is not defined.
Deferred for the same cache-invalidation reason as A1, and recorded here so the number is
read with it in mind.

### A3. A transient provider throttle was classified as permanent

**Count: 0 in the published run — 1 before the fix** (`am-08`). Verified from the
preserved error.

Groq refused with HTTP 413, code `rate_limit_exceeded`: *"on tokens per minute (TPM): Limit
8000, Requested 8092"*. Neither 413 nor the underscored code matched the retryable-error
list, so a throttle that a retry would have cleared killed the question for good.

Root cause was mine twice over: the local token estimate ran 2% under Groq's own count
(6,914 against 7,035 across 36 calls), which under-books the per-minute bucket until the
drift trips the provider's limiter. **Both fixed** — 413 and `rate_limit` are now
retryable, and the limiter pads estimates by 10%, deliberately more than the measured 2%
because under-booking loses a question while over-booking only wastes headroom.

**A third fix, and a reversal.** The retry above only covers *within* a run; `am-08` still
died because a throttle was not classified as a **transport** error, so a resume reloaded it
as permanently failed. An earlier version of this repo asserted that on purpose — a test
named `test_a_throttle_is_not_transport` recorded the throttle as "a property of the quota
rather than of the machine". That reasoning was wrong, and it confused *whose fault* the
failure was with *whether it is a finding about the architecture*. The limiter estimated
8,092 tokens against an 8,000 cap and the model never saw the question, so the system under
test did not fail — the arithmetic in this repo did. Reclassified, `am-08` was re-attempted
and answered correctly, and the test now asserts the opposite of what it once did, with the
reversal written into its docstring rather than quietly swapped. Canon's error rate moved
3.8% to 2.5% as a result.

### A4. A scoring convention penalised the baseline twice

**Count: affected 3 questions before the fix** (`t2-07`, `t2-08`, `t3-10`). Fixed.

Gold rows for comparison questions are keyed `current`/`comparison`, because gold was
written from the metric cards and that is Canon's output convention. A system answering
*"March vs the month before"* with two month-keyed rows has answered correctly and was
being marked wrong on a convention it had no way to know. Found by reading the baseline's
SQL rather than its score. Patched in two places — value comparison, then shape comparison
— and I found the second only because I went looking after the first. Baseline shape
accuracy moved 51/55 → 53/55 as a result, i.e. against Canon's interest.

---

## B. Failures of metric selection and definition

### B1. Wrong metric selected

**Canon: 0 of 55.** Every answerable question Canon answered used the gold metric
(`metric_correct` is true across all 55; semantic accuracy 100%). This is the claim the
enum design exists to make structurally, so a zero here is the expected result rather than a
surprising one — a wrong metric is unrepresentable, not merely rare.
**Baseline: not directly measurable** — it declares no metric, which is
§3.2's point. Where a wrong number is traceable to another question's gold value, the
attribution instrument names it, subject to the coverage limit reported in RESULTS.md.

### B2. Correct metric, wrong grain

**Canon: 0 of 55.** The compiler makes the illegal cases unrepresentable — `avg_order_value` by
`product_category` is refused before SQL exists — so a grain failure here would have to be
a *legal* pair used inappropriately, e.g. answering a monthly question at daily grain.

### B3. Filter mis-specified — the baseline's dominant failure

**Baseline: 35 of 55 answerable questions had the right shape and the wrong value.**
Shape accuracy 53/55 against execution accuracy 18/55. Verified over the complete arm.

This is the thesis, stated as two numbers rather than asserted. The baseline overwhelmingly
asks for the right thing and computes it under the wrong definition. Read individually, the
pattern is almost always a **missing default filter**:

- *"How much freight revenue did we bill in 2018?"* — summed `freight_value` with no
  order-status filter. The card excludes canceled and unavailable orders.
- *"What was the average item price in 2018?"* — averaged `price`, no status filter.
- *"How many sellers made a sale in 2018?"* — counted distinct `seller_id`, no status filter.
- *"What was net revenue in 2018?"* — summed `order_payments.payment_value` (8,699,763.05)
  where the card sums `order_items.price` (7,341,182.41). A different table entirely.
- *"How many orders were delivered in 2018?"* — counted by `order_delivered_customer_date`
  rather than purchase date.

Each is defensible in isolation. None matches the canonical definition, and the baseline had
no way to know the definition exists. **Canon: 0 of 55** — no Canon answer failed on a
filter. The default filters come from the card, so there is nothing for the model to omit.

### B5. Right number, wrong presentation — Canon's only answer-class failures

**Canon: 2 of 55** (`t2-03`, `t2-06`). Both were scored wrong, and in both the arithmetic is
*exactly* gold's. I found this by re-executing the stored SQL rather than trusting the
score, and it changes what the failures mean.

- `t2-03` — *"Which five product categories brought in the most gross revenue in 2018?"*
  Canon grouped by `product_category` (Portuguese) where gold uses
  `product_category_english`, which needs a third join through the translation table. The
  five revenue figures are identical to gold's, in the same order, to the cent
  (885191.12, 771986.75, 650794.70, 621999.34, …); only the labels differ —
  `beleza_saude` against `health_beauty`. A real failure, and a narrow one: the model picked
  the untranslated sibling dimension, not the wrong metric, filter, ranking or grain.
- `t2-06` — *"Show net revenue by month in 2018 for São Paulo and Rio de Janeiro states
  only."* Gold groups by month **and** state; Canon filtered to SP and RJ and grouped by
  month alone. Canon's monthly totals equal gold's rows summed over state, exactly
  (2018-01: 495257.48 = 374404.45 + 120853.03). The reading is defensible — *"by month"*
  names the breakdown and *"for SP and RJ only"* names a filter — and the question is
  weaker than I intended. Left as written rather than edited after seeing results.

**Both are dimension-selection failures, not definition failures**, which is why
`shape_correct` catches `t2-06` (98.1% against 90.9% execution) and misses `t2-03`, whose
column count and grain are right. Worth stating against my own interest: a benchmark
reporting only execution accuracy would call these two the same kind of failure as the
baseline's missing-filter cases, and they are not.

### B4. Caveat not surfaced

**Canon: 1 of 8 cases requiring one** (`t3-10`). Asserted per question by `requires_caveat`; the tier-3 cases are where it
bites, especially `t3-02` (the 2018-01-01 restatement must fire) paired against `t3-03`
(the same metric inside one regime, where it must stay silent).

---

## C. Failures of the refusal decision

### C1. Over-refusal — asked when it should have answered

**Canon: 3 of 55** (`t1-09`, `t1-10`, `t2-05`) — the largest single category of Canon
failure, and only two of the three are the A1 artefact.

`t2-05` is different and I had it filed wrongly until I read it. *"What was net revenue in
São Paulo state in 2018?"* Canon asked: *"net revenue for orders placed by customers in São
Paulo (`customer_state`) or for sales made by sellers located in São Paulo
(`seller_state`)?"* Both dimensions exist, both are legal, and the question does not say.
Gold resolves it to `customer_state` and its note asserts only that *"the state code is
SP"* — which answers the code, not the fork. **Canon's clarifying question is the better
answer and is scored a failure.** This is an eval-set weakness of the same species as
`t1-04` in section D, and it is left as written for the same reason: editing a question
after seeing it cost Canon a point is worse than reporting the point.

### C2. Under-refusal — answered when it should have asked

**Baseline: 18 of 25**, verified over the complete arm. It used its `CLARIFY` option on 7
questions (`am-05`, `am-08`, `am-09`, `am-21`, `am-23`, `am-24`, `am-25`) — so the
permission to refuse is doing real work rather than serving as an alibi, and the remaining
18 are genuine guesses at underspecified questions. **Canon: 0 of 25** — under-refusal
is the failure the project most wants at zero, and in this run it is.

### C3. Refuse instead of clarify

**Canon: 1 of 25**, with 3 more in the opposite direction (refuse-class questions answered
with a clarifying question — a *scope* failure, not a trust one). The 3×3 behaviour matrix
separates this from a correct refusal: declining a
resolvable question by claiming the metric is undefined is a *capability* failure, distinct
from answering one that should have been asked about, which is a *trust* failure. Summing
them into one refusal-accuracy number hides which kind of system you have.

### C4. Comparison window wrong

**Canon: 0.** The compiler derives both windows, so a failure here means the model set the
wrong anchor range or the wrong comparison kind, not that the arithmetic slipped.

---

## D. Cases where the gold answer itself was wrong

**Found so far: none of the class I can check mechanically.**

All 55 gold queries apply their card's default filters, with the right values, and compute
over the card's own columns — asserted in the test suite
(`TestGoldAgreesWithTheCards`). That check exists because reading the baseline's completed
arm showed its largest failure mode was one my own gold could have committed, and I went
looking for it in my own work.

**What that check cannot see, stated plainly:**

- It matches substrings, so it proves *mention*, not correct use. Gold referencing
  `order_status` in a comment, or filtering the right column with an inverted predicate,
  passes.
- It cannot see a conceptual error inside the card itself. If a card says `net_revenue`
  should exclude returns and the warehouse has no returns concept, gold and card agree with
  each other and both are wrong about the business.
- It cannot establish that any card matches what a real finance team means, because I wrote
  the cards, the questions, and the gold. No mechanical check reaches outside my own
  judgment.

One near-miss worth recording, because it is the shape a real gold error would take:
`t1-04` — *"How many orders were delivered in 2018?"* — is genuinely ambiguous between
purchase date and delivery date. Gold resolves it to purchase date, following the card, and
the card says so in a caveat. The baseline chose delivery date and was scored wrong. Gold is
defensible and the *question* is weaker than I intended; a sharper wording would say "orders
placed in 2018 that were delivered". It is left as written rather than edited after seeing
results, which would be a worse sin than an imprecise question.

**The gold review, now run.** Both arms are complete, so the check this section promised —
every question where Canon and the baseline agree with each other and disagree with gold —
has been executed by re-running all three SQL strings against the warehouse and comparing
rows directly.

**Two questions have both arms wrong: `t2-03` and `t2-06`. On neither do the two arms
agree with each other.** The strongest available signal of a gold error fired nowhere.

That is a weaker result than it sounds, and I would rather say so than bank it:

- The check has a sample size of two. With 55 answerable questions and Canon failing 2 of
  them, there was almost no opportunity for the signal to appear. A benchmark where the
  system under test is accurate is a benchmark where this particular check is nearly blind.
- Both arms being wrong for *unrelated* reasons is the common case and is what happened
  here. On `t2-03` Canon's revenue figures match gold to the cent and only the category
  labels differ, while the baseline's figures are wrong outright (772,238.15 against
  885,191.12 on the top row — it omits freight). On `t2-06` Canon reconciles exactly to gold
  summed over state, while the baseline is wrong again on the order-status filter. Two
  different failures that happen to land on the same question are not corroboration.
- So the honest reading is **not** "gold is verified". It is: the one mechanical check that
  could have caught a gold error I cannot see had almost nothing to work with, and found
  nothing. Section D's three stated blind spots are untouched by it.

What the review *did* surface is in B5 and C1: two scored failures (`t2-03`, `t2-06`) where
Canon's arithmetic is exactly right and the disagreement is about presentation, and one
(`t2-05`) where Canon's clarifying question is better than my gold. None was visible in the
score; all three needed the SQL re-executed. **A score tells you a question failed. Only
the query tells you what failed**, which is the argument this project makes about analytics
agents, turned on the benchmark itself.
