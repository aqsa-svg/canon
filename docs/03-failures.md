# 03 — Failure taxonomy

Every failure read individually, from the per-question records in
`benchmark/runs/*/checkpoint.jsonl` and `decisions.jsonl` — not summarised from scores.

**Status: IN PROGRESS.** The baseline arm is complete (80/80). The Canon arm is not, so
every Canon *count* below is `TBD` and will be filled by reading each failed unit once the
arm is whole. Individual cases named here are verified against the checkpoint and cited by
question id; aggregate rates live only in
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

**Count: 1** (`am-08`). Verified from the preserved error.

Groq refused with HTTP 413, code `rate_limit_exceeded`: *"on tokens per minute (TPM): Limit
8000, Requested 8092"*. Neither 413 nor the underscored code matched the retryable-error
list, so a throttle that a retry would have cleared killed the question for good.

Root cause was mine twice over: the local token estimate ran 2% under Groq's own count
(6,914 against 7,035 across 36 calls), which under-books the per-minute bucket until the
drift trips the provider's limiter. **Both fixed** — 413 and `rate_limit` are now
retryable, and the limiter pads estimates by 10%, deliberately more than the measured 2%
because under-booking loses a question while over-booking only wastes headroom. The fix
lands for future runs; `am-08` remains an error in this one.

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

**Canon: TBD.** **Baseline: not directly measurable** — it declares no metric, which is
§3.2's point. Where a wrong number is traceable to another question's gold value, the
attribution instrument names it, subject to the coverage limit reported in RESULTS.md.

### B2. Correct metric, wrong grain

**Canon: TBD.** The compiler makes the illegal cases unrepresentable — `avg_order_value` by
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
no way to know the definition exists. **Canon: TBD.**

### B4. Caveat not surfaced

**Canon: TBD.** Asserted per question by `requires_caveat`; the tier-3 cases are where it
bites, especially `t3-02` (the 2018-01-01 restatement must fire) paired against `t3-03`
(the same metric inside one regime, where it must stay silent).

---

## C. Failures of the refusal decision

### C1. Over-refusal — asked when it should have answered

**Canon: TBD.** Two cases are already known and attributable to A1 rather than to the
policy.

### C2. Under-refusal — answered when it should have asked

**Baseline: 18 of 25**, verified over the complete arm. It used its `CLARIFY` option on 7
questions (`am-05`, `am-08`, `am-09`, `am-21`, `am-23`, `am-24`, `am-25`) — so the
permission to refuse is doing real work rather than serving as an alibi, and the remaining
18 are genuine guesses at underspecified questions. **Canon: TBD.**

### C3. Refuse instead of clarify

**Canon: TBD.** The 3×3 behaviour matrix separates this from a correct refusal: declining a
resolvable question by claiming the metric is undefined is a *capability* failure, distinct
from answering one that should have been asked about, which is a *trust* failure. Summing
them into one refusal-accuracy number hides which kind of system you have.

### C4. Comparison window wrong

**Canon: TBD.** The compiler derives both windows, so a failure here means the model set the
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

**Remaining gold review, once the Canon arm completes:** every question where Canon and the
baseline agree with each other and disagree with gold. Two independent implementations
agreeing against gold is the strongest available signal of a gold error, and it is the check
I most expect to find something.
