# Dry run -- projected quota and wall-clock

Measured locally with tiktoken; no provider was contacted, so this cost nothing. Counts include the system prompt, the question, the JSON schema sent with the call, and a flat 24-token message-framing allowance.

Output tokens are **assumed** at 700/structured call and 300/SQL call -- no call has been made yet. Deliberately generous, so the budget projection errs toward pessimism.

## anthropic — `claude-opus-5` (**PAID -- not $0 reproducible**)

- config hash: `509b26ae06506aca`
- context limit: 1,000,000 tokens
- token counts: **approximate** (`o200k_base`)
- limits: 50 rpm, 40000 tpm, — rpd, — tpd — verified: **False**
  - ⚠ limits unverified on this machine. Source: Paid-tier limits vary by account; these are placeholders and not a free tier.
- spent today: 0 requests / 0 tokens; remaining: — requests / — tokens

### Prompt sizes (one call)

| Component | Tokens |
|---|---|
| Canon full-stuffing system prompt | 5,196 |
| Canon stage-A system prompt (short cards) | 2,078 |
| Canon stage-B system prompt (smallest card) | 1,928 |
| Canon stage-B system prompt (largest card) | 2,190 |
| Baseline system prompt (DDL) | 742 |
| AgentDecision schema | 1,685 |
| MetricSelection schema (stage A) | 580 |
| IntentFill schema (stage B) | 1,468 |

### Sweep projections — 80 questions x 1 run(s)

| Strategy | calls | prompt tokens | total tokens | tokens/call | days at cap | wall-clock (min) | fits in a day |
|---|---|---|---|---|---|---|---|
| canon_full | 80 | 553,305 | 609,305 | 7,616 | — | 15.23 | yes |
| canon_two_stage | 135 | 330,059 | 424,559 | 3,145 | — | 10.61 | yes |
| baseline | 80 | 62,185 | 86,185 | 1,077 | — | 2.15 | yes |

Binding daily constraint for anthropic: **per-minute rates only**.

**Headline sweep (canon full-stuffing + baseline, 1 run(s)): 695,490 tokens.**

Two-stage would use 424,559 tokens against full-stuffing's 609,305 — a saving of 184,746 tokens (30%), at the cost of 55 extra requests. Whether it costs accuracy is the ablation, not an assumption.

## gemini — `gemini-2.5-flash` (free tier)

- config hash: `12266b91599bf919`
- context limit: 1,048,576 tokens
- token counts: **approximate** (`o200k_base`)
- limits: 10 rpm, 250000 tpm, 250 rpd, — tpd — verified: **False**
  - ⚠ limits unverified on this machine. Source: Google AI Studio free tier for the current Flash model, from provider docs at time of writing. Free-tier quotas for Gemini have changed repeatedly; re-check before quoting.
- spent today: 0 requests / 0 tokens; remaining: 250 requests / — tokens

### Prompt sizes (one call)

| Component | Tokens |
|---|---|
| Canon full-stuffing system prompt | 5,196 |
| Canon stage-A system prompt (short cards) | 2,078 |
| Canon stage-B system prompt (smallest card) | 1,928 |
| Canon stage-B system prompt (largest card) | 2,190 |
| Baseline system prompt (DDL) | 742 |
| AgentDecision schema | 1,685 |
| MetricSelection schema (stage A) | 580 |
| IntentFill schema (stage B) | 1,468 |

### Sweep projections — 80 questions x 1 run(s)

| Strategy | calls | prompt tokens | total tokens | tokens/call | days at cap | wall-clock (min) | fits in a day |
|---|---|---|---|---|---|---|---|
| canon_full | 80 | 553,305 | 609,305 | 7,616 | — | 8.0 | yes |
| canon_two_stage | 135 | 330,059 | 424,559 | 3,145 | — | 13.5 | yes |
| baseline | 80 | 62,185 | 86,185 | 1,077 | — | 8.0 | yes |

Binding daily constraint for gemini: **requests_per_day**.

**Headline sweep (canon full-stuffing + baseline, 1 run(s)): 695,490 tokens.**

Two-stage would use 424,559 tokens against full-stuffing's 609,305 — a saving of 184,746 tokens (30%), at the cost of 55 extra requests. Whether it costs accuracy is the ablation, not an assumption.

## groq — `openai/gpt-oss-120b` (free tier)

- config hash: `72af3a5ba084b045`
- context limit: 131,072 tokens
- token counts: **exact** (`o200k_harmony`)
- limits: 30 rpm, 8000 tpm, 1000 rpd, 200,000 tpd — verified: **False**
  - ⚠ limits unverified on this machine. Source: MIXED PROVENANCE -- read the two claims separately, because only one of them is actually verified.
tokens_per_minute = 8000 is PROVIDER-CONFIRMED. Groq refused a call on 2026-09-08 during run tag sweep-groq-r1 (question am-08) and quoted its own limit verbatim: "Request too large for model openai/gpt-oss-120b in organization ... service tier on_demand on tokens per minute (TPM): Limit 8000, Requested 8092". The full error is preserved in benchmark/runs/sweep-groq-r1/checkpoint.jsonl under am-08. A provider quoting its own ceiling while enforcing it is the strongest evidence available short of reading the dashboard.
tokens_per_day = 200000 is NOT verified; it is a FLOOR. On 2026-09-04, run tag sweep-groq-r1 consumed 199,516 tokens across 95 requests in one UTC day with zero quota refusals, so the real daily cap is >= 199,516. It is not confirmed to be exactly 200,000: both "exhaustion" events were this project's own limiter refusing pre-emptively because config said 200,000, never Groq refusing. Treating that as verification would verify the config against itself. What would actually verify it is a 429 or a tokens-per-day rate_limit_exceeded from Groq, which has not happened -- by design, since the limiter stops first.
requests_per_minute and requests_per_day remain unverified: neither has been approached, let alone hit.
- spent today: 0 requests / 0 tokens; remaining: 1000 requests / 200,000 tokens

### Prompt sizes (one call)

| Component | Tokens |
|---|---|
| Canon full-stuffing system prompt | 5,196 |
| Canon stage-A system prompt (short cards) | 2,078 |
| Canon stage-B system prompt (smallest card) | 1,928 |
| Canon stage-B system prompt (largest card) | 2,190 |
| Baseline system prompt (DDL) | 742 |
| AgentDecision schema | 1,685 |
| MetricSelection schema (stage A) | 580 |
| IntentFill schema (stage B) | 1,468 |

### Sweep projections — 80 questions x 1 run(s)

| Strategy | calls | prompt tokens | total tokens | tokens/call | days at cap | wall-clock (min) | fits in a day |
|---|---|---|---|---|---|---|---|
| canon_full | 80 | 553,305 | 609,305 | 7,616 | 3.05 | 76.16 | **no** |
| canon_two_stage | 135 | 330,059 | 424,559 | 3,145 | 2.12 | 53.07 | **no** |
| baseline | 80 | 62,185 | 86,185 | 1,077 | 0.43 | 10.77 | yes |

Binding daily constraint for groq: **tokens_per_day**.

**Headline sweep (canon full-stuffing + baseline, 1 run(s)): 695,490 tokens.**
That is 3.5 days at 200,000 tokens/day, or 348% of a single day's budget.

Two-stage would use 424,559 tokens against full-stuffing's 609,305 — a saving of 184,746 tokens (30%), at the cost of 55 extra requests. Whether it costs accuracy is the ablation, not an assumption.

