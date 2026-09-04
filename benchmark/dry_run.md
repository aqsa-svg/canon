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
| AgentDecision schema | 1,684 |
| MetricSelection schema (stage A) | 579 |
| IntentFill schema (stage B) | 1,468 |

### Sweep projections — 80 questions x 1 run(s)

| Strategy | calls | prompt tokens | total tokens | tokens/call | days at cap | wall-clock (min) | fits in a day |
|---|---|---|---|---|---|---|---|
| canon_full | 80 | 553,225 | 609,225 | 7,615 | — | 15.23 | yes |
| canon_two_stage | 135 | 329,979 | 424,479 | 3,144 | — | 10.61 | yes |
| baseline | 80 | 62,185 | 86,185 | 1,077 | — | 2.15 | yes |

Binding daily constraint for anthropic: **per-minute rates only**.

**Headline sweep (canon full-stuffing + baseline, 1 run(s)): 695,410 tokens.**

Two-stage would use 424,479 tokens against full-stuffing's 609,225 — a saving of 184,746 tokens (30%), at the cost of 55 extra requests. Whether it costs accuracy is the ablation, not an assumption.

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
| AgentDecision schema | 1,684 |
| MetricSelection schema (stage A) | 579 |
| IntentFill schema (stage B) | 1,468 |

### Sweep projections — 80 questions x 1 run(s)

| Strategy | calls | prompt tokens | total tokens | tokens/call | days at cap | wall-clock (min) | fits in a day |
|---|---|---|---|---|---|---|---|
| canon_full | 80 | 553,225 | 609,225 | 7,615 | — | 8.0 | yes |
| canon_two_stage | 135 | 329,979 | 424,479 | 3,144 | — | 13.5 | yes |
| baseline | 80 | 62,185 | 86,185 | 1,077 | — | 8.0 | yes |

Binding daily constraint for gemini: **requests_per_day**.

**Headline sweep (canon full-stuffing + baseline, 1 run(s)): 695,410 tokens.**

Two-stage would use 424,479 tokens against full-stuffing's 609,225 — a saving of 184,746 tokens (30%), at the cost of 55 extra requests. Whether it costs accuracy is the ablation, not an assumption.

## groq — `openai/gpt-oss-120b` (free tier)

- config hash: `72af3a5ba084b045`
- context limit: 131,072 tokens
- token counts: **exact** (`o200k_harmony`)
- limits: 30 rpm, 8000 tpm, 1000 rpd, 200,000 tpd — verified: **False**
  - ⚠ limits unverified on this machine. Source: Groq free-tier limits for openai/gpt-oss-120b as stated in the project brief. tokens_per_minute is the conservative figure; the binding constraint is tokens_per_day.
- spent today: 0 requests / 0 tokens; remaining: 1000 requests / 200,000 tokens

### Prompt sizes (one call)

| Component | Tokens |
|---|---|
| Canon full-stuffing system prompt | 5,196 |
| Canon stage-A system prompt (short cards) | 2,078 |
| Canon stage-B system prompt (smallest card) | 1,928 |
| Canon stage-B system prompt (largest card) | 2,190 |
| Baseline system prompt (DDL) | 742 |
| AgentDecision schema | 1,684 |
| MetricSelection schema (stage A) | 579 |
| IntentFill schema (stage B) | 1,468 |

### Sweep projections — 80 questions x 1 run(s)

| Strategy | calls | prompt tokens | total tokens | tokens/call | days at cap | wall-clock (min) | fits in a day |
|---|---|---|---|---|---|---|---|
| canon_full | 80 | 553,225 | 609,225 | 7,615 | 3.05 | 76.15 | **no** |
| canon_two_stage | 135 | 329,979 | 424,479 | 3,144 | 2.12 | 53.06 | **no** |
| baseline | 80 | 62,185 | 86,185 | 1,077 | 0.43 | 10.77 | yes |

Binding daily constraint for groq: **tokens_per_day**.

**Headline sweep (canon full-stuffing + baseline, 1 run(s)): 695,410 tokens.**
That is 3.5 days at 200,000 tokens/day, or 348% of a single day's budget.

Two-stage would use 424,479 tokens against full-stuffing's 609,225 — a saving of 184,746 tokens (30%), at the cost of 55 extra requests. Whether it costs accuracy is the ablation, not an assumption.

