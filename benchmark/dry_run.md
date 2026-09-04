# Dry run -- projected quota and wall-clock

Measured locally with tiktoken; no provider was contacted, so this cost nothing. Counts include the system prompt, the question, the JSON schema sent with the call, and a flat 24-token message-framing allowance.

Output tokens are **assumed** at 700/structured call and 300/SQL call -- no call has been made yet. Deliberately generous, so the budget projection errs toward pessimism.

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

### Sweep projections — 80 questions x 2 run(s)

| Strategy | calls | prompt tokens | total tokens | tokens/call | days at cap | wall-clock (min) | fits in a day |
|---|---|---|---|---|---|---|---|
| canon_full | 160 | 1,106,450 | 1,218,450 | 7,615 | — | 16.0 | yes |
| canon_two_stage | 270 | 659,958 | 848,958 | 3,144 | — | 27.0 | **no (requests)** |
| baseline | 160 | 124,370 | 172,370 | 1,077 | — | 16.0 | yes |

Binding daily constraint for gemini: **requests_per_day**.

**Headline sweep (canon full-stuffing + baseline, 2 run(s)): 1,390,820 tokens.**

Two-stage would use 848,958 tokens against full-stuffing's 1,218,450 — a saving of 369,492 tokens (30%), at the cost of 110 extra requests. Whether it costs accuracy is the ablation, not an assumption.

