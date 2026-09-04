"""Project the cost of a sweep without spending any quota.

Everything here is measured locally. No provider is contacted — a dry-run that called a
``count_tokens`` endpoint would spend the request budget it exists to protect.

What it answers, per provider and per prompting strategy:

* how many tokens one question costs, and the whole sweep;
* how many days that is against the provider's daily token cap, which is the binding
  free-tier constraint;
* wall-clock from RPM and TPM;
* how much budget is left today, read from the persistent ledger.

Token counts for Groq are exact (the GPT-OSS harmony tokenizer is available locally).
For Gemini and Anthropic they are approximate and labelled as such.

Run:  uv run python scripts/dry_run.py
      uv run python scripts/dry_run.py --provider groq --runs 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmark.harness.baseline import NaiveBaseline
from benchmark.harness.dataset import load_questions
from canon.agent.prompt import (
    DataWindow,
    build_stage_a_prompt,
    build_stage_b_prompt,
    build_system_prompt,
)
from canon.agent.providers import (
    FREE_TIER,
    QuotaLedger,
    available_providers,
    count_call,
    count_tokens,
    load_config,
    project,
)
from canon.schemas.decision import AgentDecision, IntentFill, MetricSelection
from canon.semantic.loader import load_semantic_layer

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "canon.duckdb"
OUTPUT = REPO_ROOT / "benchmark" / "dry_run.md"

ESTIMATED_COMPLETION_TOKENS = 700
"""Assumed output length per structured call.

Not measured -- no call has been made. Deliberately generous: a QueryIntent with scored
candidates and a reading runs a few hundred tokens, and over-estimating output makes the
budget projection conservative, which is the safe direction. Replace with the measured
mean after the first real run.
"""

BASELINE_COMPLETION_TOKENS = 300
"""A SQL query is shorter than a scored structured object."""


def _schema_json(schema: type) -> str:
    return json.dumps(schema.model_json_schema(), sort_keys=True)


def _window() -> DataWindow:
    import duckdb

    connection = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        first, last = connection.execute(
            "SELECT min(order_purchase_timestamp), max(order_purchase_timestamp) FROM orders"
        ).fetchone()
    finally:
        connection.close()
    return DataWindow(first_order=first.date(), last_order=last.date())


def measure(provider: str) -> dict[str, Any]:
    """Token counts for every system and strategy, for one provider's tokenizer."""
    layer = load_semantic_layer()
    window = _window()
    questions = load_questions()

    full_system = build_system_prompt(layer, window)
    stage_a_system = build_stage_a_prompt(layer, window)

    decision_schema = _schema_json(AgentDecision)
    selection_schema = _schema_json(MetricSelection)
    fill_schema = _schema_json(IntentFill)

    baseline = NaiveBaseline(db_path=DB_PATH)
    try:
        baseline_system = baseline.system_prompt
    finally:
        baseline.close()

    # Per-metric stage-B prompt sizes, so the estimate uses the card that would actually
    # be fetched rather than an average that hides the spread.
    stage_b_tokens: dict[str, int] = {}
    for name in layer.metrics:
        stage_b_tokens[name] = count_call(
            system=build_stage_b_prompt(layer, window, name),
            user="",
            schema_json=fill_schema,
            provider=provider,
        ).tokens

    canon_full = 0
    canon_stage_a = 0
    canon_stage_b = 0
    stage_b_calls = 0
    baseline_total = 0
    accuracy = "exact"

    for item in questions.questions:
        full = count_call(
            system=full_system,
            user=item.question,
            schema_json=decision_schema,
            provider=provider,
        )
        accuracy = full.accuracy.value
        canon_full += full.tokens

        canon_stage_a += count_call(
            system=stage_a_system,
            user=item.question,
            schema_json=selection_schema,
            provider=provider,
        ).tokens

        # Stage B runs only when stage A commits to a metric. The eval set's own label is
        # the best available predictor: an `answer` case resolves to its expected metric,
        # and a case where asking is correct should stop after stage A.
        if item.expected_behavior.should_answer and item.expected_metric:
            canon_stage_b += (
                stage_b_tokens[item.expected_metric] + count_tokens(item.question, provider).tokens
            )
            stage_b_calls += 1

        baseline_total += count_call(
            system=baseline_system, user=item.question, schema_json="", provider=provider
        ).tokens

    total_questions = len(questions.questions)
    return {
        "provider": provider,
        "accuracy": accuracy,
        "questions": total_questions,
        "prompt_sizes": {
            "canon_full_system": count_tokens(full_system, provider).tokens,
            "canon_stage_a_system": count_tokens(stage_a_system, provider).tokens,
            "canon_stage_b_system_min": min(stage_b_tokens.values()),
            "canon_stage_b_system_max": max(stage_b_tokens.values()),
            "baseline_system": count_tokens(baseline_system, provider).tokens,
            "decision_schema": count_tokens(decision_schema, provider).tokens,
            "selection_schema": count_tokens(selection_schema, provider).tokens,
            "fill_schema": count_tokens(fill_schema, provider).tokens,
        },
        "strategies": {
            "canon_full": {"calls": total_questions, "prompt_tokens": canon_full},
            "canon_two_stage": {
                "calls": total_questions + stage_b_calls,
                "prompt_tokens": canon_stage_a + canon_stage_b,
                "stage_a_calls": total_questions,
                "stage_b_calls": stage_b_calls,
                "stage_a_tokens": canon_stage_a,
                "stage_b_tokens": canon_stage_b,
            },
            "baseline": {"calls": total_questions, "prompt_tokens": baseline_total},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=None, help="one provider, or all if omitted")
    parser.add_argument("--runs", type=int, default=1, help="independent runs to project")
    args = parser.parse_args()

    # The report is written as UTF-8 regardless; this only stops a Windows console
    # from aborting the run on a character it cannot render.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not DB_PATH.exists():
        print(f"{DB_PATH} not built -- run: uv run python scripts/load_data.py")
        return 1

    providers = [args.provider] if args.provider else list(available_providers())
    ledger = QuotaLedger()
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("# Dry run -- projected quota and wall-clock")
    emit()
    emit(
        "Measured locally with tiktoken; no provider was contacted, so this cost nothing. "
        "Counts include the system prompt, the question, the JSON schema sent with the "
        "call, and a flat 24-token message-framing allowance."
    )
    emit()
    emit(
        f"Output tokens are **assumed** at {ESTIMATED_COMPLETION_TOKENS}/structured call and "
        f"{BASELINE_COMPLETION_TOKENS}/SQL call -- no call has been made yet. Deliberately "
        "generous, so the budget projection errs toward pessimism."
    )
    emit()

    for name in providers:
        config = load_config(name)
        limits = config.limits
        measurement = measure(name)
        sizes = measurement["prompt_sizes"]
        free = "free tier" if name in FREE_TIER else "**PAID -- not $0 reproducible**"

        emit(f"## {name} — `{config.model_id}` ({free})")
        emit()
        emit(f"- config hash: `{config.config_hash}`")
        emit(f"- context limit: {config.context_limit:,} tokens")
        emit(
            f"- token counts: **{measurement['accuracy']}** (`{count_tokens('x', name).encoding}`)"
        )
        emit(
            f"- limits: {limits.requests_per_minute} rpm, "
            f"{limits.tokens_per_minute or '—'} tpm, "
            f"{limits.requests_per_day or '—'} rpd, "
            f"{(limits.tokens_per_day and f'{limits.tokens_per_day:,}') or '—'} tpd "
            f"— verified: **{limits.verified}**"
        )
        if not limits.verified:
            emit(f"  - ⚠ limits unverified on this machine. Source: {limits.source.strip()}")
        requests_left, tokens_left = ledger.remaining(name, limits)
        used = ledger.usage(name)
        emit(
            f"- spent today: {used.requests} requests / {used.tokens:,} tokens; "
            f"remaining: {requests_left if requests_left is not None else '—'} requests / "
            f"{f'{tokens_left:,}' if tokens_left is not None else '—'} tokens"
        )
        emit()

        emit("### Prompt sizes (one call)")
        emit()
        emit("| Component | Tokens |")
        emit("|---|---|")
        for label, key in (
            ("Canon full-stuffing system prompt", "canon_full_system"),
            ("Canon stage-A system prompt (short cards)", "canon_stage_a_system"),
            ("Canon stage-B system prompt (smallest card)", "canon_stage_b_system_min"),
            ("Canon stage-B system prompt (largest card)", "canon_stage_b_system_max"),
            ("Baseline system prompt (DDL)", "baseline_system"),
            ("AgentDecision schema", "decision_schema"),
            ("MetricSelection schema (stage A)", "selection_schema"),
            ("IntentFill schema (stage B)", "fill_schema"),
        ):
            emit(f"| {label} | {sizes[key]:,} |")
        emit()

        emit(f"### Sweep projections — 80 questions x {args.runs} run(s)")
        emit()
        emit(
            "| Strategy | calls | prompt tokens | total tokens | tokens/call | days at cap | "
            "wall-clock (min) | fits in a day |"
        )
        emit("|---|---|---|---|---|---|---|---|")

        projections: dict[str, Any] = {}
        for strategy, data in measurement["strategies"].items():
            completion = (
                BASELINE_COMPLETION_TOKENS
                if strategy == "baseline"
                else ESTIMATED_COMPLETION_TOKENS
            )
            calls = data["calls"] * args.runs
            prompt_tokens = data["prompt_tokens"] * args.runs
            projection = project(
                provider=name,
                limits=limits,
                calls=calls,
                prompt_tokens=prompt_tokens,
                completion_tokens_per_call=completion,
                accuracy=measurement["accuracy"],
                tokens_remaining_today=tokens_left,
            )
            projections[strategy] = projection.model_dump(mode="json")
            fits = "yes" if projection.fits_in_one_day else "**no**"
            if not projection.requests_per_day_ok:
                fits = "**no (requests)**"
            emit(
                f"| {strategy} | {projection.calls} | {projection.prompt_tokens:,} | "
                f"{projection.total_tokens:,} | {projection.tokens_per_call:,.0f} | "
                f"{projection.days_required if projection.days_required is not None else '—'} | "
                f"{projection.wall_clock_minutes} | {fits} |"
            )
        emit()

        # Which ceiling actually binds differs by provider, and saying which one keeps a
        # reader from planning against the wrong constraint.
        binding = (
            "tokens_per_day"
            if limits.tokens_per_day
            else ("requests_per_day" if limits.requests_per_day else "per-minute rates only")
        )
        emit(f"Binding daily constraint for {name}: **{binding}**.")
        emit()

        headline = projections.get("canon_full")
        baseline_projection = projections.get("baseline")
        two_stage = projections.get("canon_two_stage")
        if headline and baseline_projection:
            combined = headline["total_tokens"] + baseline_projection["total_tokens"]
            emit(
                f"**Headline sweep (canon full-stuffing + baseline, {args.runs} run(s)): "
                f"{combined:,} tokens.**"
            )
            if limits.tokens_per_day:
                emit(
                    f"That is {combined / limits.tokens_per_day:.1f} days at "
                    f"{limits.tokens_per_day:,} tokens/day, or "
                    f"{combined / limits.tokens_per_day * 100:.0f}% of a single day's budget."
                )
            emit()
        if two_stage and headline:
            saving = headline["total_tokens"] - two_stage["total_tokens"]
            pct = 100 * saving / headline["total_tokens"] if headline["total_tokens"] else 0
            emit(
                f"Two-stage would use {two_stage['total_tokens']:,} tokens against "
                f"full-stuffing's {headline['total_tokens']:,} — a "
                f"{'saving' if saving > 0 else 'increase'} of {abs(saving):,} tokens "
                f"({abs(pct):.0f}%), at the cost of "
                f"{two_stage['calls'] - headline['calls']} extra requests. Whether it costs "
                "accuracy is the ablation, not an assumption."
            )
            emit()

        if config.context_limit < sizes["canon_full_system"] + 2000:
            emit(
                f"⚠ **The full-stuffing prompt does not comfortably fit** {name}'s "
                f"{config.context_limit:,}-token context. Two-stage is not optional here."
            )
            emit()

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten to {OUTPUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
