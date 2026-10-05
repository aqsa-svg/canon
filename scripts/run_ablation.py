"""Run the full-stuffing vs two-stage ablation on the stratified subset.

The question under test is narrow and worth stating precisely: **does splitting the prompt
into "choose a metric from short cards" then "fill the intent from one full card" cost
accuracy?** The dry run says it saves about 30% of tokens. Whether that saving is free is
not something a token count can answer.

Only the two-stage arm is executed here. The single-stage arm for these same questions was
already run as part of ``sweep-groq-r1`` and is read from that run's records, so the
comparison costs nothing beyond the new arm and -- more importantly -- compares against the
*published* numbers rather than against a fresh run that might differ for unrelated reasons.

Resumable on the same terms as the main harness: completed units are skipped, transport
failures are re-attempted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.harness.dataset import Question, QuestionSet, load_questions
from benchmark.harness.run import _load_gold, score_canon
from benchmark.harness.score import QuestionScore
from canon.agent.llm import CallContext, LlmClient
from canon.agent.pipeline import CanonPipeline
from canon.agent.policy import PolicyThresholds
from canon.agent.providers import DEFAULT_PROVIDER, is_transport_error
from canon.semantic.loader import load_semantic_layer

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBSET = REPO_ROOT / "benchmark" / "ablation_subset.json"
BASELINE_RUN = REPO_ROOT / "benchmark" / "runs" / "sweep-groq-r1" / "canon.jsonl"
OUT_DIR = REPO_ROOT / "benchmark" / "runs" / "ablation-twostage"
OUTPUT = REPO_ROOT / "benchmark" / "ABLATION.md"

ARM = "canon_two_stage"


def _load_subset() -> tuple[tuple[str, ...], dict[str, Any]]:
    payload = json.loads(SUBSET.read_text(encoding="utf-8"))
    return tuple(payload["questions"]), payload


def _load_single_stage() -> dict[str, QuestionScore]:
    """The already-published single-stage arm, read rather than re-run."""
    if not BASELINE_RUN.exists():
        raise FileNotFoundError(
            f"{BASELINE_RUN} not found. The ablation compares against the published "
            "single-stage arm; run the main harness first."
        )
    scores: dict[str, QuestionScore] = {}
    for line in BASELINE_RUN.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            scores[record["question_id"]] = QuestionScore(
                **{k: v for k, v in record.items() if not k.startswith("_")}
            )
    return scores


def _load_checkpoint(path: Path) -> dict[str, QuestionScore]:
    """Completed two-stage units. Transport failures are left out so they re-run."""
    if not path.exists():
        return {}
    done: dict[str, QuestionScore] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("outcome") == "errored" and is_transport_error(record.get("error") or ""):
            continue
        done[record["question_id"]] = QuestionScore(
            **{k: v for k, v in record.items() if not k.startswith("_")}
        )
    return done


def _append(path: Path, score: QuestionScore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(score.model_dump(mode="json")) + "\n")


def _agrees(left: QuestionScore, right: QuestionScore) -> bool:
    """Whether the two strategies behaved the same way on one question."""
    return left.outcome == right.outcome and left.value_correct == right.value_correct


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=None, help="run only the first N subset questions"
    )
    args = parser.parse_args()

    ids, manifest = _load_subset()
    if args.limit:
        ids = ids[: args.limit]

    layer = load_semantic_layer()
    questions = load_questions()
    questions.validate_against(layer)
    gold = _load_gold()
    by_id: dict[str, Question] = {item.id: item for item in questions.questions}

    missing = [name for name in ids if name not in by_id]
    if missing:
        print(f"subset names unknown questions: {', '.join(missing)}")
        return 1

    single = _load_single_stage()
    absent = [name for name in ids if name not in single]
    if absent:
        print(f"no published single-stage result for: {', '.join(absent)}")
        return 1

    checkpoint = OUT_DIR / "checkpoint.jsonl"
    done = _load_checkpoint(checkpoint)
    todo = [name for name in ids if name not in done]
    print(f"{len(ids)} subset questions; {len(done)} already done, {len(todo)} to run")

    if todo:
        client = LlmClient(provider=args.provider, use_cache=not args.no_cache)
        if not client.has_credentials():
            print(f"no credentials for provider {client.name!r}. Add the key to .env.")
            return 1
        pipeline = CanonPipeline(layer=layer, client=client, thresholds=PolicyThresholds())
        try:
            for index, name in enumerate(todo, start=1):
                question = by_id[name]
                context = CallContext(system=ARM, question_id=name, run_index=0)
                result = pipeline.answer_two_stage(question.question, context=context)
                score = score_canon(question, result, gold, QuestionSet(questions=(question,)))
                _append(checkpoint, score)
                done[name] = score
                print(
                    f"[{ARM} {index}/{len(todo)}] {name:8} {score.outcome.value:9} "
                    f"value={score.value_correct} metric={score.metric_used or '-'}"
                )
        finally:
            pipeline.close()

    scored = [name for name in ids if name in done]
    if len(scored) != len(ids):
        print(f"INCOMPLETE: {len(scored)}/{len(ids)} -- not writing {OUTPUT.name}")
        return 2
    if args.limit:
        # --limit is a smoke switch. A report built from part of the subset would carry
        # the same filename and the same authority as the real one, and the per-tier
        # stratification that makes the subset meaningful would be silently gone.
        print(f"--limit was used: ran {len(ids)} questions, not writing {OUTPUT.name}")
        return 0

    _write_report(ids, manifest, {name: done[name] for name in ids}, single)
    print(f"wrote {OUTPUT}")
    return 0


def _half_correct(score: QuestionScore) -> bool:
    """Whether one half of a pair got the response its question required."""
    if score.expected_behavior == "answer":
        return score.value_correct is True
    return bool(score.refusal_correct)


def _paired_section(
    manifest: dict[str, Any],
    two: dict[str, QuestionScore],
    single: dict[str, QuestionScore],
) -> str:
    """The reflex-immune comparison, which is why the pairs were force-included."""
    pairs = manifest["force_included_pairs"]
    rows = [
        "Each pair looks alike and requires opposite responses, so a system with a reflex "
        "scores zero. This is the comparison the subset was built around.",
        "",
        "| Pair | Single-stage | Two-stage |",
        "|---|---|---|",
    ]
    tally = {"single": 0, "two": 0}
    for pair in pairs:
        left, right = pair["left"], pair["right"]
        verdicts = {}
        for key, data in (("single", single), ("two", two)):
            won = _half_correct(data[left]) and _half_correct(data[right])
            tally[key] += won
            detail = " / ".join(
                f"{name} {'ok' if _half_correct(data[name]) else '**WRONG**'}"
                for name in (left, right)
            )
            verdicts[key] = f"{'PASS' if won else 'FAIL'} — {detail}"
        rows.append(f"| {pair['pair']} | {verdicts['single']} | {verdicts['two']} |")
    rows.append(
        f"| **total** | **{tally['single']}/{len(pairs)}** | **{tally['two']}/{len(pairs)}** |"
    )
    return "\n".join(rows)


def _rate(scores: list[QuestionScore], predicate: Any) -> str:
    eligible = [s for s in scores if predicate(s) is not None]
    if not eligible:
        return "n/a"
    hits = sum(1 for s in eligible if predicate(s))
    return f"{hits}/{len(eligible)}"


def _write_report(
    ids: tuple[str, ...],
    manifest: dict[str, Any],
    two: dict[str, QuestionScore],
    single: dict[str, QuestionScore],
) -> None:
    two_list = [two[name] for name in ids]
    one_list = [single[name] for name in ids]

    lines: list[str] = [
        "# Ablation — full-stuffing vs two-stage prompting",
        "",
        "Generated by `scripts/run_ablation.py`. Both arms answer the same "
        f"{len(ids)} questions and share the same policy, compiler and verifier; the "
        "prompting strategy is the only variable.",
        "",
        "The single-stage column is **not re-run** — it is read from the published "
        "`sweep-groq-r1` records, so these are the same numbers `RESULTS.md` reports.",
        "",
        "## Subset",
        "",
        f"{len(ids)} questions, stratified by tier. {manifest['deliberate_oversampling']}",
        "",
        "## Agreement",
        "",
    ]

    disagree = [name for name in ids if not _agrees(two[name], single[name])]
    lines += [
        f"The two strategies behaved identically on **{len(ids) - len(disagree)} of "
        f"{len(ids)}** questions.",
        "",
    ]
    if disagree:
        lines += ["| Question | Single-stage | Two-stage |", "|---|---|---|"]
        for name in disagree:
            a, b = single[name], two[name]
            lines.append(
                f"| `{name}` | {a.outcome.value}, value={a.value_correct} "
                f"| {b.outcome.value}, value={b.value_correct} |"
            )
        lines.append("")
    else:
        lines += [
            "No question changed outcome. On this subset the saving is free.",
            "",
        ]

    lines += ["## Paired discrimination", "", _paired_section(manifest, two, single), ""]

    lines += [
        "## Scores on the subset",
        "",
        "| Measure | Single-stage | Two-stage |",
        "|---|---|---|",
    ]
    for label, predicate in (
        ("Execution accuracy (answerable)", lambda s: s.value_correct),
        ("Shape accuracy (answerable)", lambda s: s.shape_correct),
        ("Metric correct (answerable)", lambda s: s.metric_correct),
        ("Refusal correct (all)", lambda s: s.refusal_correct),
    ):
        lines.append(f"| {label} | {_rate(one_list, predicate)} | {_rate(two_list, predicate)} |")

    errors_one = sum(1 for s in one_list if s.outcome.value == "errored")
    errors_two = sum(1 for s in two_list if s.outcome.value == "errored")
    lines += [f"| Errors | {errors_one}/{len(ids)} | {errors_two}/{len(ids)} |", ""]

    lines += [
        "## Diagnosis — read before quoting any number above",
        "",
        "The three questions two-stage lost (`t2-01`, `t2-04`, `t3-02`) fail for one "
        "shared reason, visible in the clarifying questions it asked:",
        "",
        '> *"...or would you like a total net revenue for each month using a custom time '
        'grouping?"* (`t2-01`)',
        "",
        '> *"Would you like the total reported revenue for each quarter (time '
        'aggregation) **even though quarter is not a listed breakdown dimension**?"* '
        "(`t3-02`)",
        "",
        "All three name a time grain — month, month, quarter. The stage-A short card ends "
        "with `breaks down by: <valid_dimensions>`, and **time grain is not a dimension** "
        "in this semantic layer; it is a separate field on the intent. The full prompt "
        "says so across 28 lines; the stage-A prompt mentions grain only incidentally "
        'inside two metric descriptions. So stage A looks for "month" among the '
        "breakdown dimensions, does not find it, and correctly concludes it has been "
        "asked for something it cannot see how to express.",
        "",
        "**That is a defect in this repo's stage-A prompt, not a property of two-stage "
        "prompting.** The ablation measured my implementation. Stated plainly because the "
        'honest conclusion from the table above is *not* "splitting the prompt costs '
        'accuracy" — it is "splitting the prompt costs accuracy if you drop information '
        'the model needs, and I dropped some".',
        "",
        "**The one question two-stage won should not be counted as a win.** On `t2-05` "
        'single-stage asked whether *"net revenue in São Paulo state"* meant '
        "`customer_state` or `seller_state`, and two-stage answered. "
        "[`docs/03-failures.md`](../docs/03-failures.md) §C1 records that question as a "
        "case where the clarifying question is *better than my gold* — both dimensions "
        "exist and the question does not say which. So two-stage scores a point here for "
        "failing to notice a real ambiguity, and the `refusal correct` row above is "
        "flattering it by one.",
        "",
        "## What this does and does not establish",
        "",
        f"The subset is {len(ids)} questions, so a one-question change moves any rate here "
        f"by {100 / len(ids):.1f} percentage points. Treat the agreement count as the "
        "finding and the per-measure rates as supporting detail; neither is powered to "
        "detect a small difference.",
        "",
        "The paired-discrimination row is the exception worth taking seriously even at "
        "this sample size, because it is the one measure a hedging reflex cannot game: "
        "two-stage asked on `t3-02` and answered `t3-03`, which is precisely the "
        "distinction the pair exists to test. Those two questions use the same metric "
        "and differ only in whether the 2018-01-01 restatement falls inside the window.",
        "",
        "Token cost is not re-measured here. `scripts/dry_run.py` projects the saving "
        "offline from the real prompts, and that projection is the cost side of this "
        "trade-off; this document is only the accuracy side.",
        "",
        "**Status: this is iteration 1, before the stage-A prompt defect is fixed.** The "
        "number to compare a future run against is the one above, recorded before the "
        "cause was known, so a later improvement cannot be mistaken for the original "
        "result.",
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
