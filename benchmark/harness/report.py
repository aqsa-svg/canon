"""Render the report as markdown, in the shape ``docs/01-evaluation.md`` §4 fixed.

Every cell is a measured value or ``n/a``. Nothing is inferred, nothing is rounded into
looking better, and a rate with a zero denominator prints ``n/a`` rather than 0 — a rate
of nothing is not zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark.harness.score import Aggregate, Outcome, QuestionScore

if TYPE_CHECKING:
    from benchmark.harness.run import Provenance


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _delta(canon: float | None, baseline: float | None, higher_is_better: bool) -> str:
    """The delta, with its direction marked so a reader cannot misread the sign."""
    if canon is None or baseline is None:
        return "n/a"
    difference = (canon - baseline) * 100
    good = difference > 0 if higher_is_better else difference < 0
    marker = "" if abs(difference) < 0.05 else (" ✓" if good else " ✗")
    return f"{difference:+.1f}pp{marker}"


def write_report(
    path: Path,
    provenance: Provenance,
    aggregates: list[Aggregate],
    scores: dict[str, list[QuestionScore]],
) -> None:
    by_system = {item.system: item for item in aggregates}
    canon = by_system.get("canon")
    baseline = by_system.get("baseline")
    lines: list[str] = []

    lines.append("# Canon benchmark run\n")
    lines.append(
        "> Numbers below are measured. Provenance is recorded because a figure without "
        "it gets deleted rather than trusted (`CLAUDE.md`).\n"
    )

    lines.append("## Provenance\n")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    for key, value in provenance.model_dump(mode="json").items():
        rendered = (
            ", ".join(f"{k}={v}" for k, v in value.items())
            if isinstance(value, dict)
            else str(value)
        )
        lines.append(f"| {key} | `{rendered}` |")
    lines.append("")

    lines.append("## Headline\n")
    header = "| Metric | Denominator |"
    divider = "|---|---|"
    for name in ("canon", "baseline"):
        if name in by_system:
            header += f" {name} |"
            divider += "---|"
    if canon and baseline:
        header += " delta |"
        divider += "---|"
    lines.append(header)
    lines.append(divider)

    rows: list[tuple[str, str, str, bool]] = [
        ("Execution accuracy", "answerable", "execution_accuracy", True),
        ("**Semantic accuracy**", "answerable, measurable", "semantic_accuracy", True),
        ("Refusal accuracy", "all", "refusal_accuracy", True),
        ("— under-refusal rate", "should-not-answer", "under_refusal_rate", False),
        ("— over-refusal rate", "answerable", "over_refusal_rate", False),
        ("— names the fork (proxy)", "should-not-answer", "names_the_fork_rate", True),
        ("Caveat surfaced", "cases requiring one", "caveat_surfaced_rate", True),
        ("**Confident-wrong rate**", "all", "confident_wrong_rate", False),
        ("Error rate", "all", "error_rate", False),
    ]
    for label, denominator, field, higher_better in rows:
        line = f"| {label} | {denominator} |"
        for name in ("canon", "baseline"):
            if name in by_system:
                line += f" {_pct(getattr(by_system[name], field))} |"
        if canon and baseline:
            line += f" {_delta(getattr(canon, field), getattr(baseline, field), higher_better)} |"
        lines.append(line)

    for label, field, digits in (
        ("Mean latency / question (s)", "mean_latency_seconds", 2),
        ("Mean cost / question (USD)", "mean_cost_usd", 6),
    ):
        line = f"| {label} | measured calls |"
        for name in ("canon", "baseline"):
            if name in by_system:
                line += f" {_num(getattr(by_system[name], field), digits)} |"
        if canon and baseline:
            line += " |"
        lines.append(line)
    lines.append("")

    if baseline is not None and baseline.semantic_measurable == 0:
        lines.append(
            "**On the baseline's semantic accuracy.** It is `n/a` because the baseline "
            "declares no metric — there is nothing to compare against gold. That is the "
            "finding rather than a gap in the harness: a system with no semantic layer "
            "cannot say which definition produced its number, which is exactly the blind "
            "spot `docs/01-evaluation.md` §3.2 argues Spider and BIRD share. For the "
            "baseline, a wrong metric choice surfaces as an execution-accuracy failure, "
            "and the `matched_other_gold` column below names the question whose gold "
            "value its answer *did* equal — direct evidence of which definition it "
            "actually computed.\n"
        )

    lines.append("## Per tier\n")
    tiers = sorted({tier for item in aggregates for tier in item.per_tier})
    lines.append("| System | Tier | n | Execution | Semantic | Refusal | Confident-wrong |")
    lines.append("|---|---|---|---|---|---|---|")
    for item in aggregates:
        for tier in tiers:
            entry = item.per_tier.get(tier)
            if entry is None:
                continue
            lines.append(
                f"| {item.system} | {tier} | {entry.count} | "
                f"{_pct(entry.execution_accuracy)} | {_pct(entry.semantic_accuracy)} | "
                f"{_pct(entry.refusal_accuracy)} | {_pct(entry.confident_wrong_rate)} |"
            )
    lines.append("")

    lines.append("## Confident-wrong across thresholds\n")
    lines.append(
        "The 0.7 threshold is fixed in advance. The sweep is here so that choice is "
        "inspectable rather than load-bearing.\n"
    )
    lines.append("| System | ≥0.5 | ≥0.6 | ≥0.7 | ≥0.8 | ≥0.9 |")
    lines.append("|---|---|---|---|---|---|")
    from benchmark.harness.score import confident_wrong_curve

    for system, entries in scores.items():
        curve = confident_wrong_curve(entries)
        cells = " | ".join(_pct(curve[key]) for key in ("0.5", "0.6", "0.7", "0.8", "0.9"))
        lines.append(f"| {system} | {cells} |")
    lines.append("")

    lines.append("## Wrong answers, in full\n")
    lines.append(
        "Every case where a system produced a number that was wrong, or produced one at "
        "all when it should have asked. This is the list that matters.\n"
    )
    any_wrong = False
    for system, entries in scores.items():
        wrong = [entry for entry in entries if entry.is_wrong_answer]
        if not wrong:
            continue
        any_wrong = True
        lines.append(f"### {system}\n")
        lines.append("| id | question | expected | metric used | matched other gold | conf |")
        lines.append("|---|---|---|---|---|---|")
        for entry in wrong:
            lines.append(
                f"| {entry.question_id} | {entry.question[:60]} | "
                f"{entry.expected_behavior.value} | {entry.metric_used or '—'} | "
                f"{entry.matched_other_gold or '—'} | {entry.confidence:.2f} |"
            )
        lines.append("")
    if not any_wrong:
        lines.append("_None._\n")

    lines.append("## Clarifying questions, for hand-grading\n")
    lines.append(
        "The `names the fork` proxy above is necessary and not sufficient, so the actual "
        "text is reproduced here for grading against the §3.3 rubric (2 = names the fork "
        "and offers alternatives, 1 = signals ambiguity without identifying it, "
        "0 = generic or wrong fork).\n"
    )
    for system, entries in scores.items():
        asked = [entry for entry in entries if entry.outcome is Outcome.ASKED]
        if not asked:
            continue
        lines.append(f"### {system}\n")
        for entry in asked:
            verdict = "expected" if not entry.expected_behavior.should_answer else "OVER-REFUSAL"
            proxy = {True: "proxy: names fork", False: "proxy: incomplete", None: ""}[
                entry.names_the_fork
            ]
            lines.append(f"- **{entry.question_id}** ({verdict}{', ' + proxy if proxy else ''})")
            lines.append(f"  - Q: {entry.question}")
            lines.append(f"  - Asked: {entry.clarifying_question}")
        lines.append("")

    errored: list[tuple[str, QuestionScore]] = [
        (system, entry)
        for system, entries in scores.items()
        for entry in entries
        if entry.outcome is Outcome.ERRORED
    ]
    if errored:
        lines.append("## Errors\n")
        lines.append("| System | id | error |")
        lines.append("|---|---|---|")
        for system, entry in errored:
            lines.append(f"| {system} | {entry.question_id} | {(entry.error or '')[:120]} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def summarise_to_stdout(aggregates: list[Aggregate]) -> None:
    """Terse console summary, for when the markdown is not what you want to read."""
    for item in aggregates:
        print(f"\n{item.system}: {item.total_questions} questions")
        print(f"  execution accuracy : {_pct(item.execution_accuracy)}")
        print(
            f"  semantic accuracy  : {_pct(item.semantic_accuracy)} "
            f"(measurable on {item.semantic_measurable})"
        )
        print(f"  refusal accuracy   : {_pct(item.refusal_accuracy)}")
        print(f"    under-refusal    : {_pct(item.under_refusal_rate)}")
        print(f"    over-refusal     : {_pct(item.over_refusal_rate)}")
        print(f"  confident-wrong    : {_pct(item.confident_wrong_rate)}")
        print(f"  errors             : {_pct(item.error_rate)}")


__all__: list[Any] = ["summarise_to_stdout", "write_report"]
