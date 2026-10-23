"""Render the report as markdown, in the shape ``docs/01-evaluation.md`` §4 fixed.

Every cell is a measured value or ``n/a``. Nothing is inferred, nothing is rounded into
looking better, and a rate with a zero denominator prints ``n/a`` rather than 0 — a rate
of nothing is not zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark.harness.score import Aggregate, Outcome, QuestionScore, latency_comparison

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
    *,
    matrices: dict[str, dict[str, dict[str, int]]] | None = None,
    pairs: dict[str, dict[str, Any]] | None = None,
    attribution: dict[str, Any] | None = None,
    variance: dict[str, dict[str, Any]] | None = None,
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
        ("**Silent-wrong rate**", "all", "silent_wrong_rate", False),
        ("Execution accuracy", "answerable", "execution_accuracy", True),
        ("**Semantic accuracy**", "answerable, measurable", "semantic_accuracy", True),
        ("Behaviour accuracy (3x3 diagonal)", "all", "behavior_accuracy", True),
        ("Refusal accuracy (ask vs answer)", "all", "refusal_accuracy", True),
        ("— under-refusal rate", "should-not-answer", "under_refusal_rate", False),
        ("— over-refusal rate", "answerable", "over_refusal_rate", False),
        ("— names the fork (proxy)", "should-not-answer", "names_the_fork_rate", True),
        ("Caveat surfaced", "cases requiring one", "caveat_surfaced_rate", True),
        ("Confident-wrong rate [1]", "all", "confident_wrong_rate", False),
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

    for label, field, digits, denominator in (
        ("Mean latency / question (s)", "mean_latency_seconds", 2, "all units, uncached only"),
        ("Mean cost / question (USD)", "mean_cost_usd", 6, "measured calls"),
    ):
        line = f"| {label} | {denominator} |"
        for name in ("canon", "baseline"):
            if name in by_system:
                line += f" {_num(getattr(by_system[name], field), digits)} |"
        if canon and baseline:
            line += " |"
        lines.append(line)
    lines.append("")

    # Live-call timing, always with its n. Separate from the table above because it has a
    # different denominator, and merging the two is what produced the misleading figure.
    lines += ["### Latency over live calls", ""]
    for name in ("canon", "baseline"):
        aggregate = by_system.get(name)
        if aggregate is None:
            continue
        live = aggregate.live_latency
        if live is None:
            lines.append(
                f"- **{name}**: not measured. Every unit was served from cache, so no call "
                "in this arm was timed. Not reported as 0.0s, which would read as fast."
            )
        else:
            lines.append(
                f"- **{name}**: mean {live.mean_seconds:.2f}s, median "
                f"{live.median_seconds:.2f}s, max {live.max_seconds:.2f}s over "
                f"**n={live.n}** live calls ({live.untimed} of {live.total} units "
                "untimed: "
                f"{live.cached} cache-served, {live.untimed - live.cached} errored)."
                + ("" if live.covers_whole_arm else " A per-question mean is not available.")
            )
    comparison = latency_comparison(by_system)
    lines += [
        "",
        comparison
        if comparison
        else (
            "**No cross-arm latency comparison is emitted**: it requires at least one live "
            "call in every arm, and at least one arm has none."
        ),
        "",
    ]

    lines.append(
        "**Silent-wrong** is the headline safety number: the system handed over a wrong "
        "number with neither a verification flag nor a clarifying question attached. It "
        "compares observable behaviour, so it needs no assumption about how confident "
        "either system was.\n"
    )
    lines.append(
        "[1] **Confident-wrong** is secondary because it depends on an imputation: the "
        "baseline reports no confidence, so every answer it gives is scored at 1.0. That "
        "is a defensible reading -- a figure handed over with no hedge is an assertion -- "
        "but it is still an assumption, and a reviewer is right to discount a headline "
        "that rests on one. Silent-wrong does not.\n"
    )

    if canon is not None or baseline is not None:
        lines.append("### Wrong answers: caught vs. silent\n")
        lines.append(
            "The mechanisms each system has for not being silently wrong. The baseline's "
            "zeros are structural -- it has no gate and no refusal path beyond the one "
            "the prompt grants it -- which is the finding rather than a scoring artefact.\n"
        )
        lines.append("| System | silent wrong | gate-flagged | declined to answer |")
        lines.append("|---|---|---|---|")
        for item in aggregates:
            lines.append(
                f"| {item.system} | {item.silent_wrong_count} | {item.gate_flagged_count} | "
                f"{item.caught_wrong_count} |"
            )
        lines.append("")

    if baseline is not None and baseline.semantic_measurable == 0:
        lines.append(
            "**On the baseline's semantic accuracy.** It is `n/a` because the baseline "
            "declares no metric — there is nothing to compare against gold. That is the "
            "finding rather than a gap in the harness: a system with no semantic layer "
            "cannot say which definition produced its number, which is exactly the blind "
            "spot `docs/01-evaluation.md` §3.2 argues Spider and BIRD share. For the "
            "baseline, a wrong metric choice surfaces as an execution-accuracy failure "
            "instead, and the attribution instrument below tries to name the definition "
            "it actually computed — subject to the coverage limit stated there.\n"
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

    if matrices:
        lines.append("## Behaviour matrix (3x3)\n")
        lines.append(
            "Target behaviour against what the system actually did. Not collapsed to "
            "2x2, because the off-diagonals are different failures:\n\n"
            "- **answer when it should clarify** — a *trust* failure. A number reaches "
            "someone who had no way to know the question was ambiguous.\n"
            "- **clarify when it should answer** — a *usability* failure. Annoying, "
            "safe, and cheap to fix by lowering the margin.\n"
            "- **refuse when it should clarify** — a *capability* failure. The system "
            "said it could not help with something it could have resolved by asking.\n"
            "- **clarify when it should refuse** — a *scope* failure. It offered to "
            "resolve a question the warehouse cannot answer at any level of detail.\n"
        )
        for system, matrix in matrices.items():
            lines.append(f"### {system}\n")
            lines.append("| target ↓ / actual → | answer | clarify | refuse | error |")
            lines.append("|---|---|---|---|---|")
            for target in ("answer", "clarify", "refuse"):
                row = matrix[target]
                cells = " | ".join(
                    str(row[actual]) for actual in ("answer", "clarify", "refuse", "error")
                )
                lines.append(f"| **{target}** | {cells} |")
            lines.append("")

    if pairs:
        lines.append("## Paired discrimination\n")
        lines.append(
            "Deliberately paired items, scored as **both halves correct**. Each pair "
            "looks alike and requires opposite responses, so a system with a reflex gets "
            "exactly one half of each pair and scores zero here. **This is the number "
            "immune to a reflex strategy** — always-answer and always-ask both score 0%, "
            "verified in the test suite.\n"
        )
        lines.append("| System | pairs | both halves correct | rate |")
        lines.append("|---|---|---|---|")
        for system, result in pairs.items():
            lines.append(
                f"| {system} | {result['pairs_scored']} | {result['both_correct']} | "
                f"{_pct(result['rate'])} |"
            )
        lines.append("")
        for system, result in pairs.items():
            lines.append(f"**{system}** per pair:\n")
            for entry in result["pairs"]:
                left = "correct" if entry["left_correct"] else "WRONG"
                right = "correct" if entry["right_correct"] else "WRONG"
                verdict = "PASS" if entry["both_correct"] else "FAIL"
                lines.append(
                    f"- {entry['pair']}: {entry['left']} {left}, {entry['right']} {right} "
                    f"→ **{verdict}**"
                )
            lines.append("")

    if attribution is not None:
        lines.append("## Attribution instrument — stated error bar\n")
        lines.append(
            "When a system returns a wrong number, the harness checks whether that number "
            "is another question's gold answer, which would say *which* definition it "
            "actually computed. That inference is only sound where the matched value is "
            "unique across the eval set and large enough not to collide by chance, so "
            "coverage is measured and reported rather than assumed.\n"
        )
        lines.append("| Property | Value |")
        lines.append("|---|---|")
        lines.append(f"| gold values | {attribution['total_gold_values']} |")
        lines.append(
            f"| shared between questions | {attribution['questions_with_shared_values']} |"
        )
        lines.append(
            f"| below magnitude threshold ({attribution['magnitude_threshold']:g}) | "
            f"{attribution['questions_below_magnitude_threshold']} |"
        )
        lines.append(
            f"| **can support an attribution** | "
            f"{attribution['questions_supporting_attribution']} "
            f"({_pct(attribution['attribution_coverage'])}) |"
        )
        lines.append("")
        if attribution["shared_value_groups"]:
            lines.append("Questions sharing a gold value, so a match cannot identify one:\n")
            for group in attribution["shared_value_groups"]:
                lines.append(f"- {', '.join(group)}")
            lines.append("")
        for item in aggregates:
            lines.append(
                f"- {item.system}: {item.attributable_wrong_count} attributable, "
                f"{item.unattributable_wrong_count} unattributable"
            )
        lines.append("")

    if variance:
        lines.append("## Variance across runs\n")
        for system, data in variance.items():
            if data.get("rates") is None:
                lines.append(
                    f"- **{system}**: {data.get('note', 'no variance measured')}. "
                    "No spread is reported rather than printing 0.0, which would claim a "
                    "stability that was never measured.\n"
                )
                continue
            lines.append(f"### {system} — {data['runs']} independent runs\n")
            lines.append("| Metric | mean | min | max | range |")
            lines.append("|---|---|---|---|---|")
            for field, stats in data["rates"].items():
                if stats is None:
                    lines.append(f"| {field} | n/a | n/a | n/a | n/a |")
                    continue
                lines.append(
                    f"| {field} | {_pct(stats['mean'])} | {_pct(stats['min'])} | "
                    f"{_pct(stats['max'])} | {stats['range'] * 100:.1f}pp |"
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
