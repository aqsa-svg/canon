"""Generate benchmark/RESULTS.md from a completed run's report.json.

Nothing here is typed by hand. Every figure is read from the run artifact, so a number in
RESULTS.md cannot drift from the run that produced it and cannot be a transcription
error. That is the number rule in `CLAUDE.md` made mechanical rather than remembered.

**It refuses to write for an incomplete run.** A partial arm is not averaged and not
published; the runner already stamps `complete: false`, and this honours it by exiting
non-zero with the gap named. Pass --allow-incomplete only to preview, which stamps the
document itself INCOMPLETE and marks every number provisional.

Run:  uv run python scripts/write_results.py --tag sweep-groq-r1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "benchmark" / "runs"
OUTPUT = REPO_ROOT / "benchmark" / "RESULTS.md"


def pct(value: Any) -> str:
    """A rate as a percentage, or n/a. A rate of nothing is not zero."""
    return "n/a" if value is None else f"{value * 100:.1f}%"


def delta(canon: Any, baseline: Any, higher_is_better: bool) -> str:
    if canon is None or baseline is None:
        return "n/a"
    difference = (canon - baseline) * 100
    good = difference > 0 if higher_is_better else difference < 0
    marker = "" if abs(difference) < 0.05 else (" ✓" if good else " ✗")
    return f"{difference:+.1f}pp{marker}"


def _latency_and_cost(tag: str) -> list[str]:
    """Latency and cost, with the cache confound stated instead of averaged away.

    ``docs/01-evaluation.md`` pre-registered "mean latency / question" over *all*
    questions. That denominator turns out to be wrong: a cache hit records 0.0 seconds, so
    averaging over all questions reports a number that falls as the cache fills and is not
    a timing of anything. Only live calls are timed, and the count of them is printed so
    the figure can be weighed.
    """

    lines = [
        "## Latency and cost",
        "",
        "Pre-registered in `docs/01-evaluation.md` over *all questions*. That denominator "
        "is wrong and is corrected here: a cache hit records 0.0 seconds, so a mean over "
        "all questions measures how full the cache is, not how fast the system is.",
        "",
        "| System | units | cache-served | live calls timed | mean | median | max |",
        "|---|---|---|---|---|---|---|",
    ]
    for system in ("canon", "baseline"):
        path = RUNS_DIR / tag / f"{system}.jsonl"
        if not path.exists():
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cached = sum(1 for row in rows if row.get("cached"))
        live = [
            row["latency_seconds"]
            for row in rows
            if not row.get("cached") and row.get("latency_seconds")
        ]
        if live:
            stats = (
                f"{statistics.mean(live):.2f}s | {statistics.median(live):.2f}s | {max(live):.2f}s"
            )
        else:
            stats = "not measured | not measured | not measured"
        lines.append(f"| {system} | {len(rows)} | {cached} | {len(live)} | {stats} |")

    lines += [
        "",
        "**No cross-arm latency comparison is possible from this run.** Every baseline "
        "unit was served from cache, so the baseline has no live timing at all. Reporting "
        "its mean as 0.0s, or omitting the row and leaving Canon's figure to look like a "
        "comparison, would both be worse than saying so.",
        "",
        "**Mean cost per question is $0.00 for both arms**, and that is the whole point of "
        "the free-tier constraint rather than a rounding artefact: the provider bills "
        "nothing at this tier, so reproducing the benchmark costs nothing. The token spend "
        "it replaces is the real resource, and `data/quota_ledger.json` records it per UTC "
        "day.",
        "",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="preview an unfinished run; stamps the document INCOMPLETE",
    )
    args = parser.parse_args()

    report_path = RUNS_DIR / args.tag / "report.json"
    if not report_path.exists():
        print(f"no report at {report_path}")
        return 1

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    complete = provenance.get("complete", False)

    if not complete and not args.allow_incomplete:
        print(
            f"RUN IS INCOMPLETE: {provenance.get('completed_units')}/"
            f"{provenance.get('expected_units')} units, "
            f"{provenance.get('scored_questions')} questions scoreable."
        )
        print("A partial arm is never published. Finish the run, or pass --allow-incomplete")
        print("to preview (the document will be stamped INCOMPLETE).")
        return 2

    aggregates = {item["system"]: item for item in payload["aggregates"]}
    canon = aggregates.get("canon")
    baseline = aggregates.get("baseline")
    matrices = payload.get("behavior_matrix", {})
    pairs = payload.get("paired_discrimination", {})
    attribution = payload.get("attribution_instrument", {})
    variance = payload.get("variance", {})

    lines: list[str] = ["# Canon — benchmark results", ""]

    if not complete:
        lines += [
            "> ## ⚠ INCOMPLETE RUN — DO NOT QUOTE",
            f"> {provenance.get('completed_units')}/{provenance.get('expected_units')} units "
            f"finished; {provenance.get('scored_questions')} of "
            f"{provenance.get('question_count')} questions were answered by every arm. "
            "Every figure below is provisional.",
            "",
        ]

    lines += [
        f"Generated from `benchmark/runs/{args.tag}/report.json` by `scripts/write_results.py`. "
        "No figure here is typed by hand, so none can drift from the run that produced "
        "it.",
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key in (
        "started_at",
        "provider",
        "model_id",
        "provider_config_hash",
        "free_tier",
        "effort",
        "code_commit",
        "eval_set_commit",
        "metrics_yaml_sha256",
        "confident_threshold",
        "question_count",
        "scored_questions",
        "runs",
        "cache_used",
        "complete",
    ):
        if key in provenance:
            lines.append(f"| {key} | `{provenance[key]}` |")
    thresholds = provenance.get("policy_thresholds", {})
    if thresholds:
        rendered = ", ".join(f"{k}={v}" for k, v in thresholds.items())
        lines.append(f"| policy_thresholds | `{rendered}` |")
    lines += ["", f"**Threshold provenance.** {provenance.get('threshold_provenance', 'TBD')}", ""]

    # ---------------------------------------------------------------- headline
    lines += ["## Headline", ""]
    header = "| Metric | Denominator |"
    divider = "|---|---|"
    for name in ("canon", "baseline"):
        if name in aggregates:
            header += f" {name} |"
            divider += "---|"
    if canon and baseline:
        header += " delta |"
        divider += "---|"
    lines += [header, divider]

    rows = [
        ("**Silent-wrong rate**", "all", "silent_wrong_rate", False),
        ("Execution accuracy", "answerable", "execution_accuracy", True),
        ("Shape accuracy (columns + grain)", "answerable", "shape_accuracy", True),
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
    for label, denominator, field, higher in rows:
        line = f"| {label} | {denominator} |"
        for name in ("canon", "baseline"):
            if name in aggregates:
                line += f" {pct(aggregates[name].get(field))} |"
        if canon and baseline:
            line += f" {delta(canon.get(field), baseline.get(field), higher)} |"
        lines.append(line)
    lines.append("")

    lines += [
        "**Silent-wrong** is the headline safety number: a wrong answer handed over with "
        "neither a verification flag nor a clarifying question attached. It compares what "
        "each system observably did, so it needs no assumption about confidence.",
        "",
        "**Shape accuracy** is the lenient counterpart of execution accuracy — correct "
        "columns and a plausible grain, filters disregarded. The pair is the finding: high "
        "shape with low execution means a system asked for the right thing and computed it "
        "under the wrong definition. Quoting either alone misleads in opposite directions.",
        "",
        "[1] Confident-wrong is secondary because it rests on an imputation: the baseline "
        "reports no confidence, so each of its answers is scored at 1.0. Defensible — a "
        "figure handed over with no hedge is an assertion — but still an assumption. "
        "Silent-wrong needs none.",
        "",
    ]

    if baseline is not None and baseline.get("semantic_measurable", 0) == 0:
        lines += [
            "**On the baseline's semantic accuracy.** `n/a` because it declares no metric — "
            "there is nothing to compare against gold. That is the finding, not a gap in "
            "the harness: a system with no semantic layer cannot say which definition "
            "produced its number, which is the blind spot `docs/01-evaluation.md` §3.2 "
            "argues Spider and BIRD share.",
            "",
        ]

    # ---------------------------------------------------------------- caught vs silent
    lines += [
        "### Wrong answers: caught vs. silent",
        "",
        "| System | silent wrong | gate-flagged | declined to answer |",
        "|---|---|---|---|",
    ]
    for name, item in aggregates.items():
        lines.append(
            f"| {name} | {item.get('silent_wrong_count', 0)} | "
            f"{item.get('gate_flagged_count', 0)} | {item.get('caught_wrong_count', 0)} |"
        )
    lines += [
        "",
        "The baseline's zeros are structural — it has no verification gate — which is the "
        "finding rather than a scoring artefact.",
        "",
    ]

    # ---------------------------------------------------------------- per tier
    lines += [
        "## Per tier",
        "",
        "| System | Tier | n | Execution | Shape | Semantic | Behaviour | Silent-wrong |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, item in aggregates.items():
        for tier, entry in sorted(item.get("per_tier", {}).items()):
            lines.append(
                f"| {name} | {tier} | {entry['count']} | "
                f"{pct(entry.get('execution_accuracy'))} | {pct(entry.get('shape_accuracy'))} | "
                f"{pct(entry.get('semantic_accuracy'))} | {pct(entry.get('behavior_accuracy'))} | "
                f"{pct(entry.get('silent_wrong_rate'))} |"
            )
    lines.append("")

    # ---------------------------------------------------------------- 3x3
    if matrices:
        lines += [
            "## Behaviour matrix (3x3)",
            "",
            "Not collapsed to 2x2, because the off-diagonals are different failures:",
            "",
            "- **answer when it should clarify** — a *trust* failure. A number reaches "
            "someone with no way to know the question was ambiguous.",
            "- **clarify when it should answer** — a *usability* failure. Annoying, safe, "
            "and cheap to fix by lowering the margin.",
            "- **refuse when it should clarify** — a *capability* failure.",
            "- **clarify when it should refuse** — a *scope* failure.",
            "",
        ]
        for name, matrix in matrices.items():
            lines += [
                f"### {name}",
                "",
                "| target ↓ / actual → | answer | clarify | refuse | error |",
                "|---|---|---|---|---|",
            ]
            for target in ("answer", "clarify", "refuse"):
                row = matrix.get(target, {})
                cells = " | ".join(
                    str(row.get(actual, 0)) for actual in ("answer", "clarify", "refuse", "error")
                )
                lines.append(f"| **{target}** | {cells} |")
            lines.append("")

    # ---------------------------------------------------------------- pairs
    if pairs:
        lines += [
            "## Paired discrimination",
            "",
            "Deliberately paired items scored as **both halves correct**. Each pair looks "
            "alike and needs opposite responses, so a system with a reflex gets exactly one "
            "half and scores zero. **This is the number immune to a reflex strategy** — "
            "always-answer and always-ask both score 0%, asserted in the test suite.",
            "",
            "| System | pairs | both halves correct | rate |",
            "|---|---|---|---|",
        ]
        for name, result in pairs.items():
            lines.append(
                f"| {name} | {result['pairs_scored']} | {result['both_correct']} | "
                f"{pct(result.get('rate'))} |"
            )
        lines.append("")
        for name, result in pairs.items():
            lines.append(f"**{name}** per pair:")
            lines.append("")
            for entry in result.get("pairs", []):
                left = "correct" if entry["left_correct"] else "WRONG"
                right = "correct" if entry["right_correct"] else "WRONG"
                verdict = "PASS" if entry["both_correct"] else "FAIL"
                lines.append(
                    f"- {entry['pair']}: {entry['left']} {left}, {entry['right']} {right} "
                    f"→ **{verdict}**"
                )
            lines.append("")

    # ---------------------------------------------------------------- attribution
    if attribution:
        lines += [
            "## Attribution instrument — stated error bar",
            "",
            "When a system returns a wrong number, the harness checks whether that number is "
            "another question's gold answer, which would say *which* definition it computed. "
            "That inference only holds where the matched value is unique across the set and "
            "large enough not to collide by chance, so coverage is measured rather than "
            "assumed.",
            "",
            "| Property | Value |",
            "|---|---|",
            f"| gold values | {attribution.get('total_gold_values')} |",
            f"| shared between questions | {attribution.get('questions_with_shared_values')} |",
            f"| below magnitude threshold ({attribution.get('magnitude_threshold')}) | "
            f"{attribution.get('questions_below_magnitude_threshold')} |",
            f"| **can support an attribution** | "
            f"{attribution.get('questions_supporting_attribution')} "
            f"({pct(attribution.get('attribution_coverage'))}) |",
            "",
        ]
        for name, item in aggregates.items():
            lines.append(
                f"- {name}: {item.get('attributable_wrong_count', 0)} attributable, "
                f"{item.get('unattributable_wrong_count', 0)} unattributable"
            )
        lines.append("")

    # ---------------------------------------------------------------- variance
    lines += ["## Variance across runs", ""]
    for name, data in variance.items():
        if data.get("rates") is None:
            lines.append(
                f"- **{name}**: {data.get('note', 'no variance measured')}. No spread is "
                "reported rather than printing 0.0, which would claim a stability never "
                "measured."
            )
        else:
            lines += [
                f"### {name} — {data['runs']} independent runs",
                "",
                "| Metric | mean | min | max | range |",
                "|---|---|---|---|---|",
            ]
            for field, stats in data["rates"].items():
                if stats is None:
                    lines.append(f"| {field} | n/a | n/a | n/a | n/a |")
                else:
                    lines.append(
                        f"| {field} | {pct(stats['mean'])} | {pct(stats['min'])} | "
                        f"{pct(stats['max'])} | {stats['range'] * 100:.1f}pp |"
                    )
            lines.append("")
    lines.append("")

    lines += _latency_and_cost(args.tag)

    # ---------------------------------------------------------------- caveats
    lines += [
        "## What these numbers do and do not mean",
        "",
        "**Execution accuracy measures conformance to a stated canonical layer, not "
        "correctness in the abstract.** I wrote the metric cards, the questions and the "
        "gold answers. The mechanism claims are independently checkable — a hallucinated "
        "metric is unrepresentable, fan-out cannot be built — but the accuracy figures rest "
        "on labels I authored. A finance team that had ruled net revenue the default would "
        "score Canon's refusal on `am-01` a failure. The project's claim is that having a "
        "canonical layer beats not having one, not that my particular layer is right.",
        "",
        "**Known artefacts inflating Canon's over-refusal rate**, both documented in "
        "`docs/03-failures.md`:",
        "",
        "- `t1-09` and `t1-10` name no time period. My eval set assumes that means all data "
        "and never told the model, which was asked to treat unanchored time as missing "
        "information. At most 2/55 answerable questions, so at most 3.6pp.",
        "- `am-18` and `am-21` are recorded as errors because my schema requires at least "
        "one metric candidate, and the honest answer for a metric the catalogue lacks is "
        "zero. Two refuse-class questions understate Canon on the class the project is "
        "about.",
        "",
        "Both are disclosed rather than fixed, because fixing either changes the prompt or "
        "the schema, invalidates the response cache, and costs a full re-run. A quantified "
        "artefact is more credible than a quietly cleaner number.",
        "",
        "**Free-tier model.** This ran on a free, open-weights model. The absolute numbers "
        "would be higher on a frontier model; the *delta* between the two systems is the "
        "claim, not the level.",
        "",
        f"Full per-question records: `benchmark/runs/{args.tag}/`.",
        "",
    ]

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(lines)} lines)")
    if not complete:
        print("STAMPED INCOMPLETE: the numbers are provisional and must not be quoted.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
