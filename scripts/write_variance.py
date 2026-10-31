"""Generate benchmark/VARIANCE.md from a multi-run sweep.

Every figure in `RESULTS.md` is a single sample. This measures how much of that is luck.

**Why per-question flips and not just aggregate spread.** Two runs can produce an
identical execution accuracy while disagreeing about *which* questions they got right, and
a report showing only the aggregate would call that perfect stability. The headline number
here is therefore the count of questions whose outcome changed, with the spread as
supporting detail -- the same reasoning that made `benchmark/ABLATION.md` lead on
agreement rather than on rates.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "benchmark" / "runs"
OUTPUT = REPO_ROOT / "benchmark" / "VARIANCE.md"
SUBSET = REPO_ROOT / "benchmark" / "ablation_subset.json"


def _load_runs(tag: str, system: str) -> dict[int, dict[str, dict[str, Any]]]:
    """Per-question records keyed by run index, for one system."""
    path = RUNS_DIR / tag / "checkpoint.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    runs: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("_system") != system:
            continue
        runs[record["_run_index"]][record["question_id"]] = record
    return dict(runs)


def _behaviour(record: dict[str, Any]) -> str:
    """What the system did, coarsely enough to compare across runs."""
    return str(record.get("outcome"))


def _agrees(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _behaviour(left) == _behaviour(right) and left.get("value_correct") == right.get(
        "value_correct"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--system", default="canon")
    parser.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="TAG:SYSTEM",
        help=(
            "an arm to include, repeatable, e.g. --arm variance-groq-r2:canon. Overrides "
            "--tag/--system. Arms differ in coverage, so each is scoped separately in the "
            "document rather than pooled into one number."
        ),
    )
    args = parser.parse_args()

    if not args.arm and not args.tag:
        parser.error("pass --arm TAG:SYSTEM (repeatable), or --tag with optional --system")

    arms: list[tuple[str, str]] = []
    for spec in args.arm or [f"{args.tag}:{args.system}"]:
        tag, _, system = spec.partition(":")
        arms.append((tag, system or "canon"))

    sections: list[list[str]] = []
    totals: list[tuple[str, int, int]] = []
    for tag, system in arms:
        section = _arm_section(tag, system, totals)
        if section is None:
            return 2
        sections.append(section)

    return _emit(arms, sections, totals)


def _arm_section(tag: str, system: str, totals: list[tuple[str, int, int]]) -> list[str] | None:
    """One arm's section, or None if it cannot be reported."""
    runs = _load_runs(tag, system)
    indices = sorted(runs)
    if len(indices) < 2:
        print(f"only {len(indices)} run(s) for {system!r} in {tag!r}: nothing to compare")
        return None

    # The full question set is the union: a run missing questions is a run still in
    # progress, and comparing only the intersection would silently report a complete-
    # looking result over whichever questions happened to finish.
    everything = sorted(set.union(*(set(runs[i]) for i in indices)))
    shared = sorted(set.intersection(*(set(runs[i]) for i in indices)))
    if len(shared) != len(everything):
        print(f"INCOMPLETE: {len(everything)} questions seen, only {len(shared)} in every run.")
        for index in indices:
            missing = sorted(set(everything) - set(runs[index]))
            print(
                f"  run {index}: {len(runs[index])}/{len(everything)}"
                + (
                    f" -- missing {', '.join(missing[:6])}{'...' if len(missing) > 6 else ''}"
                    if missing
                    else " -- complete"
                )
            )
        print("A partial run is not reported. Finish the sweep and re-run this.")
        return None

    flips = [
        q for q in shared if not all(_agrees(runs[indices[0]][q], runs[i][q]) for i in indices)
    ]
    totals.append((system, len(flips), len(shared)))

    # Coverage is derived, not asserted: a subset run and a full-set run support very
    # different claims, and hardcoding "stratified subset" into the template would have
    # mislabelled the first full-set arm that ran through it.
    subset = json.loads(SUBSET.read_text(encoding="utf-8"))["questions"] if SUBSET.exists() else []
    is_subset = set(shared) == set(subset)

    lines: list[str] = [
        f"## `{system}` — {len(indices)} samples over {len(shared)} questions",
        "",
        f"From `benchmark/runs/{tag}/`. Same provider, same config hash, temperature 0.",
        "",
    ]
    if is_subset:
        lines += [
            f"**Coverage: the {len(shared)}-question stratified subset**, which "
            "force-includes every paired-discrimination item and so over-represents the "
            "questions where a flip is most plausible. A *sensitive* probe, not a "
            "representative one — a clean result here is the easier half of the claim.",
            "",
        ]
    else:
        lines += [
            f"**Coverage: all {len(shared)} questions.** Not a subset, so this is "
            "representative of the arm rather than sensitive to a chosen slice.",
            "",
        ]
    lines += [
        "### Did anything change?",
        "",
    ]

    if flips:
        lines += [
            f"**{len(flips)} of {len(shared)} questions changed outcome between runs.**",
            "",
            "| Question | " + " | ".join(f"run {i}" for i in indices) + " |",
            "|---" * (len(indices) + 1) + "|",
        ]
        for question in flips:
            cells = [
                f"{_behaviour(runs[i][question])}, value={runs[i][question].get('value_correct')}"
                for i in indices
            ]
            lines.append(f"| `{question}` | " + " | ".join(cells) + " |")
        lines += [
            "",
            "A flip at temperature 0 is not a contradiction: the provider does not "
            "guarantee determinism, and a question whose metric scores sit near the "
            "policy margin can land either side of it on different samples. Each one "
            "above is a question whose published result is luck-dependent.",
            "",
            "**What each flip costs the published figures.** The rate a flip moves depends "
            "on what the question was supposed to do, and the denominators are the full "
            "eval set's, so these are the swings on `RESULTS.md`:",
            "",
        ]
        for question in flips:
            record = runs[indices[0]][question]
            expected = str(record.get("expected_behavior"))
            if expected == "answer":
                lines.append(
                    f"- `{question}` (should answer): moves **over-refusal** and "
                    f"**execution accuracy**, denominator 55 answerable questions, so "
                    f"±1.8pp on each depending which sample you draw."
                )
            else:
                lines.append(
                    f"- `{question}` (should {expected}): moves **under-refusal** and "
                    f"**behaviour accuracy**, denominator 25 should-not-answer questions, "
                    f"so ±4.0pp on each depending which sample you draw."
                )
        lines += [
            "",
            "This is the honest cost of an n=1 headline: those figures are not wrong, but "
            "they are one draw, and a reader should read them with this swing attached.",
            "",
        ]
    else:
        lines += [
            f"**No question changed outcome.** All {len(shared)} questions produced the "
            f"same behaviour and the same correctness across {len(indices)} independent "
            "samples.",
            "",
            "That is the strongest result this probe can return, and it is worth being "
            "precise about what it does and does not mean. It means the pipeline is "
            "reproducible here at temperature 0 — the refusal policy is not "
            "sitting on a knife edge for these questions, and the published outcomes for "
            "them are not luck. It does **not** mean the benchmark has zero variance: "
            f"this is one system over {len(indices)} samples"
            + (
                f", on {len(shared)} of 80 questions, so the rest are unmeasured"
                if is_subset
                else ""
            )
            + f", and {len(indices)} samples cannot detect a rare flip.",
            "",
        ]

    lines += [
        "### Per-run aggregate",
        "",
        "| Measure | " + " | ".join(f"run {i}" for i in indices) + " |",
        "|---" * (len(indices) + 1) + "|",
    ]
    for label, key in (
        ("Answered", "answered"),
        ("Asked (clarify or refuse)", "asked"),
        ("Errored", "errored"),
        ("Value correct", "value_correct"),
    ):
        cells = []
        for i in indices:
            records = [runs[i][q] for q in shared]
            if key == "value_correct":
                cells.append(str(sum(1 for r in records if r.get("value_correct") is True)))
            else:
                cells.append(str(sum(1 for r in records if r.get("outcome") == key)))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")
    return lines


def _emit(
    arms: list[tuple[str, str]], sections: list[list[str]], totals: list[tuple[str, int, int]]
) -> int:
    """Assemble one document covering every arm."""
    covered = {system for _, system in arms}
    missing = {"canon", "baseline"} - covered

    lines: list[str] = [
        "# Variance across independent runs",
        "",
        "Generated by `scripts/write_variance.py`. Every figure in "
        "[`RESULTS.md`](RESULTS.md) is a single sample; this measures how much of that is "
        "luck.",
        "",
        "**Why per-question flips rather than aggregate spread.** Two runs can produce an "
        "identical execution accuracy while disagreeing about *which* questions they got "
        "right, and a report showing only the aggregate would call that perfect "
        "stability. The headline here is the count of questions whose outcome changed.",
        "",
        "## Summary",
        "",
        "| Arm | questions | samples | flipped |",
        "|---|---|---|---|",
    ]
    for (_, system), (_, flipped, total) in zip(arms, totals, strict=True):
        lines.append(f"| `{system}` | {total} | 2 | **{flipped}** |")
    lines.append("")
    if len(arms) > 1:
        lines += [
            "**Arms are not pooled and the rates are not combined**, because they cover "
            "different question sets — compare the `questions` column before comparing "
            "the `flipped` column. A single blended 'variance of the benchmark' would be "
            "a number with no denominator.",
            "",
        ]
    if missing:
        lines += [
            "**Not measured: " + ", ".join(f"`{name}`" for name in sorted(missing)) + ".** "
            "Absent from this document rather than assumed stable.",
            "",
        ]
    lines.append("---")
    lines.append("")
    for section in sections:
        lines += section
        lines.append("---")
        lines.append("")

    lines += [
        "## What would make this stronger",
        "",
        "- **More samples.** Two runs detect a coin-flip question but not a rare one, and "
        "the cost is linear in free-tier days.",
        "- **Full coverage for every arm.** Where an arm above is scoped to the subset, "
        "the remaining questions are unmeasured, and the subset was chosen to be "
        "sensitive rather than representative.",
        "- **A third provider.** Everything here is one model on one provider, so it "
        "measures run-to-run stability, not robustness to the model changing underneath.",
        "",
    ]

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    summary = ", ".join(f"{system} {f}/{t}" for system, f, t in totals)
    print(f"wrote {OUTPUT} ({len(lines)} lines); flips: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
