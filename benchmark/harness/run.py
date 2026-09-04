"""Run the benchmark and emit the report.

Usage:
  uv run python -m benchmark.harness.run --limit 10          # smoke test
  uv run python -m benchmark.harness.run                     # full set
  uv run python -m benchmark.harness.run --system canon      # one system only
  uv run python -m benchmark.harness.run --no-cache          # force fresh calls

Every run records its provenance -- UTC timestamp, code commit, eval-set commit, model
id, effort, policy thresholds, and the metrics.yaml hash -- because a number without
provenance gets deleted rather than trusted (`CLAUDE.md`).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from benchmark.harness.baseline import NaiveBaseline
from benchmark.harness.dataset import Behavior, Question, QuestionSet, load_questions
from benchmark.harness.report import write_report
from benchmark.harness.score import (
    CONFIDENT_THRESHOLD,
    Aggregate,
    Outcome,
    QuestionScore,
    aggregate,
    confident_wrong_curve,
    find_matching_gold,
    rows_match,
)
from canon.agent.llm import MODEL_ID, LlmClient
from canon.agent.pipeline import CanonPipeline, PipelineResult
from canon.agent.policy import PolicyThresholds
from canon.semantic.loader import load_semantic_layer, semantic_layer_hash

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = REPO_ROOT / "data" / "canon.duckdb"
GOLD_VALUES = REPO_ROOT / "benchmark" / "questions" / "gold_values.json"
RUNS_DIR = REPO_ROOT / "benchmark" / "runs"

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class Provenance(BaseModel):
    """What produced these numbers. Without it the numbers are not evidence."""

    model_config = _FROZEN

    started_at: str
    code_commit: str
    eval_set_commit: str
    metrics_yaml_sha256: str
    model_id: str
    effort: str
    policy_thresholds: PolicyThresholds
    confident_threshold: float
    question_count: int
    cache_used: bool


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_gold() -> dict[str, list[list[Any]]]:
    if not GOLD_VALUES.exists():
        raise FileNotFoundError(
            f"{GOLD_VALUES} missing -- run: uv run python scripts/verify_gold.py"
        )
    return json.loads(GOLD_VALUES.read_text(encoding="utf-8"))


def _names_the_fork(question: Question, asked: str | None) -> bool:
    """Automatic proxy for rubric level 2: are all the required terms named?

    Case-insensitive substring match, and underscores are treated as spaces so a
    clarification saying "gross revenue" satisfies a must_mention of "gross_revenue".
    """
    if asked is None:
        return False
    haystack = asked.lower().replace("_", " ")
    return all(term.lower().replace("_", " ") in haystack for term in question.must_mention)


def _caveats_surfaced(question: Question, caveat_text: str) -> bool | None:
    if not question.requires_caveat:
        return None
    lowered = caveat_text.lower()
    return all(term.lower() in lowered for term in question.requires_caveat)


def score_canon(
    question: Question,
    result: PipelineResult,
    gold: dict[str, list[list[Any]]],
    questions: QuestionSet,
) -> QuestionScore:
    """Score one Canon result."""
    usage = result.usage
    common: dict[str, Any] = {
        "question_id": question.id,
        "question": question.question,
        "tier": question.tier.value,
        "expected_behavior": question.expected_behavior,
        "expected_metric": question.expected_metric,
        "latency_seconds": usage.latency_seconds,
        "cost_usd": usage.cost_usd,
        "cached": usage.cached,
    }

    if result.error is not None or result.answer is None:
        return QuestionScore(
            outcome=Outcome.ERRORED, error=result.error or "no answer produced", **common
        )

    answer = result.answer
    confidence = answer.confidence

    if answer.is_refusal:
        return QuestionScore(
            outcome=Outcome.ASKED,
            refusal_correct=not question.expected_behavior.should_answer,
            names_the_fork=(
                _names_the_fork(question, answer.clarifying_question)
                if not question.expected_behavior.should_answer
                else None
            ),
            confidence=confidence,
            clarifying_question=answer.clarifying_question,
            **common,
        )

    metric_used = answer.metric_definition_used.name if answer.metric_definition_used else None
    actual = answer.answer_value.rows if answer.answer_value else ()

    value_correct: bool | None = None
    if question.expected_behavior is Behavior.ANSWER and question.id in gold:
        expected = tuple(tuple(row) for row in gold[question.id])
        value_correct = rows_match(question, actual, expected)

    metric_correct = metric_used == question.expected_metric if question.expected_metric else None
    caveat_text = " ".join(caveat.text for caveat in answer.caveats)

    score = QuestionScore(
        outcome=Outcome.ANSWERED,
        value_correct=value_correct,
        metric_correct=metric_correct,
        metric_used=metric_used,
        refusal_correct=question.expected_behavior.should_answer,
        caveats_surfaced=_caveats_surfaced(question, caveat_text),
        confidence=confidence,
        matched_other_gold=find_matching_gold(actual, question.id, gold, questions),
        sql=answer.sql,
        **common,
    )
    return score.model_copy(
        update={"confident_wrong": score.is_wrong_answer and confidence >= CONFIDENT_THRESHOLD}
    )


def score_baseline(
    question: Question,
    output: Any,
    gold: dict[str, list[list[Any]]],
    questions: QuestionSet,
) -> QuestionScore:
    """Score one baseline result.

    ``metric_correct`` stays None throughout: the baseline declares no metric, so there
    is nothing to compare. That is the structural finding, not a gap in the harness.

    Confidence is fixed at 1.0 for every answer it gives. The baseline reports no
    confidence, and treating a bare number as a confident claim is the honest reading --
    a system that hands over a figure with no hedge is asserting it. This is why the
    confident-wrong comparison is meaningful rather than an artefact of scoring.
    """
    usage = output.usage
    common: dict[str, Any] = {
        "question_id": question.id,
        "question": question.question,
        "tier": question.tier.value,
        "expected_behavior": question.expected_behavior,
        "expected_metric": question.expected_metric,
        "latency_seconds": usage.latency_seconds,
        "cost_usd": usage.cost_usd,
        "cached": usage.cached,
    }

    if output.error is not None:
        return QuestionScore(outcome=Outcome.ERRORED, error=output.error, **common)

    if output.clarifying_question is not None:
        return QuestionScore(
            outcome=Outcome.ASKED,
            refusal_correct=not question.expected_behavior.should_answer,
            names_the_fork=(
                _names_the_fork(question, output.clarifying_question)
                if not question.expected_behavior.should_answer
                else None
            ),
            confidence=0.0,
            clarifying_question=output.clarifying_question,
            **common,
        )

    if output.execution_error is not None:
        return QuestionScore(
            outcome=Outcome.ERRORED,
            error=f"SQL failed: {output.execution_error}",
            sql=output.sql,
            **common,
        )

    value_correct: bool | None = None
    if question.expected_behavior is Behavior.ANSWER and question.id in gold:
        expected = tuple(tuple(row) for row in gold[question.id])
        value_correct = rows_match(question, output.rows, expected)

    score = QuestionScore(
        outcome=Outcome.ANSWERED,
        value_correct=value_correct,
        metric_correct=None,
        metric_used=None,
        refusal_correct=question.expected_behavior.should_answer,
        confidence=1.0,
        matched_other_gold=find_matching_gold(output.rows, question.id, gold, questions),
        sql=output.sql,
        **common,
    )
    return score.model_copy(
        update={"confident_wrong": score.is_wrong_answer and CONFIDENT_THRESHOLD <= 1.0}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N")
    parser.add_argument("--system", choices=["canon", "baseline", "both"], default="both")
    parser.add_argument("--no-cache", action="store_true", help="ignore the response cache")
    parser.add_argument("--tag", default=None, help="label for the output directory")
    args = parser.parse_args()

    layer = load_semantic_layer()
    questions = load_questions()
    questions.validate_against(layer)
    gold = _load_gold()

    selected = questions.questions[: args.limit] if args.limit else questions.questions
    subset = QuestionSet(questions=selected)

    thresholds = PolicyThresholds()
    started = datetime.now(UTC)
    provenance = Provenance(
        started_at=started.isoformat(),
        code_commit=_git("rev-parse", "--short", "HEAD"),
        eval_set_commit=_git("log", "-1", "--format=%h", "--", "benchmark/questions"),
        metrics_yaml_sha256=semantic_layer_hash(),
        model_id=MODEL_ID,
        effort=LlmClient().effort,
        policy_thresholds=thresholds,
        confident_threshold=CONFIDENT_THRESHOLD,
        question_count=len(selected),
        cache_used=not args.no_cache,
    )

    tag = args.tag or started.strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUNS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregates: list[Aggregate] = []
    all_scores: dict[str, list[QuestionScore]] = {}

    if args.system in ("canon", "both"):
        client = LlmClient(use_cache=not args.no_cache)
        pipeline = CanonPipeline(
            db_path=DB_PATH,
            layer=layer,
            client=client,
            thresholds=thresholds,
            decision_log=out_dir / "decisions.jsonl",
        )
        scores: list[QuestionScore] = []
        try:
            for index, question in enumerate(selected, start=1):
                result = pipeline.answer(question.question)
                score = score_canon(question, result, gold, subset)
                scores.append(score)
                print(
                    f"[canon {index}/{len(selected)}] {question.id:8} "
                    f"{score.outcome.value:8} "
                    f"value={score.value_correct} metric={score.metric_used or '-'}"
                )
        finally:
            pipeline.close()
        all_scores["canon"] = scores
        aggregates.append(aggregate("canon", scores, subset))

    if args.system in ("baseline", "both"):
        client = LlmClient(use_cache=not args.no_cache)
        baseline = NaiveBaseline(db_path=DB_PATH, client=client)
        scores = []
        try:
            for index, question in enumerate(selected, start=1):
                output = baseline.answer(question.question)
                score = score_baseline(question, output, gold, subset)
                scores.append(score)
                print(
                    f"[baseline {index}/{len(selected)}] {question.id:8} "
                    f"{score.outcome.value:8} value={score.value_correct}"
                )
        finally:
            baseline.close()
        all_scores["baseline"] = scores
        aggregates.append(aggregate("baseline", scores, subset))

    for system, scores in all_scores.items():
        with (out_dir / f"{system}.jsonl").open("w", encoding="utf-8") as handle:
            for score in scores:
                handle.write(json.dumps(score.model_dump(mode="json"), sort_keys=True) + "\n")

    payload = {
        "provenance": provenance.model_dump(mode="json"),
        "aggregates": [item.model_dump(mode="json") for item in aggregates],
        "confident_wrong_curve": {
            system: confident_wrong_curve(scores) for system, scores in all_scores.items()
        },
    }
    (out_dir / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(out_dir / "report.md", provenance, aggregates, all_scores)

    print(f"\nwrote {out_dir.relative_to(REPO_ROOT)}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
