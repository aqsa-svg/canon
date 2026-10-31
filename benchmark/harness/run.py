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
    attribute_wrong_answer,
    behavior_matrix,
    confident_wrong_curve,
    gold_collision_report,
    paired_discrimination,
    rows_match,
    shape_matches,
)
from canon.agent.llm import CallContext, LlmClient
from canon.agent.pipeline import CanonPipeline, PipelineResult
from canon.agent.policy import PolicyThresholds
from canon.agent.providers import (
    DEFAULT_PROVIDER,
    QuotaExhausted,
    is_free_tier,
    is_transport_error,
)
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
    """When the session that produced the ANSWERS began.

    Preserved across re-scores. A resumed sweep re-aggregates from the checkpoint without
    calling the model at all, and an earlier version re-stamped this field every time --
    so re-scoring the September run on 2026-10-23 made it claim it ran that day. The
    scoring session records itself in ``scored_at`` instead.
    """

    code_commit: str
    """HEAD when the answering session started. See ``answering_span`` for why a single
    commit understates a sweep that could not finish inside one day's token budget."""

    eval_set_commit: str
    metrics_yaml_sha256: str

    scored_at: str = ""
    """When this report was last written. Differs from ``started_at`` whenever a finished
    run is re-scored, which costs no quota and answers nothing."""

    scoring_commit: str = ""
    """HEAD at the last re-score. A change here with ``code_commit`` unchanged means the
    numbers were re-derived by newer scoring code from the same answers."""

    answering_sessions: tuple[str, ...] = ()
    """Every session that answered at least one unit, oldest first, as
    ``<iso timestamp>@<commit>``.

    A free-tier sweep cannot finish inside one day's token cap, so it resumes across days
    and a single ``code_commit`` is a convenient fiction. This records the truth. Empty on
    runs that predate the field, where ``data/quota_ledger.json`` is the fallback record
    of which UTC days had live calls.
    """

    @property
    def answering_span(self) -> tuple[str, str] | None:
        """First and last answering session, or None when only one (or none) is recorded."""
        if len(self.answering_sessions) < 2:
            return None
        return self.answering_sessions[0], self.answering_sessions[-1]

    provider: str
    """Which backend answered. A result from one provider is not a result from another,
    so this is part of the run's identity rather than a footnote."""

    provider_config_hash: str
    """Fingerprint of model id and decoding settings. Two runs with different hashes are
    not comparable, and the hash makes that checkable instead of assumed."""

    free_tier: bool
    """Whether this run cost nothing. The project's constraint is a $0 reproduction, so a
    run that quietly used a paid provider must be visible as such."""

    model_id: str
    effort: str
    policy_thresholds: PolicyThresholds
    threshold_provenance: str
    """Where the two refusal thresholds were tuned. Stated on every run because a
    threshold fitted to the benchmark would inflate refusal accuracy, the project's most
    distinctive claim."""

    confident_threshold: float
    question_count: int
    runs: int
    cache_used: bool

    complete: bool = True
    """False when the sweep did not finish -- quota exhausted, or interrupted. A partial
    sweep is not averaged and not reported as a result; the report is stamped INCOMPLETE
    and says how much is missing."""

    completed_units: int = 0
    expected_units: int = 0

    scored_questions: int = 0
    """Questions in the balanced subset every arm answered. Equals question_count only on
    a complete run."""


TUNING_RESULT = REPO_ROOT / "benchmark" / "dev" / "tuning_result.json"


def _prior_provenance(out_dir: Path) -> dict[str, Any]:
    """Answering provenance from an earlier session of this run, if there is one.

    Returns ``{}`` for a new run. Read from the existing ``report.json`` so a resume or a
    re-score inherits when the answers were actually produced rather than overwriting it
    with the current clock.
    """
    report = out_dir / "report.json"
    if not report.exists():
        return {}
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    existing = payload.get("provenance")
    return existing if isinstance(existing, dict) else {}


def threshold_provenance() -> str:
    """Where the refusal thresholds come from, derived rather than asserted.

    Recorded on every run because a threshold fitted to the eval set would inflate
    refusal accuracy, and refusal accuracy is the project's most distinctive claim. A
    reader has to be able to check that the tuning set and the test set are disjoint.

    Read from the tuning artifact rather than hardcoded, so the sentence cannot drift
    from what actually happened. No artifact means no validation, and the run says so
    instead of implying otherwise.
    """
    if not TUNING_RESULT.exists():
        return (
            "UNVALIDATED: chosen a priori from the design and never fitted to data. "
            "No eval-set question was used, so refusal accuracy is not inflated by "
            "threshold leakage -- but the values have also never been checked against "
            "real model output. Run scripts/tune_thresholds.py against "
            "benchmark/dev/questions.yaml (14 held-out questions) to validate them."
        )

    payload = json.loads(TUNING_RESULT.read_text(encoding="utf-8"))
    chosen = payload["chosen"]
    return (
        f"tuned on benchmark/dev/questions.yaml ({payload['dev_question_count']} held-out "
        f"questions, disjoint from the 80-question eval set) at {payload['tuned_at']} on "
        f"{payload['model_id']}; chose margin={chosen['metric_score_margin']} "
        f"min_score={chosen['minimum_top_score']} at "
        f"{payload['chosen_dev_accuracy']:.0%} dev accuracy. "
        f"eval_set_used={payload['eval_set_used']}. Frozen thereafter."
    )


_TRACKED_RATES = (
    "execution_accuracy",
    "semantic_accuracy",
    "refusal_accuracy",
    "silent_wrong_rate",
    "under_refusal_rate",
    "over_refusal_rate",
)


def _variance_across_runs(
    system: str, runs: list[list[QuestionScore]], subset: QuestionSet
) -> dict[str, Any]:
    """Spread of each headline rate across independent runs.

    Returns ``null`` for every field when there is only one run: a single sample has no
    spread, and printing 0.0 would claim a stability that was never measured.
    """
    if len(runs) < 2:
        return {"runs": len(runs), "note": "single run -- no variance measured", "rates": None}

    aggregates = [aggregate(system, scores, subset) for scores in runs]
    rates: dict[str, Any] = {}
    for field in _TRACKED_RATES:
        values = [getattr(item, field) for item in aggregates]
        present = [value for value in values if value is not None]
        if not present:
            rates[field] = None
            continue
        mean = sum(present) / len(present)
        spread = max(present) - min(present)
        rates[field] = {
            "mean": round(mean, 4),
            "min": round(min(present), 4),
            "max": round(max(present), 4),
            "range": round(spread, 4),
            "per_run": present,
        }
    return {"runs": len(runs), "rates": rates}


class WorkItem(BaseModel):
    """One unit of work: a system, a run index, and a question."""

    model_config = _FROZEN

    system: str
    run_index: int
    question_id: str

    @property
    def key(self) -> str:
        return f"{self.system}|{self.run_index}|{self.question_id}"


def _load_checkpoint(path: Path) -> dict[str, QuestionScore]:
    """Rehydrate completed work from a previous session.

    Keyed by system, run index and question id, so resuming re-does nothing and the
    quota spent yesterday is not spent again.

    **Transport failures are deliberately NOT loaded**, so they become pending again and
    are re-attempted. A blocked SSL library or a dropped connection says nothing about the
    system under test, and freezing it as a permanent error understates the arm for a
    reason that is not about the architecture. A schema rejection or a model failure IS
    kept, because that is a finding.
    """
    if not path.exists():
        return {}
    done: dict[str, QuestionScore] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("outcome") == "errored" and is_transport_error(record.get("error") or ""):
            continue
        item = WorkItem(
            system=record["_system"],
            run_index=record["_run_index"],
            question_id=record["question_id"],
        )
        done[item.key] = QuestionScore(**{k: v for k, v in record.items() if not k.startswith("_")})
    return done


def _append_checkpoint(path: Path, item: WorkItem, score: QuestionScore, provider: str) -> None:
    """Write one result immediately.

    After every question rather than at the end: a session that dies to quota, a network
    drop, or a keyboard interrupt must lose at most the question in flight.

    The provider is recorded per unit so a later resume can prove the whole checkpoint
    came from one backend.
    """
    record = score.model_dump(mode="json")
    record["_system"] = item.system
    record["_run_index"] = item.run_index
    record["_provider"] = provider
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _checkpoint_providers(path: Path) -> set[str]:
    """Every provider that appears in an existing checkpoint.

    More than one means the file already mixes backends and nothing in it is scoreable.
    Units written before this field existed report as 'unrecorded'.
    """
    if not path.exists():
        return set()
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            seen.add(json.loads(line).get("_provider", "unrecorded"))
    return seen


def provider_conflict(checkpoint_path: Path, provider: str) -> str | None:
    """Why this checkpoint must not be extended on ``provider``, or None if it is safe.

    A tag is bound to whichever provider first wrote to it. Mixing two backends inside one
    comparison would make the delta between Canon and the baseline measure the backend
    rather than the architecture, and the mixing would be invisible in the report -- so it
    is refused up front rather than detected afterwards.

    Extracted from main() so it can be tested without credentials for a second provider.
    """
    existing = _checkpoint_providers(checkpoint_path)
    known = existing - {"unrecorded"}

    if len(known) > 1:
        return (
            f"REFUSING TO RUN: checkpoint {checkpoint_path} already mixes providers "
            f"({', '.join(sorted(known))}). Nothing in it is scoreable."
        )

    foreign = known - {provider}
    if foreign:
        return (
            f"REFUSING TO RUN: checkpoint {checkpoint_path} contains units from "
            f"{', '.join(sorted(foreign))}, and this run is on {provider!r}. Mixing "
            "providers inside one comparison would make the delta measure the backend "
            f"rather than the architecture. Use a different --tag for {provider}."
        )

    if "unrecorded" in existing and known != {provider}:
        return (
            f"REFUSING TO RUN: checkpoint {checkpoint_path} has units written before the "
            "provider was recorded, so it cannot be proven to be "
            f"{provider!r}. Re-run under a fresh --tag, or backfill _provider if you know "
            "which backend produced them."
        )
    return None


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
            gate_flagged=result.verification is not None and not result.verification.passed,
            **common,
        )

    metric_used = answer.metric_definition_used.name if answer.metric_definition_used else None
    actual = answer.answer_value.rows if answer.answer_value else ()

    value_correct: bool | None = None
    shape_correct: bool | None = None
    if question.expected_behavior is Behavior.ANSWER and question.id in gold:
        expected = tuple(tuple(row) for row in gold[question.id])
        value_correct = rows_match(question, actual, expected)
        shape_correct = shape_matches(question, actual, expected)

    metric_correct = metric_used == question.expected_metric if question.expected_metric else None
    caveat_text = " ".join(caveat.text for caveat in answer.caveats)

    attribution = attribute_wrong_answer(actual, question.id, gold, questions)
    score = QuestionScore(
        outcome=Outcome.ANSWERED,
        value_correct=value_correct,
        shape_correct=shape_correct,
        metric_correct=metric_correct,
        metric_used=metric_used,
        refusal_correct=question.expected_behavior.should_answer,
        caveats_surfaced=_caveats_surfaced(question, caveat_text),
        confidence=confidence,
        gate_flagged=False,
        attribution=attribution.status.value,
        matched_other_gold=attribution.matched_question_id,
        attribution_reason=attribution.reason,
        sql=answer.sql,
        **common,
    )
    return score.model_copy(
        update={
            "confident_wrong": score.is_wrong_answer and confidence >= CONFIDENT_THRESHOLD,
            "silent_wrong": score.is_silent_wrong,
        }
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

    Confidence is imputed as 1.0 for every answer it gives, because the baseline reports
    none. That imputation is defensible -- handing over a figure with no hedge is an
    assertion -- but it is still an imputation, which is why confident-wrong is no longer
    the headline. **Silent-wrong** is, and it needs no imputation: it asks only whether a
    wrong number arrived with any warning attached, and the baseline has no mechanism to
    attach one.
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
    shape_correct: bool | None = None
    if question.expected_behavior is Behavior.ANSWER and question.id in gold:
        expected = tuple(tuple(row) for row in gold[question.id])
        value_correct = rows_match(question, output.rows, expected)
        shape_correct = shape_matches(question, output.rows, expected)

    attribution = attribute_wrong_answer(output.rows, question.id, gold, questions)
    score = QuestionScore(
        outcome=Outcome.ANSWERED,
        value_correct=value_correct,
        shape_correct=shape_correct,
        metric_correct=None,
        metric_used=None,
        refusal_correct=question.expected_behavior.should_answer,
        confidence=1.0,
        # Structurally False: the baseline has no verification gate. Recorded rather than
        # assumed, because that absence is exactly what silent-wrong measures.
        gate_flagged=False,
        attribution=attribution.status.value,
        matched_other_gold=attribution.matched_question_id,
        attribution_reason=attribution.reason,
        sql=output.sql,
        **common,
    )
    return score.model_copy(
        update={
            "confident_wrong": score.is_wrong_answer and CONFIDENT_THRESHOLD <= 1.0,
            "silent_wrong": score.is_silent_wrong,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N")
    parser.add_argument(
        "--ids",
        default=None,
        help=(
            "comma-separated question ids. Preferred over --limit for a smoke run: the "
            "first N questions in load order are all from ambiguous.yaml, which exercises "
            "the refusal path and says nothing about grain or metric selection."
        ),
    )
    parser.add_argument("--system", choices=["canon", "baseline", "both"], default="both")
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help="which backend answers; both systems always share it",
    )
    parser.add_argument("--runs", type=int, default=1, help="independent samples per question")
    parser.add_argument("--no-cache", action="store_true", help="ignore the response cache")
    parser.add_argument("--tag", default=None, help="label for the output directory")
    args = parser.parse_args()

    layer = load_semantic_layer()
    questions = load_questions()
    questions.validate_against(layer)
    gold = _load_gold()

    if args.ids:
        wanted = [item.strip() for item in args.ids.split(",") if item.strip()]
        by_id = {item.id: item for item in questions.questions}
        missing = [name for name in wanted if name not in by_id]
        if missing:
            print(f"unknown question ids: {', '.join(missing)}")
            return 1
        selected = tuple(by_id[name] for name in wanted)
    elif args.limit:
        selected = questions.questions[: args.limit]
    else:
        selected = questions.questions
    subset = QuestionSet(questions=selected)

    thresholds = PolicyThresholds()
    provenance_client = LlmClient(provider=args.provider, use_cache=not args.no_cache)

    # Never mix providers inside one comparison. Canon and the baseline must answer on
    # the same backend or the delta between them measures the backend, not the
    # architecture -- so both are constructed from this single client's provider.
    if args.system == "both" and not provenance_client.has_credentials():
        print(
            f"no credentials for provider {provenance_client.name!r}. "
            "Add the key to .env, or run scripts/dry_run.py which needs none."
        )
        return 1

    started = datetime.now(UTC)
    head = _git("rev-parse", "--short", "HEAD")
    prior = _prior_provenance(RUNS_DIR / (args.tag or ""))

    provenance = Provenance(
        # Preserved from the first answering session, so re-scoring a finished run does
        # not restamp it with today's date and today's commit.
        started_at=prior.get("started_at") or started.isoformat(),
        code_commit=prior.get("code_commit") or head,
        scored_at=started.isoformat(),
        scoring_commit=head,
        answering_sessions=tuple(prior.get("answering_sessions") or ()),
        eval_set_commit=_git("log", "-1", "--format=%h", "--", "benchmark/questions"),
        metrics_yaml_sha256=semantic_layer_hash(),
        provider=provenance_client.name,
        provider_config_hash=provenance_client.config_hash,
        free_tier=is_free_tier(provenance_client.name),
        model_id=provenance_client.model,
        effort=provenance_client.effort,
        policy_thresholds=thresholds,
        threshold_provenance=threshold_provenance(),
        confident_threshold=CONFIDENT_THRESHOLD,
        question_count=len(selected),
        runs=args.runs,
        cache_used=not args.no_cache,
    )

    tag = args.tag or started.strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUNS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / "checkpoint.jsonl"
    done = _load_checkpoint(checkpoint_path)
    if done:
        print(f"resuming: {len(done)} units already complete in {checkpoint_path.name}")

    refusal = provider_conflict(checkpoint_path, provenance_client.name)
    if refusal is not None:
        print(refusal)
        return 3

    systems = (
        ["baseline", "canon"]
        if args.system == "both"
        else ["canon" if args.system == "canon" else "baseline"]
    )
    # Interleaved by question, so a quota stop leaves N complete question-pairs rather
    # than one whole arm and a stub of the other. The Groq run ordered baseline-first and
    # its day-1 stop left 80/80 against 4/80 -- two arms, nothing comparable.
    work: list[WorkItem] = [
        WorkItem(system=system, run_index=run_index, question_id=question.id)
        for run_index in range(args.runs)
        for question in selected
        for system in systems
    ]
    expected_units = len(work)
    pending = [item for item in work if item.key not in done]
    print(f"{expected_units} units total, {len(pending)} to do\n")

    by_id = {item.id: item for item in selected}
    pipeline: CanonPipeline | None = None
    baseline: NaiveBaseline | None = None
    complete = True
    answered_this_session = bool(pending)
    """Whether this session produced any new answers. False on a pure re-score, which
    must not add itself to the answering history or move ``started_at``."""

    try:
        for position, item in enumerate(pending, start=1):
            question = by_id[item.question_id]
            context = CallContext(
                system=item.system, question_id=item.question_id, run_index=item.run_index
            )
            try:
                if item.system == "canon":
                    if pipeline is None:
                        pipeline = CanonPipeline(
                            db_path=DB_PATH,
                            layer=layer,
                            client=LlmClient(
                                provider=provenance_client.provider,
                                use_cache=not args.no_cache,
                            ),
                            thresholds=thresholds,
                            decision_log=out_dir / "decisions.jsonl",
                        )
                    score = score_canon(
                        question, pipeline.answer(question.question, context), gold, subset
                    )
                else:
                    if baseline is None:
                        baseline = NaiveBaseline(
                            db_path=DB_PATH,
                            client=LlmClient(
                                provider=provenance_client.provider,
                                use_cache=not args.no_cache,
                            ),
                        )
                    score = score_baseline(
                        question, baseline.answer(question.question, context), gold, subset
                    )
            except QuotaExhausted as exhausted:
                # The expected end of a session, not a failure. Everything so far is on
                # disk; re-run the identical command tomorrow to continue.
                print(f"\nQUOTA EXHAUSTED after {position - 1} of {len(pending)} pending units")
                print(f"  {exhausted}")
                print(f"  progress saved to {checkpoint_path}")
                print(f"  resume with the SAME --tag {tag}")
                complete = False
                break
            except KeyboardInterrupt:
                print("\ninterrupted; progress is checkpointed")
                complete = False
                break

            done[item.key] = score
            _append_checkpoint(checkpoint_path, item, score, provenance_client.name)
            print(
                f"[{item.system} r{item.run_index} {position}/{len(pending)}] "
                f"{item.question_id:8} {score.outcome.value:8} "
                f"value={score.value_correct} metric={score.metric_used or '-'}"
            )
    finally:
        if pipeline is not None:
            pipeline.close()
        if baseline is not None:
            baseline.close()

    aggregates: list[Aggregate] = []
    all_scores: dict[str, list[QuestionScore]] = {}
    per_run: dict[str, list[list[QuestionScore]]] = {}

    # Score the BALANCED subset: questions every system and run answered. Interleaving
    # makes that subset meaningful -- both arms cover the same questions, so the delta is
    # a real comparison rather than two different samples.
    #
    # It is still not a complete run. Provenance records complete=false whenever the
    # subset is smaller than the selection, and RESULTS.md publishes only complete arms;
    # this exists so a stopped run can be inspected, not quoted.
    # A re-attempted transport failure leaves two lines for the same key; `done` already
    # holds the later one because the file is read in order and the retry is appended
    # last. Nothing else to do, but stated because a duplicated key is easy to misread as
    # a double count.
    balanced = [
        question
        for question in selected
        if all(
            WorkItem(system=system, run_index=run_index, question_id=question.id).key in done
            for system in systems
            for run_index in range(args.runs)
        )
    ]
    if len(balanced) < len(selected):
        complete = False
        missing_per_tier: dict[str, int] = {}
        for question in selected:
            if question not in balanced:
                missing_per_tier[question.tier.value] = (
                    missing_per_tier.get(question.tier.value, 0) + 1
                )
        print(
            f"\n  balanced subset: {len(balanced)}/{len(selected)} questions answered by "
            f"every arm. Missing per tier: {missing_per_tier}"
        )
        print("  the report covers that subset only and is stamped INCOMPLETE")

    scoreable = QuestionSet(questions=tuple(balanced)) if balanced else None
    if scoreable is not None:
        for system in systems:
            runs = [
                [
                    done[WorkItem(system=system, run_index=run_index, question_id=q.id).key]
                    for q in balanced
                ]
                for run_index in range(args.runs)
            ]
            per_run[system] = runs
            all_scores[system] = runs[0]
            aggregates.append(aggregate(system, runs[0], scoreable))
        subset = scoreable
    else:
        complete = False
        print("\n  no question has been answered by every arm yet; nothing to score")

    completed_units = len(done)
    # Record this session only if it actually answered something. A re-score answers
    # nothing and must not appear in the answering history.
    sessions = provenance.answering_sessions
    if answered_this_session:
        sessions = (*sessions, f"{started.isoformat()}@{head}")

    provenance = provenance.model_copy(
        update={
            "complete": complete and completed_units == expected_units,
            "completed_units": completed_units,
            "expected_units": expected_units,
            "scored_questions": len(balanced),
            "answering_sessions": sessions,
        }
    )

    for system, runs in per_run.items():
        for run_index, scores in enumerate(runs):
            name = f"{system}.jsonl" if run_index == 0 else f"{system}.run{run_index}.jsonl"
            with (out_dir / name).open("w", encoding="utf-8") as handle:
                for score in scores:
                    handle.write(json.dumps(score.model_dump(mode="json"), sort_keys=True) + "\n")

    payload = {
        "provenance": provenance.model_dump(mode="json"),
        "aggregates": [item.model_dump(mode="json") for item in aggregates],
        "per_run_aggregates": {
            system: [aggregate(system, scores, subset).model_dump(mode="json") for scores in runs]
            for system, runs in per_run.items()
        },
        "variance": {
            system: _variance_across_runs(system, runs, subset) for system, runs in per_run.items()
        },
        "behavior_matrix": {
            system: behavior_matrix(scores) for system, scores in all_scores.items()
        },
        "paired_discrimination": {
            system: paired_discrimination(scores) for system, scores in all_scores.items()
        },
        "attribution_instrument": gold_collision_report(gold, subset),
        "confident_wrong_curve": {
            system: confident_wrong_curve(scores) for system, scores in all_scores.items()
        },
    }
    (out_dir / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(
        out_dir / "report.md",
        provenance,
        aggregates,
        all_scores,
        matrices=payload["behavior_matrix"],
        pairs=payload["paired_discrimination"],
        attribution=payload["attribution_instrument"],
        variance=payload["variance"],
    )

    print(f"\nwrote {out_dir.relative_to(REPO_ROOT)}/report.md")
    if not provenance.complete:
        print(
            f"\nRUN IS INCOMPLETE: {provenance.completed_units}/{provenance.expected_units} "
            "units. The report is stamped INCOMPLETE and its numbers must not be quoted."
        )
        print(f"Resume with: uv run python -m benchmark.harness.run --tag {tag} ...")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
