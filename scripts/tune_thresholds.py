"""Tune the two refusal thresholds on the dev set, then freeze them.

Why this is a separate script and a separate question file: the thresholds decide when
Canon asks instead of answering, so tuning them on the 80-question eval set would fit
them to the test set and inflate refusal accuracy — the project's most distinctive claim
and therefore the worst place to leak.

How it works, and why it is cheap: one structured call per dev question collects the
model's ``AgentDecision`` (14 calls, cached). The threshold sweep then runs *offline*
over those cached decisions, because the policy is deterministic Python over the scores
the decision already contains. Sweeping 100 threshold pairs costs nothing.

The output is written to ``benchmark/dev/tuning_result.json``, which is what lets the
harness make an honest provenance claim: if the file is absent, the run says the
thresholds were never validated. A claim that cannot be checked is not made.

Run:  uv run python scripts/tune_thresholds.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from canon.agent.llm import CallContext, LlmClient, LlmError
from canon.agent.pipeline import CanonPipeline
from canon.agent.policy import PolicyOutcome, PolicyThresholds, decide
from canon.schemas.decision import AgentDecision
from canon.semantic.loader import load_semantic_layer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_QUESTIONS = REPO_ROOT / "benchmark" / "dev" / "questions.yaml"
RESULT = REPO_ROOT / "benchmark" / "dev" / "tuning_result.json"
DB_PATH = REPO_ROOT / "data" / "canon.duckdb"

MARGIN_GRID = [round(0.05 * step, 2) for step in range(1, 13)]  # 0.05 .. 0.60
MIN_SCORE_GRID = [round(0.05 * step, 2) for step in range(4, 15)]  # 0.20 .. 0.70


class DevQuestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str
    expected_behavior: str
    expected_metric: str | None = None
    notes: str

    @property
    def should_answer(self) -> bool:
        return self.expected_behavior == "answer"


def load_dev() -> tuple[DevQuestion, ...]:
    raw = yaml.safe_load(DEV_QUESTIONS.read_text(encoding="utf-8"))
    return tuple(DevQuestion(**item) for item in raw)


def main() -> int:
    dev = load_dev()
    layer = load_semantic_layer()
    client = LlmClient()
    pipeline = CanonPipeline(db_path=DB_PATH, layer=layer, client=client, decision_log=None)

    print(f"collecting decisions for {len(dev)} dev questions ({client.name} / {client.model})")
    decisions: dict[str, AgentDecision] = {}
    try:
        for item in dev:
            try:
                decision, usage = client.structured(
                    system=pipeline.system_prompt,
                    question=item.question,
                    output_format=AgentDecision,
                    context=CallContext(system="dev-tuning", question_id=item.id, run_index=0),
                )
            except LlmError as error:
                print(f"  {item.id}: FAILED -- {error}")
                return 1
            decisions[item.id] = decision
            marker = "cached" if usage.cached else f"{usage.latency_seconds:.1f}s"
            print(
                f"  {item.id}: top={decision.top.metric} score={decision.top.score:.2f} "
                f"gap={decision.score_gap:.2f} intent={'yes' if decision.intent else 'no'} "
                f"({marker})"
            )
    finally:
        pipeline.close()

    print(f"\nsweeping {len(MARGIN_GRID)} x {len(MIN_SCORE_GRID)} threshold pairs offline")
    results: list[dict[str, Any]] = []
    for margin in MARGIN_GRID:
        for minimum in MIN_SCORE_GRID:
            thresholds = PolicyThresholds(metric_score_margin=margin, minimum_top_score=minimum)
            correct = 0
            over = 0
            under = 0
            for item in dev:
                verdict = decide(
                    question=item.question,
                    decision=decisions[item.id],
                    layer=layer,
                    thresholds=thresholds,
                )
                answered = verdict.outcome is PolicyOutcome.ANSWER
                if answered == item.should_answer:
                    correct += 1
                elif item.should_answer:
                    over += 1
                else:
                    under += 1
            results.append(
                {
                    "metric_score_margin": margin,
                    "minimum_top_score": minimum,
                    "correct": correct,
                    "accuracy": round(correct / len(dev), 4),
                    "over_refusals": over,
                    "under_refusals": under,
                }
            )

    # Selection rule, fixed before looking at the sweep: maximise dev accuracy; break
    # ties toward fewer under-refusals, because a confidently wrong number is worse than
    # an unnecessary question; then toward the *lower* margin, which is the less
    # refusal-happy setting and so the more conservative claim to make.
    best = min(
        results,
        key=lambda row: (
            -row["correct"],
            row["under_refusals"],
            row["metric_score_margin"],
            row["minimum_top_score"],
        ),
    )

    top = sorted(results, key=lambda row: (-row["correct"], row["under_refusals"]))[:8]
    print("\nbest settings by dev accuracy:")
    for row in top:
        print(
            f"  margin={row['metric_score_margin']:.2f} min_score={row['minimum_top_score']:.2f}"
            f"  correct={row['correct']}/{len(dev)}  over={row['over_refusals']}"
            f" under={row['under_refusals']}"
        )

    current = PolicyThresholds()
    payload = {
        "tuned_at": datetime.now(UTC).isoformat(),
        "model_id": client.model,
        "provider": client.name,
        "effort": client.effort,
        "dev_question_count": len(dev),
        "dev_question_ids": [item.id for item in dev],
        "eval_set_used": False,
        "selection_rule": (
            "maximise dev accuracy; ties broken toward fewer under-refusals, then the "
            "lower margin. Fixed before the sweep was run."
        ),
        "grid": {"metric_score_margin": MARGIN_GRID, "minimum_top_score": MIN_SCORE_GRID},
        "chosen": {
            "metric_score_margin": best["metric_score_margin"],
            "minimum_top_score": best["minimum_top_score"],
        },
        "chosen_dev_accuracy": best["accuracy"],
        "chosen_over_refusals": best["over_refusals"],
        "chosen_under_refusals": best["under_refusals"],
        "previous_defaults": {
            "metric_score_margin": current.metric_score_margin,
            "minimum_top_score": current.minimum_top_score,
        },
        "all_results": results,
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nchosen: margin={best['metric_score_margin']} min_score={best['minimum_top_score']}")
    print(f"dev accuracy {best['correct']}/{len(dev)} = {best['accuracy']:.1%}")
    print(f"written to {RESULT.relative_to(REPO_ROOT)}")
    if (
        best["metric_score_margin"] != current.metric_score_margin
        or best["minimum_top_score"] != current.minimum_top_score
    ):
        print(
            "\nACTION REQUIRED: update the defaults in canon/agent/policy.py to the "
            "chosen values, then FREEZE them. Do not tune again."
        )
    else:
        print("\nthe a-priori defaults are already optimal on the dev set; nothing to change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
