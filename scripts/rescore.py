"""Re-score a completed run from cache, without spending any quota.

Scoring code changes more often than model responses do. When a new metric is added --
``shape_correct``, say -- every unit already on disk should get it without re-asking the
model. The LLM responses are cached, DuckDB is local, and the compiler is pure, so a
full re-score is free.

The guarantee is enforced rather than hoped for: the client runs in **cache-only** mode
and raises on a cache miss. If re-scoring would need a live call, this exits non-zero
instead of quietly spending the day's budget.

Run:  uv run python scripts/rescore.py --tag sweep-groq-r1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmark.harness.baseline import NaiveBaseline
from benchmark.harness.dataset import load_questions
from benchmark.harness.run import (
    DB_PATH,
    RUNS_DIR,
    _load_gold,
    score_baseline,
    score_canon,
)
from canon.agent.llm import CacheMiss, CallContext, LlmClient
from canon.agent.pipeline import CanonPipeline
from canon.semantic.loader import load_semantic_layer

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="run directory under benchmark/runs")
    parser.add_argument(
        "--provider",
        default=None,
        help="override the provider recorded in the checkpoint (rarely needed)",
    )
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.tag
    checkpoint = run_dir / "checkpoint.jsonl"
    if not checkpoint.exists():
        print(f"no checkpoint at {checkpoint}")
        return 1

    records = [
        json.loads(line)
        for line in checkpoint.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    providers = {record.get("_provider", "unrecorded") for record in records}
    known = providers - {"unrecorded"}
    if len(known) > 1:
        print(f"checkpoint mixes providers ({', '.join(sorted(known))}); refusing")
        return 3
    provider = args.provider or (next(iter(known)) if known else "groq")

    layer = load_semantic_layer()
    gold = _load_gold()
    questions = load_questions()
    by_id = {item.id: item for item in questions.questions}

    # Cache-only: a miss raises rather than calling out. This is what makes the
    # zero-quota promise a property of the code and not of my care.
    client = LlmClient(provider=provider, cache_only=True)
    pipeline = CanonPipeline(db_path=DB_PATH, layer=layer, client=client, decision_log=None)
    baseline = NaiveBaseline(db_path=DB_PATH, client=client)

    rescored: list[dict[str, Any]] = []
    misses: list[str] = []
    changed = 0

    try:
        for record in records:
            system = record["_system"]
            question_id = record["question_id"]
            question = by_id.get(question_id)
            if question is None:
                rescored.append(record)
                continue

            # An errored unit has no cached response to replay; keep it as it stands.
            if record.get("outcome") == "errored":
                rescored.append(record)
                continue

            context = CallContext(
                system=system, question_id=question_id, run_index=record["_run_index"]
            )
            try:
                if system == "canon":
                    score = score_canon(
                        question, pipeline.answer(question.question, context), gold, questions
                    )
                else:
                    score = score_baseline(
                        question, baseline.answer(question.question, context), gold, questions
                    )
            except CacheMiss:
                misses.append(f"{system}/{question_id}")
                rescored.append(record)
                continue

            fresh = score.model_dump(mode="json")
            fresh["_system"] = system
            fresh["_run_index"] = record["_run_index"]
            fresh["_provider"] = record.get("_provider", provider)
            if fresh != record:
                changed += 1
            rescored.append(fresh)
    finally:
        pipeline.close()
        baseline.close()

    if misses:
        print(f"{len(misses)} unit(s) had no cached response and were left unchanged:")
        for name in misses[:10]:
            print(f"  {name}")
        if len(misses) > 10:
            print(f"  ... and {len(misses) - 10} more")

    backup = checkpoint.with_suffix(".jsonl.bak")
    backup.write_text(checkpoint.read_text(encoding="utf-8"), encoding="utf-8")
    checkpoint.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in rescored) + "\n",
        encoding="utf-8",
    )

    print(f"\nre-scored {len(rescored)} units from cache; {changed} changed")
    print(f"previous checkpoint saved to {backup.name}")
    print(
        f"\nNow regenerate the report:\n  uv run python -m benchmark.harness.run --tag {args.tag}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
