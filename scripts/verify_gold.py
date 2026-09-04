"""Execute every gold SQL and record what it returned.

Gold values are never typed by hand. ``docs/01-evaluation.md`` fixes the rule: gold SQL
is written by a human reading the metric card, and the gold value is whatever that SQL
returns against ``data/canon.duckdb``. This script closes that loop and writes the
results to ``benchmark/questions/gold_values.json``, which is committed so a data reload
that changes a number shows up in the diff.

It also fails loudly on the things that quietly rot an eval set:

* gold SQL that does not execute;
* gold SQL that returns no rows, which almost always means a filter typo rather than a
  genuinely empty answer;
* a question naming a metric or dimension the canonical layer does not define;
* an `answer` case whose expected metric/dimension pair the metric card forbids.

Run:  uv run python scripts/verify_gold.py
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from benchmark.harness.dataset import Behavior, load_questions
from canon.semantic.loader import load_semantic_layer

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "canon.duckdb"
OUTPUT = REPO_ROOT / "benchmark" / "questions" / "gold_values.json"


def _jsonable(value: Any) -> Any:
    """Serialise a result cell without losing exactness.

    Decimals become strings, not floats: money is compared as DECIMAL, and a float
    round-trip through JSON would silently defeat that.
    """
    if isinstance(value, Decimal):
        return {"__decimal__": str(value)}
    if hasattr(value, "isoformat"):
        return {"__timestamp__": str(value)}
    return value


def main() -> int:
    if not DB_PATH.exists():
        print(f"{DB_PATH} not built -- run: uv run python scripts/load_data.py")
        return 1

    layer = load_semantic_layer()
    questions = load_questions()

    print(f"loaded {len(questions.questions)} questions")
    questions.validate_against(layer)
    print("every expected metric and dimension exists in metrics.yaml")

    counts = {behavior: len(questions.by_behavior(behavior)) for behavior in Behavior}
    print(
        f"  answer:  {counts[Behavior.ANSWER]}\n  clarify: {counts[Behavior.CLARIFY]}\n  refuse:  {counts[Behavior.REFUSE]}"
    )

    connection = duckdb.connect(str(DB_PATH), read_only=True)
    recorded: dict[str, list[list[Any]]] = {}
    problems: list[str] = []

    try:
        for item in questions.answerable:
            if item.gold_sql is None:  # pragma: no cover - validator guarantees this
                continue
            try:
                rows = connection.execute(item.gold_sql).fetchall()
            except duckdb.Error as error:
                problems.append(f"{item.id}: gold SQL failed -- {error}")
                continue

            if not rows:
                problems.append(
                    f"{item.id}: gold SQL returned no rows. An empty gold answer is "
                    "almost always a filter typo, so this is treated as an error."
                )
                continue

            if all(cell is None for row in rows for cell in row):
                problems.append(f"{item.id}: gold SQL returned only nulls")
                continue

            recorded[item.id] = [[_jsonable(cell) for cell in row] for row in rows]
    finally:
        connection.close()

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems:
            print(f"  - {line}")
        return 1

    OUTPUT.write_text(json.dumps(recorded, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nall {len(recorded)} gold queries ran and returned rows")
    print(f"values written to {OUTPUT.relative_to(REPO_ROOT)}")

    print("\nsample of what was recorded:")
    for question_id in sorted(recorded)[:6]:
        rows = recorded[question_id]
        first = rows[0]
        shown = ", ".join(
            str(cell.get("__decimal__") or cell.get("__timestamp__"))
            if isinstance(cell, dict)
            else str(cell)
            for cell in first
        )
        suffix = f"  (+{len(rows) - 1} more rows)" if len(rows) > 1 else ""
        print(f"  {question_id:8} {shown}{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
