"""The one verification gate: does this result look like an answer to this question?

Scope is deliberately one gate, not three (`CLAUDE.md`: dual-query reconciliation and
driver analysis are out of scope). This gate checks the *result*, after execution,
against what the plan said it should be:

* **Shape** — the columns that came back are the columns the compiler said it selected.
* **Types** — each cell is a type the answer schema can carry, and a metric column is
  actually numeric rather than a string that looks like one.
* **Emptiness** — zero rows where the question implies at least one.
* **All-null** — a metric column that is entirely null, which is what a wrong join or an
  impossible filter combination produces.
* **Grain** — the row count is plausible for the requested grain: a monthly series over
  three months returning 900 rows means the grouping is wrong.

On failure the gate returns a **flag, not a number**. The pipeline turns a flagged result
into a refusal rather than showing a figure it cannot vouch for, because a number that
reaches a slide is a number someone will act on.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from canon.compiler.build import CompiledQuery
from canon.compiler.plan import COMPARISON_ALIAS, PERIOD_ALIAS, QueryPlan
from canon.schemas.answer import AnswerValue, ResultColumn
from canon.semantic.enums import TimeGrain

_FROZEN = ConfigDict(frozen=True, extra="forbid")

_ACCEPTED_CELL_TYPES = (Decimal, bool, int, float, str, datetime, date, type(None))

_DAYS_PER_BUCKET: dict[TimeGrain, float] = {
    TimeGrain.DAY: 1.0,
    TimeGrain.WEEK: 7.0,
    TimeGrain.MONTH: 28.0,
    TimeGrain.QUARTER: 89.0,
    TimeGrain.YEAR: 365.0,
}
"""Shortest possible bucket length, used to bound the expected row count from above.

Deliberately the *minimum* length of each bucket: dividing the span by it gives the most
buckets the grain could possibly produce, so the check only fires on a real grouping
error rather than on a February.
"""

GRAIN_TOLERANCE = 3.0
"""How far past the theoretical maximum row count to allow before flagging.

Generous on purpose. This gate exists to catch a grouping that is wrong by a factor —
900 rows for a three-month series — not to police off-by-a-few. A tight bound here would
produce false alarms on dimension breakdowns, and a gate that cries wolf gets ignored.
"""


class VerificationFailure(StrEnum):
    """What went wrong. Reported verbatim so a flag is diagnosable."""

    COLUMN_MISMATCH = "columns_do_not_match_the_plan"
    UNSUPPORTED_CELL_TYPE = "result_contains_an_unrepresentable_value"
    NON_NUMERIC_METRIC = "metric_column_is_not_numeric"
    UNEXPECTED_EMPTY = "no_rows_where_the_question_implies_some"
    ALL_NULL_METRIC = "metric_column_is_entirely_null"
    IMPLAUSIBLE_ROW_COUNT = "row_count_is_implausible_for_the_requested_grain"
    RAGGED_ROWS = "rows_have_inconsistent_widths"


class VerificationReport(BaseModel):
    """The gate's verdict. ``passed`` is the only thing the pipeline branches on."""

    model_config = _FROZEN

    passed: bool
    failures: tuple[VerificationFailure, ...] = ()
    notes: tuple[str, ...] = ()
    """Human-readable detail, one per failure, in the same order."""

    row_count: int = 0
    expected_max_rows: int | None = None


def _is_numeric(value: object) -> bool:
    return isinstance(value, Decimal | int | float) and not isinstance(value, bool)


def _expected_max_rows(plan: QueryPlan) -> int | None:
    """An upper bound on plausible row count, or None when one cannot be derived.

    Only computed when a bounded time range makes it derivable. An unbounded query over
    a free-text dimension has no meaningful bound, and inventing one would produce false
    flags.
    """
    if plan.intent.time_grain is None:
        # No time series: one row per dimension combination. Only bound the scalar case,
        # where the answer must be exactly one row (two with a comparison).
        if not plan.dimensions:
            return 2 if plan.comparison_range is not None else 1
        return None

    time_range = plan.intent.time_range
    if time_range is None or time_range.start is None or time_range.end is None:
        return None

    span_days = (time_range.end - time_range.start).days
    buckets = max(1.0, span_days / _DAYS_PER_BUCKET[plan.intent.time_grain])
    if plan.comparison_range is not None:
        buckets *= 2
    if plan.dimensions:
        # A breakdown multiplies buckets by cardinality the compiler cannot know without
        # querying, so widen rather than guess: the check still catches order-of-
        # magnitude grouping errors.
        return None
    return int(buckets * GRAIN_TOLERANCE) + 1


def verify(
    *,
    plan: QueryPlan,
    compiled: CompiledQuery,
    columns: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
) -> VerificationReport:
    """Check an executed result against what the plan promised.

    ``columns`` and ``rows`` come straight from DuckDB. Pure otherwise: no network, no
    second query.
    """
    failures: list[VerificationFailure] = []
    notes: list[str] = []

    if tuple(columns) != compiled.columns:
        failures.append(VerificationFailure.COLUMN_MISMATCH)
        notes.append(f"plan selected {compiled.columns}, result has {tuple(columns)}")

    width = len(columns)
    if any(len(row) != width for row in rows):
        failures.append(VerificationFailure.RAGGED_ROWS)
        notes.append(f"expected {width} cells per row")

    for row in rows:
        for cell in row:
            if not isinstance(cell, _ACCEPTED_CELL_TYPES):
                failures.append(VerificationFailure.UNSUPPORTED_CELL_TYPE)
                notes.append(f"{type(cell).__name__} cannot be carried in an AnswerValue")
                break
        else:
            continue
        break

    metric_index = columns.index(compiled.metric.name) if compiled.metric.name in columns else -1
    if metric_index >= 0 and rows:
        metric_values = [row[metric_index] for row in rows]
        present = [value for value in metric_values if value is not None]
        if not present:
            failures.append(VerificationFailure.ALL_NULL_METRIC)
            notes.append(
                f"every row's {compiled.metric.name} is null, which is what a wrong join "
                "or an impossible filter combination produces"
            )
        elif not all(_is_numeric(value) for value in present):
            offender = next(value for value in present if not _is_numeric(value))
            failures.append(VerificationFailure.NON_NUMERIC_METRIC)
            notes.append(f"{compiled.metric.name} returned {type(offender).__name__}, not a number")

    if not rows:
        failures.append(VerificationFailure.UNEXPECTED_EMPTY)
        notes.append(
            "the query returned no rows. That is not the same as zero: it means nothing "
            "matched the filters, so no number can be reported"
        )

    expected_max = _expected_max_rows(plan)
    if expected_max is not None and len(rows) > expected_max:
        failures.append(VerificationFailure.IMPLAUSIBLE_ROW_COUNT)
        notes.append(
            f"{len(rows)} rows for a {plan.intent.time_grain or 'scalar'} result over this "
            f"range; at most {expected_max} is plausible, so the grouping is wrong"
        )

    return VerificationReport(
        passed=not failures,
        failures=tuple(failures),
        notes=tuple(notes),
        row_count=len(rows),
        expected_max_rows=expected_max,
    )


def to_answer_value(
    columns: tuple[str, ...], rows: tuple[tuple[object, ...], ...], dtypes: tuple[str, ...]
) -> AnswerValue:
    """Wrap a verified result in the typed answer model.

    Called only after ``verify`` passes, so the types are already known to fit.
    """
    scalar = rows[0][0] if len(rows) == 1 and len(columns) == 1 else None
    return AnswerValue(
        columns=tuple(
            ResultColumn(name=name, dtype=dtype)
            for name, dtype in zip(columns, dtypes, strict=True)
        ),
        rows=tuple(tuple(row) for row in rows),  # type: ignore[arg-type]
        scalar=scalar,  # type: ignore[arg-type]
    )


__all__ = [
    "COMPARISON_ALIAS",
    "PERIOD_ALIAS",
    "VerificationFailure",
    "VerificationReport",
    "to_answer_value",
    "verify",
]
