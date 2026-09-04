"""``QueryIntent`` -- the only thing the LLM ever produces.

This is Canon's first core idea in one object. The model does not write SQL, does not
name a column, and does not choose a join. It fills in these fields, every one of which
is either an enum generated from ``metrics.yaml``, a date, or a bounded number.

What that buys, precisely:

* ``metric`` and ``dimensions`` are enum members, so an undefined metric or dimension
  cannot be represented at all -- not rejected later, unrepresentable now.
* ``filters`` reference a *dimension*, never a raw column, so the model cannot reach a
  column the semantic layer has not exposed.
* filter ``values`` are the only free-form content, and they reach SQL exclusively as
  typed literal nodes in a syntax tree. They are never concatenated into SQL text.

Everything the compiler needs beyond this -- expressions, joins, grain, default filters
-- comes from the canonical layer, not the model.
"""

from __future__ import annotations

from calendar import monthrange
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from canon.semantic.enums import (
    ComparisonKind,
    DimensionName,
    FilterOp,
    MetricName,
    SortDirection,
    SortTarget,
    TimeGrain,
)

_FROZEN = ConfigDict(frozen=True, extra="forbid")

FilterValue = str | int | float
"""What a filter can compare against. Deliberately not a date: time is expressed by
``time_range``, against the metric's declared time column, so there is exactly one way
to say "when"."""


class IntentFilter(BaseModel):
    """A filter on a dimension the metric is allowed to be sliced by."""

    model_config = _FROZEN

    dimension: DimensionName
    op: FilterOp
    values: tuple[FilterValue, ...] = ()

    @model_validator(mode="after")
    def _check_arity(self) -> IntentFilter:
        arity = self.op.arity
        count = len(self.values)
        if arity == "none" and count != 0:
            raise ValueError(f"{self.op.value} takes no values, got {count}")
        if arity == "one" and count != 1:
            raise ValueError(f"{self.op.value} takes exactly one value, got {count}")
        if arity == "many" and count == 0:
            raise ValueError(f"{self.op.value} needs at least one value")
        return self


class TimeRange(BaseModel):
    """A half-open interval: ``start`` inclusive, ``end`` exclusive.

    Half-open on purpose. It is the only convention under which consecutive periods
    tile without overlapping and without a gap, so "January" and "February" cannot
    both claim the same midnight. ``None`` on either side means unbounded.
    """

    model_config = _FROZEN

    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def _check_order(self) -> TimeRange:
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError(
                f"time_range start {self.start} must be before end {self.end}; "
                "end is exclusive, so a single day is start=D, end=D+1"
            )
        return self

    def spans(self, moment: date) -> bool:
        """Whether this range straddles ``moment``, with data on both sides.

        Used to decide whether a metric with a mid-series definition change needs its
        caveat attached. An unbounded side counts as covering everything on that side.
        """
        after_start = self.start is None or self.start < moment
        before_end = self.end is None or self.end > moment
        return after_start and before_end


class Comparison(BaseModel):
    """A period-over-period comparison against a derived range.

    The model names the *kind* of comparison; the compiler derives the second range from
    the anchor range. That split is the same principle as the rest of the design: the
    model supplies intent, deterministic code supplies arithmetic.
    """

    model_config = _FROZEN

    kind: ComparisonKind

    def derive(self, anchor: TimeRange) -> TimeRange:
        """The range to compare against.

        Requires a bounded anchor -- "compared to the previous period" is meaningless
        without knowing which period, and guessing one would invent a number. Validation
        turns the unbounded case into a clarifying question instead.
        """
        if anchor.start is None or anchor.end is None:
            raise ValueError("a comparison needs a time_range with both start and end")

        if self.kind is ComparisonKind.PREVIOUS_PERIOD:
            months = _aligned_month_span(anchor)
            if months is not None:
                # Month-aligned ranges shift by whole calendar months, so the month
                # before March is February and not "the 31 days before 1 March". A
                # day-count shift would start February on 29 January and quietly
                # report a number nobody asked for.
                return TimeRange(start=_shift_months(anchor.start, months), end=anchor.start)
            span = anchor.end - anchor.start
            return TimeRange(start=anchor.start - span, end=anchor.start)

        return TimeRange(start=_shift_months(anchor.start, 12), end=_shift_months(anchor.end, 12))


def _aligned_month_span(anchor: TimeRange) -> int | None:
    """Whole months covered by ``anchor``, or None if it is not month-aligned.

    Month-aligned means both bounds fall on the first of a month, which is what a
    question about a month, quarter, or year produces.
    """
    if anchor.start is None or anchor.end is None:
        return None
    if anchor.start.day != 1 or anchor.end.day != 1:
        return None
    months = (anchor.end.year - anchor.start.year) * 12 + (anchor.end.month - anchor.start.month)
    return months if months >= 1 else None


def _shift_months(value: date, months: int) -> date:
    """``value`` moved back by ``months``, clamping to the last valid day.

    31 March has no counterpart on a 30-day month and 29 February none in a common
    year. Clamping is the convention finance systems use; raising would fail a
    legitimate question about a month end.
    """
    total = (value.year * 12 + value.month - 1) - months
    year, month = divmod(total, 12)
    month += 1
    day = min(value.day, monthrange(year, month)[1])
    return date(year, month, day)


class SortSpec(BaseModel):
    """How to order the result."""

    model_config = _FROZEN

    target: SortTarget
    direction: SortDirection = SortDirection.DESC
    dimension: DimensionName | None = None
    """Required when ``target`` is ``dimension``, forbidden otherwise."""

    @model_validator(mode="after")
    def _check_dimension(self) -> SortSpec:
        if self.target is SortTarget.DIMENSION and self.dimension is None:
            raise ValueError("sorting by a dimension requires naming it")
        if self.target is not SortTarget.DIMENSION and self.dimension is not None:
            raise ValueError(f"sort target {self.target.value} must not name a dimension")
        return self


class QueryIntent(BaseModel):
    """A fully-specified analytical request. The LLM's entire output surface."""

    model_config = _FROZEN

    metric: MetricName
    dimensions: tuple[DimensionName, ...] = ()
    time_grain: TimeGrain | None = None
    """Set to return a time series bucketed at this grain; leave unset for a total."""

    time_range: TimeRange | None = None
    """Unset means the whole warehouse, which for a metric with a definition change
    guarantees the range spans it and the caveat fires."""

    filters: tuple[IntentFilter, ...] = ()
    comparison: Comparison | None = None
    """Set to return the metric for the requested range and a derived earlier range,
    labelled, in one result. The delta is arithmetic on those rows, not more SQL."""

    sort: SortSpec | None = None
    limit: int | None = Field(default=None, gt=0, le=10_000)

    @model_validator(mode="after")
    def _check_dimensions(self) -> QueryIntent:
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("dimensions contains duplicates")
        if self.comparison is not None and (
            self.time_range is None or self.time_range.start is None or self.time_range.end is None
        ):
            raise ValueError(
                "a comparison needs a bounded time_range: 'compared to the previous "
                "period' has no meaning without knowing which period"
            )
        return self
