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

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from canon.semantic.enums import (
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
    sort: SortSpec | None = None
    limit: int | None = Field(default=None, gt=0, le=10_000)

    @model_validator(mode="after")
    def _check_dimensions(self) -> QueryIntent:
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("dimensions contains duplicates")
        return self
