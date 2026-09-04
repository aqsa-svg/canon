"""``QueryPlan`` -- a validated intent, resolved against the canonical layer.

The plan exists so that building SQL and deciding what is legal are separate jobs.
``validate.py`` produces a plan or raises ``InvalidIntent``; ``build.py`` only ever
renders a plan. Nothing in the builder needs to check legality, because an illegal
query has no plan to render.

That split is what makes the guarantee in CLAUDE.md concrete: illegal metric/dimension
pairs are rejected before SQL exists, not before it executes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from canon.schemas.intent import QueryIntent, TimeRange
from canon.semantic.cards import (
    DimensionCard,
    JoinEdge,
    MetricCard,
    SemanticFilter,
)
from canon.semantic.enums import FilterOp

_FROZEN = ConfigDict(frozen=True, extra="forbid")

PERIOD_ALIAS = "period"
"""Output column name for the time bucket. Fixed so charts and tests can rely on it."""

COMPARISON_ALIAS = "comparison_period"
"""Output column labelling which period a comparison row belongs to."""

CURRENT_LABEL = "current"
COMPARISON_LABEL = "comparison"


class ResolvedFilter(BaseModel):
    """A filter with its SQL expression already looked up.

    ``expression`` comes from a dimension card or a metric's default filter -- that is,
    from the trusted YAML. ``values`` may originate with the model, and reach SQL only
    as typed literal nodes.
    """

    model_config = _FROZEN

    expression: str
    op: FilterOp
    values: tuple[str | int | float, ...]
    source_dimension: str | None = None
    """The dimension this came from, or None for a metric default filter."""

    from_default: bool = False
    """True when the metric card supplied it, so the answer can say it was applied."""

    @classmethod
    def from_semantic(cls, semantic_filter: SemanticFilter) -> ResolvedFilter:
        return cls(
            expression=semantic_filter.column,
            op=semantic_filter.op,
            values=semantic_filter.values,
            from_default=True,
        )


class QueryPlan(BaseModel):
    """Everything needed to emit SQL, and nothing that still needs deciding."""

    model_config = _FROZEN

    intent: QueryIntent
    metric: MetricCard
    dimensions: tuple[DimensionCard, ...]
    joins: tuple[JoinEdge, ...]
    """In emission order. Every edge is many_to_one, so no join duplicates a base row."""

    filters: tuple[ResolvedFilter, ...]
    """Metric default filters first, then the question's own. Ordering is fixed so the
    generated SQL is byte-identical across runs."""

    needs_time_column: bool
    """Whether the time column is used, for bucketing or for filtering."""

    comparison_range: TimeRange | None = None
    """The derived earlier range, computed by the compiler rather than the model.

    Present iff the intent asked for a comparison. Both ranges go into one query with a
    label column, so the metric expression is evaluated per period unchanged -- which is
    the only approach that stays correct for ratio metrics, where a difference of sums
    is not the difference of the ratios.
    """

    @property
    def metric_alias(self) -> str:
        return self.metric.name

    @property
    def applied_defaults(self) -> tuple[ResolvedFilter, ...]:
        return tuple(item for item in self.filters if item.from_default)
