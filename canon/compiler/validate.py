"""Legality checks. Everything here happens before any SQL exists.

Two independent protections, and they are deliberately not redundant:

1.  **Semantic legality** -- the dimension must appear in the metric card's
    ``valid_dimensions``. This is human judgment about meaning. ``avg_order_value`` by
    ``product_category`` is excluded here even though the SQL would run fine, because
    727 orders contain items from more than one category and attributing a whole
    order's value to one of them needs an allocation rule nobody has written down.

2.  **Structural safety** -- the dimension's table must be reachable from the metric's
    base over ``many_to_one`` edges only. This is mechanical. It is what stops a query
    from counting a split-payment order twice.

A check that passes (1) can still fail (2), and vice versa. Both raise
``InvalidIntent`` carrying the alternatives, so the agent can ask a question that names
the actual fork instead of quietly substituting a dimension that happens to work.
"""

from __future__ import annotations

from canon.compiler.plan import QueryPlan, ResolvedFilter
from canon.schemas.errors import (
    CompilerBug,
    InvalidIntent,
    InvalidIntentDetail,
    InvalidIntentReason,
)
from canon.schemas.intent import QueryIntent
from canon.semantic.cards import DimensionCard, JoinEdge, SemanticLayer


def _fail(
    reason: InvalidIntentReason,
    message: str,
    *,
    metric: str | None = None,
    dimension: str | None = None,
    suggestions: tuple[str, ...] = (),
) -> InvalidIntent:
    return InvalidIntent(
        InvalidIntentDetail(
            reason=reason,
            message=message,
            metric=metric,
            dimension=dimension,
            suggestions=suggestions,
        )
    )


def _metrics_supporting(layer: SemanticLayer, dimension: str) -> tuple[str, ...]:
    """Metrics that legally accept this dimension -- the useful half of a refusal.

    A clarifying question is only correct if it offers a way forward, so a rejected
    pair should be able to say which metrics *can* answer the question as asked.
    """
    return tuple(
        sorted(name for name, card in layer.metrics.items() if dimension in card.valid_dimensions)
    )


def _resolve_dimensions(intent: QueryIntent, layer: SemanticLayer) -> tuple[DimensionCard, ...]:
    metric = layer.metrics[str(intent.metric)]
    resolved: list[DimensionCard] = []

    for dimension in intent.dimensions:
        name = str(dimension)
        if name not in layer.dimensions:
            raise _fail(
                InvalidIntentReason.DIMENSION_NOT_VALID_FOR_METRIC,
                f"{name!r} is not a dimension in the canonical layer",
                metric=metric.name,
                dimension=name,
                suggestions=tuple(sorted(layer.dimensions)),
            )
        if name not in metric.valid_dimensions:
            card = layer.dimensions[name]
            reachable = layer.join_path(metric.base, card.table) is not None
            reason = (
                InvalidIntentReason.DIMENSION_NOT_VALID_FOR_METRIC
                if reachable
                else InvalidIntentReason.FAN_OUT_RISK
            )
            explanation = (
                (
                    f"{metric.name} cannot be sliced by {name}. The join is structurally "
                    "possible but the result would not mean anything: the metric card "
                    "excludes this pair."
                )
                if reachable
                else (
                    f"{metric.name} cannot be sliced by {name}. Reaching {card.table} from "
                    f"{metric.base} requires a one-to-many join, which would count "
                    f"{metric.base} rows more than once and inflate the number."
                )
            )
            caveat = next(
                (text for text in metric.caveats if name in text or card.table in text), None
            )
            raise _fail(
                reason,
                explanation if caveat is None else f"{explanation} Card says: {caveat}",
                metric=metric.name,
                dimension=name,
                suggestions=_metrics_supporting(layer, name),
            )
        resolved.append(layer.dimensions[name])

    return tuple(resolved)


def _resolve_filters(intent: QueryIntent, layer: SemanticLayer) -> tuple[ResolvedFilter, ...]:
    """Metric defaults first, then the question's own filters.

    A filter on a dimension is subject to the same legality rules as grouping by it: a
    filter narrows the population the metric is computed over, so filtering
    ``avg_order_value`` to one product category is exactly as ill-defined as grouping
    by it -- it would pull in the whole value of every order that merely touches the
    category.
    """
    metric = layer.metrics[str(intent.metric)]
    resolved = [ResolvedFilter.from_semantic(item) for item in metric.default_filters]

    for intent_filter in intent.filters:
        name = str(intent_filter.dimension)
        if name not in metric.valid_dimensions:
            raise _fail(
                InvalidIntentReason.DIMENSION_NOT_VALID_FOR_METRIC,
                f"{metric.name} cannot be filtered by {name} for the same reason it "
                "cannot be grouped by it: the filter would change which rows the metric "
                "covers, and the metric card excludes this pair.",
                metric=metric.name,
                dimension=name,
                suggestions=_metrics_supporting(layer, name),
            )
        # Arity was validated when IntentFilter was constructed; re-checked because the
        # compiler must not depend on how its caller built the object.
        arity = intent_filter.op.arity
        count = len(intent_filter.values)
        if (
            (arity == "none" and count != 0)
            or (arity == "one" and count != 1)
            or (arity == "many" and count == 0)
        ):
            raise _fail(
                InvalidIntentReason.FILTER_ARITY,
                f"operator {intent_filter.op.value} cannot take {count} values",
                metric=metric.name,
                dimension=name,
            )
        resolved.append(
            ResolvedFilter(
                expression=layer.dimensions[name].expression,
                op=intent_filter.op,
                values=intent_filter.values,
                source_dimension=name,
            )
        )

    return tuple(resolved)


def _required_tables(
    intent: QueryIntent,
    layer: SemanticLayer,
    dimensions: tuple[DimensionCard, ...],
    needs_time: bool,
) -> tuple[str, ...]:
    """Every table this query touches, base excluded, in deterministic order."""
    metric = layer.metrics[str(intent.metric)]
    required: set[str] = set()

    # Tables the metric's own expression reads -- unique_customers is defined on orders
    # but counts a column that lives on customers, so the expression must be inspected
    # rather than assumed to stay inside the base table.
    required |= {name for name in layer.tables if f"{name}." in metric.expression}

    if needs_time:
        required.add(metric.time_table)

    for semantic_filter in metric.default_filters:
        required.add(semantic_filter.qualified.table)

    for card in dimensions:
        required.add(card.table)

    for intent_filter in intent.filters:
        required.add(layer.dimensions[str(intent_filter.dimension)].table)

    required.discard(metric.base)
    return tuple(sorted(required))


def _resolve_joins(
    intent: QueryIntent, layer: SemanticLayer, required: tuple[str, ...]
) -> tuple[JoinEdge, ...]:
    """Union the many_to_one paths to every required table, prerequisites first.

    Walking each path in order guarantees an edge is only emitted once the table it
    starts from has already been joined, and the sorted table order plus BFS
    tie-breaking make the emitted join list identical on every run.
    """
    metric = layer.metrics[str(intent.metric)]
    edges: list[JoinEdge] = []
    joined = {metric.base}

    for table in required:
        path = layer.join_path(metric.base, table)
        if path is None:
            raise _fail(
                InvalidIntentReason.FAN_OUT_RISK,
                f"{metric.name} needs table {table!r}, which cannot be reached from "
                f"{metric.base!r} without a one-to-many join. Answering would duplicate "
                f"{metric.base} rows.",
                metric=metric.name,
                suggestions=_metrics_by_base(layer, table),
            )
        for edge in path:
            if edge.to not in joined:
                edges.append(edge)
                joined.add(edge.to)

    return tuple(edges)


def _metrics_by_base(layer: SemanticLayer, table: str) -> tuple[str, ...]:
    """Metrics based on the table we could not reach -- what to ask for instead."""
    return tuple(sorted(name for name, card in layer.metrics.items() if card.base == table))


def validate_intent(intent: QueryIntent, layer: SemanticLayer) -> QueryPlan:
    """Resolve an intent into a plan, or raise ``InvalidIntent``.

    Pure: no database, no clock, no network. Same intent and same layer give the same
    plan, which is what lets the compiler be tested without a warehouse.
    """
    metric_name = str(intent.metric)
    if metric_name not in layer.metrics:
        # Unreachable through MetricName, which is generated from this same YAML. Kept
        # as a guard against an enum built from a different revision of the file.
        raise _fail(
            InvalidIntentReason.UNKNOWN_METRIC,
            f"{metric_name!r} is not defined in the canonical layer",
            metric=metric_name,
            suggestions=tuple(sorted(layer.metrics)),
        )

    metric = layer.metrics[metric_name]

    if intent.sort is not None:
        sort = intent.sort
        if sort.target.value == "period" and intent.time_grain is None:
            raise _fail(
                InvalidIntentReason.SORT_TARGET_MISSING,
                "cannot sort by period when no time_grain is requested",
                metric=metric.name,
            )
        if sort.dimension is not None and sort.dimension not in intent.dimensions:
            raise _fail(
                InvalidIntentReason.SORT_TARGET_MISSING,
                f"cannot sort by {sort.dimension} because the query does not group by it",
                metric=metric.name,
                dimension=str(sort.dimension),
                suggestions=tuple(str(name) for name in intent.dimensions),
            )

    comparison_range = None
    if intent.comparison is not None:
        if intent.limit is not None:
            # A LIMIT is applied after grouping, so it truncates whole periods: ordering
            # puts the current period first and the limit cuts the comparison rows off
            # entirely, leaving a result that looks like a comparison and contains one
            # period. Top-N-per-period needs a window function the compiler does not
            # emit, so this is refused rather than silently mangled.
            raise _fail(
                InvalidIntentReason.BAD_LIMIT,
                "a limit cannot be combined with a comparison: the limit would cut off "
                "one of the two periods and the result would look complete while "
                "missing half its rows. Ask for the comparison without a limit, or ask "
                "for the top rows of a single period.",
                metric=metric.name,
            )
        # QueryIntent's validator already guarantees a bounded anchor, so derive cannot
        # fail here; the overlap check is the part that needs saying out loud.
        anchor = intent.time_range
        if anchor is None:  # pragma: no cover - QueryIntent's validator forbids this
            raise CompilerBug("a comparison reached validation without a time_range")
        comparison_range = intent.comparison.derive(anchor)
        if (
            comparison_range.end is not None
            and anchor.start is not None
            and comparison_range.end > anchor.start
        ):
            raise _fail(
                InvalidIntentReason.OVERLAPPING_COMPARISON,
                f"a {intent.comparison.kind.value} comparison of "
                f"{anchor.start}..{anchor.end} would cover "
                f"{comparison_range.start}..{comparison_range.end}, which overlaps the "
                "range it is compared against. A row inside the overlap belongs to both "
                "periods, so the labels would be arbitrary. Narrow the range to less "
                "than a year, or compare against the previous period instead.",
                metric=metric.name,
            )

    dimensions = _resolve_dimensions(intent, layer)
    filters = _resolve_filters(intent, layer)
    needs_time = (
        intent.time_grain is not None
        or intent.time_range is not None
        or comparison_range is not None
    )
    required = _required_tables(intent, layer, dimensions, needs_time)
    joins = _resolve_joins(intent, layer, required)

    return QueryPlan(
        intent=intent,
        metric=metric,
        dimensions=dimensions,
        joins=joins,
        filters=filters,
        needs_time_column=needs_time,
        comparison_range=comparison_range,
    )
