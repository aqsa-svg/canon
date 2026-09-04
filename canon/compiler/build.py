"""Deterministic SQL construction from a validated plan.

How this upholds "the LLM never writes SQL":

* Identifiers -- tables, columns, expressions -- come only from ``metrics.yaml``. The
  model selects enum members; it never supplies a name.
* Filter values, the one thing that can originate with the model, are placed into the
  syntax tree as typed literal nodes (``exp.Literal``, ``exp.Cast`` to DATE). They are
  never concatenated into SQL text, so a value cannot escape its literal and become
  syntax. That is why the emitted SQL is a single runnable string rather than a
  parameterised one: the safety comes from the tree, not from placeholders, and the
  user gets SQL they can paste into DuckDB and re-run.
* After generation, ``_assert_only_known_tables`` re-parses the output and checks every
  table and qualified column against the semantic layer. If the compiler ever emitted
  something unknown, that is a ``CompilerBug`` and it crashes rather than executing.

The module is pure: no database, no clock, no network. The same plan yields
byte-identical SQL on every run, which is what makes the compiler testable without a
warehouse and what makes a diff in generated SQL meaningful.
"""

from __future__ import annotations

import sqlglot
from pydantic import BaseModel, ConfigDict
from sqlglot import exp

from canon.compiler.plan import PERIOD_ALIAS, QueryPlan, ResolvedFilter
from canon.compiler.validate import validate_intent
from canon.schemas.answer import Caveat, ChartKind, ChartSpec
from canon.schemas.errors import CompilerBug
from canon.schemas.intent import QueryIntent
from canon.semantic.cards import MetricCard, SemanticLayer
from canon.semantic.enums import CaveatSource, FilterOp, SortDirection, SortTarget
from canon.semantic.loader import load_semantic_layer

DIALECT = "duckdb"

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class CompiledQuery(BaseModel):
    """SQL plus everything the answer needs to be trustworthy."""

    model_config = _FROZEN

    sql: str
    metric: MetricCard
    caveats: tuple[Caveat, ...]
    chart_spec: ChartSpec | None
    columns: tuple[str, ...]
    """Output column names in order: period, dimensions, then the metric."""


def _parse(sql_fragment: str) -> exp.Expression:
    """Parse a trusted fragment from the semantic layer into an AST node."""
    parsed = sqlglot.maybe_parse(sql_fragment, dialect=DIALECT)
    if parsed is None:
        raise CompilerBug(f"metrics.yaml fragment did not parse: {sql_fragment!r}")
    return parsed


def _literal(value: str | int | float) -> exp.Expression:
    """A filter value as a typed literal node -- never as text spliced into SQL."""
    if isinstance(value, bool):
        return exp.Boolean(this=value)
    if isinstance(value, int | float):
        return exp.Literal.number(value)
    return exp.Literal.string(value)


def _date_literal(value: object) -> exp.Expression:
    """A date as ``CAST('YYYY-MM-DD' AS DATE)``, so comparison semantics are explicit."""
    return exp.cast(exp.Literal.string(str(value)), "DATE", dialect=DIALECT)


def _condition(resolved: ResolvedFilter) -> exp.Expression:
    """Render one filter as an AST node."""
    left = _parse(resolved.expression)
    values = [_literal(value) for value in resolved.values]

    match resolved.op:
        case FilterOp.EQ:
            return exp.EQ(this=left, expression=values[0])
        case FilterOp.NE:
            return exp.NEQ(this=left, expression=values[0])
        case FilterOp.GT:
            return exp.GT(this=left, expression=values[0])
        case FilterOp.GTE:
            return exp.GTE(this=left, expression=values[0])
        case FilterOp.LT:
            return exp.LT(this=left, expression=values[0])
        case FilterOp.LTE:
            return exp.LTE(this=left, expression=values[0])
        case FilterOp.IN:
            return exp.In(this=left, expressions=values)
        case FilterOp.NOT_IN:
            return exp.Not(this=exp.In(this=left, expressions=values))
        case FilterOp.IS_NULL:
            return exp.Is(this=left, expression=exp.Null())
        case FilterOp.IS_NOT_NULL:
            return exp.Not(this=exp.Is(this=left, expression=exp.Null()))
    raise CompilerBug(f"unhandled filter operator {resolved.op!r}")


def _time_conditions(plan: QueryPlan) -> list[exp.Expression]:
    """Half-open range: ``>= start`` and ``< end``.

    Half-open is why consecutive months tile exactly once. A ``BETWEEN`` here would
    double-count midnight on the boundary, and on a timestamp column it would also
    silently drop everything after 00:00:00 on the end date.
    """
    time_range = plan.intent.time_range
    if time_range is None:
        return []

    column = _parse(plan.metric.time_column)
    conditions: list[exp.Expression] = []
    if time_range.start is not None:
        conditions.append(exp.GTE(this=column, expression=_date_literal(time_range.start)))
    if time_range.end is not None:
        conditions.append(exp.LT(this=column.copy(), expression=_date_literal(time_range.end)))
    return conditions


def _projections(plan: QueryPlan) -> tuple[list[exp.Expression], list[exp.Expression]]:
    """Select list and GROUP BY list.

    Grouping is by the full expressions rather than by output alias. DuckDB accepts
    aliases, but grouping by the expression is what the SQL standard guarantees, and
    the emitted query is meant to survive being pasted somewhere else.
    """
    select: list[exp.Expression] = []
    group: list[exp.Expression] = []

    if plan.intent.time_grain is not None:
        bucket = exp.func(
            "DATE_TRUNC",
            exp.Literal.string(plan.intent.time_grain.value),
            _parse(plan.metric.time_column),
            dialect=DIALECT,
        )
        select.append(bucket.as_(PERIOD_ALIAS))
        group.append(bucket.copy())

    for card in plan.dimensions:
        expression = _parse(card.expression)
        select.append(expression.as_(card.name))
        group.append(expression.copy())

    select.append(_parse(plan.metric.expression).as_(plan.metric_alias))
    return select, group


def _order_by(plan: QueryPlan, grouped: bool) -> list[exp.Ordered]:
    """Ordering, always deterministic.

    An explicit sort wins. Otherwise a time series reads oldest-first, and a grouped
    result reads largest-first, which is what a person asking for a breakdown means.
    Every grouped ordering gets the group keys appended as tie-breakers, because
    without them two rows with equal values could swap places between runs and a test
    comparing results would be flaky for no real reason.
    """
    if not grouped:
        return []

    keys: list[exp.Ordered] = []
    tiebreakers = [exp.column(PERIOD_ALIAS)] if plan.intent.time_grain is not None else []
    tiebreakers += [exp.column(card.name) for card in plan.dimensions]

    sort = plan.intent.sort
    if sort is not None:
        descending = sort.direction is SortDirection.DESC
        match sort.target:
            case SortTarget.METRIC:
                primary = exp.column(plan.metric_alias)
            case SortTarget.PERIOD:
                primary = exp.column(PERIOD_ALIAS)
            case SortTarget.DIMENSION:
                primary = exp.column(str(sort.dimension))
        keys.append(exp.Ordered(this=primary, desc=descending))
    elif plan.intent.time_grain is not None:
        keys.append(exp.Ordered(this=exp.column(PERIOD_ALIAS), desc=False))
    else:
        keys.append(exp.Ordered(this=exp.column(plan.metric_alias), desc=True))

    already = {key.this.sql(dialect=DIALECT) for key in keys}
    for tiebreaker in tiebreakers:
        if tiebreaker.sql(dialect=DIALECT) not in already:
            keys.append(exp.Ordered(this=tiebreaker, desc=False))
    return keys


def _assert_only_known_tables(sql: str, plan: QueryPlan, layer: SemanticLayer) -> None:
    """Re-parse the output and prove every name in it came from the semantic layer.

    A belt-and-braces check on the compiler itself. If model-supplied text had ever
    reached SQL as an identifier, or if a refactor introduced a stray table, this fails
    as a ``CompilerBug`` instead of running against the warehouse.
    """
    tree = sqlglot.parse_one(sql, dialect=DIALECT)

    for table in tree.find_all(exp.Table):
        if table.name not in layer.tables:
            raise CompilerBug(f"generated SQL references unknown table {table.name!r}")

    allowed_bare = {plan.metric_alias, PERIOD_ALIAS} | {card.name for card in plan.dimensions}
    for column in tree.find_all(exp.Column):
        qualifier = column.table
        if qualifier:
            if qualifier not in layer.tables:
                raise CompilerBug(
                    f"generated SQL references unknown table {qualifier!r} in a column"
                )
        elif column.name not in allowed_bare:
            raise CompilerBug(
                f"generated SQL has unqualified column {column.name!r}, which is not one "
                "of this query's output aliases"
            )


def _caveats(plan: QueryPlan) -> tuple[Caveat, ...]:
    """Card caveats, then what the compiler itself did to the query.

    The compiler-sourced ones matter as much as the card's: a reader who is not told
    that canceled orders were excluded by default cannot reconcile this number against
    one that included them.
    """
    caveats = [Caveat(source=CaveatSource.METRIC_CARD, text=text) for text in plan.metric.caveats]

    defaults = plan.applied_defaults
    if defaults:
        described = "; ".join(
            f"{item.expression} {item.op.value}"
            + (f" ({', '.join(str(value) for value in item.values)})" if item.values else "")
            for item in defaults
        )
        caveats.append(
            Caveat(
                source=CaveatSource.COMPILER,
                text=(
                    f"Default filters from the {plan.metric.name} definition were applied: "
                    f"{described}. A figure computed without them will not match."
                ),
            )
        )

    change = plan.metric.definition_change
    if change is not None:
        time_range = plan.intent.time_range
        spans = time_range.spans(change.effective_date) if time_range else True
        if spans:
            caveats.append(Caveat(source=CaveatSource.COMPILER, text=change.caveat))

    return tuple(caveats)


def _chart_spec(plan: QueryPlan) -> ChartSpec | None:
    """A chart only when it helps. A single scalar does not need one."""
    metric = plan.metric_alias
    dimension_names = [card.name for card in plan.dimensions]

    if plan.intent.time_grain is not None:
        return ChartSpec(
            kind=ChartKind.LINE,
            x=PERIOD_ALIAS,
            y=metric,
            series=dimension_names[0] if dimension_names else None,
        )
    if dimension_names:
        return ChartSpec(
            kind=ChartKind.BAR,
            x=dimension_names[0],
            y=metric,
            series=dimension_names[1] if len(dimension_names) > 1 else None,
        )
    return None


def compile_plan(plan: QueryPlan, layer: SemanticLayer) -> CompiledQuery:
    """Render a validated plan as SQL. Never raises ``InvalidIntent``."""
    select, group = _projections(plan)

    query = exp.Select().select(*select).from_(exp.to_table(plan.metric.base))

    for edge in plan.joins:
        query = query.join(
            exp.to_table(edge.to),
            on=_parse(edge.join_on),
            join_type="LEFT",
            dialect=DIALECT,
        )

    # Order matters for readability, not for semantics: the metric's own default
    # filters first, then the question's, then time. A reader sees the definition's
    # constraints before the question's.
    for resolved in plan.filters:
        query = query.where(_condition(resolved))
    for condition in _time_conditions(plan):
        query = query.where(condition)

    if group:
        query = query.group_by(*group)

    ordering = _order_by(plan, grouped=bool(group))
    if ordering:
        query = query.order_by(*ordering)

    if plan.intent.limit is not None:
        query = query.limit(plan.intent.limit)

    sql = query.sql(dialect=DIALECT, pretty=True)
    _assert_only_known_tables(sql, plan, layer)

    columns: list[str] = []
    if plan.intent.time_grain is not None:
        columns.append(PERIOD_ALIAS)
    columns += [card.name for card in plan.dimensions]
    columns.append(plan.metric_alias)

    return CompiledQuery(
        sql=sql,
        metric=plan.metric,
        caveats=_caveats(plan),
        chart_spec=_chart_spec(plan),
        columns=tuple(columns),
    )


def compile_intent(intent: QueryIntent, layer: SemanticLayer | None = None) -> CompiledQuery:
    """Validate then compile. The one entry point callers should use.

    Raises ``InvalidIntent`` when the request is illegal -- always before SQL exists.
    """
    resolved_layer = layer if layer is not None else load_semantic_layer()
    plan = validate_intent(intent, resolved_layer)
    return compile_plan(plan, resolved_layer)
