"""Pydantic models for the canonical metric layer.

These are the validated, in-memory form of ``metrics.yaml``. Every model is frozen and
forbids extra keys: a typo in the YAML must be a loud parse failure, not a key that is
silently ignored while the metric it belongs to quietly does the wrong thing.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from canon.semantic.enums import Cardinality, FilterOp

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class QualifiedColumn(BaseModel):
    """A ``table.column`` reference, split so the compiler can resolve join paths."""

    model_config = _FROZEN

    table: str
    column: str

    @classmethod
    def parse(cls, reference: str) -> QualifiedColumn:
        table, separator, column = reference.partition(".")
        if not separator or not table or not column:
            raise ValueError(f"{reference!r} must be qualified as table.column")
        return cls(table=table, column=column)

    def __str__(self) -> str:
        return f"{self.table}.{self.column}"


class SemanticFilter(BaseModel):
    """A filter declared in the YAML (a metric's ``default_filters``).

    ``column`` is a qualified column rather than a dimension name because default
    filters routinely constrain columns that are not dimensions at all -- an
    ``is_not_null`` guard on a delivery timestamp, for instance.
    """

    model_config = _FROZEN

    column: str
    op: FilterOp
    values: tuple[str | int | float, ...] = ()

    @model_validator(mode="after")
    def _check_arity(self) -> SemanticFilter:
        arity = self.op.arity
        count = len(self.values)
        if arity == "none" and count != 0:
            raise ValueError(f"{self.op} takes no values, got {count}")
        if arity == "one" and count != 1:
            raise ValueError(f"{self.op} takes exactly one value, got {count}")
        if arity == "many" and count == 0:
            raise ValueError(f"{self.op} needs at least one value")
        return self

    @property
    def qualified(self) -> QualifiedColumn:
        return QualifiedColumn.parse(self.column)


class JoinEdge(BaseModel):
    """One directed edge of the join graph."""

    model_config = _FROZEN

    to: str
    join_on: str
    cardinality: Cardinality


class TableCard(BaseModel):
    """One physical table and the edges leading out of it."""

    model_config = _FROZEN

    name: str
    description: str
    primary_key: tuple[str, ...]
    joins: tuple[JoinEdge, ...] = ()

    @field_validator("primary_key")
    @classmethod
    def _non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("primary_key must name at least one column")
        return value


class DimensionCard(BaseModel):
    """One dimension: how to express it, and which table it needs."""

    model_config = _FROZEN

    name: str
    description: str
    expression: str
    table: str


class DefinitionChange(BaseModel):
    """A metric whose meaning changes at a date, with the caveat that must be told.

    Modelled rather than merely commented because the compiler acts on it: when the
    queried range spans ``effective_date``, the caveat is attached to the answer. A
    total that silently sums two definitions is the failure this exists to prevent.
    """

    model_config = _FROZEN

    effective_date: date
    before: str
    after: str
    caveat: str


class MetricCard(BaseModel):
    """The canonical definition of one metric. The authoritative unit of this project.

    Travels intact into the answer, so a reader sees the definition that produced the
    number without opening another file.
    """

    model_config = _FROZEN

    name: str
    description: str
    expression: str
    base: str
    grain: str
    time_column: str
    valid_dimensions: tuple[str, ...] = ()
    default_filters: tuple[SemanticFilter, ...] = ()
    caveats: tuple[str, ...] = ()
    synonyms: tuple[str, ...] = ()
    owner: str
    definition_change: DefinitionChange | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> MetricCard:
        if len(set(self.valid_dimensions)) != len(self.valid_dimensions):
            raise ValueError(f"metric {self.name}: valid_dimensions contains duplicates")
        QualifiedColumn.parse(self.time_column)
        return self

    @property
    def time_table(self) -> str:
        """The table holding this metric's time column."""
        return QualifiedColumn.parse(self.time_column).table


class AmbiguousTerm(BaseModel):
    """A term with no canonical default, and the metrics it could mean.

    This is refusal expressed as data. When a question turns on one of these and offers
    nothing else to settle it, the correct answer is a clarifying question naming these
    candidates -- a decision grounded in the semantic layer rather than in whatever the
    model felt about the phrasing.
    """

    model_config = _FROZEN

    term: str
    candidates: tuple[str, ...]
    why: str

    @field_validator("candidates")
    @classmethod
    def _needs_a_fork(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) < 2:
            raise ValueError("an ambiguous term needs at least two candidates to be ambiguous")
        return value


class SemanticLayer(BaseModel):
    """The whole canonical layer, validated and cross-checked.

    The model validator is where the moat is defended. It proves, at load time, that
    every metric's declared dimensions are actually reachable from its base table over
    many_to_one edges. A metric card that promises a slice the join graph cannot deliver
    without fan-out fails here -- before a question is ever asked.
    """

    model_config = _FROZEN

    version: int
    tables: dict[str, TableCard]
    dimensions: dict[str, DimensionCard]
    metrics: dict[str, MetricCard]
    ambiguous_terms: tuple[AmbiguousTerm, ...] = Field(default=())

    @model_validator(mode="after")
    def _check_referential_integrity(self) -> SemanticLayer:
        problems: list[str] = []

        for name, table in self.tables.items():
            if name != table.name:
                problems.append(f"table {name!r} is keyed differently from its name {table.name!r}")
            for edge in table.joins:
                if edge.to not in self.tables:
                    problems.append(f"table {name}: join targets unknown table {edge.to!r}")

        for name, dimension in self.dimensions.items():
            if name != dimension.name:
                problems.append(f"dimension {name!r} is keyed differently from {dimension.name!r}")
            if dimension.table not in self.tables:
                problems.append(f"dimension {name}: unknown table {dimension.table!r}")

        for name, metric in self.metrics.items():
            if name != metric.name:
                problems.append(f"metric {name!r} is keyed differently from {metric.name!r}")
            if metric.base not in self.tables:
                problems.append(f"metric {name}: unknown base table {metric.base!r}")
                continue

            for dimension_name in metric.valid_dimensions:
                if dimension_name not in self.dimensions:
                    problems.append(f"metric {name}: unknown dimension {dimension_name!r}")
                    continue
                target = self.dimensions[dimension_name].table
                if self.join_path(metric.base, target) is None:
                    problems.append(
                        f"metric {name}: declares dimension {dimension_name!r} on table "
                        f"{target!r}, which is not reachable from base {metric.base!r} over "
                        "many_to_one edges. Slicing by it would fan out the base rows."
                    )

            for required in self._tables_in(metric.expression) | {metric.time_table}:
                if required not in self.tables:
                    problems.append(f"metric {name}: expression references unknown {required!r}")
                elif self.join_path(metric.base, required) is None:
                    problems.append(
                        f"metric {name}: needs table {required!r}, unreachable from base "
                        f"{metric.base!r} over many_to_one edges"
                    )

            for semantic_filter in metric.default_filters:
                filter_table = semantic_filter.qualified.table
                if filter_table not in self.tables:
                    problems.append(
                        f"metric {name}: default filter on unknown table {filter_table!r}"
                    )
                elif self.join_path(metric.base, filter_table) is None:
                    problems.append(
                        f"metric {name}: default filter needs {filter_table!r}, unreachable "
                        f"from base {metric.base!r}"
                    )

        for term in self.ambiguous_terms:
            for candidate in term.candidates:
                if candidate not in self.metrics:
                    problems.append(
                        f"ambiguous term {term.term!r}: unknown candidate metric {candidate!r}"
                    )

        if problems:
            listed = "\n  - ".join(problems)
            raise ValueError(f"metrics.yaml is not internally consistent:\n  - {listed}")
        return self

    def _tables_in(self, expression: str) -> set[str]:
        """Table names referenced as ``table.`` inside a SQL expression.

        Substring matching over the known table names, which is sound because the set
        of names is closed and declared: an expression cannot reference a table this
        layer has never heard of without the reference simply not being found, and the
        reachability check above then fails on the base table instead.
        """
        return {name for name in self.tables if f"{name}." in expression}

    def join_path(self, base: str, target: str) -> tuple[JoinEdge, ...] | None:
        """Shortest many_to_one path from ``base`` to ``target``, or None.

        Breadth-first, with neighbours visited in sorted order so the path -- and
        therefore the emitted SQL -- is identical on every run. None means the target
        is reachable only through a one_to_many edge, or not at all: either way,
        answering would require duplicating base rows.
        """
        if base == target:
            return ()
        if base not in self.tables or target not in self.tables:
            return None

        queue: list[tuple[str, tuple[JoinEdge, ...]]] = [(base, ())]
        seen = {base}
        while queue:
            current, path = queue.pop(0)
            edges = sorted(
                (
                    edge
                    for edge in self.tables[current].joins
                    if edge.cardinality is Cardinality.MANY_TO_ONE
                ),
                key=lambda edge: edge.to,
            )
            for edge in edges:
                if edge.to in seen:
                    continue
                extended = (*path, edge)
                if edge.to == target:
                    return extended
                seen.add(edge.to)
                queue.append((edge.to, extended))
        return None
