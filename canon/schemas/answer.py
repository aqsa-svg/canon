"""The response contract from ``docs/01-evaluation.md``, as types.

The exclusivity invariant is the important part: an ``Answer`` is either *answered* or
*refused*, and there is no third state. A number with a hedge attached ("here is gross
revenue, though you may have meant net") is the behaviour this project argues against,
so the schema makes it impossible to construct rather than merely discouraged.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from canon.semantic.cards import MetricCard
from canon.semantic.enums import CaveatSource

_FROZEN = ConfigDict(frozen=True, extra="forbid")

Scalar = Decimal | bool | int | float | str | datetime | date | None
"""One cell. ``Decimal`` leads the union because money must stay exact -- the eval
compares money as DECIMAL, and a float round-trip would quietly defeat that."""


class ResultColumn(BaseModel):
    """One output column: its name and the SQL type it came back as."""

    model_config = _FROZEN

    name: str
    dtype: str


class AnswerValue(BaseModel):
    """A result set. A scalar is the 1x1 case, not a separate shape."""

    model_config = _FROZEN

    columns: tuple[ResultColumn, ...]
    rows: tuple[tuple[Scalar, ...], ...]
    scalar: Scalar = None
    """Set if and only if the result is exactly one row by one column."""

    @model_validator(mode="after")
    def _check_rectangular(self) -> AnswerValue:
        width = len(self.columns)
        for index, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(f"row {index} has {len(row)} cells but there are {width} columns")

        is_single_cell = len(self.rows) == 1 and width == 1
        if is_single_cell:
            if self.scalar != self.rows[0][0]:
                raise ValueError("scalar must equal the single cell it summarises")
        elif self.scalar is not None:
            raise ValueError("scalar is only meaningful for a 1x1 result")
        return self

    @property
    def is_empty(self) -> bool:
        """No rows. Distinct from a zero: the eval treats NULL and 0 as different."""
        return len(self.rows) == 0


class Caveat(BaseModel):
    """Something true about this answer that the reader needs in order to use it."""

    model_config = _FROZEN

    source: CaveatSource
    text: str


class ChartKind(StrEnum):
    """The only renderings the demo supports. A closed set, so the UI cannot be handed
    a chart type it has no branch for."""

    LINE = "line"
    BAR = "bar"
    TABLE = "table"


class ChartSpec(BaseModel):
    """How to draw the result. Four typed fields, not a Vega blob.

    A JSON blob would be an untyped dict crossing a module boundary, which CLAUDE.md
    forbids, and Streamlit needs nothing more than this.
    """

    model_config = _FROZEN

    kind: ChartKind
    x: str
    y: str
    series: str | None = None


class Answer(BaseModel):
    """What Canon returns for a question. Answered or refused, never both.

    ``confidence`` is required in both branches: a refusal can be confident too, and
    the confident-wrong rate in the eval needs the number to exist unconditionally.
    """

    model_config = _FROZEN

    answer_value: AnswerValue | None = None
    sql: str | None = None
    metric_definition_used: MetricCard | None = None
    caveats: tuple[Caveat, ...] = ()
    chart_spec: ChartSpec | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    clarifying_question: str | None = None

    @model_validator(mode="after")
    def _check_exclusivity(self) -> Answer:
        answered_parts = (self.answer_value, self.sql, self.metric_definition_used)
        is_refusal = self.clarifying_question is not None

        if is_refusal:
            populated = [
                name
                for name, value in zip(
                    ("answer_value", "sql", "metric_definition_used"),
                    answered_parts,
                    strict=True,
                )
                if value is not None
            ]
            if populated:
                raise ValueError(
                    "a refusal must not carry an answer; found "
                    f"{', '.join(populated)} alongside clarifying_question. "
                    "Answering with a hedge is the failure mode this schema forbids."
                )
            if self.chart_spec is not None:
                raise ValueError("a refusal has nothing to chart")
            return self

        missing = [
            name
            for name, value in zip(
                ("answer_value", "sql", "metric_definition_used"), answered_parts, strict=True
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"an answer needs {', '.join(missing)}; if the question cannot be "
                "answered, populate clarifying_question instead"
            )
        return self

    @property
    def is_refusal(self) -> bool:
        return self.clarifying_question is not None
