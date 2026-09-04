"""Typed errors. Structured enough for the agent layer to turn into a question.

``InvalidIntent`` is not a generic failure. It carries which metric, which dimension,
and what the alternatives are, because its whole purpose is to be converted into a
clarifying question that names the actual fork. An error that only says "invalid" would
force the agent to guess, which is the behaviour this project exists to remove.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class InvalidIntentReason(StrEnum):
    """Why an intent cannot be compiled. Each maps to a different question to ask."""

    UNKNOWN_METRIC = "unknown_metric"
    """The metric is not in the canonical layer. Unreachable via the enums, kept as a
    guard against an enum built from a different version of the YAML than the loader."""

    DIMENSION_NOT_VALID_FOR_METRIC = "dimension_not_valid_for_metric"
    """The pair is semantically illegal: the metric card excludes this dimension because
    the answer would be well-formed SQL and still meaningless. The wrong-grain case."""

    FAN_OUT_RISK = "fan_out_risk"
    """The dimension's table is reachable only across a one_to_many edge, so answering
    would duplicate base rows and inflate the number."""

    FILTER_ARITY = "filter_arity"
    """The operator was given the wrong number of values."""

    EMPTY_TIME_RANGE = "empty_time_range"
    """The range cannot contain any row."""

    BAD_LIMIT = "bad_limit"
    """A limit that is not a positive integer."""

    SORT_TARGET_MISSING = "sort_target_missing"
    """Sorting by something the query does not select."""

    OVERLAPPING_COMPARISON = "overlapping_comparison"
    """The comparison range overlaps the range it is compared against, so a row would
    belong to both periods and the label would be a coin flip."""


class InvalidIntentDetail(BaseModel):
    """The serialisable payload of an ``InvalidIntent``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: InvalidIntentReason
    message: str
    metric: str | None = None
    dimension: str | None = None
    suggestions: tuple[str, ...] = ()
    """Legal alternatives, for a clarifying question that offers a way forward."""


class InvalidIntent(Exception):
    """Raised when an intent is illegal. Always before any SQL is built or executed.

    The point of raising this early is that "compiles but returns nonsense" is the
    outcome the whole design exists to prevent. By the time a query reaches DuckDB it
    has already been proven legal.
    """

    def __init__(self, detail: InvalidIntentDetail) -> None:
        self.detail = detail
        super().__init__(detail.message)


class CompilerBug(RuntimeError):
    """A compiler invariant was violated -- not the user's fault and not recoverable.

    Distinct from ``InvalidIntent``: an invalid intent is a legitimate outcome that
    becomes a clarifying question, whereas this means the compiler emitted something it
    should have been unable to emit. It must crash loudly rather than degrade.
    """
