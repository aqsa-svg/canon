"""Enums generated from metrics.yaml at import time.

This module is the mechanism behind Canon's first core idea. ``MetricName`` and
``DimensionName`` are built from the YAML, so a metric the semantic layer does not
define cannot be constructed, cannot be validated, and cannot appear in a
``QueryIntent``. When these enums are handed to the LLM as a JSON schema, the set of
expressible metrics is exactly the set of defined ones -- a hallucinated metric is not
caught downstream, it is unrepresentable.

Reading YAML at import time is a deliberate trade. The alternative -- a hand-maintained
enum that mirrors the YAML -- can drift, and the drift would be silent and would
reintroduce exactly the failure this design removes. Only names are read here; the
validated cards are built in ``loader.py``.

This is the deepest module in the package: it imports nothing from ``canon``.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import yaml

METRICS_YAML: Final[Path] = Path(__file__).with_name("metrics.yaml")


def _raw_layer() -> dict[str, Any]:
    """Parse metrics.yaml. Any is unavoidable at a file boundary; loader.py types it."""
    with METRICS_YAML.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)
    if not isinstance(parsed, dict):
        raise TypeError(f"{METRICS_YAML} must contain a mapping at the top level")
    return parsed


def _metric_names(layer: dict[str, Any]) -> tuple[str, ...]:
    metrics = layer.get("metrics") or []
    return tuple(str(metric["name"]) for metric in metrics)


def _dimension_names(layer: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(name) for name in (layer.get("dimensions") or {}))


_LAYER: Final[dict[str, Any]] = _raw_layer()

METRIC_NAMES: Final[tuple[str, ...]] = _metric_names(_LAYER)
DIMENSION_NAMES: Final[tuple[str, ...]] = _dimension_names(_LAYER)

if not METRIC_NAMES:
    raise ValueError(f"{METRICS_YAML} defines no metrics; the enums would be empty")
if len(set(METRIC_NAMES)) != len(METRIC_NAMES):
    raise ValueError(f"{METRICS_YAML} defines duplicate metric names")
if not DIMENSION_NAMES:
    raise ValueError(f"{METRICS_YAML} defines no dimensions")

# Functional StrEnum construction: the members exist only at runtime, because their
# identity comes from the YAML. A type checker cannot see MetricName.NET_REVENUE, which
# is the price of the guarantee that the enum and the definitions cannot disagree.
MetricName = StrEnum("MetricName", {name.upper(): name for name in METRIC_NAMES})
"""Every metric the canonical layer defines. Nothing else is expressible."""

DimensionName = StrEnum("DimensionName", {name.upper(): name for name in DIMENSION_NAMES})
"""Every dimension the canonical layer defines. Per-metric legality is separate."""


class TimeGrain(StrEnum):
    """Bucket size for a time series. Applied to the metric's declared time_column."""

    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class FilterOp(StrEnum):
    """Comparison operators, shared by YAML default_filters and intent filters.

    One enum for both so a filter the semantic layer can express is a filter a question
    can express, and the compiler has a single code path to render.
    """

    EQ = "eq"
    NE = "ne"
    IN = "in"
    NOT_IN = "not_in"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"

    @property
    def arity(self) -> str:
        """How many values this operator takes: 'none', 'one', or 'many'."""
        if self in (FilterOp.IS_NULL, FilterOp.IS_NOT_NULL):
            return "none"
        if self in (FilterOp.IN, FilterOp.NOT_IN):
            return "many"
        return "one"


class Cardinality(StrEnum):
    """Direction of a join edge, relative to the table declaring it.

    ``MANY_TO_ONE`` means many rows of the declaring table map to one row of the
    target, so the join cannot duplicate the declaring table's rows. The compiler
    traverses these and only these, which is what makes fan-out structurally
    impossible rather than a mistake to be careful about.
    """

    MANY_TO_ONE = "many_to_one"
    ONE_TO_MANY = "one_to_many"


class CaveatSource(StrEnum):
    """Where a caveat came from. Lets a reader judge how much weight it carries."""

    METRIC_CARD = "from_metric_card"
    COMPILER = "from_compiler"
    DATA = "from_data"


class SortDirection(StrEnum):
    ASC = "asc"
    DESC = "desc"


class SortTarget(StrEnum):
    """What a sort is applied to.

    Named targets rather than a free-text column, so a sort cannot smuggle in an
    identifier the semantic layer never exposed.
    """

    METRIC = "metric"
    PERIOD = "period"
    DIMENSION = "dimension"


class ComparisonKind(StrEnum):
    """A period-over-period comparison.

    The comparison range is *derived* from the requested range, never supplied by the
    model: "compared to last month" is a transformation of the anchor range, and letting
    the model compute the second range would put arithmetic it can get wrong on the
    critical path.
    """

    PREVIOUS_PERIOD = "previous_period"
    """Immediately preceding window of the same length. MoM for a month, QoQ for a
    quarter -- the grain follows the range the question already asked for."""

    PREVIOUS_YEAR = "previous_year"
    """The same window shifted back one calendar year. YoY."""
