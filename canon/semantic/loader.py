"""Load metrics.yaml into a validated ``SemanticLayer``.

Parsed once and cached. The cache is not merely an optimisation: the compiler is
required to be pure, and it can only be pure if the definitions it compiles against do
not change underneath it during a run.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

from canon.semantic.cards import (
    AmbiguousTerm,
    DimensionCard,
    MetricCard,
    SemanticLayer,
    TableCard,
)
from canon.semantic.enums import METRICS_YAML, _raw_layer


def _tables(raw: dict[str, Any]) -> dict[str, TableCard]:
    return {name: TableCard(name=name, **body) for name, body in (raw.get("tables") or {}).items()}


def _dimensions(raw: dict[str, Any]) -> dict[str, DimensionCard]:
    return {
        name: DimensionCard(name=name, **body)
        for name, body in (raw.get("dimensions") or {}).items()
    }


def _metrics(raw: dict[str, Any]) -> dict[str, MetricCard]:
    return {metric["name"]: MetricCard(**metric) for metric in (raw.get("metrics") or [])}


@lru_cache(maxsize=1)
def load_semantic_layer() -> SemanticLayer:
    """Parse, validate and cross-check the canonical layer.

    Raises ``pydantic.ValidationError`` if the YAML is malformed, or ``ValueError`` if
    it is internally inconsistent -- a metric promising a dimension its base table
    cannot reach without fan-out, for instance. Both happen at load time, so a broken
    definition can never reach a question.
    """
    raw = _raw_layer()
    return SemanticLayer(
        version=int(raw["version"]),
        tables=_tables(raw),
        dimensions=_dimensions(raw),
        metrics=_metrics(raw),
        ambiguous_terms=tuple(AmbiguousTerm(**term) for term in (raw.get("ambiguous_terms") or [])),
    )


@lru_cache(maxsize=1)
def semantic_layer_hash() -> str:
    """SHA-256 of metrics.yaml, for stamping benchmark runs with their definitions.

    A result is only interpretable against the definitions that produced it, so every
    recorded run carries this hash (see docs/01-evaluation.md).
    """
    return hashlib.sha256(METRICS_YAML.read_bytes()).hexdigest()
