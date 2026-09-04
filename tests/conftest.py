"""Shared fixtures. No LLM is involved anywhere in this suite."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from canon.semantic.cards import SemanticLayer
from canon.semantic.loader import load_semantic_layer

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "canon.duckdb"


@pytest.fixture(scope="session")
def layer() -> SemanticLayer:
    """The canonical layer. Loading it at all proves it is internally consistent."""
    return load_semantic_layer()


@pytest.fixture(scope="session")
def warehouse() -> duckdb.DuckDBPyConnection:
    """Read-only connection to the built warehouse.

    Read-only on purpose: a test must not be able to change the data another test is
    asserting against.
    """
    if not DB_PATH.exists():
        pytest.skip(f"{DB_PATH} not built -- run: uv run python scripts/load_data.py")
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    yield connection
    connection.close()
