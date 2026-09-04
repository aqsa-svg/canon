"""The naive baseline: schema in the prompt, SQL out, executed directly.

This is what the industry default looks like, and it has to be a fair version of it or
the comparison is worthless. Fair means:

* **The real DDL**, generated from the warehouse rather than paraphrased, with row counts
  and the distinct values of the low-cardinality columns a text-to-SQL prompt would
  normally include. Withholding `order_status` values would make the baseline fail on
  cancellations for a reason that has nothing to do with a semantic layer.
* **The same data window** Canon gets, so it is not penalised for not knowing when the
  data ends.
* **Permission to refuse.** The prompt tells it to ask instead of guessing when a
  question is ambiguous or unanswerable. Without this, refusal accuracy on the ambiguous
  half would be zero by construction and the headline delta would be an artefact of the
  prompt rather than of the architecture.
* **The same model, effort, and cache** as Canon.

What it does *not* get is the thing under test: canonical metric definitions, a join
graph with cardinality, validation before execution, and a result gate. It writes SQL as
free text and that SQL is executed as-is.

One consequence is worth stating rather than hiding: the baseline cannot populate
``metric_definition_used``, because it has no notion of a metric. That is not an unfair
handicap — it is the finding. A system with no semantic layer has nothing to cite, so
"which definition produced this number" has no answer, and semantic accuracy is not
directly measurable for it. See ``docs/01-evaluation.md`` §3.2.
"""

from __future__ import annotations

import re
from functools import cached_property
from pathlib import Path

import duckdb
from pydantic import BaseModel, ConfigDict

from canon.agent.llm import LlmClient, LlmError, Usage
from canon.agent.prompt import DataWindow

_FROZEN = ConfigDict(frozen=True, extra="forbid")

CLARIFY_TOKEN = "CLARIFY:"

_LOW_CARDINALITY = (
    ("orders", "order_status"),
    ("order_payments", "payment_type"),
)
"""Columns whose distinct values go in the prompt.

A competent engineer writing a text-to-SQL prompt would include these, so withholding
them would rig the comparison. Deliberately not extended to product_category (73 values)
or customer_state (27), which would start to approximate a semantic layer.
"""

_SKIP_TABLES = ("_canon_load_manifest",)

INSTRUCTIONS = f"""\
You are a SQL analyst answering business questions against a DuckDB warehouse. The full
schema is below.

Reply in exactly one of two ways:

1. A single DuckDB SQL query and nothing else. No prose, no explanation, no markdown
   fences. The query must return the answer to the question.
2. The token {CLARIFY_TOKEN} followed by one short question, when the question is
   genuinely ambiguous or cannot be answered from this schema. Use this when you would
   otherwise have to guess at what the user meant, or when the data needed does not
   exist. Do not use it for questions you can answer.

Guidance:
- Prefer a query that returns exactly the number asked for.
- Half-open date ranges are safest: `>= DATE '2018-01-01' AND < DATE '2019-01-01'`.
- Be careful with joins that can duplicate rows.
"""


class BaselineOutput(BaseModel):
    """What the baseline produced for one question."""

    model_config = _FROZEN

    question: str
    raw_response: str
    sql: str | None
    clarifying_question: str | None
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[object, ...], ...] = ()
    execution_error: str | None = None
    usage: Usage
    error: str | None = None

    @property
    def answered(self) -> bool:
        """It answered if it produced SQL that ran and returned something."""
        return (
            self.clarifying_question is None
            and self.sql is not None
            and self.execution_error is None
            and self.error is None
        )


def _strip_fences(text: str) -> str:
    """Remove markdown fences the model may add despite being told not to.

    Tolerated rather than scored: a fence is a formatting slip, not a SQL error, and
    failing the baseline for it would be the kind of strawmanning that makes a
    comparison meaningless.
    """
    fenced = re.findall(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (fenced[0] if fenced else text).strip()


class NaiveBaseline:
    """Schema-in-prompt text-to-SQL, with no semantic layer and no validation."""

    def __init__(
        self,
        *,
        db_path: Path,
        client: LlmClient | None = None,
    ) -> None:
        self.db_path = db_path
        self.client = client or LlmClient()
        self._connection: duckdb.DuckDBPyConnection | None = None

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        if self._connection is None:
            self._connection = duckdb.connect(str(self.db_path), read_only=True)
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    @cached_property
    def window(self) -> DataWindow:
        first, last = self.connection.execute(
            "SELECT min(order_purchase_timestamp), max(order_purchase_timestamp) FROM orders"
        ).fetchone()  # type: ignore[misc]
        return DataWindow(first_order=first.date(), last_order=last.date())

    @cached_property
    def ddl(self) -> str:
        """CREATE TABLE statements, row counts, and low-cardinality value lists.

        Read from the warehouse itself so it cannot drift from reality.
        """
        tables = [
            name
            for (name,) in self.connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
            if name not in _SKIP_TABLES
        ]

        blocks: list[str] = []
        for table in tables:
            columns = self.connection.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position",
                [table],
            ).fetchall()
            (count,) = self.connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()  # type: ignore[misc]
            body = ",\n".join(f"  {name} {dtype}" for name, dtype in columns)
            blocks.append(f"CREATE TABLE {table} (\n{body}\n);  -- {count:,} rows")

        values: list[str] = []
        for table, column in _LOW_CARDINALITY:
            distinct = self.connection.execute(
                f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL ORDER BY 1'
            ).fetchall()
            listed = ", ".join(repr(str(value)) for (value,) in distinct)
            values.append(f"{table}.{column} values: {listed}")

        return "\n\n".join(blocks) + "\n\n" + "\n".join(values)

    @cached_property
    def system_prompt(self) -> str:
        return (
            f"{INSTRUCTIONS}\n"
            f"Orders span {self.window.first_order} to {self.window.last_order} inclusive; "
            "there is no data outside that range and no notion of today.\n\n"
            f"## SCHEMA\n\n{self.ddl}\n"
        )

    def answer(self, question: str) -> BaselineOutput:
        """One call, then execute whatever came back. No validation in between."""
        try:
            text, usage = self.client.freeform(system=self.system_prompt, question=question)
        except LlmError as error:
            return BaselineOutput(
                question=question,
                raw_response="",
                sql=None,
                clarifying_question=None,
                usage=Usage(),
                error=str(error),
            )

        cleaned = text.strip()
        if CLARIFY_TOKEN.lower() in cleaned.lower():
            index = cleaned.lower().index(CLARIFY_TOKEN.lower())
            asked = cleaned[index + len(CLARIFY_TOKEN) :].strip()
            return BaselineOutput(
                question=question,
                raw_response=text,
                sql=None,
                clarifying_question=asked or "Could you clarify?",
                usage=usage,
            )

        sql = _strip_fences(cleaned)
        try:
            relation = self.connection.sql(sql)
            columns = tuple(relation.columns)
            rows = tuple(tuple(row) for row in relation.fetchall())
        except (duckdb.Error, RuntimeError) as error:
            return BaselineOutput(
                question=question,
                raw_response=text,
                sql=sql,
                clarifying_question=None,
                execution_error=str(error),
                usage=usage,
            )

        return BaselineOutput(
            question=question,
            raw_response=text,
            sql=sql,
            clarifying_question=None,
            columns=columns,
            rows=rows,
            usage=usage,
        )
