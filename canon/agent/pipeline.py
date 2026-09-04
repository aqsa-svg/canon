"""Question in, verified answer or clarifying question out.

The whole path, in order: one structured LLM call fills a ``QueryIntent``; the compiler
turns it into SQL or rejects it; the policy decides whether to answer at all; DuckDB
executes; the gate checks the result; an ``Answer`` comes back. Every step is typed and
every step can refuse.

The LLM appears exactly once, at the top, and produces no SQL.
"""

from __future__ import annotations

from datetime import date
from functools import cached_property
from pathlib import Path

import duckdb
from pydantic import BaseModel, ConfigDict

from canon.agent.llm import LlmClient, LlmError, Usage
from canon.agent.policy import (
    PolicyOutcome,
    PolicyThresholds,
    RefusalDecision,
    decide,
    log_decision,
)
from canon.agent.prompt import DataWindow, build_system_prompt
from canon.agent.verify import VerificationReport, to_answer_value, verify
from canon.compiler.build import CompiledQuery, compile_plan
from canon.compiler.validate import validate_intent
from canon.schemas.answer import Answer, ChartSpec
from canon.schemas.decision import AgentDecision
from canon.schemas.errors import InvalidIntent, InvalidIntentDetail
from canon.schemas.intent import QueryIntent
from canon.semantic.cards import SemanticLayer
from canon.semantic.loader import load_semantic_layer

_FROZEN = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

DEFAULT_DB = Path(__file__).resolve().parent.parent.parent / "data" / "canon.duckdb"
DEFAULT_DECISION_LOG = (
    Path(__file__).resolve().parent.parent.parent / "benchmark" / "runs" / "decisions.jsonl"
)


class PipelineResult(BaseModel):
    """Everything one question produced, for the harness and for the demo.

    ``answer`` is the deliverable; the rest is the audit trail. The harness needs the
    intent to score semantic accuracy, the decision to score refusal accuracy, and the
    verification report to tell a refusal-because-ambiguous from a
    refusal-because-the-result-looked-wrong.
    """

    model_config = _FROZEN

    question: str
    answer: Answer | None
    decision: RefusalDecision | None
    intent: QueryIntent | None
    sql: str | None
    usage: Usage
    verification: VerificationReport | None = None
    error: str | None = None
    """Set when the pipeline could not reach a decision at all — an API failure or a
    schema-invalid response. Distinct from a refusal, and scored as an error."""

    @property
    def refused(self) -> bool:
        return self.answer is not None and self.answer.is_refusal


class CanonPipeline:
    """The Canon system under test."""

    def __init__(
        self,
        *,
        db_path: Path = DEFAULT_DB,
        layer: SemanticLayer | None = None,
        client: LlmClient | None = None,
        thresholds: PolicyThresholds | None = None,
        decision_log: Path | None = DEFAULT_DECISION_LOG,
    ) -> None:
        self.db_path = db_path
        self.layer = layer or load_semantic_layer()
        self.client = client or LlmClient()
        self.thresholds = thresholds or PolicyThresholds()
        self.decision_log = decision_log
        self._connection: duckdb.DuckDBPyConnection | None = None

    # -- warehouse ---------------------------------------------------------------
    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        if self._connection is None:
            if not self.db_path.exists():
                raise FileNotFoundError(
                    f"{self.db_path} not built -- run: uv run python scripts/load_data.py"
                )
            self._connection = duckdb.connect(str(self.db_path), read_only=True)
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    @cached_property
    def window(self) -> DataWindow:
        """The warehouse's real span, read once from the data.

        Read rather than hardcoded so the prompt cannot drift from the warehouse; cached
        so the prompt stays byte-identical across questions and the prompt cache hits.
        """
        first, last = self.connection.execute(
            "SELECT min(order_purchase_timestamp), max(order_purchase_timestamp) FROM orders"
        ).fetchone()  # type: ignore[misc]
        return DataWindow(first_order=_as_date(first), last_order=_as_date(last))

    @cached_property
    def system_prompt(self) -> str:
        return build_system_prompt(self.layer, self.window)

    # -- the path ----------------------------------------------------------------
    def answer(self, question: str) -> PipelineResult:
        """One question, one structured call, one verified answer or one question back."""
        try:
            decision, usage = self.client.structured(
                system=self.system_prompt,
                question=question,
                output_format=AgentDecision,
            )
        except LlmError as error:
            return PipelineResult(
                question=question,
                answer=None,
                decision=None,
                intent=None,
                sql=None,
                usage=Usage(),
                error=str(error),
            )

        compiled: CompiledQuery | None = None
        plan = None
        invalid: InvalidIntentDetail | None = None

        if decision.intent is not None:
            try:
                plan = validate_intent(decision.intent, self.layer)
                compiled = compile_plan(plan, self.layer)
            except InvalidIntent as rejected:
                invalid = rejected.detail

        verdict = decide(
            question=question,
            decision=decision,
            layer=self.layer,
            invalid=invalid,
            thresholds=self.thresholds,
        )
        if self.decision_log is not None:
            log_decision(verdict, self.decision_log)

        if verdict.outcome is PolicyOutcome.CLARIFY:
            return PipelineResult(
                question=question,
                answer=Answer(
                    confidence=verdict.confidence,
                    clarifying_question=verdict.clarifying_question,
                ),
                decision=verdict,
                intent=decision.intent,
                sql=None,
                usage=usage,
            )

        if compiled is None or plan is None:  # pragma: no cover - policy guarantees this
            raise RuntimeError("the policy chose to answer without a compiled query")

        relation = self.connection.sql(compiled.sql)
        columns = tuple(relation.columns)
        dtypes = tuple(str(dtype) for dtype in relation.types)
        rows = tuple(tuple(row) for row in relation.fetchall())

        report = verify(plan=plan, compiled=compiled, columns=columns, rows=rows)
        if not report.passed:
            # A flag, not a number. The result did not look like an answer to this
            # question, so nothing is reported that someone could paste into a deck.
            return PipelineResult(
                question=question,
                answer=Answer(
                    confidence=verdict.confidence,
                    clarifying_question=(
                        "I ran the query but the result did not pass verification "
                        f"({', '.join(failure.value for failure in report.failures)}): "
                        f"{report.notes[0] if report.notes else 'no detail'}. "
                        "I would rather ask than report a number I cannot vouch for — "
                        "could you confirm the period and breakdown you need?"
                    ),
                ),
                decision=verdict,
                intent=decision.intent,
                sql=compiled.sql,
                usage=usage,
                verification=report,
            )

        return PipelineResult(
            question=question,
            answer=Answer(
                answer_value=to_answer_value(columns, rows, dtypes),
                sql=compiled.sql,
                metric_definition_used=compiled.metric,
                caveats=compiled.caveats,
                chart_spec=_chart_for(compiled, len(rows)),
                confidence=verdict.confidence,
            ),
            decision=verdict,
            intent=decision.intent,
            sql=compiled.sql,
            usage=usage,
            verification=report,
        )


def _chart_for(compiled: CompiledQuery, row_count: int) -> ChartSpec | None:
    """Drop the chart for a single row: a one-bar bar chart is noise, not information."""
    return compiled.chart_spec if row_count > 1 else None


def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    raise TypeError(f"expected a date from the warehouse, got {type(value).__name__}")
