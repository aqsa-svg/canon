"""The compiler's contract, tested without a database and without an LLM.

These are the properties the rest of the project leans on: illegal pairs are rejected
before SQL exists, the same intent always produces the same SQL, and a filter value can
never become syntax.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
import sqlglot
from sqlglot import exp

from canon.compiler import compile_intent, validate_intent
from canon.schemas.answer import Answer, AnswerValue, ResultColumn
from canon.schemas.decision import AgentDecision
from canon.schemas.errors import InvalidIntent, InvalidIntentReason
from canon.schemas.intent import (
    Comparison,
    IntentFilter,
    QueryIntent,
    SortSpec,
    TimeRange,
)
from canon.semantic.cards import SemanticLayer
from canon.semantic.enums import (
    ComparisonKind,
    DimensionName,
    FilterOp,
    MetricName,
    SortDirection,
    SortTarget,
    TimeGrain,
)


class TestHallucinationIsUnrepresentable:
    """Core idea 1: the LLM cannot name a metric the semantic layer has not defined."""

    def test_unknown_metric_cannot_be_constructed(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            QueryIntent(metric="revenue_i_just_made_up")  # type: ignore[arg-type]

    def test_unknown_dimension_cannot_be_constructed(self) -> None:
        with pytest.raises(ValueError):
            QueryIntent(
                metric=MetricName.NET_REVENUE,
                dimensions=("marketing_channel",),  # type: ignore[arg-type]
            )

    def test_the_enum_matches_the_yaml_exactly(self, layer: SemanticLayer) -> None:
        """No drift is possible, because there is only one source."""
        assert {str(name) for name in MetricName} == set(layer.metrics)
        assert {str(name) for name in DimensionName} == set(layer.dimensions)


class TestIllegalPairsAreRejectedBeforeSql:
    """Core idea, second half: validation happens before any SQL is built."""

    def test_wrong_grain_is_refused_even_though_the_join_would_work(
        self, layer: SemanticLayer
    ) -> None:
        """avg_order_value by product_category: legal SQL, meaningless answer.

        order_items -> products is many_to_one, so nothing fans out and DuckDB would
        happily return numbers. It is refused because 727 orders span more than one
        category, so an order's value cannot be attributed to one of them. This is the
        check the join graph cannot make.
        """
        intent = QueryIntent(
            metric=MetricName.AVG_ORDER_VALUE,
            dimensions=(DimensionName.PRODUCT_CATEGORY,),
        )
        assert layer.join_path("order_items", "products") is not None

        with pytest.raises(InvalidIntent) as raised:
            compile_intent(intent, layer)

        detail = raised.value.detail
        assert detail.reason is InvalidIntentReason.DIMENSION_NOT_VALID_FOR_METRIC
        assert detail.metric == "avg_order_value"
        assert detail.dimension == "product_category"
        assert "avg_item_price" in detail.suggestions

    def test_fan_out_is_refused_with_a_different_reason(self, layer: SemanticLayer) -> None:
        """order_count by payment_type would count split-payment orders twice."""
        with pytest.raises(InvalidIntent) as raised:
            compile_intent(
                QueryIntent(
                    metric=MetricName.ORDER_COUNT,
                    dimensions=(DimensionName.PAYMENT_TYPE,),
                ),
                layer,
            )

        detail = raised.value.detail
        assert detail.reason is InvalidIntentReason.FAN_OUT_RISK
        assert "one-to-many" in detail.message
        assert "total_payment_value" in detail.suggestions

    def test_the_two_refusals_are_distinguishable(self, layer: SemanticLayer) -> None:
        """The agent needs to tell a meaning problem from a structure problem.

        They lead to different clarifying questions: one offers a different metric at
        the same grain, the other says the slice is not answerable at all.
        """
        reasons = set()
        for metric, dimension in (
            (MetricName.AVG_ORDER_VALUE, DimensionName.PRODUCT_CATEGORY),
            (MetricName.ORDER_COUNT, DimensionName.PAYMENT_TYPE),
        ):
            with pytest.raises(InvalidIntent) as raised:
                compile_intent(QueryIntent(metric=metric, dimensions=(dimension,)), layer)
            reasons.add(raised.value.detail.reason)
        assert len(reasons) == 2

    def test_filtering_is_policed_like_grouping(self, layer: SemanticLayer) -> None:
        """A filter narrows the population, so an illegal slice is illegal as a filter."""
        with pytest.raises(InvalidIntent) as raised:
            compile_intent(
                QueryIntent(
                    metric=MetricName.AVG_ORDER_VALUE,
                    filters=(
                        IntentFilter(
                            dimension=DimensionName.PRODUCT_CATEGORY,
                            op=FilterOp.EQ,
                            values=("beleza_saude",),
                        ),
                    ),
                ),
                layer,
            )
        assert raised.value.detail.reason is InvalidIntentReason.DIMENSION_NOT_VALID_FOR_METRIC

    def test_sorting_by_an_unselected_dimension_is_refused(self, layer: SemanticLayer) -> None:
        with pytest.raises(InvalidIntent) as raised:
            compile_intent(
                QueryIntent(
                    metric=MetricName.NET_REVENUE,
                    dimensions=(DimensionName.CUSTOMER_STATE,),
                    sort=SortSpec(
                        target=SortTarget.DIMENSION,
                        dimension=DimensionName.CUSTOMER_CITY,
                        direction=SortDirection.ASC,
                    ),
                ),
                layer,
            )
        assert raised.value.detail.reason is InvalidIntentReason.SORT_TARGET_MISSING

    def test_no_sql_is_produced_when_an_intent_is_invalid(self, layer: SemanticLayer) -> None:
        """The failure is in validation, so there is no SQL object to leak."""
        intent = QueryIntent(
            metric=MetricName.AVG_ORDER_VALUE, dimensions=(DimensionName.PRODUCT_CATEGORY,)
        )
        with pytest.raises(InvalidIntent):
            validate_intent(intent, layer)


class TestDeterminism:
    """The compiler is pure, so identical inputs must give byte-identical SQL."""

    def test_repeated_compilation_is_byte_identical(self, layer: SemanticLayer) -> None:
        intent = QueryIntent(
            metric=MetricName.GROSS_REVENUE,
            dimensions=(DimensionName.CUSTOMER_STATE, DimensionName.PRODUCT_CATEGORY),
            time_grain=TimeGrain.MONTH,
            time_range=TimeRange(start="2018-01-01", end="2018-06-01"),
            filters=(
                IntentFilter(
                    dimension=DimensionName.CUSTOMER_STATE, op=FilterOp.IN, values=("SP", "RJ")
                ),
            ),
        )
        first = compile_intent(intent, layer).sql
        for _ in range(5):
            assert compile_intent(intent, layer).sql == first

    def test_join_order_does_not_depend_on_dimension_order(self, layer: SemanticLayer) -> None:
        """Two orderings of the same dimensions must give the same joins.

        Join emission is driven by a sorted table list, so the FROM clause cannot drift
        with the order the model happened to list dimensions in.
        """
        one = compile_intent(
            QueryIntent(
                metric=MetricName.GROSS_REVENUE,
                dimensions=(DimensionName.CUSTOMER_STATE, DimensionName.SELLER_STATE),
            ),
            layer,
        )
        other = compile_intent(
            QueryIntent(
                metric=MetricName.GROSS_REVENUE,
                dimensions=(DimensionName.SELLER_STATE, DimensionName.CUSTOMER_STATE),
            ),
            layer,
        )
        joins_of = lambda sql: [  # noqa: E731 - a local reader, not an API
            line.strip() for line in sql.splitlines() if line.startswith("LEFT JOIN")
        ]
        assert joins_of(one.sql) == joins_of(other.sql)

    def test_every_metric_compiles_and_parses(self, layer: SemanticLayer) -> None:
        """Every card in the file produces valid DuckDB SQL for a bare total."""
        for name in MetricName:
            compiled = compile_intent(QueryIntent(metric=name), layer)
            parsed = sqlglot.parse_one(compiled.sql, dialect="duckdb")
            assert isinstance(parsed, exp.Select), name

    def test_every_declared_dimension_compiles_for_every_metric_that_allows_it(
        self, layer: SemanticLayer
    ) -> None:
        """The moat's promises are all keepable.

        If a card lists a dimension, compiling that pair must work. Load-time validation
        proves reachability; this proves the SQL actually builds.
        """
        checked = 0
        for metric_name, card in layer.metrics.items():
            for dimension_name in card.valid_dimensions:
                compiled = compile_intent(
                    QueryIntent(
                        metric=MetricName(metric_name),
                        dimensions=(DimensionName(dimension_name),),
                    ),
                    layer,
                )
                assert dimension_name in compiled.columns
                checked += 1
        assert checked >= 80, f"only {checked} metric/dimension pairs exercised"


class TestValuesCannotBecomeSyntax:
    """The invariant that keeps "the LLM never writes SQL" true in practice."""

    def test_a_quote_in_a_value_is_escaped_not_interpolated(self, layer: SemanticLayer) -> None:
        compiled = compile_intent(
            QueryIntent(
                metric=MetricName.NET_REVENUE,
                filters=(
                    IntentFilter(
                        dimension=DimensionName.CUSTOMER_CITY,
                        op=FilterOp.EQ,
                        values=("santa barbara d'oeste",),
                    ),
                ),
            ),
            layer,
        )
        assert "d''oeste" in compiled.sql
        assert isinstance(sqlglot.parse_one(compiled.sql, dialect="duckdb"), exp.Select)

    def test_an_injection_attempt_stays_a_single_string_literal(self, layer: SemanticLayer) -> None:
        """The classic attack, applied to the one field a model can fill freely.

        The value reaches SQL as an ``exp.Literal`` node, so it cannot terminate the
        string it lives in. The proof is structural: the output is one SELECT, and no
        DROP or second statement appears anywhere in its tree.
        """
        attack = "'); DROP TABLE orders; --"
        compiled = compile_intent(
            QueryIntent(
                metric=MetricName.NET_REVENUE,
                filters=(
                    IntentFilter(
                        dimension=DimensionName.CUSTOMER_CITY,
                        op=FilterOp.EQ,
                        values=(attack,),
                    ),
                ),
            ),
            layer,
        )

        statements = sqlglot.parse(compiled.sql, dialect="duckdb")
        assert len(statements) == 1
        tree = statements[0]
        assert isinstance(tree, exp.Select)
        assert not list(tree.find_all(exp.Drop))

        literals = [node.this for node in tree.find_all(exp.Literal) if node.is_string]
        assert attack in literals, "the value must survive intact as one literal"

    def test_generated_sql_only_names_tables_from_the_semantic_layer(
        self, layer: SemanticLayer
    ) -> None:
        for name in MetricName:
            compiled = compile_intent(QueryIntent(metric=name, time_grain=TimeGrain.YEAR), layer)
            tree = sqlglot.parse_one(compiled.sql, dialect="duckdb")
            for table in tree.find_all(exp.Table):
                assert table.name in layer.tables


class TestTimeHandling:
    """Half-open ranges, and the definition-change caveat."""

    def test_range_is_half_open_so_months_tile_exactly_once(self, layer: SemanticLayer) -> None:
        compiled = compile_intent(
            QueryIntent(
                metric=MetricName.NET_REVENUE,
                time_range=TimeRange(start="2018-01-01", end="2018-02-01"),
            ),
            layer,
        )
        assert ">= CAST('2018-01-01' AS DATE)" in compiled.sql
        assert "< CAST('2018-02-01' AS DATE)" in compiled.sql
        assert "BETWEEN" not in compiled.sql.upper()

    def test_inverted_range_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="must be before"):
            TimeRange(start="2018-06-01", end="2018-01-01")

    def test_definition_change_caveat_fires_when_the_range_spans_it(
        self, layer: SemanticLayer
    ) -> None:
        compiled = compile_intent(
            QueryIntent(
                metric=MetricName.REPORTED_REVENUE,
                time_range=TimeRange(start="2017-06-01", end="2018-06-01"),
            ),
            layer,
        )
        assert any("restatement" in caveat.text for caveat in compiled.caveats)

    def test_definition_change_caveat_is_silent_inside_one_regime(
        self, layer: SemanticLayer
    ) -> None:
        """A range wholly after the change has one consistent definition, so no caveat.

        A caveat that fires unconditionally teaches the reader to ignore caveats.
        """
        compiled = compile_intent(
            QueryIntent(
                metric=MetricName.REPORTED_REVENUE,
                time_range=TimeRange(start="2018-02-01", end="2018-06-01"),
            ),
            layer,
        )
        assert not any("restatement" in caveat.text for caveat in compiled.caveats)

    def test_unbounded_range_spans_the_change(self, layer: SemanticLayer) -> None:
        """No time filter means all data, which straddles the restatement."""
        compiled = compile_intent(QueryIntent(metric=MetricName.REPORTED_REVENUE), layer)
        assert any("restatement" in caveat.text for caveat in compiled.caveats)


class TestDefaultFiltersAreDisclosed:
    def test_applied_defaults_produce_a_compiler_caveat(self, layer: SemanticLayer) -> None:
        """A reader who is not told about the status filter cannot reconcile the number."""
        compiled = compile_intent(QueryIntent(metric=MetricName.NET_REVENUE), layer)
        compiler_caveats = [c for c in compiled.caveats if c.source.value == "from_compiler"]
        assert any("canceled" in caveat.text for caveat in compiler_caveats)

    def test_a_metric_without_defaults_says_nothing(self, layer: SemanticLayer) -> None:
        compiled = compile_intent(QueryIntent(metric=MetricName.GROSS_REVENUE), layer)
        assert not [
            caveat
            for caveat in compiled.caveats
            if caveat.source.value == "from_compiler" and "Default filters" in caveat.text
        ]


class TestAnswerExclusivity:
    """The schema must make a hedged answer impossible to construct."""

    def test_an_answer_with_a_clarifying_question_is_rejected(self, layer: SemanticLayer) -> None:
        compiled = compile_intent(QueryIntent(metric=MetricName.NET_REVENUE), layer)
        value = AnswerValue(
            columns=(ResultColumn(name="net_revenue", dtype="DECIMAL"),),
            rows=((1,),),
            scalar=1,
        )
        with pytest.raises(ValueError, match="must not carry an answer"):
            Answer(
                answer_value=value,
                sql=compiled.sql,
                metric_definition_used=compiled.metric,
                confidence=0.9,
                clarifying_question="Did you mean gross or net revenue?",
            )

    def test_a_refusal_needs_no_answer_parts(self) -> None:
        answer = Answer(
            confidence=0.2,
            clarifying_question=(
                "Do you mean gross_revenue (includes freight and canceled orders) or "
                "net_revenue (merchandise only, excluding canceled)?"
            ),
        )
        assert answer.is_refusal

    def test_a_partial_answer_is_rejected(self, layer: SemanticLayer) -> None:
        compiled = compile_intent(QueryIntent(metric=MetricName.NET_REVENUE), layer)
        with pytest.raises(ValueError, match="an answer needs"):
            Answer(sql=compiled.sql, metric_definition_used=compiled.metric, confidence=0.9)


class TestComparison:
    """Period-over-period, with the second range derived rather than supplied."""

    def test_previous_period_is_calendar_aware(self) -> None:
        """March compared to the previous period is February, not "the last 31 days".

        A day-count shift would start the comparison window on 29 January and report a
        number nobody asked for.
        """
        march = TimeRange(start="2018-03-01", end="2018-04-01")
        derived = Comparison(kind=ComparisonKind.PREVIOUS_PERIOD).derive(march)
        assert (derived.start, derived.end) == (date(2018, 2, 1), date(2018, 3, 1))

    def test_quarter_over_quarter_shifts_three_months(self) -> None:
        q1 = TimeRange(start="2018-01-01", end="2018-04-01")
        derived = Comparison(kind=ComparisonKind.PREVIOUS_PERIOD).derive(q1)
        assert (derived.start, derived.end) == (date(2017, 10, 1), date(2018, 1, 1))

    def test_an_unaligned_window_shifts_by_its_own_length(self) -> None:
        week = TimeRange(start="2018-03-08", end="2018-03-15")
        derived = Comparison(kind=ComparisonKind.PREVIOUS_PERIOD).derive(week)
        assert (derived.start, derived.end) == (date(2018, 3, 1), date(2018, 3, 8))

    def test_leap_day_clamps_rather_than_raising(self) -> None:
        leap = TimeRange(start="2016-02-29", end="2016-03-01")
        derived = Comparison(kind=ComparisonKind.PREVIOUS_YEAR).derive(leap)
        assert derived.start == date(2015, 2, 28)

    def test_a_comparison_without_an_anchor_cannot_be_constructed(self) -> None:
        """ "Compared to the previous period" is meaningless with no period."""
        with pytest.raises(ValueError, match="bounded time_range"):
            QueryIntent(
                metric=MetricName.NET_REVENUE,
                comparison=Comparison(kind=ComparisonKind.PREVIOUS_PERIOD),
            )

    def test_both_periods_land_in_one_query_with_a_label(self, layer: SemanticLayer) -> None:
        compiled = compile_intent(
            QueryIntent(
                metric=MetricName.NET_REVENUE,
                time_range=TimeRange(start="2018-03-01", end="2018-04-01"),
                comparison=Comparison(kind=ComparisonKind.PREVIOUS_PERIOD),
            ),
            layer,
        )
        assert compiled.columns == ("comparison_period", "net_revenue")
        # One query, not two: two queries would be two chances to filter differently.
        assert len(sqlglot.parse(compiled.sql, dialect="duckdb")) == 1
        assert compiled.sql.count("SELECT") == 1

    def test_the_metric_is_recomputed_per_period_not_differenced(
        self, layer: SemanticLayer
    ) -> None:
        """A ratio metric must be evaluated inside each period.

        Differencing two ratios' components would give the ratio of differences, which
        is a different and wrong number. Grouping by the period label sidesteps it
        entirely: the card's expression runs unchanged, once per period.
        """
        compiled = compile_intent(
            QueryIntent(
                metric=MetricName.CANCELLATION_RATE,
                time_range=TimeRange(start="2018-01-01", end="2018-04-01"),
                comparison=Comparison(kind=ComparisonKind.PREVIOUS_YEAR),
            ),
            layer,
        )
        assert (
            compiled.metric.expression in compiled.sql.replace("\n  ", " ").replace("\n", " ")
            or "NULLIF" in compiled.sql
        )

    def test_overlapping_comparison_is_refused(self, layer: SemanticLayer) -> None:
        """A range longer than a year cannot be compared year-over-year.

        The windows would overlap, and a row in the overlap belongs to both periods, so
        the label would be arbitrary.
        """
        with pytest.raises(InvalidIntent) as raised:
            compile_intent(
                QueryIntent(
                    metric=MetricName.NET_REVENUE,
                    time_range=TimeRange(start="2017-01-01", end="2018-06-01"),
                    comparison=Comparison(kind=ComparisonKind.PREVIOUS_YEAR),
                ),
                layer,
            )
        assert raised.value.detail.reason is InvalidIntentReason.OVERLAPPING_COMPARISON

    def test_a_limit_with_a_comparison_is_refused(self, layer: SemanticLayer) -> None:
        """A limit truncates whole periods, leaving a half-result that looks whole."""
        with pytest.raises(InvalidIntent) as raised:
            compile_intent(
                QueryIntent(
                    metric=MetricName.ORDER_COUNT,
                    dimensions=(DimensionName.CUSTOMER_STATE,),
                    time_range=TimeRange(start="2018-01-01", end="2018-04-01"),
                    comparison=Comparison(kind=ComparisonKind.PREVIOUS_YEAR),
                    limit=5,
                ),
                layer,
            )
        assert raised.value.detail.reason is InvalidIntentReason.BAD_LIMIT

    def test_a_comparison_discloses_both_windows(self, layer: SemanticLayer) -> None:
        compiled = compile_intent(
            QueryIntent(
                metric=MetricName.NET_REVENUE,
                time_range=TimeRange(start="2018-03-01", end="2018-04-01"),
                comparison=Comparison(kind=ComparisonKind.PREVIOUS_PERIOD),
            ),
            layer,
        )
        text = " ".join(caveat.text for caveat in compiled.caveats)
        assert "2018-02-01" in text and "2018-03-01" in text


class TestSortSpecTolerance:
    """Strictness that discards a well-formed intent is pedantry, not safety.

    Asked "which customer states had the worst late delivery rate?", the model returned
    target=metric with dimension=customer_state -- sort by the metric, with the breakdown
    named redundantly. The meaning is unambiguous. The stricter validator rejected it and
    the question was lost to a hard error (t2-14 in sweep-groq-r1).
    """

    def test_a_redundant_dimension_is_dropped_not_fatal(self) -> None:
        spec = SortSpec.model_validate(
            {"target": "metric", "dimension": "customer_state", "direction": "desc"}
        )
        assert spec.target is SortTarget.METRIC
        assert spec.dimension is None, "the redundant value is dropped, not honoured"

    def test_a_dimension_sort_still_requires_its_dimension(self) -> None:
        """The genuine error is still an error."""
        with pytest.raises(ValueError, match="requires naming it"):
            SortSpec.model_validate({"target": "dimension", "direction": "asc"})

    def test_a_dimension_sort_keeps_its_dimension(self) -> None:
        spec = SortSpec.model_validate(
            {"target": "dimension", "dimension": "customer_state", "direction": "asc"}
        )
        assert spec.dimension is not None

    def test_the_wire_schema_is_unchanged_by_the_relaxation(self) -> None:
        """The rule lives in a model_validator, so the response cache stays valid.

        If this ever appears in the JSON schema, relaxing it would invalidate every
        cached response and cost a full re-run.
        """
        schema = json.dumps(AgentDecision.model_json_schema(), sort_keys=True)
        assert "must not name a dimension" not in schema
