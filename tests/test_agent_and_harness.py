"""Day 2's layers, tested without a single LLM call.

The promise from CLAUDE.md holds: no test in this suite touches the network. The policy,
the verification gate, the scorer and the eval set are all pure, so they can be tested
exhaustively — and the parts that would need the API are tested through their schemas
instead, which is where the guarantees actually live.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from benchmark.harness.dataset import Behavior, QuestionSet, Tier, load_questions
from benchmark.harness.score import (
    Outcome,
    QuestionScore,
    aggregate,
    cells_equal,
    confident_wrong_curve,
    find_matching_gold,
    order_matters,
    rows_match,
)
from canon.agent.policy import (
    PolicyOutcome,
    PolicyRule,
    PolicyThresholds,
    decide,
    log_decision,
)
from canon.agent.prompt import DataWindow, build_system_prompt
from canon.agent.verify import VerificationFailure, verify
from canon.compiler import CompiledQuery, QueryPlan, compile_intent, validate_intent
from canon.schemas.decision import AgentDecision, MetricCandidate
from canon.schemas.intent import QueryIntent, TimeRange
from canon.semantic.cards import SemanticLayer
from canon.semantic.enums import DimensionName, MetricName, TimeGrain

GOLD_VALUES = (
    Path(__file__).resolve().parent.parent / "benchmark" / "questions" / "gold_values.json"
)


@pytest.fixture(scope="module")
def questions():
    return load_questions()


@pytest.fixture(scope="module")
def gold() -> dict[str, list[list[object]]]:
    if not GOLD_VALUES.exists():
        pytest.skip("gold_values.json missing -- run: uv run python scripts/verify_gold.py")
    return json.loads(GOLD_VALUES.read_text(encoding="utf-8"))


def _decision(
    *,
    top: MetricName = MetricName.NET_REVENUE,
    top_score: float = 0.95,
    second: MetricName | None = None,
    second_score: float = 0.3,
    intent: QueryIntent | None = None,
    missing: tuple[str, ...] = (),
) -> AgentDecision:
    candidates = [MetricCandidate(metric=top, score=top_score, why="fits")]
    if second is not None:
        candidates.append(MetricCandidate(metric=second, score=second_score, why="also fits"))
    if intent is None and not missing:
        intent = QueryIntent(metric=top, time_range=TimeRange(start="2018-01-01", end="2019-01-01"))
    return AgentDecision(
        reading="a reading",
        candidates=tuple(candidates),
        intent=intent,
        missing_information=missing,
    )


class TestEvalSetIntegrity:
    """The eval set is held to the same standard as the code it scores."""

    def test_the_set_is_the_size_it_claims(self, questions) -> None:
        assert len(questions.questions) == 80
        assert len(questions.answerable) == 55
        assert len(questions.should_not_answer) == 25

    def test_tier_sizes(self, questions) -> None:
        assert len(questions.by_tier(Tier.TIER_1)) == 20
        assert len(questions.by_tier(Tier.TIER_2)) == 20
        assert len(questions.by_tier(Tier.TIER_3)) == 15
        assert len(questions.by_tier(Tier.AMBIGUOUS)) == 25

    def test_it_agrees_with_the_canonical_layer(self, questions, layer: SemanticLayer) -> None:
        """Catches eval rot: a metric renamed in the YAML while the questions lag."""
        questions.validate_against(layer)

    def test_every_answer_case_has_gold_sql_and_a_metric(self, questions) -> None:
        for item in questions.answerable:
            assert item.gold_sql, item.id
            assert item.expected_metric, item.id

    def test_every_refusal_case_can_be_graded(self, questions) -> None:
        for item in questions.should_not_answer:
            assert item.must_mention, item.id
            assert item.gold_sql is None, item.id

    def test_every_question_explains_itself(self, questions) -> None:
        for item in questions.questions:
            assert len(item.notes) > 40, f"{item.id}: notes too thin to maintain"

    def test_gold_values_exist_for_every_answer_case(self, questions, gold) -> None:
        missing = [item.id for item in questions.answerable if item.id not in gold]
        assert not missing, f"gold values missing for {missing}"

    def test_gold_values_are_never_empty_or_all_null(self, gold) -> None:
        for question_id, rows in gold.items():
            assert rows, question_id
            assert any(cell is not None for row in rows for cell in row), question_id


class TestTrapsDiscriminate:
    """A trap only works if the two readings give different numbers."""

    def test_gross_and_net_revenue_differ_materially(self, questions, gold) -> None:
        by_id = {item.id: item for item in questions.questions}
        net = tuple(tuple(row) for row in gold["t1-01"])  # net revenue 2018
        gross = tuple(tuple(row) for row in gold["t3-01"])  # gross revenue 2018
        assert not rows_match(by_id["t1-01"], gross, net), (
            "if these matched, picking the wrong revenue metric would score as correct "
            "and the whole trap would be decorative"
        )

    def test_the_paired_grain_cases_disagree_about_legality(
        self, questions, layer: SemanticLayer
    ) -> None:
        """t3-08 answers avg_order_value by state; am-13 refuses it by category."""
        by_id = {item.id: item for item in questions.questions}
        assert by_id["t3-08"].expected_behavior is Behavior.ANSWER
        assert by_id["am-13"].expected_behavior is Behavior.CLARIFY

        legal = compile_intent(
            QueryIntent(
                metric=MetricName.AVG_ORDER_VALUE, dimensions=(DimensionName.CUSTOMER_STATE,)
            ),
            layer,
        )
        assert legal.sql
        from canon.schemas.errors import InvalidIntent

        with pytest.raises(InvalidIntent):
            compile_intent(
                QueryIntent(
                    metric=MetricName.AVG_ORDER_VALUE,
                    dimensions=(DimensionName.PRODUCT_CATEGORY,),
                ),
                layer,
            )

    def test_the_paired_fanout_cases_disagree_about_legality(
        self, questions, layer: SemanticLayer
    ) -> None:
        """t2-12 answers payment_type on the payments base; am-14 refuses it on orders."""
        from canon.schemas.errors import InvalidIntent

        assert compile_intent(
            QueryIntent(
                metric=MetricName.TOTAL_PAYMENT_VALUE, dimensions=(DimensionName.PAYMENT_TYPE,)
            ),
            layer,
        ).sql
        with pytest.raises(InvalidIntent):
            compile_intent(
                QueryIntent(
                    metric=MetricName.ORDER_COUNT, dimensions=(DimensionName.PAYMENT_TYPE,)
                ),
                layer,
            )


class TestScoringRules:
    """The comparison rules from the eval doc, fixed before any run."""

    def test_null_is_not_zero(self) -> None:
        assert not cells_equal(None, 0)
        assert not cells_equal(0, None)
        assert cells_equal(None, None)

    def test_money_compares_exactly_as_decimal(self) -> None:
        assert cells_equal(Decimal("7341182.41"), Decimal("7341182.41"))
        assert not cells_equal(Decimal("7341182.41"), Decimal("7341182.42"))

    def test_a_gold_decimal_rehydrates_from_json(self) -> None:
        assert cells_equal({"__decimal__": "145.00"}, Decimal("145.00"))

    def test_floats_compare_within_tolerance(self) -> None:
        assert cells_equal(137.13889914255293, 137.13889914255299)
        assert not cells_equal(137.13, 137.14)

    def test_order_matters_only_when_the_question_implies_it(self, questions) -> None:
        by_id = {item.id: item for item in questions.questions}
        assert order_matters(by_id["t2-03"])  # "Which five ... the most"
        assert not order_matters(by_id["t2-02"])  # a plain breakdown

    def test_row_count_mismatch_fails(self, questions) -> None:
        by_id = {item.id: item for item in questions.questions}
        question = by_id["t2-01"]
        expected = ((date(2018, 1, 1), Decimal("1")), (date(2018, 2, 1), Decimal("2")))
        assert not rows_match(question, ((date(2018, 1, 1), Decimal("1")),), expected)

    def test_matching_another_questions_gold_is_detected(self, questions, gold) -> None:
        """The baseline's only semantic signal: which definition did it actually compute?

        Feeding t3-01's gold value (gross revenue 2018) in as an answer to t1-01 (net
        revenue 2018) must point at t3-01.
        """
        gross_rows = tuple(tuple(row) for row in gold["t3-01"])
        matched = find_matching_gold(gross_rows, "t1-01", gold, questions)
        assert matched is not None
        assert matched in {"t3-01", "t3-05"}  # both are gross revenue 2018


class TestDegenerateStrategiesArePunished:
    """Neither always-ask nor always-answer can score well. This is the whole design."""

    def _scores(
        self, questions: QuestionSet, outcome: Outcome, **overrides: object
    ) -> list[QuestionScore]:
        return [
            QuestionScore(
                question_id=item.id,
                question=item.question,
                tier=item.tier.value,
                expected_behavior=item.expected_behavior,
                outcome=outcome,
                refusal_correct=(
                    not item.expected_behavior.should_answer
                    if outcome is Outcome.ASKED
                    else item.expected_behavior.should_answer
                ),
                **overrides,
            )
            for item in questions.questions
        ]

    def test_hedging_everything_scores_zero_on_execution(self, questions) -> None:
        result = aggregate("hedge", self._scores(questions, Outcome.ASKED), questions)
        assert result.execution_accuracy == 0.0
        assert result.over_refusal_rate == 1.0
        assert result.refusal_accuracy is not None
        assert result.refusal_accuracy < 0.4

    def test_answering_everything_scores_zero_on_refusal(self, questions) -> None:
        scores = self._scores(questions, Outcome.ANSWERED, value_correct=True, confidence=1.0)
        scores = [
            item.model_copy(update={"confident_wrong": item.is_wrong_answer}) for item in scores
        ]
        result = aggregate("always-answer", scores, questions)
        assert result.under_refusal_rate == 1.0
        assert result.confident_wrong_rate is not None
        assert result.confident_wrong_rate > 0.3

    def test_confident_wrong_curve_is_monotonic(self, questions) -> None:
        scores = self._scores(questions, Outcome.ANSWERED, value_correct=True, confidence=0.8)
        scores = [
            item.model_copy(update={"confident_wrong": item.is_wrong_answer}) for item in scores
        ]
        curve = confident_wrong_curve(scores)
        values = [curve[key] or 0.0 for key in ("0.5", "0.6", "0.7", "0.8", "0.9")]
        assert values == sorted(values, reverse=True)


class TestRefusalPolicy:
    """Deterministic over the model's scores, and tunable in one place."""

    def test_a_clear_question_is_answered(self, layer: SemanticLayer) -> None:
        verdict = decide(question="net revenue 2018?", decision=_decision(), layer=layer)
        assert verdict.outcome is PolicyOutcome.ANSWER
        assert verdict.rule is PolicyRule.ANSWERED
        assert verdict.clarifying_question is None

    def test_close_scores_trigger_a_clarification(self, layer: SemanticLayer) -> None:
        verdict = decide(
            question="how much revenue?",
            decision=_decision(top_score=0.55, second=MetricName.GROSS_REVENUE, second_score=0.52),
            layer=layer,
        )
        assert verdict.outcome is PolicyOutcome.CLARIFY
        assert verdict.rule is PolicyRule.CLOSE_CANDIDATES
        assert "gross_revenue" in (verdict.clarifying_question or "")
        assert "net_revenue" in (verdict.clarifying_question or "")

    def test_the_threshold_is_what_decides(self, layer: SemanticLayer) -> None:
        """The same decision flips on the threshold alone, with no new API call.

        This is why the margin lives in one place: re-tuning re-scores a cached run.
        """
        decision = _decision(top_score=0.7, second=MetricName.GROSS_REVENUE, second_score=0.5)
        strict = decide(
            question="q",
            decision=decision,
            layer=layer,
            thresholds=PolicyThresholds(metric_score_margin=0.30),
        )
        lenient = decide(
            question="q",
            decision=decision,
            layer=layer,
            thresholds=PolicyThresholds(metric_score_margin=0.10),
        )
        assert strict.outcome is PolicyOutcome.CLARIFY
        assert lenient.outcome is PolicyOutcome.ANSWER

    def test_a_weak_top_score_clarifies_even_uncontested(self, layer: SemanticLayer) -> None:
        """Nothing in the catalogue fits well, so answering would be a substitution."""
        verdict = decide(
            question="what is our margin?", decision=_decision(top_score=0.25), layer=layer
        )
        assert verdict.outcome is PolicyOutcome.CLARIFY

    def test_missing_information_clarifies(self, layer: SemanticLayer) -> None:
        verdict = decide(
            question="revenue last month?",
            decision=_decision(missing=("which month, since there is no today",)),
            layer=layer,
        )
        assert verdict.outcome is PolicyOutcome.CLARIFY
        assert verdict.rule is PolicyRule.NO_INTENT

    def test_an_undefined_metric_is_not_offered_a_substitute(self, layer: SemanticLayer) -> None:
        """Asked for margin, it must not propose net_revenue as the answer.

        Proposing the nearest available metric is the exact failure this project exists
        to refuse, so the clarification says the metric is not defined instead.
        """
        verdict = decide(
            question="What was our profit margin?",
            decision=_decision(
                top_score=0.2, missing=("profit margin is not defined and there is no cost data",)
            ),
            layer=layer,
        )
        assert "net_revenue" not in (verdict.clarifying_question or "")
        assert "cannot answer" in (verdict.clarifying_question or "").lower()

    def test_a_compiler_rejection_becomes_a_question_with_alternatives(
        self, layer: SemanticLayer
    ) -> None:
        from canon.schemas.errors import InvalidIntent

        try:
            validate_intent(
                QueryIntent(
                    metric=MetricName.AVG_ORDER_VALUE,
                    dimensions=(DimensionName.PRODUCT_CATEGORY,),
                ),
                layer,
            )
        except InvalidIntent as rejected:
            detail = rejected.detail
        else:  # pragma: no cover
            pytest.fail("expected InvalidIntent")

        verdict = decide(
            question="average order value by category?",
            decision=_decision(top=MetricName.AVG_ORDER_VALUE),
            layer=layer,
            invalid=detail,
        )
        assert verdict.rule is PolicyRule.ILLEGAL_INTENT
        assert verdict.invalid_intent_reason == detail.reason.value
        asked = verdict.clarifying_question or ""
        assert "avg_item_price" in asked

    def test_alternatives_are_trimmed_to_three(self, layer: SemanticLayer) -> None:
        """A question offering ten choices scores as a weak refusal under the rubric."""
        from canon.schemas.errors import InvalidIntent

        try:
            validate_intent(
                QueryIntent(
                    metric=MetricName.AVG_ORDER_VALUE,
                    dimensions=(DimensionName.PRODUCT_CATEGORY,),
                ),
                layer,
            )
        except InvalidIntent as rejected:
            detail = rejected.detail
        else:  # pragma: no cover
            pytest.fail("expected InvalidIntent")

        assert len(detail.suggestions) > 3, "the raw suggestion list should be long"
        verdict = decide(
            question="q",
            decision=_decision(top=MetricName.AVG_ORDER_VALUE),
            layer=layer,
            invalid=detail,
        )
        asked = verdict.clarifying_question or ""
        named = [name for name in detail.suggestions if name in asked]
        assert len(named) <= 3

    def test_decisions_are_logged_as_readable_json(
        self, layer: SemanticLayer, tmp_path: Path
    ) -> None:
        log = tmp_path / "decisions.jsonl"
        log_decision(decide(question="q", decision=_decision(), layer=layer), log)
        log_decision(decide(question="q2", decision=_decision(), layer=layer), log)
        entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert len(entries) == 2
        assert entries[0]["rule"] == "answered"
        assert "score_gap" in entries[0]
        assert "thresholds" in entries[0]


class TestVerificationGate:
    """A flag, not a number, when the result does not look like an answer."""

    def _plan_and_compiled(
        self, layer: SemanticLayer, **kwargs: object
    ) -> tuple[QueryPlan, CompiledQuery]:
        intent = QueryIntent(metric=MetricName.NET_REVENUE, **kwargs)
        plan = validate_intent(intent, layer)
        return plan, compile_intent(intent, layer)

    def test_a_good_scalar_result_passes(self, layer: SemanticLayer) -> None:
        plan, compiled = self._plan_and_compiled(layer)
        report = verify(
            plan=plan, compiled=compiled, columns=("net_revenue",), rows=((Decimal("1.00"),),)
        )
        assert report.passed
        assert report.failures == ()

    def test_an_empty_result_is_flagged(self, layer: SemanticLayer) -> None:
        """The am-20 case: correct SQL, no qualifying rows, so no number to report."""
        plan, compiled = self._plan_and_compiled(layer)
        report = verify(plan=plan, compiled=compiled, columns=("net_revenue",), rows=())
        assert not report.passed
        assert VerificationFailure.UNEXPECTED_EMPTY in report.failures

    def test_an_all_null_metric_is_flagged(self, layer: SemanticLayer) -> None:
        plan, compiled = self._plan_and_compiled(layer)
        report = verify(plan=plan, compiled=compiled, columns=("net_revenue",), rows=((None,),))
        assert not report.passed
        assert VerificationFailure.ALL_NULL_METRIC in report.failures

    def test_a_non_numeric_metric_is_flagged(self, layer: SemanticLayer) -> None:
        plan, compiled = self._plan_and_compiled(layer)
        report = verify(
            plan=plan, compiled=compiled, columns=("net_revenue",), rows=(("not a number",),)
        )
        assert not report.passed
        assert VerificationFailure.NON_NUMERIC_METRIC in report.failures

    def test_columns_that_disagree_with_the_plan_are_flagged(self, layer: SemanticLayer) -> None:
        plan, compiled = self._plan_and_compiled(layer)
        report = verify(
            plan=plan, compiled=compiled, columns=("something_else",), rows=((Decimal("1"),),)
        )
        assert not report.passed
        assert VerificationFailure.COLUMN_MISMATCH in report.failures

    def test_an_implausible_row_count_for_the_grain_is_flagged(self, layer: SemanticLayer) -> None:
        """Three months of monthly buckets cannot be 900 rows."""
        plan, compiled = self._plan_and_compiled(
            layer,
            time_grain=TimeGrain.MONTH,
            time_range=TimeRange(start="2018-01-01", end="2018-04-01"),
        )
        rows = tuple((date(2018, 1, 1), Decimal("1")) for _ in range(900))
        report = verify(plan=plan, compiled=compiled, columns=("period", "net_revenue"), rows=rows)
        assert not report.passed
        assert VerificationFailure.IMPLAUSIBLE_ROW_COUNT in report.failures

    def test_a_plausible_monthly_series_passes(self, layer: SemanticLayer) -> None:
        plan, compiled = self._plan_and_compiled(
            layer,
            time_grain=TimeGrain.MONTH,
            time_range=TimeRange(start="2018-01-01", end="2018-04-01"),
        )
        rows = tuple((date(2018, month, 1), Decimal("1000.00")) for month in (1, 2, 3))
        report = verify(plan=plan, compiled=compiled, columns=("period", "net_revenue"), rows=rows)
        assert report.passed


class TestPromptAndSchema:
    """What reaches the API is what makes hallucination impossible."""

    def test_the_prompt_is_byte_stable(self, layer: SemanticLayer) -> None:
        """A varying byte would invalidate the cache on every request."""
        window = DataWindow(first_order=date(2016, 9, 4), last_order=date(2018, 10, 17))
        assert build_system_prompt(layer, window) == build_system_prompt(layer, window)

    def test_the_prompt_carries_the_whole_catalogue(self, layer: SemanticLayer) -> None:
        """No retrieval, by decision: 24 metrics fit, so all 24 go in."""
        window = DataWindow(first_order=date(2016, 9, 4), last_order=date(2018, 10, 17))
        prompt = build_system_prompt(layer, window)
        for name in layer.metrics:
            assert name in prompt
        for term in layer.ambiguous_terms:
            assert term.term in prompt

    def test_the_prompt_states_the_data_window(self, layer: SemanticLayer) -> None:
        """Without it the model anchors relative time on a today that is not in the data."""
        window = DataWindow(first_order=date(2016, 9, 4), last_order=date(2018, 10, 17))
        prompt = build_system_prompt(layer, window)
        assert "2016-09-04" in prompt
        assert "2018-10-17" in prompt

    def test_the_wire_schema_constrains_the_metric_to_the_enum(self) -> None:
        """The guarantee is enforced by the API, not by our validation.

        The schema sent with the request lists the legal metric names, so a metric the
        catalogue does not define cannot come back at all.
        """
        from anthropic.lib._parse._transform import transform_schema
        from pydantic import TypeAdapter

        schema = json.dumps(transform_schema(TypeAdapter(AgentDecision).json_schema()))
        assert '"net_revenue"' in schema
        assert '"gross_revenue"' in schema
        assert "profit_margin" not in schema
        assert "margin" not in schema

    def test_a_decision_must_explain_itself_when_it_declines(self) -> None:
        with pytest.raises(ValueError, match="missing_information"):
            AgentDecision(
                reading="something",
                candidates=(
                    MetricCandidate(metric=MetricName.NET_REVENUE, score=0.5, why="maybe"),
                ),
                intent=None,
            )

    def test_score_gap_is_one_when_only_one_metric_is_plausible(self) -> None:
        contested = _decision(top_score=0.95, second=MetricName.GROSS_REVENUE, second_score=0.30)
        assert contested.score_gap == pytest.approx(0.65)

        # With nothing to be confused with, the gap is 1.0 rather than 0.0 -- an
        # uncontested reading is maximally unambiguous, not minimally.
        single = AgentDecision(
            reading="r",
            candidates=(MetricCandidate(metric=MetricName.NET_REVENUE, score=0.9, why="only fit"),),
            intent=QueryIntent(metric=MetricName.NET_REVENUE),
        )
        assert single.score_gap == 1.0
