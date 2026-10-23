"""Day 2's layers, tested without a single LLM call.

The promise from CLAUDE.md holds: no test in this suite touches the network. The policy,
the verification gate, the scorer and the eval set are all pure, so they can be tested
exhaustively — and the parts that would need the API are tested through their schemas
instead, which is where the guarantees actually live.
"""

from __future__ import annotations

import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.harness.dataset import Behavior, QuestionSet, Tier, load_questions
from benchmark.harness.run import (
    Provenance,
    _load_checkpoint,
    _prior_provenance,
    provider_conflict,
)
from benchmark.harness.score import (
    Attribution,
    LiveLatency,
    Outcome,
    QuestionScore,
    actual_behavior,
    aggregate,
    attribute_wrong_answer,
    behavior_correct,
    behavior_matrix,
    cells_equal,
    confident_wrong_curve,
    gold_collision_report,
    latency_comparison,
    order_matters,
    paired_discrimination,
    rows_match,
    shape_matches,
)
from canon.agent.policy import (
    PolicyOutcome,
    PolicyRule,
    PolicyThresholds,
    decide,
    log_decision,
)
from canon.agent.prompt import DataWindow, build_system_prompt
from canon.agent.providers import is_transport_error
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

    def test_a_shared_gold_value_cannot_be_attributed(self, questions, gold) -> None:
        """Gross revenue 2018 is the gold answer for two questions, so a match is mute.

        The unguarded version of this instrument named whichever id sorted first, which
        asserted a definition it could not actually distinguish. Now it says so.
        """
        gross_rows = tuple(tuple(row) for row in gold["t3-01"])
        result = attribute_wrong_answer(gross_rows, "t1-01", gold, questions)
        assert result.status is Attribution.UNATTRIBUTABLE
        assert result.matched_question_id is None
        assert "t3-01" in (result.reason or "") and "t3-05" in (result.reason or "")

    def test_a_small_value_match_is_not_attributed(self, questions, gold) -> None:
        """Counts in the hundreds collide by chance, so a match proves nothing."""
        small = tuple(tuple(row) for row in gold["t1-05"])  # 334 canceled orders
        result = attribute_wrong_answer(small, "t1-03", gold, questions)
        assert result.status is Attribution.UNATTRIBUTABLE
        assert "magnitude" in (result.reason or "")

    def test_a_unique_large_value_is_attributed(self, questions, gold) -> None:
        """A distinctive, unshared gold value can identify the definition computed.

        t1-02 is gross revenue for 2017 -- seven figures, and the gold answer for no
        other question -- so answering it in place of something else is traceable.
        """
        target = "t1-02"
        report = gold_collision_report(gold)
        shared = {qid for group in report["shared_value_groups"] for qid in group}
        assert target not in shared, f"{target} is no longer a unique value; pick another"

        result = attribute_wrong_answer(
            tuple(tuple(row) for row in gold[target]), "t1-01", gold, questions
        )
        assert result.status is Attribution.ATTRIBUTABLE
        assert result.matched_question_id == target
        assert result.matched_count == 1

    def test_the_instrument_reports_its_own_coverage(self, gold) -> None:
        """An instrument with no stated error bar should not be quoted."""
        report = gold_collision_report(gold)
        assert report["total_gold_values"] == 55
        assert report["attribution_coverage"] is not None
        # Most of this eval set cannot support an attribution, which is worth knowing
        # before any conclusion is drawn from one.
        assert 0.0 < report["attribution_coverage"] < 1.0
        assert report["questions_supporting_attribution"] < report["total_gold_values"]


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


class TestCacheKeyingMakesRunsIndependent:
    """Fix 2: runs 2 and 3 must be fresh samples, not replays of run 1."""

    def test_run_index_changes_the_cache_key(self) -> None:
        from canon.agent.llm import CallContext, _cache_key

        base = {
            "provider": "groq",
            "model_id": "openai/gpt-oss-120b",
            "config_hash": "abc123",
            "system": "sys",
            "question": "q",
            "schema_name": "AgentDecision",
            "schema": "{}",
        }
        first = _cache_key(
            **base, context=CallContext(system="canon", question_id="t1-01", run_index=0)
        )
        second = _cache_key(
            **base, context=CallContext(system="canon", question_id="t1-01", run_index=1)
        )
        assert first != second, (
            "identical keys across runs would make every reported variance exactly 0.0, "
            "which looks like a measurement and is an artefact of the cache"
        )

    def test_the_two_systems_do_not_share_cache_entries(self) -> None:
        from canon.agent.llm import CallContext, _cache_key

        base = {
            "provider": "groq",
            "model_id": "openai/gpt-oss-120b",
            "config_hash": "abc123",
            "system": "sys",
            "question": "q",
            "schema_name": "AgentDecision",
            "schema": "{}",
        }
        canon = _cache_key(
            **base, context=CallContext(system="canon", question_id="t1-01", run_index=0)
        )
        baseline = _cache_key(
            **base, context=CallContext(system="baseline", question_id="t1-01", run_index=0)
        )
        assert canon != baseline

    def test_two_providers_never_share_a_cached_answer(self) -> None:
        """Otherwise a cross-provider robustness check could compare a provider to itself."""
        from canon.agent.llm import CallContext, _cache_key

        common = {
            "system": "sys",
            "question": "q",
            "schema_name": "AgentDecision",
            "schema": "{}",
            "context": CallContext(system="canon", question_id="t1-01", run_index=0),
        }
        groq = _cache_key(
            provider="groq", model_id="openai/gpt-oss-120b", config_hash="a", **common
        )
        gemini = _cache_key(
            provider="gemini", model_id="gemini-2.5-flash", config_hash="b", **common
        )
        assert groq != gemini

    def test_no_context_keeps_content_only_keying(self) -> None:
        """The demo wants cache hits, so an absent context must not vary the key."""
        from canon.agent.llm import _cache_key

        base = {
            "provider": "groq",
            "model_id": "openai/gpt-oss-120b",
            "config_hash": "abc123",
            "system": "sys",
            "question": "q",
            "schema_name": "AgentDecision",
            "schema": "{}",
        }
        assert _cache_key(**base, context=None) == _cache_key(**base, context=None)


class TestSilentWrongIsTheHeadline:
    """Fix 3: compare observable behaviour, not an imputed confidence."""

    def _score(self, **kwargs: object) -> QuestionScore:
        defaults: dict[str, object] = {
            "question_id": "x-01",
            "question": "q",
            "tier": "1",
            "expected_behavior": Behavior.ANSWER,
            "outcome": Outcome.ANSWERED,
            "value_correct": False,
        }
        defaults.update(kwargs)
        score = QuestionScore(**defaults)  # type: ignore[arg-type]
        return score.model_copy(update={"silent_wrong": score.is_silent_wrong})

    def test_a_wrong_answer_with_no_warning_is_silent(self) -> None:
        assert self._score().silent_wrong

    def test_a_gate_flagged_wrong_answer_is_not_silent(self) -> None:
        """The gate said something, so nobody was misled."""
        assert not self._score(gate_flagged=True).silent_wrong

    def test_a_clarified_case_is_not_silent(self) -> None:
        score = self._score(
            outcome=Outcome.ASKED, clarifying_question="which revenue did you mean?"
        )
        assert not score.silent_wrong

    def test_a_correct_answer_is_not_silent(self) -> None:
        assert not self._score(value_correct=True).silent_wrong

    def test_silent_wrong_needs_no_imputed_confidence(self) -> None:
        """The same verdict at any confidence -- which is the point of replacing
        confident-wrong as the headline."""
        assert self._score(confidence=0.0).silent_wrong
        assert self._score(confidence=1.0).silent_wrong


class TestBehaviorMatrixAndPairs:
    """Fix: a 3x3 matrix, because the off-diagonals are different failures."""

    def _asked(self, expected: Behavior, text: str) -> QuestionScore:
        return QuestionScore(
            question_id="x",
            question="q",
            tier="ambiguous",
            expected_behavior=expected,
            outcome=Outcome.ASKED,
            clarifying_question=text,
        )

    def test_a_clarification_and_a_refusal_are_told_apart(self) -> None:
        """Both ask, but one names a resolvable fork and the other says it cannot."""
        clarified = self._asked(Behavior.CLARIFY, "Do you mean gross_revenue or net_revenue?")
        refused = self._asked(Behavior.REFUSE, "I cannot answer that: margin is not defined.")
        assert actual_behavior(clarified) == "clarify"
        assert actual_behavior(refused) == "refuse"

    def test_refusing_when_it_should_clarify_is_off_diagonal(self) -> None:
        """A usability failure -- safe, and still not correct."""
        score = self._asked(Behavior.CLARIFY, "I cannot answer that, it is not defined.")
        assert not behavior_correct(score)
        matrix = behavior_matrix([score])
        assert matrix["clarify"]["refuse"] == 1
        assert matrix["clarify"]["clarify"] == 0

    def test_answering_when_it_should_clarify_is_the_trust_failure(self) -> None:
        score = QuestionScore(
            question_id="x",
            question="q",
            tier="ambiguous",
            expected_behavior=Behavior.CLARIFY,
            outcome=Outcome.ANSWERED,
            value_correct=None,
        )
        matrix = behavior_matrix([score])
        assert matrix["clarify"]["answer"] == 1

    def test_the_matrix_is_three_by_three_plus_errors(self) -> None:
        matrix = behavior_matrix([])
        assert set(matrix) == {"answer", "clarify", "refuse"}
        assert set(matrix["answer"]) == {"answer", "clarify", "refuse", "error"}

    def test_a_reflex_strategy_cannot_win_the_paired_items(self, questions) -> None:
        """The paired score is the number immune to always-answer and always-ask.

        Each pair needs opposite responses, so a system with one reflex gets exactly one
        half of each pair and scores zero on both-halves-correct.
        """
        always_answer = [
            QuestionScore(
                question_id=item.id,
                question=item.question,
                tier=item.tier.value,
                expected_behavior=item.expected_behavior,
                outcome=Outcome.ANSWERED,
                value_correct=True if item.expected_behavior.should_answer else None,
                metric_correct=True if item.expected_metric else None,
                # A system with no caveat mechanism -- what the scorer computes for the
                # baseline on any question that requires one.
                caveats_surfaced=False if item.requires_caveat else None,
            )
            for item in questions.questions
        ]
        always_ask = [
            QuestionScore(
                question_id=item.id,
                question=item.question,
                tier=item.tier.value,
                expected_behavior=item.expected_behavior,
                outcome=Outcome.ASKED,
                clarifying_question="Do you mean gross_revenue or net_revenue?",
            )
            for item in questions.questions
        ]
        answered = paired_discrimination(always_answer)
        asked = paired_discrimination(always_ask)
        assert answered["pairs_scored"] >= 2
        assert answered["both_correct"] == 0, "always-answer must fail every pair"
        assert asked["both_correct"] == 0, "always-ask must fail every pair"


class TestThresholdProvenanceIsHonest:
    """Fix 1: the provenance sentence has to be checkable, not asserted."""

    def test_it_says_unvalidated_when_no_tuning_artifact_exists(self) -> None:
        from benchmark.harness.run import TUNING_RESULT, threshold_provenance

        claim = threshold_provenance()
        if TUNING_RESULT.exists():
            assert "tuned on benchmark/dev/questions.yaml" in claim
            assert "eval_set_used=False" in claim
        else:
            assert claim.startswith("UNVALIDATED")

    def test_the_dev_set_is_disjoint_from_the_eval_set(self, questions) -> None:
        """The whole point of the dev set. Overlap would be leakage."""
        import yaml

        dev_path = Path(__file__).resolve().parent.parent / "benchmark" / "dev" / "questions.yaml"
        dev = yaml.safe_load(dev_path.read_text(encoding="utf-8"))
        dev_texts = {item["question"].strip().lower() for item in dev}
        eval_texts = {item.question.strip().lower() for item in questions.questions}
        assert not (dev_texts & eval_texts), "a dev question also appears in the eval set"
        dev_ids = {item["id"] for item in dev}
        assert not (dev_ids & {item.id for item in questions.questions})
        assert 12 <= len(dev) <= 15

    def test_the_dev_set_covers_every_behavior_class(self) -> None:
        import yaml

        dev_path = Path(__file__).resolve().parent.parent / "benchmark" / "dev" / "questions.yaml"
        dev = yaml.safe_load(dev_path.read_text(encoding="utf-8"))
        behaviors = {item["expected_behavior"] for item in dev}
        assert behaviors == {"answer", "clarify", "refuse"}


class TestGoldAgreesWithTheCards:
    """Gold SQL must compute what the metric card says, not merely run.

    Gold was already verified two ways -- the SQL executes, and its value is committed --
    and neither catches a *conceptual* error: gold that quietly disagrees with the card it
    was written from. Gold omitting a card's default filter is wrong in exactly the way a
    schema-only system is wrong, and it would mark correct answers as failures. This check
    was added after mining the completed baseline arm made the gap obvious.
    """

    def test_every_gold_query_applies_its_cards_default_filters(
        self, questions, layer: SemanticLayer
    ) -> None:
        problems: list[str] = []
        for item in questions.answerable:
            card = layer.metrics[item.expected_metric]
            sql = " ".join((item.gold_sql or "").split()).lower()
            for semantic_filter in card.default_filters:
                column = semantic_filter.qualified.column.lower()
                if column not in sql:
                    problems.append(
                        f"{item.id}: {card.name} always filters on {column}, "
                        "absent from the gold SQL"
                    )
                    continue
                for value in semantic_filter.values:
                    if str(value).lower() not in sql:
                        problems.append(
                            f"{item.id}: {card.name} filters {column} on {value!r}, "
                            "absent from the gold SQL"
                        )
        assert not problems, "gold disagrees with the cards:\n  " + "\n  ".join(problems)

    def test_every_gold_query_computes_over_its_cards_columns(
        self, questions, layer: SemanticLayer
    ) -> None:
        """A gold query for net_revenue that never mentions price computes something else."""
        pattern = re.compile(
            r"order_items\.(\w+)|orders\.(\w+)|order_payments\.(\w+)"
            r"|order_reviews\.(\w+)|customers\.(\w+)"
        )
        problems: list[str] = []
        for item in questions.answerable:
            card = layer.metrics[item.expected_metric]
            sql = " ".join((item.gold_sql or "").split()).lower()
            for match in pattern.findall(card.expression):
                column = next(part for part in match if part)
                if column not in sql:
                    problems.append(
                        f"{item.id}: {card.name} computes over {column}, "
                        "which the gold SQL does not mention"
                    )
        assert not problems, "gold does not compute the card:\n  " + "\n  ".join(problems)


class TestShapeAccuracy:
    """shape_correct separates "asked for the wrong thing" from "wrong definition".

    Exact-value equality scores those identically, which hides the finding: the baseline
    almost always gets the shape right and the value wrong.
    """

    def _q(self, questions, question_id: str):  # noqa: ANN202 - test helper
        return next(item for item in questions.questions if item.id == question_id)

    def test_right_shape_wrong_value_is_shape_correct(self, questions) -> None:
        """The whole point: a missing default filter is not a wrong question."""
        question = self._q(questions, "t1-01")  # scalar money
        gold = ((Decimal("7341182.41"),),)
        wrong_value = ((Decimal("8699763.05"),),)  # what the baseline actually returned
        assert not rows_match(question, wrong_value, gold)
        assert shape_matches(question, wrong_value, gold) is True

    def test_a_scalar_answered_as_a_series_is_not_shape_correct(self, questions) -> None:
        question = self._q(questions, "t1-01")
        gold = ((Decimal("1"),),)
        series = ((date(2018, 1, 1), Decimal("1")), (date(2018, 2, 1), Decimal("2")))
        assert shape_matches(question, series, gold) is False

    def test_a_series_answered_as_a_scalar_is_not_shape_correct(self, questions) -> None:
        question = self._q(questions, "t2-01")
        gold = ((date(2018, 1, 1), Decimal("1")), (date(2018, 2, 1), Decimal("2")))
        assert shape_matches(question, ((Decimal("3"),),), gold) is False

    def test_a_series_may_span_a_different_number_of_buckets(self, questions) -> None:
        """A different filter legitimately changes how many groups exist."""
        question = self._q(questions, "t2-01")
        gold = tuple((date(2018, month, 1), Decimal("1")) for month in range(1, 10))
        ten = tuple((date(2018, month, 1), Decimal("1")) for month in range(1, 11))
        assert shape_matches(question, ten, gold) is True

    def test_an_order_of_magnitude_row_count_is_a_grouping_error(self, questions) -> None:
        question = self._q(questions, "t2-01")
        gold = tuple((date(2018, month, 1), Decimal("1")) for month in range(1, 10))
        many = tuple((date(2018, 1, 1), Decimal("1")) for _ in range(500))
        assert shape_matches(question, many, gold) is False

    def test_a_comparison_ignores_the_key_convention(self, questions) -> None:
        """Consistent with rows_match: the label column is a convention, not an answer."""
        question = self._q(questions, "t2-07")
        gold = (("current", Decimal("981051.06")), ("comparison", Decimal("837895.43")))
        month_keyed = (
            (date(2018, 2, 1), Decimal("992463.34")),
            (date(2018, 3, 1), Decimal("1159652.12")),
        )
        assert shape_matches(question, month_keyed, gold) is True

    def test_no_rows_is_never_shape_correct(self, questions) -> None:
        question = self._q(questions, "t1-01")
        assert shape_matches(question, (), ((Decimal("1"),),)) is False


class TestProviderGuard:
    """A tag is bound to one provider. Mixing backends inside one comparison is refused."""

    def _checkpoint(self, tmp_path: Path, providers: list[str]) -> Path:
        path = tmp_path / "checkpoint.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for index, provider in enumerate(providers):
                handle.write(
                    json.dumps(
                        {
                            "question_id": f"q-{index}",
                            "_system": "canon",
                            "_run_index": 0,
                            "_provider": provider,
                        }
                    )
                    + "\n"
                )
        return path

    def test_same_provider_is_allowed(self, tmp_path: Path) -> None:
        path = self._checkpoint(tmp_path, ["groq", "groq"])
        assert provider_conflict(path, "groq") is None

    def test_a_foreign_provider_is_refused(self, tmp_path: Path) -> None:
        """Resuming a Groq tag on Gemini would make the delta measure the backend."""
        path = self._checkpoint(tmp_path, ["groq", "groq"])
        message = provider_conflict(path, "gemini")
        assert message is not None
        assert "groq" in message and "gemini" in message

    def test_an_already_mixed_checkpoint_is_refused(self, tmp_path: Path) -> None:
        path = self._checkpoint(tmp_path, ["groq", "gemini"])
        message = provider_conflict(path, "groq")
        assert message is not None
        assert "mixes providers" in message

    def test_an_absent_checkpoint_is_fine(self, tmp_path: Path) -> None:
        assert provider_conflict(tmp_path / "nothing.jsonl", "groq") is None

    def test_unrecorded_units_cannot_be_claimed_for_a_provider(self, tmp_path: Path) -> None:
        """Units written before the field existed cannot be proven to be anything."""
        path = tmp_path / "checkpoint.jsonl"
        path.write_text(
            json.dumps({"question_id": "q", "_system": "canon", "_run_index": 0}) + "\n",
            encoding="utf-8",
        )
        assert provider_conflict(path, "groq") is not None


class TestTransportErrorsAreNotFindings:
    """A blocked DLL says nothing about the architecture; a schema rejection does.

    Two tier-2 answerable questions were lost to the environment rather than to the
    system under test -- one to APIConnectionError and one to a Windows Application
    Control policy blocking the _ssl module. Freezing those as permanent errors
    understates the arm for a reason that is not about the design.
    """

    def test_a_dropped_connection_is_transport(self) -> None:
        assert is_transport_error("APIConnectionError: Connection error.")

    def test_a_blocked_ssl_library_is_transport(self) -> None:
        assert is_transport_error(
            "ImportError: DLL load failed while importing _ssl: An Application Control "
            "policy has blocked this file."
        )

    def test_a_schema_rejection_is_not_transport(self) -> None:
        """This one is a real finding about the harness and must stay recorded."""
        assert not is_transport_error(
            "BadRequestError: Error code: 400 - Tool call validation failed: "
            "`/candidates`: minimum 1 items required, but found 0 items"
        )

    def test_a_quota_overage_is_transport(self) -> None:
        """Reverses an earlier call in this file, which had it the other way round.

        The first reading was that a throttle is "a property of the quota rather than of
        the machine", so not transport. That confuses whose fault it is with whether it
        is a finding. The limiter estimated 8092 tokens against a 8000 cap and the model
        never saw the question, so the system under test did not fail; the arithmetic in
        this repo did. am-08 was frozen as a permanent error on exactly that basis and
        charged to Canon's error rate.
        """
        assert is_transport_error(
            "Request too large ... on tokens per minute (TPM): Limit 8000, Requested 8092"
        )

    def test_a_transport_failure_is_reattempted_on_resume(self, tmp_path: Path) -> None:
        """It must not be loaded as done, so the work list picks it up again."""
        path = tmp_path / "checkpoint.jsonl"
        lines = [
            {
                "question_id": "t2-14",
                "question": "q",
                "tier": "2",
                "expected_behavior": "answer",
                "outcome": "errored",
                "error": "APIConnectionError: Connection error.",
                "_system": "canon",
                "_run_index": 0,
                "_provider": "groq",
            },
            {
                "question_id": "am-18",
                "question": "q",
                "tier": "ambiguous",
                "expected_behavior": "refuse",
                "outcome": "errored",
                "error": "BadRequestError: 400 - `/candidates`: minimum 1 items required",
                "_system": "canon",
                "_run_index": 0,
                "_provider": "groq",
            },
        ]
        path.write_text("\n".join(json.dumps(item) for item in lines) + "\n", encoding="utf-8")
        done = _load_checkpoint(path)

        assert "canon|0|t2-14" not in done, "a transport failure must be retried"
        assert "canon|0|am-18" in done, "a schema rejection is a finding and stays recorded"

    def test_a_successful_unit_is_always_loaded(self, tmp_path: Path) -> None:
        path = tmp_path / "checkpoint.jsonl"
        path.write_text(
            json.dumps(
                {
                    "question_id": "t1-01",
                    "question": "q",
                    "tier": "1",
                    "expected_behavior": "answer",
                    "outcome": "answered",
                    "value_correct": True,
                    "_system": "canon",
                    "_run_index": 0,
                    "_provider": "groq",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        assert "canon|0|t1-01" in _load_checkpoint(path)


class TestLatencyCannotBeMisreported:
    """Fix: the instrument must not be able to state what the data does not support.

    A published arm once reported ``mean_latency_seconds = 2.869`` under the label "mean
    latency / question" while 36 of its 80 units were cache replays recording 0.0s. The
    document was corrected by hand; these tests correct the thing that produced it.
    """

    def _score(self, *, cached: bool, latency: float) -> QuestionScore:
        return QuestionScore(
            question_id="x",
            question="q",
            tier="1",
            expected_behavior=Behavior.ANSWER,
            outcome=Outcome.ANSWERED,
            cached=cached,
            latency_seconds=latency,
        )

    def _questions(self) -> QuestionSet:
        return QuestionSet(questions=())

    def test_a_fully_cached_arm_has_no_live_latency(self) -> None:
        """None, not 0.0: "we did not measure" must not render like "it was fast"."""
        scores = [self._score(cached=True, latency=0.0) for _ in range(5)]
        result = aggregate("baseline", scores, self._questions())
        assert result.live_latency is None
        assert result.mean_latency_seconds is None

    def test_a_partly_cached_arm_refuses_a_per_question_mean(self) -> None:
        """The exact shape of the published Canon arm: some live, most replayed."""
        scores = [self._score(cached=False, latency=3.0) for _ in range(2)]
        scores += [self._score(cached=True, latency=0.0) for _ in range(3)]
        result = aggregate("canon", scores, self._questions())
        assert result.mean_latency_seconds is None, "a diluted mean must not be produced"
        assert result.live_latency is not None
        assert result.live_latency.n == 2
        assert result.live_latency.cached == 3
        assert not result.live_latency.covers_whole_arm

    def test_a_fully_live_arm_does_get_a_per_question_mean(self) -> None:
        """The field is not disabled -- it is conditioned on the denominator holding."""
        scores = [self._score(cached=False, latency=2.0) for _ in range(4)]
        result = aggregate("canon", scores, self._questions())
        assert result.mean_latency_seconds == 2.0
        assert result.live_latency is not None
        assert result.live_latency.covers_whole_arm

    def test_live_latency_cannot_exist_without_its_sample_size(self) -> None:
        """n is required and positive, so the figure cannot be quoted bare."""
        with pytest.raises(ValidationError):
            LiveLatency(
                n=0, cached=5, total=5, mean_seconds=3.0, median_seconds=3.0, max_seconds=3.0
            )

    def test_no_cross_arm_comparison_when_one_arm_is_fully_cached(self) -> None:
        """The published run's case: the baseline was 80/80 cache-served."""
        live = aggregate(
            "canon",
            [self._score(cached=False, latency=3.0)],
            self._questions(),
        )
        dead = aggregate(
            "baseline",
            [self._score(cached=True, latency=0.0)],
            self._questions(),
        )
        assert latency_comparison({"canon": live, "baseline": dead}) is None

    def test_a_comparison_is_emitted_when_both_arms_have_live_calls(self) -> None:
        one = aggregate("canon", [self._score(cached=False, latency=3.0)], self._questions())
        two = aggregate("baseline", [self._score(cached=False, latency=1.0)], self._questions())
        comparison = latency_comparison({"canon": one, "baseline": two})
        assert comparison is not None
        assert "n=1" in comparison

    def test_the_arm_size_is_carried_not_inferred(self) -> None:
        """Regression: an errored unit is neither cached nor timed.

        The first version of this model rendered "36 of 78 units" for an arm of 80,
        because it inferred the denominator as n + cached.
        """
        scores = [self._score(cached=False, latency=3.0) for _ in range(2)]
        scores += [self._score(cached=True, latency=0.0) for _ in range(3)]
        scores.append(self._score(cached=False, latency=0.0))  # errored: no timing at all
        result = aggregate("canon", scores, self._questions())
        assert result.live_latency is not None
        assert result.live_latency.total == 6
        assert result.live_latency.n == 2
        assert result.live_latency.cached == 3
        assert result.live_latency.untimed == 4


class TestProvenanceIsNotRestamped:
    """Fix: re-scoring a finished run must not claim it ran today.

    The published run was re-scored on 2026-10-23 and its ``started_at`` moved to that
    date, having already been overwritten several times before. A re-score costs no quota
    and answers nothing, so it has no business touching when the answers were produced.
    """

    def test_a_rescore_inherits_the_answering_provenance(self, tmp_path: Path) -> None:
        (tmp_path / "report.json").write_text(
            json.dumps(
                {
                    "provenance": {
                        "started_at": "2026-09-04",
                        "code_commit": "61d6ac4",
                        "answering_sessions": ["2026-09-04@abc1234"],
                    }
                }
            ),
            encoding="utf-8",
        )
        prior = _prior_provenance(tmp_path)
        assert prior["started_at"] == "2026-09-04"
        assert prior["code_commit"] == "61d6ac4"
        assert prior["answering_sessions"] == ["2026-09-04@abc1234"]

    def test_a_new_run_has_no_prior_to_inherit(self, tmp_path: Path) -> None:
        assert _prior_provenance(tmp_path) == {}

    def test_a_corrupt_report_does_not_crash_the_run(self, tmp_path: Path) -> None:
        """A damaged report must lose provenance, not block scoring."""
        (tmp_path / "report.json").write_text("{not json", encoding="utf-8")
        assert _prior_provenance(tmp_path) == {}

    def test_the_answering_span_needs_two_sessions(self) -> None:
        one = Provenance(
            started_at="2026-09-04",
            code_commit="a",
            eval_set_commit="b",
            metrics_yaml_sha256="c",
            provider="groq",
            provider_config_hash="d",
            free_tier=True,
            model_id="m",
            effort="medium",
            policy_thresholds=PolicyThresholds(),
            threshold_provenance="x",
            confident_threshold=0.7,
            question_count=1,
            runs=1,
            cache_used=True,
            answering_sessions=("2026-09-04@a",),
        )
        assert one.answering_span is None
        many = one.model_copy(update={"answering_sessions": ("2026-09-04@a", "2026-10-02@b")})
        assert many.answering_span == ("2026-09-04@a", "2026-10-02@b")
