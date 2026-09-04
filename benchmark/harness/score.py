"""Scoring, exactly as ``docs/01-evaluation.md`` defines it.

The comparison rules were fixed before any code ran and are not tuned to flatter a run:
money as DECIMAL, numerics to a relative tolerance, multi-row results as a multiset
unless the question implies an order, NULL distinct from 0.

One asymmetry is deliberate and is the finding rather than a flaw in the measurement.
**Semantic accuracy is only directly measurable for a system that declares which metric
it used.** Canon returns a metric card, so its metric choice can be checked against gold.
The baseline has no notion of a metric, so there is nothing to check — which is precisely
the gap §3.2 argues Spider and BIRD cannot see. For the baseline, a wrong metric choice
shows up as an execution-accuracy failure instead, and `matched_other_gold` names the
question whose gold value its number *did* match, which is direct evidence of which
definition it actually computed.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from benchmark.harness.dataset import Behavior, Question, QuestionSet

_FROZEN = ConfigDict(frozen=True, extra="forbid")

RELATIVE_TOLERANCE = 1e-6
ABSOLUTE_TOLERANCE = 1e-9
CONFIDENT_THRESHOLD = 0.7
"""Fixed in advance, per §3.4. The report also sweeps it so the choice is inspectable."""

_ORDER_SENSITIVE_CUES = ("top ", "worst", "best", "most", "least", "ranked", "five", "ten")


def _decimalise(value: Any) -> Any:
    """Normalise a cell for comparison, preserving the distinctions that matter."""
    if isinstance(value, dict):
        # A gold value rehydrated from JSON.
        if "__decimal__" in value:
            return Decimal(value["__decimal__"])
        if "__timestamp__" in value:
            return str(value["__timestamp__"])
        return value
    if hasattr(value, "isoformat"):
        return str(value)
    return value


def cells_equal(left: Any, right: Any) -> bool:
    """Compare one cell pair under the rules fixed in the eval doc."""
    left, right = _decimalise(left), _decimalise(right)

    # NULL is not 0. A month with no orders is not a month with zero revenue.
    if left is None or right is None:
        return left is None and right is None

    left_numeric = isinstance(left, Decimal | int | float) and not isinstance(left, bool)
    right_numeric = isinstance(right, Decimal | int | float) and not isinstance(right, bool)
    if left_numeric and right_numeric:
        if isinstance(left, Decimal) and isinstance(right, Decimal):
            return left == right
        a, b = float(left), float(right)
        if a == b:
            return True
        scale = max(abs(a), abs(b))
        return abs(a - b) <= max(RELATIVE_TOLERANCE * scale, ABSOLUTE_TOLERANCE)

    return str(left) == str(right)


def _rows_equal_ordered(
    actual: tuple[tuple[Any, ...], ...], expected: tuple[tuple[Any, ...], ...]
) -> bool:
    if len(actual) != len(expected):
        return False
    return all(
        len(a) == len(b) and all(cells_equal(x, y) for x, y in zip(a, b, strict=True))
        for a, b in zip(actual, expected, strict=True)
    )


def _rows_equal_multiset(
    actual: tuple[tuple[Any, ...], ...], expected: tuple[tuple[Any, ...], ...]
) -> bool:
    """Order-insensitive comparison, matching each expected row to an unused actual row."""
    if len(actual) != len(expected):
        return False
    remaining = list(actual)
    for wanted in expected:
        for index, candidate in enumerate(remaining):
            if len(candidate) == len(wanted) and all(
                cells_equal(x, y) for x, y in zip(candidate, wanted, strict=True)
            ):
                del remaining[index]
                break
        else:
            return False
    return True


def order_matters(question: Question) -> bool:
    """Whether the question implies a row order.

    "Top five" and "worst states" are claims about ordering; a plain breakdown is not.
    """
    lowered = question.question.lower()
    return any(cue in lowered for cue in _ORDER_SENSITIVE_CUES)


def rows_match(
    question: Question,
    actual: tuple[tuple[Any, ...], ...],
    expected: tuple[tuple[Any, ...], ...],
) -> bool:
    """Compare a full result against gold.

    Column names are ignored -- aliasing is cosmetic -- but the *shape* is not: a
    different number of columns means a different answer. A single-column gold value is
    matched against the last column of a wider result, because a system that returns
    ``(period, revenue)`` for a scalar question has answered a different question, while
    one returning ``(revenue)`` with an extra label column has not.
    """
    if not actual:
        return False

    gold_width = len(expected[0]) if expected else 0
    actual_width = len(actual[0])

    if actual_width != gold_width:
        # Tolerate exactly one case: the system returned the right numbers with an extra
        # leading label the question did not ask for. Compare on the metric column alone.
        if gold_width == 1 and actual_width > 1 and len(actual) == len(expected):
            trimmed = tuple((row[-1],) for row in actual)
            return (
                _rows_equal_ordered(trimmed, expected)
                if order_matters(question)
                else _rows_equal_multiset(trimmed, expected)
            )
        return False

    return (
        _rows_equal_ordered(actual, expected)
        if order_matters(question)
        else _rows_equal_multiset(actual, expected)
    )


class Outcome(StrEnum):
    """What the system did, before judging whether it was right."""

    ANSWERED = "answered"
    ASKED = "asked"
    ERRORED = "errored"
    """Could not produce either -- an API failure or SQL that would not execute. Scored
    as a failure to answer, never silently dropped."""


class QuestionScore(BaseModel):
    """The scored result for one question, written to newline-delimited JSON."""

    model_config = _FROZEN

    question_id: str
    question: str
    tier: str
    expected_behavior: Behavior
    outcome: Outcome

    value_correct: bool | None = None
    """None when the system did not answer, or when there is no gold value to compare."""

    metric_correct: bool | None = None
    """None when the system declares no metric -- the baseline's structural position."""

    metric_used: str | None = None
    expected_metric: str | None = None

    refusal_correct: bool = False
    """Whether the ask-or-answer decision matched the label."""

    names_the_fork: bool | None = None
    """Automatic proxy for rubric level 2: does the clarifying question mention every
    term in ``must_mention``? Necessary, not sufficient — the questions are also written
    out for hand-grading."""

    caveats_surfaced: bool | None = None
    """Whether every substring in ``requires_caveat`` appeared. The tier-3 assertion."""

    confidence: float = 0.0
    confident_wrong: bool = False
    matched_other_gold: str | None = None
    """The id of another question whose gold value this answer matched. Evidence of which
    definition was actually computed, and the only semantic signal available for a system
    that declares no metric."""

    sql: str | None = None
    clarifying_question: str | None = None
    error: str | None = None
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False

    @property
    def is_wrong_answer(self) -> bool:
        """Answered when it should have asked, or answered with the wrong number/metric."""
        if self.outcome is not Outcome.ANSWERED:
            return False
        if not self.expected_behavior.should_answer:
            return True
        return self.value_correct is False or self.metric_correct is False


class Aggregate(BaseModel):
    """The report table from §4. Rates are None when the denominator is zero."""

    model_config = _FROZEN

    system: str
    total_questions: int
    answerable: int
    should_not_answer: int

    execution_accuracy: float | None = None
    semantic_accuracy: float | None = None
    semantic_measurable: int = 0
    """How many questions semantic accuracy could be computed on. Zero for a system that
    declares no metric, which is reported rather than papered over."""

    refusal_accuracy: float | None = None
    under_refusal_rate: float | None = None
    over_refusal_rate: float | None = None
    names_the_fork_rate: float | None = None
    caveat_surfaced_rate: float | None = None
    caveat_measurable: int = 0
    confident_wrong_rate: float | None = None
    error_rate: float | None = None

    mean_latency_seconds: float | None = None
    mean_cost_usd: float | None = None
    cached_fraction: float | None = None

    per_tier: dict[str, TierAggregate] = Field(default_factory=dict)


class TierAggregate(BaseModel):
    model_config = _FROZEN

    tier: str
    count: int
    execution_accuracy: float | None = None
    semantic_accuracy: float | None = None
    refusal_accuracy: float | None = None
    confident_wrong_rate: float | None = None


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 4)


def find_matching_gold(
    actual: tuple[tuple[Any, ...], ...],
    question_id: str,
    gold_values: dict[str, list[list[Any]]],
    questions: QuestionSet,
) -> str | None:
    """Which other question's gold value does this answer equal?

    Cheap, deterministic evidence of a definition confusion: if a system asked for
    revenue-excluding-shipping returns the number that gold says is gross revenue for the
    same period, it computed the wrong definition. No LLM judge, no extra queries.
    """
    if not actual:
        return None
    by_id = {item.id: item for item in questions.questions}
    for other_id, other_rows in sorted(gold_values.items()):
        if other_id == question_id or other_id not in by_id:
            continue
        expected = tuple(tuple(row) for row in other_rows)
        if rows_match(by_id[other_id], actual, expected):
            return other_id
    return None


def aggregate(system: str, scores: list[QuestionScore], questions: QuestionSet) -> Aggregate:
    """Roll per-question scores into the report table."""
    answerable = [s for s in scores if s.expected_behavior.should_answer]
    ambiguous = [s for s in scores if not s.expected_behavior.should_answer]

    # Execution accuracy: correct values over ALL answerable questions, so refusing an
    # answerable question counts against it. Otherwise "refuse everything" scores 100%.
    value_correct = sum(1 for s in answerable if s.value_correct is True)

    semantic_pool = [s for s in answerable if s.metric_correct is not None]
    semantic_correct = sum(1 for s in semantic_pool if s.metric_correct)

    correct_attempts = sum(
        1 for s in answerable if s.outcome is Outcome.ANSWERED and s.value_correct is True
    )
    correct_refusals = sum(1 for s in ambiguous if s.outcome is Outcome.ASKED)
    under_refusals = sum(1 for s in ambiguous if s.outcome is Outcome.ANSWERED)
    over_refusals = sum(1 for s in answerable if s.outcome is Outcome.ASKED)

    fork_pool = [s for s in ambiguous if s.names_the_fork is not None]
    caveat_pool = [s for s in scores if s.caveats_surfaced is not None]

    per_tier: dict[str, TierAggregate] = {}
    for tier in sorted({s.tier for s in scores}):
        subset = [s for s in scores if s.tier == tier]
        tier_answerable = [s for s in subset if s.expected_behavior.should_answer]
        tier_semantic = [s for s in tier_answerable if s.metric_correct is not None]
        tier_correct_refusals = sum(
            1
            for s in subset
            if not s.expected_behavior.should_answer and s.outcome is Outcome.ASKED
        )
        tier_correct_attempts = sum(
            1 for s in tier_answerable if s.outcome is Outcome.ANSWERED and s.value_correct is True
        )
        per_tier[tier] = TierAggregate(
            tier=tier,
            count=len(subset),
            execution_accuracy=_rate(
                sum(1 for s in tier_answerable if s.value_correct is True), len(tier_answerable)
            ),
            semantic_accuracy=_rate(
                sum(1 for s in tier_semantic if s.metric_correct), len(tier_semantic)
            ),
            refusal_accuracy=_rate(tier_correct_attempts + tier_correct_refusals, len(subset)),
            confident_wrong_rate=_rate(sum(1 for s in subset if s.confident_wrong), len(subset)),
        )

    measured = [s for s in scores if not s.cached]
    return Aggregate(
        system=system,
        total_questions=len(scores),
        answerable=len(answerable),
        should_not_answer=len(ambiguous),
        execution_accuracy=_rate(value_correct, len(answerable)),
        semantic_accuracy=_rate(semantic_correct, len(semantic_pool)),
        semantic_measurable=len(semantic_pool),
        refusal_accuracy=_rate(correct_attempts + correct_refusals, len(scores)),
        under_refusal_rate=_rate(under_refusals, len(ambiguous)),
        over_refusal_rate=_rate(over_refusals, len(answerable)),
        names_the_fork_rate=_rate(sum(1 for s in fork_pool if s.names_the_fork), len(fork_pool)),
        caveat_surfaced_rate=_rate(
            sum(1 for s in caveat_pool if s.caveats_surfaced), len(caveat_pool)
        ),
        caveat_measurable=len(caveat_pool),
        confident_wrong_rate=_rate(sum(1 for s in scores if s.confident_wrong), len(scores)),
        error_rate=_rate(sum(1 for s in scores if s.outcome is Outcome.ERRORED), len(scores)),
        mean_latency_seconds=(
            round(sum(s.latency_seconds for s in measured) / len(measured), 3) if measured else None
        ),
        mean_cost_usd=(
            round(sum(s.cost_usd for s in measured) / len(measured), 6) if measured else None
        ),
        cached_fraction=_rate(sum(1 for s in scores if s.cached), len(scores)),
        per_tier=per_tier,
    )


def confident_wrong_curve(scores: list[QuestionScore]) -> dict[str, float | None]:
    """Confident-wrong rate across thresholds, so 0.7 is inspectable, not load-bearing."""
    return {
        f"{threshold:.1f}": _rate(
            sum(1 for s in scores if s.is_wrong_answer and s.confidence >= threshold),
            len(scores),
        )
        for threshold in (0.5, 0.6, 0.7, 0.8, 0.9)
    }
