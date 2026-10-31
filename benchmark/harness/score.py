"""Scoring, exactly as ``docs/01-evaluation.md`` defines it.

The comparison rules were fixed before any code ran and are not tuned to flatter a run:
money as DECIMAL, numerics to a relative tolerance, multi-row results as a multiset
unless the question implies an order, NULL distinct from 0.

One asymmetry is deliberate and is the finding rather than a flaw in the measurement.
**Semantic accuracy is only directly measurable for a system that declares which metric
it used.** Canon returns a metric card, so its metric choice can be checked against gold.
The baseline has no notion of a metric, so there is nothing to check — which is precisely
the gap §3.2 argues Spider and BIRD cannot see. For the baseline, a wrong metric choice
shows up as an execution-accuracy failure instead, and ``attribute_wrong_answer`` tries to
name the definition it actually computed by matching its number against other questions'
gold values. That instrument is guarded and its coverage is measured
(``gold_collision_report``), because on this eval set most gold values either repeat
across questions or are too small to distinguish — so an unguarded version would assert
far more than the data supports.

The headline safety metric is **silent-wrong rate**: wrong, with neither a gate flag nor
a clarifying question to warn anyone. It reads only what each system observably did, so
unlike confident-wrong it needs no imputed confidence for a system that reports none.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
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

    # FAIRNESS ADJUSTMENT for comparison questions. Canon labels the two periods
    # 'current' and 'comparison'; gold uses that convention because gold was written
    # from the metric cards. A system that answers "March vs the month before" with two
    # month-keyed rows has answered the question correctly and would be marked wrong
    # purely on a key convention it had no way to know. So for these questions only the
    # metric column is compared, as a multiset. Found by reading the baseline's actual
    # SQL rather than its score.
    if question.expected_comparison and expected:
        gold_values = tuple((row[-1],) for row in expected)
        actual_values = tuple((row[-1],) for row in actual)
        return _rows_equal_multiset(actual_values, gold_values)

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


def shape_matches(
    question: Question,
    actual: tuple[tuple[Any, ...], ...],
    expected: tuple[tuple[Any, ...], ...],
) -> bool | None:
    """Did the answer have the right *form*, setting aside whether the numbers agree?

    Three things must hold, and nothing about filter correctness:

    * the same number of columns -- a breakdown answered as a scalar is a different
      question, not a filter mistake;
    * the same *kinds* of column in the same positions, so a period column is matched
      against a period column and a metric against a metric;
    * a plausible row count for the grain: exact where gold is a scalar or a comparison,
      and within a tolerance where gold is a time series, because a series whose filters
      differ legitimately spans a different number of buckets.

    ``None`` when there is nothing to compare against.

    The point of the metric is to separate "asked for the wrong thing" from "asked for the
    right thing under the wrong definition". The baseline does the second constantly and
    the first rarely, and execution accuracy alone cannot tell them apart.
    """
    if not expected:
        return None
    if not actual:
        return False

    gold_width = len(expected[0])
    actual_width = len(actual[0])
    if actual_width != gold_width:
        return False

    # Same fairness adjustment as rows_match: on a comparison question Canon keys the two
    # periods 'current'/'comparison' and gold follows that convention, while a system
    # answering "March vs the month before" with two month-keyed rows has answered
    # correctly. Judging the key column's *type* here would penalise that convention a
    # second time, having already excused it for the value comparison.
    if question.expected_comparison:
        return len(actual) == len(expected)

    def kind(cell: Any) -> str:
        value = _decimalise(cell)
        if value is None:
            return "null"
        if isinstance(value, Decimal | int | float) and not isinstance(value, bool):
            return "number"
        text = str(value)
        # A date-like string in the leading position is a period key, not a label.
        return "period" if text[:4].isdigit() and "-" in text[:8] else "text"

    gold_kinds = [kind(cell) for cell in expected[0]]
    actual_kinds = [kind(cell) for cell in actual[0]]
    # A null in one sample says nothing about the column's type, so it matches anything.
    for gold_kind, actual_kind in zip(gold_kinds, actual_kinds, strict=True):
        if "null" in (gold_kind, actual_kind):
            continue
        if gold_kind != actual_kind:
            return False

    if len(expected) == 1 or question.expected_comparison:
        return len(actual) == len(expected)

    # A time series or breakdown: allow a different bucket count, since a different
    # filter legitimately changes how many groups exist. Reject an order-of-magnitude
    # difference, which is a grouping error rather than a filter difference.
    lower, upper = len(expected) / 3, len(expected) * 3
    return lower <= len(actual) <= upper


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

    shape_correct: bool | None = None
    """Right columns and a plausible grain, ignoring filters.

    The lenient counterpart to value_correct. Reported only as a pair with it: alone it
    flatters a system that knows the schema and not the semantics, which is precisely the
    system it exists to describe.
    """

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

    gate_flagged: bool = False
    """Whether the verification gate rejected the result.

    The baseline is always False here, and not because it passed: it has no gate. That
    asymmetry is the finding, so it is recorded per question rather than assumed.
    """

    silent_wrong: bool = False
    """Wrong, and nothing said so.

    The headline safety number. A wrong answer the system flagged or asked about is a
    caught error; a wrong answer handed over clean is the one that reaches a slide and
    gets acted on. Unlike confident-wrong this needs no imputed confidence -- it reads
    only what each system observably did.
    """

    attribution: str = "no_match"
    """Whether a wrong answer could be traced to another question's gold value:
    no_match, attributable_wrong, or unattributable_wrong."""

    matched_other_gold: str | None = None
    """The question whose gold value this answer matched, set only when the attribution
    is ATTRIBUTABLE. Evidence of which definition was actually computed."""

    attribution_reason: str | None = None

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

    @property
    def is_silent_wrong(self) -> bool:
        """Wrong, with neither a gate flag nor a clarifying question to warn anyone."""
        return self.is_wrong_answer and not self.gate_flagged and self.clarifying_question is None

    @property
    def is_caught_wrong(self) -> bool:
        """Declined to answer a question where answering would have been wrong.

        The counterpart to ``is_silent_wrong``: both describe cases the system got no
        number right, and the difference is whether anyone was told.
        """
        return self.outcome is Outcome.ASKED and not self.expected_behavior.should_answer


class LiveLatency(BaseModel):
    """Latency over live calls, inseparable from the sample it was measured on.

    ``n`` is not optional and not a separate field a caller can forget to print: anything
    rendering this model has the denominator in hand. That is the whole design -- the
    previous shape was a bare float, and a bare float is exactly what gets quoted without
    its sample size.
    """

    model_config = _FROZEN

    n: int = Field(gt=0)
    """Live calls timed. Greater than zero by construction; an arm with none gets
    ``live_latency=None`` rather than an n of 0."""

    cached: int = Field(ge=0)
    """Units served from cache and therefore never timed. Printed beside ``n`` so the
    coverage of the measurement is visible."""

    total: int = Field(gt=0)
    """Units in the arm. Carried explicitly rather than inferred as ``n + cached``, which
    undercounts: a unit can be neither cached nor timed -- an errored one records no
    latency -- and the first version of this model rendered "36 of 78 units" for an arm of
    80 for exactly that reason.
    """

    mean_seconds: float
    median_seconds: float
    max_seconds: float

    @property
    def untimed(self) -> int:
        """Units with no timing, whether cached or errored."""
        return self.total - self.n

    @property
    def covers_whole_arm(self) -> bool:
        """Whether every unit was timed, i.e. whether a per-question mean would be sound."""
        return self.n == self.total


class Aggregate(BaseModel):
    """The report table from §4. Rates are None when the denominator is zero."""

    model_config = _FROZEN

    system: str
    total_questions: int
    answerable: int
    should_not_answer: int

    execution_accuracy: float | None = None

    shape_accuracy: float | None = None
    """Correct columns and grain over answerable questions, filters disregarded.

    The gap between this and execution accuracy is the size of the semantics problem: a
    system scoring high here and low there knows the schema and not the business rules.
    """

    shape_measurable: int = 0
    semantic_accuracy: float | None = None
    semantic_measurable: int = 0
    """How many questions semantic accuracy could be computed on. Zero for a system that
    declares no metric, which is reported rather than papered over."""

    refusal_accuracy: float | None = None
    behavior_accuracy: float | None = None
    """Fraction on the 3x3 diagonal -- did the system do the right *kind* of thing?

    Stricter than refusal accuracy, which only asks whether it answered or asked. A
    system that declines an ambiguous question by claiming the metric is undefined gets
    credit from refusal accuracy and not from this.
    """

    under_refusal_rate: float | None = None
    over_refusal_rate: float | None = None
    names_the_fork_rate: float | None = None
    caveat_surfaced_rate: float | None = None
    caveat_measurable: int = 0

    silent_wrong_rate: float | None = None
    """The headline: wrong answers handed over with no signal, over all questions."""

    silent_wrong_count: int = 0
    caught_wrong_count: int = 0
    """Cases the system declined to answer or flagged, where answering would have been
    wrong. The baseline's gate count is structurally zero."""

    gate_flagged_count: int = 0
    attributable_wrong_count: int = 0
    unattributable_wrong_count: int = 0

    confident_wrong_rate: float | None = None
    """Secondary. Depends on imputing confidence 1.0 to the baseline, which is why it is
    no longer the headline -- see the footnote in RESULTS.md."""

    error_rate: float | None = None

    mean_latency_seconds: float | None = None
    """``None`` whenever ANY unit in the arm was cache-served, which is almost always.

    This field once averaged live calls over the live-call count while the *label* said
    "mean latency / question", and an arm with 36 of 80 units replayed from cache reported
    2.869s as though it were a per-question timing. The number was not wrong so much as
    unlabelled, and a reader had no way to see the dilution.

    A partially cached arm has no defensible per-question latency: the cached units were
    never timed, and imputing anything to them -- 0.0, or the live mean -- invents data. So
    the field reports nothing at all, and the timing that IS defensible moves to
    ``live_latency``, where it cannot be read without its sample size.
    """

    live_latency: LiveLatency | None = None
    """Timing over live calls only, carrying its own n. ``None`` when nothing was timed."""

    mean_cost_usd: float | None = None
    cached_fraction: float | None = None

    per_tier: dict[str, TierAggregate] = Field(default_factory=dict)


class TierAggregate(BaseModel):
    model_config = _FROZEN

    tier: str
    count: int
    execution_accuracy: float | None = None
    shape_accuracy: float | None = None
    semantic_accuracy: float | None = None
    refusal_accuracy: float | None = None
    behavior_accuracy: float | None = None
    silent_wrong_rate: float | None = None
    confident_wrong_rate: float | None = None


def _live_latency(scores: Sequence[QuestionScore]) -> LiveLatency | None:
    """Timing over live calls only, or None when nothing in this arm was timed.

    A cache hit records 0.0 seconds, so it is excluded rather than averaged in. An arm
    served entirely from cache -- which the published baseline arm was, 80 of 80 -- yields
    None, because "we did not measure this" and "this was fast" must not render alike.
    """
    timed = [s.latency_seconds for s in scores if not s.cached and s.latency_seconds]
    if not timed:
        return None
    return LiveLatency(
        n=len(timed),
        cached=sum(1 for s in scores if s.cached),
        total=len(scores),
        mean_seconds=round(statistics.mean(timed), 3),
        median_seconds=round(statistics.median(timed), 3),
        max_seconds=round(max(timed), 3),
    )


def latency_comparison(aggregates: Mapping[str, Aggregate]) -> str | None:
    """A cross-arm latency comparison, or None when the data cannot support one.

    Emitted only when every arm has at least one live call. The published run cannot
    support it -- the baseline was wholly cache-served -- and the correct output there is
    nothing at all, not a table with one populated column that reads as a comparison.
    """
    live = {name: agg.live_latency for name, agg in aggregates.items()}
    if not live or any(value is None for value in live.values()):
        return None
    parts = [
        f"{name} {value.mean_seconds:.2f}s (n={value.n})"
        for name, value in sorted(live.items())
        if value is not None
    ]
    return "Mean latency over live calls: " + ", ".join(parts) + "."


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 4)


MIN_ATTRIBUTION_MAGNITUDE = 1000.0
"""Below this, a scalar match is not evidence of anything.

Small numbers collide by coincidence. The eval set contains counts like 334 canceled
orders and averages like a 4.086 review score, and two unrelated questions landing on
the same small value says nothing about which definition was computed. Money and volume
figures in the thousands are distinctive enough to carry an inference.
"""


class Attribution(StrEnum):
    """How much weight a value-collision finding can bear."""

    NO_MATCH = "no_match"
    """The answer matched no other question's gold value."""

    ATTRIBUTABLE = "attributable_wrong"
    """Matched exactly one other question, whose gold value is unique across the set and
    large enough not to collide by chance. Evidence of which definition was computed."""

    UNATTRIBUTABLE = "unattributable_wrong"
    """Matched something, but the match cannot identify a definition -- the value is
    shared by several questions, or too small to be distinctive. Recorded as wrong
    without a claim about *why*."""


class AttributionResult(BaseModel):
    """The finding, with its own reliability attached."""

    model_config = _FROZEN

    status: Attribution
    matched_question_id: str | None = None
    matched_count: int = 0
    reason: str | None = None


def _signature(rows: tuple[tuple[Any, ...], ...]) -> str:
    """A comparable fingerprint of a result, for detecting shared gold values."""
    return "|".join(
        ";".join(
            f"{cell:.9g}" if isinstance(cell, float) else str(_decimalise(cell)) for cell in row
        )
        for row in rows
    )


def _is_distinctive(rows: tuple[tuple[Any, ...], ...]) -> bool:
    """Whether a result is large enough that a match is unlikely to be coincidence."""
    for row in rows:
        for cell in row:
            value = _decimalise(cell)
            numeric = isinstance(value, Decimal | int | float) and not isinstance(value, bool)
            if numeric and abs(float(value)) >= MIN_ATTRIBUTION_MAGNITUDE:
                return True
    return False


def gold_collision_report(
    gold_values: dict[str, list[list[Any]]],
    questions: QuestionSet | None = None,
) -> dict[str, Any]:
    """The instrument's stated error bar.

    Measures how much of the gold set can support an attribution at all: how many gold
    values are shared between questions, and how many are too small to be distinctive.
    An instrument with no known false-positive rate should not be quoted, so this is
    computed and reported alongside anything derived from it.

    When ``questions`` is supplied, each collision group is also labelled with the
    metrics it spans, because the two kinds of collision are not equally dangerous:

    - **Same metric, different wording.** ``t1-01`` ("net revenue in 2018"), ``t3-04``
      ("revenue excluding shipping") and ``t3-06`` ("merchandise revenue") share one
      value because they are deliberate paraphrases with byte-identical gold SQL. They
      cost attribution coverage and nothing else -- matching that value tells you the
      definition was right and only the phrasing differed.
    - **Different metrics, equal value.** ``t3-03`` asks for ``reported_revenue`` and
      lands on the same number, because the 2018-01-01 restatement makes the two
      definitions agree inside 2018. *That* is the dangerous kind: a wrong answer
      matching it cannot be attributed to a definition at all, since two definitions
      produce it.

    Counting them together would overstate how much the instrument knows.
    """
    signatures: dict[str, list[str]] = {}
    for question_id, rows in gold_values.items():
        signatures.setdefault(_signature(tuple(tuple(r) for r in rows)), []).append(question_id)

    shared = {sig: ids for sig, ids in signatures.items() if len(ids) > 1}
    small = [
        question_id
        for question_id, rows in gold_values.items()
        if not _is_distinctive(tuple(tuple(r) for r in rows))
    ]
    ambiguous_ids = {qid for ids in shared.values() for qid in ids}
    usable = [qid for qid in gold_values if qid not in ambiguous_ids and qid not in set(small)]

    groups = sorted(sorted(ids) for ids in shared.values())
    report: dict[str, Any] = {
        "total_gold_values": len(gold_values),
        "shared_value_groups": groups,
        "questions_with_shared_values": len(ambiguous_ids),
        "questions_below_magnitude_threshold": len(small),
        "questions_supporting_attribution": len(usable),
        "attribution_coverage": (round(len(usable) / len(gold_values), 4) if gold_values else None),
        "magnitude_threshold": MIN_ATTRIBUTION_MAGNITUDE,
    }

    if questions is not None:
        metric_of = {item.id: item.expected_metric for item in questions.questions}
        annotated = []
        for group in groups:
            metrics = sorted({str(metric_of.get(qid)) for qid in group if metric_of.get(qid)})
            annotated.append(
                {
                    "questions": group,
                    "metrics": metrics,
                    "cross_metric": len(metrics) > 1,
                }
            )
        report["collision_groups"] = annotated
        report["cross_metric_groups"] = sum(1 for item in annotated if item["cross_metric"])

    return report


def attribute_wrong_answer(
    actual: tuple[tuple[Any, ...], ...],
    question_id: str,
    gold_values: dict[str, list[list[Any]]],
    questions: QuestionSet,
) -> AttributionResult:
    """Which other question's gold value does this answer equal, if that is knowable?

    Cheap, deterministic evidence of a definition confusion -- a system asked for
    revenue-excluding-shipping that returns the gross figure computed the wrong
    definition. Guarded three ways, because the unguarded version asserts more than the
    data supports:

    * a value shared by several questions cannot identify one of them;
    * a value below the magnitude threshold could collide by chance;
    * more than one distinct matching value means the collision is not informative.
    """
    if not actual:
        return AttributionResult(status=Attribution.NO_MATCH)

    by_id = {item.id: item for item in questions.questions}
    signature_owners: dict[str, list[str]] = {}
    for other_id, other_rows in gold_values.items():
        key = _signature(tuple(tuple(r) for r in other_rows))
        signature_owners.setdefault(key, []).append(other_id)

    matches: list[str] = []
    for other_id, other_rows in sorted(gold_values.items()):
        if other_id == question_id or other_id not in by_id:
            continue
        expected = tuple(tuple(row) for row in other_rows)
        if rows_match(by_id[other_id], actual, expected):
            matches.append(other_id)

    if not matches:
        return AttributionResult(status=Attribution.NO_MATCH)

    if not _is_distinctive(actual):
        return AttributionResult(
            status=Attribution.UNATTRIBUTABLE,
            matched_count=len(matches),
            reason=(
                f"every value is below the {MIN_ATTRIBUTION_MAGNITUDE:g} magnitude "
                "threshold, so the match could be coincidence"
            ),
        )

    distinct = {_signature(tuple(tuple(r) for r in gold_values[m])) for m in matches}
    if len(distinct) > 1:
        return AttributionResult(
            status=Attribution.UNATTRIBUTABLE,
            matched_count=len(matches),
            reason=f"matched {len(matches)} questions with differing values",
        )

    owners = [
        owner for owner in signature_owners.get(next(iter(distinct)), []) if owner != question_id
    ]
    if len(owners) > 1:
        return AttributionResult(
            status=Attribution.UNATTRIBUTABLE,
            matched_count=len(owners),
            reason=(
                "this value is the gold answer for "
                f"{', '.join(sorted(owners))}, so it cannot identify one of them"
            ),
        )

    return AttributionResult(
        status=Attribution.ATTRIBUTABLE, matched_question_id=matches[0], matched_count=1
    )


def aggregate(system: str, scores: list[QuestionScore], questions: QuestionSet) -> Aggregate:
    """Roll per-question scores into the report table."""
    answerable = [s for s in scores if s.expected_behavior.should_answer]
    ambiguous = [s for s in scores if not s.expected_behavior.should_answer]

    # Execution accuracy: correct values over ALL answerable questions, so refusing an
    # answerable question counts against it. Otherwise "refuse everything" scores 100%.
    value_correct = sum(1 for s in answerable if s.value_correct is True)

    semantic_pool = [s for s in answerable if s.metric_correct is not None]
    semantic_correct = sum(1 for s in semantic_pool if s.metric_correct)

    shape_pool = [s for s in answerable if s.shape_correct is not None]
    shape_correct = sum(1 for s in shape_pool if s.shape_correct)

    correct_attempts = sum(
        1 for s in answerable if s.outcome is Outcome.ANSWERED and s.value_correct is True
    )
    correct_refusals = sum(1 for s in ambiguous if s.outcome is Outcome.ASKED)
    under_refusals = sum(1 for s in ambiguous if s.outcome is Outcome.ANSWERED)
    over_refusals = sum(1 for s in answerable if s.outcome is Outcome.ASKED)

    fork_pool = [s for s in ambiguous if s.names_the_fork is not None]
    caveat_pool = [s for s in scores if s.caveats_surfaced is not None]

    silent_wrong = [s for s in scores if s.is_silent_wrong]
    caught = [
        s for s in scores if s.outcome is Outcome.ASKED and not s.expected_behavior.should_answer
    ]

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
            shape_accuracy=_rate(
                sum(1 for s in tier_answerable if s.shape_correct is True),
                sum(1 for s in tier_answerable if s.shape_correct is not None),
            ),
            semantic_accuracy=_rate(
                sum(1 for s in tier_semantic if s.metric_correct), len(tier_semantic)
            ),
            refusal_accuracy=_rate(tier_correct_attempts + tier_correct_refusals, len(subset)),
            behavior_accuracy=_rate(sum(1 for s in subset if behavior_correct(s)), len(subset)),
            silent_wrong_rate=_rate(sum(1 for s in subset if s.is_silent_wrong), len(subset)),
            confident_wrong_rate=_rate(sum(1 for s in subset if s.confident_wrong), len(subset)),
        )

    measured = [s for s in scores if not s.cached]
    return Aggregate(
        system=system,
        total_questions=len(scores),
        answerable=len(answerable),
        should_not_answer=len(ambiguous),
        execution_accuracy=_rate(value_correct, len(answerable)),
        shape_accuracy=_rate(shape_correct, len(shape_pool)),
        shape_measurable=len(shape_pool),
        semantic_accuracy=_rate(semantic_correct, len(semantic_pool)),
        semantic_measurable=len(semantic_pool),
        refusal_accuracy=_rate(correct_attempts + correct_refusals, len(scores)),
        behavior_accuracy=_rate(sum(1 for s in scores if behavior_correct(s)), len(scores)),
        under_refusal_rate=_rate(under_refusals, len(ambiguous)),
        over_refusal_rate=_rate(over_refusals, len(answerable)),
        names_the_fork_rate=_rate(sum(1 for s in fork_pool if s.names_the_fork), len(fork_pool)),
        caveat_surfaced_rate=_rate(
            sum(1 for s in caveat_pool if s.caveats_surfaced), len(caveat_pool)
        ),
        caveat_measurable=len(caveat_pool),
        silent_wrong_rate=_rate(len(silent_wrong), len(scores)),
        silent_wrong_count=len(silent_wrong),
        caught_wrong_count=len(caught),
        gate_flagged_count=sum(1 for s in scores if s.gate_flagged),
        attributable_wrong_count=sum(1 for s in scores if s.attribution == "attributable_wrong"),
        unattributable_wrong_count=sum(
            1 for s in scores if s.attribution == "unattributable_wrong"
        ),
        confident_wrong_rate=_rate(sum(1 for s in scores if s.confident_wrong), len(scores)),
        error_rate=_rate(sum(1 for s in scores if s.outcome is Outcome.ERRORED), len(scores)),
        # None unless EVERY unit was timed. A partially cached arm has no per-question
        # latency, so the field refuses to produce one rather than reporting a mean whose
        # denominator differs from its label.
        mean_latency_seconds=(
            round(sum(s.latency_seconds for s in measured) / len(measured), 3)
            if measured and len(measured) == len(scores)
            else None
        ),
        live_latency=_live_latency(scores),
        mean_cost_usd=(
            round(sum(s.cost_usd for s in measured) / len(measured), 6) if measured else None
        ),
        cached_fraction=_rate(sum(1 for s in scores if s.cached), len(scores)),
        per_tier=per_tier,
    )


def actual_behavior(score: QuestionScore) -> str:
    """What the system actually did, in the eval set's own vocabulary.

    Mapped to answer / clarify / refuse so target and actual are comparable in one
    matrix. A system that asks is doing the same *act* whether the question was merely
    underspecified or wholly unanswerable, so the distinction between clarify and refuse
    lives in what it said, not in whether it asked: a refusal names the metric as
    undefined, a clarification names a fork it could resolve.
    """
    if score.outcome is Outcome.ERRORED:
        return "error"
    if score.outcome is Outcome.ANSWERED:
        return "answer"
    asked = (score.clarifying_question or "").lower()
    declines = any(
        phrase in asked
        for phrase in ("not defined", "cannot answer", "not a defined", "does not exist")
    )
    return "refuse" if declines else "clarify"


def behavior_correct(score: QuestionScore) -> bool:
    """Whether the act matched the target exactly, on the 3x3 diagonal."""
    return actual_behavior(score) == score.expected_behavior.value


def behavior_matrix(scores: list[QuestionScore]) -> dict[str, dict[str, int]]:
    """Target behaviour x actual behaviour, 3x3 plus an error column.

    Not collapsed to 2x2, because the off-diagonals mean different things:
    answer-when-should-clarify is a trust failure -- a number nobody should act on --
    while refuse-when-should-clarify is a usability failure, annoying and safe. Summing
    them into one "refusal accuracy" hides which kind of system you have.
    """
    targets = ("answer", "clarify", "refuse")
    actuals = ("answer", "clarify", "refuse", "error")
    matrix = {target: dict.fromkeys(actuals, 0) for target in targets}
    for score in scores:
        target = score.expected_behavior.value
        if target in matrix:
            matrix[target][actual_behavior(score)] += 1
    return matrix


PAIRED_DISCRIMINATIONS: tuple[tuple[str, str, str], ...] = (
    (
        "avg_order_value grain",
        "t3-08",
        "am-13",
    ),
    (
        "payment_type fan-out",
        "t2-12",
        "am-14",
    ),
    (
        "reported_revenue restatement caveat",
        "t3-02",
        "t3-03",
    ),
)
"""Deliberately paired items, scored as both-halves-correct.

Each pair is one question a reflex answers wrongly: the two halves look alike and
require opposite responses. A system that always answers gets the first half of each
pair and fails the second; one that always asks does the reverse. Only a system that
actually consulted the definitions gets both, which is why this number is the one immune
to a reflex strategy.
"""


def paired_discrimination(scores: list[QuestionScore]) -> dict[str, Any]:
    """Score each pair as a single item: both halves right, or the pair is wrong."""
    by_id = {score.question_id: score for score in scores}
    results: list[dict[str, Any]] = []
    for label, left_id, right_id in PAIRED_DISCRIMINATIONS:
        left, right = by_id.get(left_id), by_id.get(right_id)
        if left is None or right is None:
            continue

        def half_correct(score: QuestionScore) -> bool:
            """One half is right only if it is right in every way that pair tests.

            The caveat clause matters: the restatement pair is two `answer` cases, so
            behaviour alone cannot separate them. What separates them is the caveat
            firing on the range that spans the definition change and staying silent on
            the range that does not. Without this clause an always-answer system won
            that pair, which would have made the headline paired number overstate.
            """
            if not behavior_correct(score):
                return False
            if score.caveats_surfaced is False:
                return False
            if score.expected_behavior.should_answer:
                return score.value_correct is True and score.metric_correct is not False
            return True

        results.append(
            {
                "pair": label,
                "left": left_id,
                "right": right_id,
                "left_correct": half_correct(left),
                "right_correct": half_correct(right),
                "both_correct": half_correct(left) and half_correct(right),
            }
        )
    scored = results
    return {
        "pairs": results,
        "pairs_scored": len(scored),
        "both_correct": sum(1 for item in scored if item["both_correct"]),
        "rate": _rate(sum(1 for item in scored if item["both_correct"]), len(scored)),
    }


def confident_wrong_curve(scores: list[QuestionScore]) -> dict[str, float | None]:
    """Confident-wrong rate across thresholds, so 0.7 is inspectable, not load-bearing."""
    return {
        f"{threshold:.1f}": _rate(
            sum(1 for s in scores if s.is_wrong_answer and s.confidence >= threshold),
            len(scores),
        )
        for threshold in (0.5, 0.6, 0.7, 0.8, 0.9)
    }
