"""The eval set: typed, loaded from YAML, and cross-checked against the semantic layer.

A question file that names a metric the canonical layer does not define, or expects a
breakdown the metric card forbids, fails to load. The eval set is held to the same
standard as the code it scores — an eval with a wrong gold answer is worse than no eval,
because it looks like a measurement.

Gold values are not written by hand. ``scripts/verify_gold.py`` executes every gold SQL
against the warehouse and records what it returned, per ``docs/01-evaluation.md``: gold
SQL is human-written, and its value is whatever that SQL says.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from canon.semantic.cards import SemanticLayer

_FROZEN = ConfigDict(frozen=True, extra="forbid")

QUESTIONS_DIR = Path(__file__).resolve().parent.parent / "questions"


class Behavior(StrEnum):
    """What the system is supposed to do. The label the scorer reads, never the agent."""

    ANSWER = "answer"
    """A number is the correct output."""

    CLARIFY = "clarify"
    """Underspecified but answerable once resolved. Asking is correct; guessing fails."""

    REFUSE = "refuse"
    """Not answerable from this warehouse at all — the metric is not defined. Asking is
    correct, and substituting the nearest available metric is the failure this class
    exists to catch."""

    @property
    def should_answer(self) -> bool:
        return self is Behavior.ANSWER


class Tier(StrEnum):
    """Difficulty band, reported separately so a headline number cannot hide a cliff."""

    TIER_1 = "1"
    TIER_2 = "2"
    TIER_3 = "3"
    AMBIGUOUS = "ambiguous"


class GoldRow(BaseModel):
    """One row of the expected result, as executed from the gold SQL."""

    model_config = _FROZEN

    cells: tuple[Any, ...]
    """Any is unavoidable: a result row holds whatever DuckDB returned. Compared by the
    scorer's own type-aware rules rather than by Python equality alone."""


class Question(BaseModel):
    """One eval case."""

    model_config = _FROZEN

    id: str = Field(pattern=r"^[a-z0-9-]+$")
    question: str
    tier: Tier
    expected_behavior: Behavior
    notes: str
    """Why this question is in the set and what it is testing. Required: a case nobody
    can explain is a case nobody can maintain."""

    expected_metric: str | None = None
    """The canonical metric a correct answer must use. Required for `answer` cases; for
    `clarify` cases it names the metric the question resolves to *after* clarification,
    or is left unset."""

    expected_dimensions: tuple[str, ...] = ()
    expected_grain: str | None = None
    expected_comparison: str | None = None

    gold_sql: str | None = None
    """Hand-written, independent of the compiler. Required for `answer` cases."""

    gold_rows: tuple[GoldRow, ...] | None = None
    """Filled by scripts/verify_gold.py from a real execution. Never typed by hand."""

    must_mention: tuple[str, ...] = ()
    """Terms a correct clarifying question has to name — the actual fork.

    Used as an automatic proxy for rubric level 2 in docs/01-evaluation.md §3.3. It is a
    proxy, not the rubric: the harness also writes every clarifying question out for
    hand-grading, because "mentions both metrics" is necessary and not sufficient.
    """

    requires_caveat: tuple[str, ...] = ()
    """Substrings that must appear in the answer's caveats. This is how a tier-3 case
    asserts that a caveat was actually surfaced rather than merely defined."""

    @model_validator(mode="after")
    def _check_completeness(self) -> Question:
        if self.expected_behavior is Behavior.ANSWER:
            if not self.gold_sql:
                raise ValueError(f"{self.id}: an `answer` case needs gold_sql")
            if not self.expected_metric:
                raise ValueError(f"{self.id}: an `answer` case needs expected_metric")
        else:
            if self.gold_sql:
                raise ValueError(
                    f"{self.id}: a `{self.expected_behavior.value}` case must not carry "
                    "gold_sql -- there is no number to produce"
                )
            if not self.must_mention:
                raise ValueError(
                    f"{self.id}: a `{self.expected_behavior.value}` case needs "
                    "must_mention, or the clarifying question cannot be scored"
                )
        if self.expected_behavior is Behavior.REFUSE and self.expected_metric:
            raise ValueError(
                f"{self.id}: a `refuse` case has no expected_metric by definition -- the "
                "metric is not in the catalogue"
            )
        return self


class QuestionSet(BaseModel):
    """The whole eval set, validated against the semantic layer it is written for."""

    model_config = _FROZEN

    questions: tuple[Question, ...]

    @model_validator(mode="after")
    def _check_unique_ids(self) -> QuestionSet:
        ids = [item.id for item in self.questions]
        duplicates = {name for name in ids if ids.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate question ids: {sorted(duplicates)}")
        return self

    def validate_against(self, layer: SemanticLayer) -> None:
        """Fail loudly if the eval set disagrees with the canonical layer.

        Catches the eval-rot failure: a metric renamed in the YAML while the questions
        still expect the old name would otherwise show up as a mysterious accuracy drop.
        """
        problems: list[str] = []
        for item in self.questions:
            if item.expected_metric and item.expected_metric not in layer.metrics:
                problems.append(f"{item.id}: unknown expected_metric {item.expected_metric!r}")
            for dimension in item.expected_dimensions:
                if dimension not in layer.dimensions:
                    problems.append(f"{item.id}: unknown dimension {dimension!r}")
                elif (
                    item.expected_metric
                    and item.expected_metric in layer.metrics
                    and dimension not in layer.metrics[item.expected_metric].valid_dimensions
                    and item.expected_behavior is Behavior.ANSWER
                ):
                    problems.append(
                        f"{item.id}: expects {item.expected_metric} broken down by "
                        f"{dimension}, which that metric card forbids"
                    )
        if problems:
            listed = "\n  - ".join(problems)
            raise ValueError(f"the eval set disagrees with metrics.yaml:\n  - {listed}")

    def by_behavior(self, behavior: Behavior) -> tuple[Question, ...]:
        return tuple(item for item in self.questions if item.expected_behavior is behavior)

    def by_tier(self, tier: Tier) -> tuple[Question, ...]:
        return tuple(item for item in self.questions if item.tier is tier)

    @property
    def answerable(self) -> tuple[Question, ...]:
        return self.by_behavior(Behavior.ANSWER)

    @property
    def should_not_answer(self) -> tuple[Question, ...]:
        """Clarify and refuse cases together: the class where asking is correct."""
        return tuple(item for item in self.questions if not item.expected_behavior.should_answer)


def load_questions(directory: Path = QUESTIONS_DIR) -> QuestionSet:
    """Load every YAML file in the questions directory, sorted by filename."""
    entries: list[Question] = []
    for path in sorted(directory.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or []
        if not isinstance(raw, list):
            raise TypeError(f"{path} must contain a list of questions")
        for record in raw:
            entries.append(Question(**record))
    if not entries:
        raise ValueError(f"no questions found in {directory}")
    return QuestionSet(questions=tuple(entries))
