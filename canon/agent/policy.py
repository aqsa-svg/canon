"""When to answer and when to ask. Deterministic, tunable, and logged.

The model scores the candidate metrics; **this module decides**. That split is the point.
If the model decided, the refusal rate would be a property of its mood on the day and
tuning it would mean rewriting a prompt and hoping. Here the rule is a number in one
place, the decision is a record you can read, and changing the threshold re-scores the
whole benchmark without another API call.

Four rules produce a clarifying question, in priority order:

1. the compiler rejected the intent as illegal (wrong grain, fan-out, overlapping
   comparison) — the typed error already knows what the fork is;
2. the model declined to produce an intent at all;
3. the model listed information the question does not supply;
4. the top two metric scores are within ``metric_score_margin`` of each other.

Otherwise the question is answered. A clarifying question is only useful if it names the
actual fork and offers a way forward, so every one built here does both — see
``docs/01-evaluation.md`` §3.3 for the rubric this is written against.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from canon.schemas.decision import AgentDecision
from canon.schemas.errors import InvalidIntentDetail
from canon.semantic.cards import SemanticLayer

_FROZEN = ConfigDict(frozen=True, extra="forbid")

MAX_ALTERNATIVES = 3
"""How many alternatives a clarifying question offers.

Three at most. The compiler's ``suggestions`` list can name ten legal metrics, and a
question listing ten choices scores as a weak refusal under the rubric — it signals
confusion without resolving it.
"""


class PolicyRule(StrEnum):
    """Which rule decided. Recorded on every decision so the split is inspectable."""

    ANSWERED = "answered"
    ILLEGAL_INTENT = "compiler_rejected_intent"
    NO_INTENT = "model_declined_to_commit"
    MISSING_INFORMATION = "required_information_unspecified"
    CLOSE_CANDIDATES = "top_two_metrics_too_close"


class PolicyThresholds(BaseModel):
    """The tunable surface. Two numbers, both recorded in every run's provenance."""

    model_config = _FROZEN

    metric_score_margin: float = Field(default=0.15, ge=0.0, le=1.0)
    """Minimum gap between the best and second-best metric before answering.

    Tuned on a handful of examples rather than fitted to the benchmark, which would be
    training on the test set. Raising it trades over-refusals for under-refusals; the
    harness reports both rates separately so the trade is visible instead of hidden
    inside one accuracy number.
    """

    minimum_top_score: float = Field(default=0.45, ge=0.0, le=1.0)
    """Below this, even an uncontested metric is too weak a match to answer on.

    Catches the case where nothing in the catalogue really fits and the model picks the
    least-bad option — the substitution failure that a system without a semantic layer
    makes silently.
    """


class PolicyOutcome(StrEnum):
    ANSWER = "answer"
    CLARIFY = "clarify"


class RefusalDecision(BaseModel):
    """The full record of one ask-or-answer decision. Appended to a JSONL log."""

    model_config = _FROZEN

    question: str
    outcome: PolicyOutcome
    rule: PolicyRule
    confidence: float = Field(ge=0.0, le=1.0)
    top_metric: str
    top_score: float
    runner_up: str | None = None
    runner_up_score: float | None = None
    score_gap: float
    reading: str
    missing_information: tuple[str, ...] = ()
    clarifying_question: str | None = None
    thresholds: PolicyThresholds
    invalid_intent_reason: str | None = None

    @property
    def is_clarification(self) -> bool:
        return self.outcome is PolicyOutcome.CLARIFY


def _first_sentence(text: str) -> str:
    """The head of a metric description, for use inside a clarifying question."""
    cleaned = " ".join(text.split())
    for stop in (". ", "; "):
        if stop in cleaned:
            return cleaned.split(stop)[0].rstrip(".")
    return cleaned.rstrip(".")


def _nearest_alternatives(
    layer: SemanticLayer, attempted: str, suggestions: tuple[str, ...]
) -> tuple[str, ...]:
    """Trim a suggestion list to the closest few.

    "Closest" means sharing the attempted metric's base table, i.e. answering at the
    same grain — the alternatives a person would actually accept as substitutes. Falls
    back to alphabetical order so the output is deterministic either way.
    """
    if attempted not in layer.metrics:
        return suggestions[:MAX_ALTERNATIVES]

    base = layer.metrics[attempted].base
    same_grain = [
        name
        for name in suggestions
        if name in layer.metrics and layer.metrics[name].base == base and name != attempted
    ]
    others = [name for name in suggestions if name not in same_grain and name != attempted]
    return tuple((same_grain + others)[:MAX_ALTERNATIVES])


def _clarify_between_metrics(layer: SemanticLayer, decision: AgentDecision) -> str:
    """Ask which of the close-scoring metrics was meant, describing each."""
    contenders = [
        candidate
        for candidate in decision.ranked[:MAX_ALTERNATIVES]
        if str(candidate.metric) in layer.metrics
    ]
    described = [
        f"{candidate.metric} ({_first_sentence(layer.metrics[str(candidate.metric)].description)})"
        for candidate in contenders
    ]
    if len(described) == 1:
        return (
            f"I can compute {described[0]}, but I am not confident that is what you meant. Is it?"
        )
    joined = ", or ".join([", ".join(described[:-1]), described[-1]])
    return f"That could mean {joined}. Which did you have in mind?"


def _clarify_missing(
    layer: SemanticLayer, decision: AgentDecision, limits: PolicyThresholds
) -> str:
    """Ask about information the question does not supply.

    Prefers the model's own wording when it named a fork, because it saw the question;
    falls back to the listed gaps.

    Crucially, the metric options are offered **only when the best candidate is a
    plausible fit**. When nothing in the catalogue really matches -- someone asks for
    profit margin and the closest thing is net revenue at score 0.2 -- naming that
    metric would be proposing the substitution this project exists to refuse. In that
    case the clarification says the metric is not defined and stops there.
    """
    proposed = (decision.proposed_clarification or "").strip()
    if proposed and len(proposed) > 25:
        return proposed

    gaps = "; ".join(item.strip().rstrip(".") for item in decision.missing_information)
    plausible = decision.top.score >= limits.minimum_top_score

    if gaps and plausible:
        return f"I need one more thing before I can answer: {gaps}. " + _clarify_between_metrics(
            layer, decision
        )
    if gaps:
        return (
            f"I cannot answer that from the metrics I have defined: {gaps}. "
            "Could you restate it in terms of a metric that exists, or point me at "
            "where that definition lives?"
        )
    return _clarify_between_metrics(layer, decision)


def _clarify_illegal(layer: SemanticLayer, detail: InvalidIntentDetail) -> str:
    """Turn a typed compiler rejection into a question that offers a way forward."""
    alternatives = _nearest_alternatives(layer, detail.metric or "", detail.suggestions)
    question = detail.message.rstrip()
    if alternatives:
        listed = ", ".join(alternatives)
        return (
            f"{question} I can answer it with {listed} instead — which would you like, "
            "or would you rather I change the breakdown?"
        )
    return f"{question} Could you rephrase what you need?"


def decide(
    *,
    question: str,
    decision: AgentDecision,
    layer: SemanticLayer,
    invalid: InvalidIntentDetail | None = None,
    thresholds: PolicyThresholds | None = None,
) -> RefusalDecision:
    """Answer or ask, with the reason recorded.

    Pure: no network, no clock beyond the log timestamp added at write time. The same
    decision object and thresholds always give the same outcome, which is what makes
    re-scoring a cached benchmark run meaningful.
    """
    limits = thresholds or PolicyThresholds()
    ranked = decision.ranked
    top = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    gap = decision.score_gap

    def record(
        outcome: PolicyOutcome,
        rule: PolicyRule,
        confidence: float,
        clarifying_question: str | None,
    ) -> RefusalDecision:
        return RefusalDecision(
            question=question,
            outcome=outcome,
            rule=rule,
            confidence=round(min(0.95, max(0.0, confidence)), 4),
            top_metric=str(top.metric),
            top_score=top.score,
            runner_up=str(runner_up.metric) if runner_up else None,
            runner_up_score=runner_up.score if runner_up else None,
            score_gap=round(gap, 4),
            reading=decision.reading,
            missing_information=decision.missing_information,
            clarifying_question=clarifying_question,
            thresholds=limits,
            invalid_intent_reason=invalid.reason.value if invalid else None,
        )

    # 1. The compiler already proved this cannot be answered as asked.
    if invalid is not None:
        return record(
            PolicyOutcome.CLARIFY,
            PolicyRule.ILLEGAL_INTENT,
            0.9,
            _clarify_illegal(layer, invalid),
        )

    # 2. The model would not commit to an intent.
    if decision.intent is None:
        return record(
            PolicyOutcome.CLARIFY,
            PolicyRule.NO_INTENT,
            0.85,
            _clarify_missing(layer, decision, limits),
        )

    # 3. The model named something the question leaves open.
    if decision.missing_information:
        return record(
            PolicyOutcome.CLARIFY,
            PolicyRule.MISSING_INFORMATION,
            0.85,
            _clarify_missing(layer, decision, limits),
        )

    # 4. Two readings, too close to choose between.
    if gap < limits.metric_score_margin or top.score < limits.minimum_top_score:
        # Confidence in the *refusal*: the closer the scores, the more certain the
        # question is genuinely ambiguous.
        return record(
            PolicyOutcome.CLARIFY,
            PolicyRule.CLOSE_CANDIDATES,
            1.0 - gap,
            _clarify_between_metrics(layer, decision),
        )

    # Otherwise: answer, with confidence taken from the winning metric's own score.
    # Capped below 1.0 because nothing here justifies certainty.
    return record(PolicyOutcome.ANSWER, PolicyRule.ANSWERED, top.score, None)


def log_decision(decision: RefusalDecision, log_path: Path) -> None:
    """Append one decision as JSON. Inspect with ``jq`` or by reading it.

    A log rather than a metric: aggregate refusal rates say how often the system asked,
    and this says why it asked each time, which is what you need to tune the threshold.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = decision.model_dump(mode="json")
    record["logged_at"] = datetime.now(UTC).isoformat()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
