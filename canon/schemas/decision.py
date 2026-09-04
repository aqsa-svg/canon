"""What the model returns. One structured call, no free text, no SQL.

The model does two things at once here, and both are needed for the refusal path to
work: it fills a ``QueryIntent`` when the question is answerable, and it *scores the
candidate metrics* either way. The scores are what make ambiguity measurable — a
question that could mean two metrics produces two close scores, and a deterministic
policy in ``canon/agent/policy.py`` turns that into a clarifying question.

Asking for the scores in the same call is deliberate. A second "are you sure?" call
would cost another request, and worse, it would let the model revise a judgement it
already made, which is not the same as expressing uncertainty about it.

Note what is *not* here: no confidence field. The model is not asked to rate its own
confidence, because a self-rated number is uncalibrated and unfalsifiable. Confidence is
derived from the score distribution over metrics, which is grounded in the one judgement
the model actually made.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from canon.schemas.intent import QueryIntent
from canon.semantic.enums import MetricName

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class MetricCandidate(BaseModel):
    """One metric the question might mean, and how well it fits."""

    model_config = _FROZEN

    metric: MetricName
    """An enum member. The model cannot score a metric that does not exist."""

    score: float = Field(ge=0.0, le=1.0)
    """How well this metric matches what the question asked for.

    Not a probability and not calibrated in any formal sense; what matters is the
    *ordering* and the *gap* between the top two, which is what the refusal policy
    reads.
    """

    why: str
    """One sentence. Shown in the decision log so a scoring choice can be audited."""


class AgentDecision(BaseModel):
    """The model's complete output for one question."""

    model_config = _FROZEN

    reading: str
    """The model's paraphrase of the question, in its own words.

    Kept because a wrong answer with a correct reading is a different bug from a wrong
    answer with a wrong reading, and the decision log should distinguish them.
    """

    candidates: tuple[MetricCandidate, ...] = Field(min_length=1)
    """Every plausible metric, scored, best first is not required -- the policy sorts."""

    intent: QueryIntent | None = None
    """Filled when the question can be answered as asked. ``None`` is the model
    declining to commit, which the policy treats as a vote to clarify."""

    missing_information: tuple[str, ...] = ()
    """Things the question needs to pin down and does not: which revenue metric, which
    date range, whose definition of "active". Each entry names one gap."""

    proposed_clarification: str | None = None
    """The model's suggested question to ask. Advisory: the policy decides *whether* to
    ask, and prefers a clarification built from the semantic layer when it has one."""

    @model_validator(mode="after")
    def _check_coherence(self) -> AgentDecision:
        if len(self.candidates) != len({item.metric for item in self.candidates}):
            raise ValueError("candidates must not score the same metric twice")
        if self.intent is None and not (self.missing_information or self.proposed_clarification):
            raise ValueError(
                "declining to produce an intent requires saying what is missing: fill "
                "missing_information or proposed_clarification"
            )
        return self

    @property
    def ranked(self) -> tuple[MetricCandidate, ...]:
        """Candidates by descending score, then by name so ties are not order-dependent."""
        return tuple(sorted(self.candidates, key=lambda item: (-item.score, str(item.metric))))

    @property
    def top(self) -> MetricCandidate:
        return self.ranked[0]

    @property
    def score_gap(self) -> float:
        """Distance between the best and second-best metric.

        ``1.0`` when only one metric was plausible: nothing to be confused with.
        """
        ranked = self.ranked
        if len(ranked) == 1:
            return 1.0
        return ranked[0].score - ranked[1].score
