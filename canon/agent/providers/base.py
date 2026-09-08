"""The provider boundary: one protocol, several free-tier backends.

Canon's design never depended on a particular model — the LLM fills a typed
``QueryIntent`` and writes no SQL — so swapping the backend should be a configuration
change rather than a rewrite. This module is where that claim gets cashed.

Two things here exist because the benchmark has to be *reproducible for $0*, which makes
quota a first-class concern rather than an operational detail:

* every provider declares its free-tier ``RateLimits``, so the runner can project
  wall-clock and refuse to start a sweep it cannot finish;
* every provider exposes a ``config_hash`` covering model id, limits and decoding
  settings, which goes into run provenance. A number produced under one configuration is
  not comparable to a number produced under another, and the hash is what makes that
  checkable instead of assumed.

**Note on the protocol signature.** ``complete_structured(prompt, schema)`` returns a
``StructuredResult`` rather than a bare ``BaseModel``. The parsed object alone would drop
token usage, and usage is the binding constraint on a free tier — the runner cannot pace
itself against a budget it cannot observe. The parsed model is on ``.parsed``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

_FROZEN = ConfigDict(frozen=True, extra="forbid")

ModelT = TypeVar("ModelT", bound=BaseModel)


class Prompt(BaseModel):
    """A system/user split, kept explicit rather than concatenated.

    The split is load-bearing: the system half is the stable prefix that prompt caching
    and the two-stage token measurement both key on, and folding it into one string would
    make both impossible.
    """

    model_config = _FROZEN

    system: str
    user: str

    @property
    def combined(self) -> str:
        """Everything the model reads, for token counting."""
        return f"{self.system}\n\n{self.user}"


class TokenUsage(BaseModel):
    """What one call consumed. Zeroed on a cache hit."""

    model_config = _FROZEN

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    latency_seconds: float = 0.0
    throttled_seconds: float = 0.0
    """Time spent waiting on the rate limiter or backing off. Reported separately so
    wall-clock projections can be checked against reality."""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class StructuredResult[ModelT: BaseModel](BaseModel):
    """A validated object plus what it cost to obtain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parsed: ModelT
    usage: TokenUsage
    raw_text: str | None = None
    """The unparsed response, kept for the failure taxonomy. A schema violation is
    evidence about the provider, and discarding it would make that unexaminable."""


class RateLimits(BaseModel):
    """A provider's free-tier ceilings.

    Daily caps matter more than per-minute ones here. At roughly 23k tokens a question,
    a 200k/day token budget is about eight questions per system per day — so the
    constraint that shapes the sweep plan is ``tokens_per_day``, not ``requests_per_minute``.
    """

    model_config = _FROZEN

    requests_per_minute: int = Field(gt=0)
    tokens_per_minute: int | None = Field(default=None, gt=0)
    requests_per_day: int | None = Field(default=None, gt=0)
    tokens_per_day: int | None = Field(default=None, gt=0)

    source: str
    """Where these figures came from. Surfaced in the dry-run because a wall-clock
    projection is only as trustworthy as the limits it was computed from."""

    verified: bool = False
    """True only when EVERY field has been confirmed against the provider itself.

    Deliberately all-or-nothing, so a partially-verified set cannot be presented as
    verified. Per-field status lives in ``verified_note``, because in practice the fields
    have different evidence: a provider may quote its own per-minute limit while
    refusing a call, and never say anything about the daily one.
    """

    verified_note: str = ""
    """Per-field provenance, in prose. Surfaced in the dry-run alongside any projection
    built on these numbers."""


class ProviderConfig(BaseModel):
    """Everything that could change an answer, in one hashable object."""

    model_config = _FROZEN

    name: str
    model_id: str
    context_limit: int = Field(gt=0)
    max_output_tokens: int = Field(default=8192, gt=0)
    temperature: float | None = None
    """None where the provider rejects it (current Anthropic models do). Recorded either
    way so a run is reproducible."""

    reasoning_effort: str | None = None
    limits: RateLimits
    notes: str = ""

    @property
    def config_hash(self) -> str:
        """Fingerprint of the decoding configuration, for run provenance.

        Deliberately excludes the rate limits: throttling changes how long a run takes,
        not what it answers. Including them would make two identical results look
        incomparable because a quota was raised.
        """
        payload = json.dumps(
            {
                "name": self.name,
                "model_id": self.model_id,
                "context_limit": self.context_limit,
                "max_output_tokens": self.max_output_tokens,
                "temperature": self.temperature,
                "reasoning_effort": self.reasoning_effort,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@runtime_checkable
class LLMProvider(Protocol):
    """One structured call. No SQL, no free text, no retries hidden inside."""

    config: ProviderConfig

    @property
    def name(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    @property
    def context_limit(self) -> int: ...

    def complete_structured(self, prompt: Prompt, schema: type[ModelT]) -> StructuredResult[ModelT]:
        """Fill ``schema`` from ``prompt``, or raise ``ProviderError``."""
        ...

    def complete_text(self, prompt: Prompt) -> StructuredResult[TextResponse]:
        """Free-text completion. Used only by the naive baseline, which writes SQL."""
        ...


class TextResponse(BaseModel):
    """Wrapper so the free-text path returns the same shape as the structured one."""

    model_config = _FROZEN

    text: str


class ProviderError(RuntimeError):
    """The provider did not return a usable object.

    Carries whether a retry could plausibly help, so the runner can distinguish a
    transient 429 from a schema the model cannot satisfy — retrying the latter just
    burns quota.
    """

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class QuotaExhausted(ProviderError):
    """The daily free-tier budget is spent.

    Not a failure of the run: the sweep checkpoints and resumes tomorrow. Distinct from
    a transient 429, which backoff can clear within one session.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)
