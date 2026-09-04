"""The LLM boundary: a provider, a rate limiter, a disk cache, and nothing else.

This used to be Anthropic-specific. It now delegates to whichever ``LLMProvider`` is
configured, which is what makes a $0 reproduction possible: the default is Groq's free
tier, and the paid Anthropic path is one flag away.

Four things happen here, and they are worth separating:

**Provider selection.** By name, from ``providers.yaml``. The provider's name, model id
and config hash travel with every ``Usage``, so a scored result can always be traced to
the configuration that produced it.

**Rate limiting.** Before each call the limiter checks the persistent daily ledger and
paces against RPM and TPM. A call that would exceed the daily budget raises
``QuotaExhausted`` rather than being sent and refused -- discovering a ceiling by being
rejected wastes a request and tells you nothing you could not have computed.

**Retries.** Only for errors the provider marked retryable, with jittered backoff, and
counted in ``Usage.attempts``. A retry hidden further down would be a second call the
benchmark reports as one.

**Caching.** Keyed on everything that could change the answer -- provider, model, config
hash, prompt, schema -- including the run index, so repeat runs are independent samples
rather than replays.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Final, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from canon.agent.providers import (
    DEFAULT_PROVIDER,
    LLMProvider,
    Prompt,
    ProviderError,
    QuotaExhausted,
    QuotaLedger,
    RateLimiter,
    TextResponse,
    build_provider,
    count_call,
    is_free_tier,
    load_config,
    retry_after_seconds,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent

# Explicit path rather than find_dotenv(): the latter walks the caller's frame stack and
# fails outright when the caller is stdin, which is how most ad-hoc checks run.
load_dotenv(REPO_ROOT / ".env", override=False)

CACHE_DIR: Final[Path] = REPO_ROOT / "data" / "llm_cache"
MAX_ATTEMPTS: Final[int] = 4

_FROZEN = ConfigDict(frozen=True, extra="forbid")

ModelT = TypeVar("ModelT", bound=BaseModel)

MODEL_ID: Final[str] = load_config(DEFAULT_PROVIDER).model_id
"""The default provider's model, as a module constant because provenance and several
call sites reference it by this name."""

_PRICE_PER_MTOK: Final[dict[str, tuple[float, float]]] = {
    # (input, output) USD per million tokens. Free tiers are zero, and stating that
    # explicitly is the point of the exercise rather than an omission.
    "groq": (0.0, 0.0),
    "gemini": (0.0, 0.0),
    "anthropic": (5.0, 25.0),
}


class Usage(BaseModel):
    """What one call consumed, plus which configuration produced it."""

    model_config = _FROZEN

    provider: str = ""
    model_id: str = ""
    config_hash: str = ""

    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    latency_seconds: float = 0.0
    throttled_seconds: float = 0.0
    attempts: int = 1
    """Calls actually made, including retries. Greater than 1 means a retryable error was
    hit, and the benchmark should not silently report that as one call."""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """Zero on a free tier, which is the constraint this project now runs under."""
        rates = _PRICE_PER_MTOK.get(self.provider, (0.0, 0.0))
        return (self.input_tokens * rates[0] + self.output_tokens * rates[1]) / 1_000_000

    @property
    def free(self) -> bool:
        return is_free_tier(self.provider)


class LlmError(RuntimeError):
    """The call did not produce a usable structured object.

    Distinct from a refusal by the analytics system: this is the model or the transport
    failing, and the harness records it as an error rather than scoring it as an answer.
    """


class CallContext(BaseModel):
    """Identifies *which* call this is, not just what was asked.

    Without this the cache is keyed on content alone, so asking the same question three
    times returns the same stored answer three times. That would make runs 2 and 3
    replays of run 1 and any reported variance identically zero -- a number that looks
    like a measurement and is an artefact of the cache.
    """

    model_config = _FROZEN

    system: str
    """Which system under test -- 'canon' or 'baseline'. Keeps their caches disjoint even
    where the question text is identical."""

    question_id: str
    run_index: int = Field(ge=0)
    stage: str = ""
    """Which stage of the two-stage prompt this is. Two calls for one question must not
    collide in the cache."""


def _cache_key(
    *,
    provider: str,
    model_id: str,
    config_hash: str,
    system: str,
    question: str,
    schema_name: str,
    schema: str,
    context: CallContext | None,
) -> str:
    payload = json.dumps(
        {
            "provider": provider,
            "model_id": model_id,
            "config_hash": config_hash,
            "schema_name": schema_name,
            "schema": schema,
            "system": system,
            "question": question,
            # Omitted entirely when there is no context, so the demo and ad-hoc use keep
            # content-only keying and get cache hits, which is what you want there.
            "context": context.model_dump(mode="json") if context else None,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LlmClient:
    """One structured call per question, paced against a free-tier budget."""

    def __init__(
        self,
        *,
        provider: str | LLMProvider = DEFAULT_PROVIDER,
        cache_dir: Path = CACHE_DIR,
        use_cache: bool = True,
        ledger: QuotaLedger | None = None,
    ) -> None:
        self.provider: LLMProvider = (
            build_provider(provider) if isinstance(provider, str) else provider
        )
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(
            provider=self.provider.name,
            limits=self.provider.config.limits,
            ledger=ledger,
        )

    # -- identity ---------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def model(self) -> str:
        return self.provider.model_id

    @property
    def config_hash(self) -> str:
        return self.provider.config.config_hash

    @property
    def effort(self) -> str:
        """Kept for provenance compatibility; not every provider has the concept."""
        return self.provider.config.reasoning_effort or "n/a"

    def _blank_usage(self, cached: bool = False) -> Usage:
        return Usage(
            provider=self.provider.name,
            model_id=self.provider.model_id,
            config_hash=self.config_hash,
            cached=cached,
        )

    # -- the call ---------------------------------------------------------------
    def _dispatch(
        self, prompt: Prompt, schema: type[ModelT] | None, estimated_tokens: int
    ) -> tuple[ModelT | TextResponse, Usage]:
        """Rate-limit, call, retry retryable errors, and book the usage."""
        throttled = self.limiter.acquire(estimated_tokens)
        last: ProviderError | None = None

        for attempt in range(MAX_ATTEMPTS):
            try:
                result = (
                    self.provider.complete_structured(prompt, schema)
                    if schema is not None
                    else self.provider.complete_text(prompt)
                )
            except QuotaExhausted:
                # Not a failure of the run: the sweep checkpoints and resumes tomorrow.
                raise
            except ProviderError as error:
                last = error
                if not error.retryable or attempt == MAX_ATTEMPTS - 1:
                    break
                throttled += self.limiter.backoff(attempt, retry_after_seconds(error))
                continue

            observed = result.usage
            # Book what the provider actually charged, falling back to the local estimate
            # when it reported nothing -- an unbooked call would let the ledger drift under
            # real spend and overrun the daily cap.
            self.limiter.record_actual(observed.total_tokens or estimated_tokens)
            return result.parsed, Usage(
                provider=self.provider.name,
                model_id=self.provider.model_id,
                config_hash=self.config_hash,
                input_tokens=observed.prompt_tokens,
                output_tokens=observed.completion_tokens,
                latency_seconds=observed.latency_seconds,
                throttled_seconds=round(throttled, 3),
                attempts=attempt + 1,
            )

        # Every attempt failed. Book the estimate anyway: the requests were made.
        self.limiter.record_actual(estimated_tokens)
        raise LlmError(str(last) if last else "the provider returned no usable response")

    def structured(
        self,
        *,
        system: str,
        question: str,
        output_format: type[ModelT],
        context: CallContext | None = None,
    ) -> tuple[ModelT, Usage]:
        """One call, one validated object. No retry on a schema violation.

        A schema-invalid response is information about the system under test rather than
        noise to smooth away, and retrying it burns quota to hide a finding.
        """
        schema_json = json.dumps(output_format.model_json_schema(), sort_keys=True)
        path = self.cache_dir / (
            _cache_key(
                provider=self.provider.name,
                model_id=self.provider.model_id,
                config_hash=self.config_hash,
                system=system,
                question=question,
                schema_name=output_format.__name__,
                schema=schema_json,
                context=context,
            )
            + ".json"
        )
        if self.use_cache and path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            return output_format.model_validate_json(cached["raw_json"]), self._blank_usage(True)

        estimated = count_call(
            system=system, user=question, schema_json=schema_json, provider=self.provider.name
        ).tokens
        parsed, usage = self._dispatch(
            Prompt(system=system, user=question), output_format, estimated
        )
        if not isinstance(parsed, output_format):  # pragma: no cover - provider contract
            raise LlmError(f"provider returned {type(parsed).__name__}, expected {output_format}")

        if self.use_cache:
            self._write_cache(path, parsed.model_dump_json(), usage)
        return parsed, usage

    def freeform(
        self, *, system: str, question: str, context: CallContext | None = None
    ) -> tuple[str, Usage]:
        """A plain text response. Used only by the naive baseline, which writes SQL."""
        path = self.cache_dir / (
            _cache_key(
                provider=self.provider.name,
                model_id=self.provider.model_id,
                config_hash=self.config_hash,
                system=system,
                question=question,
                schema_name="freeform",
                schema="",
                context=context,
            )
            + ".json"
        )
        if self.use_cache and path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            return str(cached["raw_json"]), self._blank_usage(True)

        estimated = count_call(
            system=system, user=question, schema_json="", provider=self.provider.name
        ).tokens
        parsed, usage = self._dispatch(Prompt(system=system, user=question), None, estimated)
        text = parsed.text if isinstance(parsed, TextResponse) else str(parsed)
        if self.use_cache:
            self._write_cache(path, text, usage)
        return text, usage

    def _write_cache(self, path: Path, raw_json: str, usage: Usage) -> None:
        path.write_text(
            json.dumps(
                {
                    "raw_json": raw_json,
                    "provider": self.provider.name,
                    "model_id": self.provider.model_id,
                    "config_hash": self.config_hash,
                    "usage": usage.model_dump(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # -- budget -----------------------------------------------------------------
    def remaining_today(self) -> tuple[int | None, int | None]:
        """Requests and tokens left in today's free-tier budget."""
        return self.limiter.ledger.remaining(self.provider.name, self.provider.config.limits)

    def has_credentials(self) -> bool:
        """Whether a key for this provider is present, without making a call."""
        variables = {
            "groq": ("GROQ_API_KEY",),
            "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            "anthropic": ("ANTHROPIC_API_KEY",),
        }.get(self.provider.name, ())
        return any(os.environ.get(name) for name in variables)
