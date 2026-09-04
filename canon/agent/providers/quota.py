"""Rate limiting and a daily quota ledger that survives process restarts.

Two separate mechanisms, because the constraints have different shapes:

**Token buckets** handle per-minute ceilings (RPM and TPM). In-memory is fine — a minute
window does not outlive a process in any way that matters.

**The ledger** handles per-day ceilings, and it must be on disk. At roughly 23k tokens a
question against a 200k/day budget, an 80-question sweep does not fit in one day, so the
runner has to know at start-up what it already spent *yesterday and today* and stop
before the provider starts refusing. A limiter that forgot across restarts would make a
resumable sweep impossible, which is the whole point.

The ledger is keyed by UTC date. That is a deliberate simplification: providers reset on
their own schedules, and assuming UTC midnight can only make the limiter more
conservative than the provider, never less.
"""

from __future__ import annotations

import json
import random
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from canon.agent.providers.base import QuotaExhausted, RateLimits

_FROZEN = ConfigDict(frozen=True, extra="forbid")

DEFAULT_LEDGER = Path(__file__).resolve().parents[3] / "data" / "quota_ledger.json"

MAX_BACKOFF_SECONDS = 120.0
MAX_ATTEMPTS = 5


class ThrottleEvent(BaseModel):
    """One wait, logged so a wall-clock projection can be checked against reality."""

    model_config = _FROZEN

    provider: str
    reason: str
    waited_seconds: float
    at: str


class DailyUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests: int = 0
    tokens: int = 0


class QuotaLedger:
    """Persistent per-provider, per-UTC-day usage.

    Written after every call rather than at the end of a run: a sweep that dies halfway
    must not resume believing it has a full day's budget left.
    """

    def __init__(self, path: Path = DEFAULT_LEDGER) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, DailyUsage]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._data = {
            provider: {day: DailyUsage(**usage) for day, usage in days.items()}
            for provider, days in raw.items()
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            provider: {day: usage.model_dump() for day, usage in days.items()}
            for provider, days in self._data.items()
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def today() -> str:
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def usage(self, provider: str, day: str | None = None) -> DailyUsage:
        return self._data.get(provider, {}).get(day or self.today(), DailyUsage())

    def record(self, provider: str, *, tokens: int, requests: int = 1) -> None:
        with self._lock:
            day = self.today()
            bucket = self._data.setdefault(provider, {}).setdefault(day, DailyUsage())
            bucket.requests += requests
            bucket.tokens += tokens
            self._save()

    def remaining(self, provider: str, limits: RateLimits) -> tuple[int | None, int | None]:
        """Requests and tokens left today, or None where the provider sets no daily cap."""
        used = self.usage(provider)
        requests_left = (
            None
            if limits.requests_per_day is None
            else max(0, limits.requests_per_day - used.requests)
        )
        tokens_left = (
            None if limits.tokens_per_day is None else max(0, limits.tokens_per_day - used.tokens)
        )
        return requests_left, tokens_left


class _Bucket:
    """A token bucket that refills continuously.

    Continuous refill rather than a fixed window because a window boundary lets a burst
    of 2x the limit through, which is exactly what trips a provider's own limiter.
    """

    def __init__(self, capacity: float, per_seconds: float) -> None:
        self.capacity = float(capacity)
        self.per_seconds = float(per_seconds)
        self.level = float(capacity)
        self.updated = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.updated
        self.level = min(self.capacity, self.level + elapsed * self.capacity / self.per_seconds)
        self.updated = now

    def wait_time(self, amount: float) -> float:
        self._refill()
        if self.level >= amount:
            return 0.0
        deficit = amount - self.level
        return deficit * self.per_seconds / self.capacity

    def consume(self, amount: float) -> None:
        self._refill()
        self.level = max(0.0, self.level - amount)


class RateLimiter:
    """Paces calls against a provider's published limits, and logs every wait."""

    def __init__(
        self,
        *,
        provider: str,
        limits: RateLimits,
        ledger: QuotaLedger | None = None,
        sleep: object | None = None,
    ) -> None:
        self.provider = provider
        self.limits = limits
        self.ledger = ledger if ledger is not None else QuotaLedger()
        self.events: list[ThrottleEvent] = []
        self._sleep = sleep if callable(sleep) else time.sleep
        self._requests = _Bucket(limits.requests_per_minute, 60.0)
        self._tokens = (
            _Bucket(limits.tokens_per_minute, 60.0)
            if limits.tokens_per_minute is not None
            else None
        )

    # -- daily budget -----------------------------------------------------------
    def check_daily_budget(self, estimated_tokens: int) -> None:
        """Raise ``QuotaExhausted`` if this call would exceed today's cap.

        Checked *before* the call, using the dry-run's own token estimate. Discovering
        the ceiling by being refused wastes a request and tells you nothing you could not
        have computed.
        """
        requests_left, tokens_left = self.ledger.remaining(self.provider, self.limits)
        if requests_left is not None and requests_left <= 0:
            raise QuotaExhausted(
                f"{self.provider}: daily request cap ({self.limits.requests_per_day}) "
                "reached. The sweep is checkpointed -- resume tomorrow."
            )
        if tokens_left is not None and estimated_tokens > tokens_left:
            raise QuotaExhausted(
                f"{self.provider}: this call needs ~{estimated_tokens:,} tokens and only "
                f"{tokens_left:,} remain of the {self.limits.tokens_per_day:,}/day budget. "
                "The sweep is checkpointed -- resume tomorrow."
            )

    # -- per-minute pacing ------------------------------------------------------
    def acquire(self, estimated_tokens: int) -> float:
        """Block until this call fits both per-minute buckets. Returns seconds waited."""
        self.check_daily_budget(estimated_tokens)

        waited = 0.0
        request_wait = self._requests.wait_time(1.0)
        token_wait = (
            self._tokens.wait_time(float(estimated_tokens)) if self._tokens is not None else 0.0
        )
        delay = max(request_wait, token_wait)
        if delay > 0:
            reason = "tokens_per_minute" if token_wait >= request_wait else "requests_per_minute"
            self._log(reason, delay)
            self._sleep(delay)
            waited = delay

        self._requests.consume(1.0)
        if self._tokens is not None:
            self._tokens.consume(float(estimated_tokens))
        return waited

    def record_actual(self, tokens: int) -> None:
        """Book real usage against the daily ledger after a call returns."""
        self.ledger.record(self.provider, tokens=tokens)

    # -- backoff ----------------------------------------------------------------
    def backoff(self, attempt: int, retry_after: float | None = None) -> float:
        """Sleep before retrying a 429, honouring ``retry-after`` when the provider sends it.

        Jittered because a fixed schedule makes several workers retry in lockstep and
        re-trip the same limit.
        """
        if retry_after is not None:
            delay = min(retry_after, MAX_BACKOFF_SECONDS)
        else:
            delay = min(MAX_BACKOFF_SECONDS, (2.0**attempt))
        delay += random.uniform(0, min(1.0, delay * 0.25))
        self._log(f"429_backoff_attempt_{attempt}", delay)
        self._sleep(delay)
        return delay

    def _log(self, reason: str, seconds: float) -> None:
        self.events.append(
            ThrottleEvent(
                provider=self.provider,
                reason=reason,
                waited_seconds=round(seconds, 3),
                at=datetime.now(UTC).isoformat(),
            )
        )

    @property
    def total_throttled_seconds(self) -> float:
        return round(sum(event.waited_seconds for event in self.events), 2)


class Projection(BaseModel):
    """What a sweep would cost in quota and wall-clock, computed without spending any."""

    model_config = _FROZEN

    provider: str
    calls: int
    prompt_tokens: int
    estimated_completion_tokens: int
    total_tokens: int
    tokens_per_call: float

    days_required: float | None = None
    """Driven by the daily token cap, which is the binding constraint on a free tier."""

    minutes_from_rpm: float
    minutes_from_tpm: float | None = None
    wall_clock_minutes: float
    fits_in_one_day: bool
    requests_per_day_ok: bool = True
    tokens_remaining_today: int | None = None
    accuracy: str = "exact"
    limits_verified: bool = False
    limits_source: str = ""


def project(
    *,
    provider: str,
    limits: RateLimits,
    calls: int,
    prompt_tokens: int,
    completion_tokens_per_call: int,
    accuracy: str,
    tokens_remaining_today: int | None = None,
) -> Projection:
    """Turn a token count into days and minutes, given a provider's ceilings."""
    completion_total = calls * completion_tokens_per_call
    total = prompt_tokens + completion_total

    days = None
    if limits.tokens_per_day:
        days = round(total / limits.tokens_per_day, 2)

    minutes_rpm = round(calls / limits.requests_per_minute, 2)
    minutes_tpm = round(total / limits.tokens_per_minute, 2) if limits.tokens_per_minute else None
    wall_clock = max(minutes_rpm, minutes_tpm or 0.0)

    return Projection(
        provider=provider,
        calls=calls,
        prompt_tokens=prompt_tokens,
        estimated_completion_tokens=completion_total,
        total_tokens=total,
        tokens_per_call=round(total / calls, 1) if calls else 0.0,
        days_required=days,
        minutes_from_rpm=minutes_rpm,
        minutes_from_tpm=minutes_tpm,
        wall_clock_minutes=round(wall_clock, 2),
        fits_in_one_day=(days is None or days <= 1.0),
        requests_per_day_ok=(limits.requests_per_day is None or calls <= limits.requests_per_day),
        tokens_remaining_today=tokens_remaining_today,
        accuracy=accuracy,
        limits_verified=limits.verified,
        limits_source=limits.source,
    )
