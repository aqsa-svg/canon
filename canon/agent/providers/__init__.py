"""Provider registry: load a backend by name from ``providers.yaml``.

One place that knows which providers exist, so adding a fourth is a YAML entry plus a
class rather than a search through the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import yaml

from canon.agent.providers.base import (
    LLMProvider,
    Prompt,
    ProviderConfig,
    ProviderError,
    QuotaExhausted,
    RateLimits,
    StructuredResult,
    TextResponse,
    TokenUsage,
)
from canon.agent.providers.implementations import (
    AnthropicProvider,
    GeminiProvider,
    GroqProvider,
    retry_after_seconds,
)
from canon.agent.providers.quota import (
    Projection,
    QuotaLedger,
    RateLimiter,
    ThrottleEvent,
    project,
)
from canon.agent.providers.tokens import CountAccuracy, TokenCount, count_call, count_tokens

CONFIG_PATH: Final[Path] = Path(__file__).with_name("providers.yaml")

DEFAULT_PROVIDER: Final[str] = "groq"
"""Groq is the default because the project must be reproducible for $0. The Anthropic
path still works and is selected explicitly."""

_CLASSES: Final[dict[str, type]] = {
    "groq": GroqProvider,
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
}

FREE_TIER: Final[frozenset[str]] = frozenset({"groq", "gemini"})
"""Providers that cost nothing to run. ``anthropic`` is excluded deliberately: a run on
it is not a $0-reproducible run, and the dry-run says so."""


@lru_cache(maxsize=1)
def _raw_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def available_providers() -> tuple[str, ...]:
    return tuple(sorted(_raw_config()))


@lru_cache(maxsize=8)
def load_config(name: str) -> ProviderConfig:
    """Parse and validate one provider's configuration."""
    raw = _raw_config()
    if name not in raw:
        raise KeyError(f"unknown provider {name!r}; known: {', '.join(available_providers())}")
    body = dict(raw[name])
    body["limits"] = RateLimits(**body["limits"])
    return ProviderConfig(**body)


def build_provider(name: str = DEFAULT_PROVIDER) -> LLMProvider:
    """Instantiate a provider. No network call and no key needed until first use."""
    config = load_config(name)
    provider_class = _CLASSES[name]
    return provider_class(config)  # type: ignore[no-any-return]


def is_free_tier(name: str) -> bool:
    return name in FREE_TIER


__all__ = [
    "CONFIG_PATH",
    "DEFAULT_PROVIDER",
    "FREE_TIER",
    "AnthropicProvider",
    "CountAccuracy",
    "GeminiProvider",
    "GroqProvider",
    "LLMProvider",
    "Projection",
    "Prompt",
    "ProviderConfig",
    "ProviderError",
    "QuotaExhausted",
    "QuotaLedger",
    "RateLimiter",
    "RateLimits",
    "StructuredResult",
    "TextResponse",
    "ThrottleEvent",
    "TokenCount",
    "TokenUsage",
    "available_providers",
    "build_provider",
    "count_call",
    "count_tokens",
    "is_free_tier",
    "load_config",
    "project",
    "retry_after_seconds",
]
