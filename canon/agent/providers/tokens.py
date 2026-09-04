"""Offline token counting, so the dry-run costs nothing.

A dry-run that called a provider's ``count_tokens`` endpoint to estimate quota would
spend the request budget it exists to protect. Everything here is local.

Accuracy differs by provider, and the difference is reported rather than smoothed over:

* **groq** — exact. ``openai/gpt-oss-120b`` uses the harmony tokenizer, which is
  tiktoken's ``o200k_harmony``, so the count is the count.
* **gemini** — approximate. There is no local SentencePiece vocabulary for it, so
  ``o200k_base`` stands in. In practice this over-counts English prose slightly, which
  is the safe direction for a budget projection.
* **anthropic** — approximate for the same reason. Anthropic's own
  ``messages.count_tokens`` is exact but needs a paid key, and never guessing Claude
  token counts from a third-party tokenizer is the standing rule; the figure is labelled
  as an estimate wherever it is shown.

Chat-completion overhead (role markers, tool-schema serialisation) is added as a flat
per-call allowance rather than modelled per provider. It is small next to a 23k-token
prompt and modelling it precisely would be false precision.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

import tiktoken
from pydantic import BaseModel, ConfigDict

_FROZEN = ConfigDict(frozen=True, extra="forbid")

MESSAGE_OVERHEAD_TOKENS = 24
"""Flat allowance for role markers and message framing, per call."""

SCHEMA_OVERHEAD_RATIO = 1.0
"""Tool/response schemas are counted verbatim: they are part of the prompt the provider
bills for, and on a 200k/day budget a 600-token schema per call is not a rounding error.
"""


class CountAccuracy(StrEnum):
    EXACT = "exact"
    APPROXIMATE = "approximate"


class TokenCount(BaseModel):
    """A count with its own reliability attached."""

    model_config = _FROZEN

    tokens: int
    accuracy: CountAccuracy
    encoding: str
    note: str = ""


_ENCODING_FOR_PROVIDER: dict[str, tuple[str, CountAccuracy, str]] = {
    "groq": (
        "o200k_harmony",
        CountAccuracy.EXACT,
        "GPT-OSS harmony tokenizer; counts are exact",
    ),
    "gemini": (
        "o200k_base",
        CountAccuracy.APPROXIMATE,
        "no local SentencePiece vocabulary; o200k_base proxy, tends to over-count English",
    ),
    "anthropic": (
        "o200k_base",
        CountAccuracy.APPROXIMATE,
        "Claude token counts are only exact via messages.count_tokens, which needs a paid key",
    ),
}


@lru_cache(maxsize=8)
def _encoder(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def count_tokens(text: str, provider: str) -> TokenCount:
    """Tokens in ``text`` under ``provider``'s tokenizer, or the closest local proxy."""
    encoding_name, accuracy, note = _ENCODING_FOR_PROVIDER.get(
        provider, ("o200k_base", CountAccuracy.APPROXIMATE, "unknown provider; generic proxy")
    )
    encoder = _encoder(encoding_name)
    return TokenCount(
        tokens=len(encoder.encode(text, disallowed_special=())),
        accuracy=accuracy,
        encoding=encoding_name,
        note=note,
    )


def count_call(*, system: str, user: str, schema_json: str = "", provider: str) -> TokenCount:
    """Total prompt tokens for one call, including schema and message framing.

    This is the number that matters for a free-tier budget: what the provider counts,
    not what the visible prompt text alone measures.
    """
    body = count_tokens(f"{system}\n\n{user}", provider)
    schema = count_tokens(schema_json, provider) if schema_json else None
    total = body.tokens + MESSAGE_OVERHEAD_TOKENS
    if schema is not None:
        total += int(schema.tokens * SCHEMA_OVERHEAD_RATIO)
    return TokenCount(
        tokens=total,
        accuracy=body.accuracy,
        encoding=body.encoding,
        note=body.note,
    )
