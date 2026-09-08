"""The three backends. Each one turns a schema into whatever its API calls structure.

The differences are real and worth naming, because they are the reason the abstraction
exists rather than a leaky detail hidden behind it:

* **Groq / GPT-OSS** exposes function calling, not a JSON-schema response mode, so the
  schema becomes a single tool and the model is told to call it. Structure is enforced by
  the tool definition.
* **Gemini** has native ``response_schema``, so the API enforces the schema server-side.
  This is the strongest of the three guarantees.
* **Anthropic** has ``messages.parse`` with a Pydantic output format, which also enforces
  server-side.

In all three the *metric enum is inside the schema*, so Canon's core claim survives the
provider swap: a metric the semantic layer does not define cannot be returned, whichever
backend answers.

No provider retries internally. Retries live in the runner, where they can be counted
against a quota and logged — a retry hidden in here would be a second call the benchmark
reports as one.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, TypeVar

from pydantic import BaseModel

from canon.agent.providers.base import (
    Prompt,
    ProviderConfig,
    ProviderError,
    StructuredResult,
    TextResponse,
    TokenUsage,
)

ModelT = TypeVar("ModelT", bound=BaseModel)

TOOL_NAME = "submit_query_intent"


MAX_INLINE_DEPTH = 12
"""Guard against a self-referential model inlining forever. Nothing in Canon's schemas
recurses, so hitting this is a bug worth surfacing rather than tolerating."""


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace every ``$ref`` with the definition it points at.

    Required for any provider whose validator does not resolve ``$ref``. Groq's does not,
    and the failure mode is not a clean error: the nested model becomes invisible, so the
    model guesses field names (``breakdown`` for ``dimensions``) and the validator
    misreads a nullable object as null-only.
    """
    definitions = schema.get("$defs", {})

    def resolve(node: Any, ref_depth: int) -> Any:
        # Only $ref resolution can recurse without bound, so only that increments the
        # counter. Counting ordinary structural nesting instead tripped the guard on a
        # perfectly finite schema.
        if isinstance(node, dict):
            reference = node.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                if ref_depth >= MAX_INLINE_DEPTH:
                    raise ProviderError(
                        "schema resolves $ref deeper than the inlining guard allows; a "
                        "recursive model cannot be sent to a provider that will not "
                        "resolve $ref itself"
                    )
                name = reference.split("/")[-1]
                if name not in definitions:
                    raise ProviderError(f"schema references unknown definition {name!r}")
                # Keep any siblings of the $ref (description, default) and let the
                # definition supply the structure.
                merged = {key: value for key, value in node.items() if key != "$ref"}
                return {**resolve(definitions[name], ref_depth + 1), **merged}
            return {key: resolve(value, ref_depth) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(item, ref_depth) for item in node]
        return node

    inlined = resolve({key: value for key, value in schema.items() if key != "$defs"}, 0)
    return inlined  # type: ignore[no-any-return]


def _strict_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """A fully inlined JSON schema with ``additionalProperties: false`` throughout.

    Two transformations, both for strict validators:

    * ``$defs`` are inlined, so a provider that will not follow ``$ref`` still sees every
      nested field. Without this the model cannot know what to fill in.
    * every object forbids extra properties, because a schema that permits them defeats
      the point -- an unexpected field is how a hallucinated metric would sneak back in.
    """
    raw = _inline_defs(schema.model_json_schema())

    def close(node: Any) -> Any:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node.setdefault("additionalProperties", False)
            return {key: close(value) for key, value in node.items()}
        if isinstance(node, list):
            return [close(item) for item in node]
        return node

    return close(raw)


# --------------------------------------------------------------------------- Groq
class GroqProvider:
    """Primary free-tier backend: ``openai/gpt-oss-120b`` via Groq."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: Any = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def context_limit(self) -> int:
        return self.config.context_limit

    @property
    def client(self) -> Any:
        if self._client is None:
            import groq

            key = os.environ.get("GROQ_API_KEY")
            if not key:
                raise ProviderError(
                    "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                    "and add it to .env. The dry-run needs no key."
                )
            self._client = groq.Groq(api_key=key)
        return self._client

    def _call(self, prompt: Prompt, tools: list[dict[str, Any]] | None) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "max_completion_tokens": self.config.max_output_tokens,
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.reasoning_effort:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": TOOL_NAME},
            }
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as error:
            raise _normalise(error) from error

    def complete_structured(self, prompt: Prompt, schema: type[ModelT]) -> StructuredResult[ModelT]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "description": "Return the structured analytical intent.",
                    "parameters": _strict_schema(schema),
                },
            }
        ]
        started = time.monotonic()
        response = self._call(prompt, tools)
        elapsed = time.monotonic() - started

        choice = response.choices[0]
        calls = getattr(choice.message, "tool_calls", None)
        if not calls:
            raise ProviderError(
                "the model returned prose instead of calling the tool "
                f"(finish_reason={choice.finish_reason}); no structured object to parse"
            )
        arguments = calls[0].function.arguments
        try:
            parsed = schema.model_validate_json(arguments)
        except ValueError as error:
            raise ProviderError(f"tool arguments failed schema validation: {error}") from error

        return StructuredResult[schema](  # type: ignore[valid-type]
            parsed=parsed, usage=_usage_from_openai(response, elapsed), raw_text=arguments
        )

    def complete_text(self, prompt: Prompt) -> StructuredResult[TextResponse]:
        started = time.monotonic()
        response = self._call(prompt, None)
        elapsed = time.monotonic() - started
        text = response.choices[0].message.content or ""
        return StructuredResult[TextResponse](
            parsed=TextResponse(text=text),
            usage=_usage_from_openai(response, elapsed),
            raw_text=text,
        )


def _usage_from_openai(response: Any, elapsed: float) -> TokenUsage:
    usage = getattr(response, "usage", None)
    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        latency_seconds=round(elapsed, 3),
    )


# ------------------------------------------------------------------------- Gemini
class GeminiProvider:
    """Second provider, for the cross-model robustness check.

    Uses native ``response_schema``, so the schema is enforced by the API. Note the
    free-tier data policy called out in ``providers.yaml`` and the README.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: Any = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def context_limit(self) -> int:
        return self.config.context_limit

    @property
    def client(self) -> Any:
        if self._client is None:
            from google import genai

            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise ProviderError(
                    "GEMINI_API_KEY is not set. Get a free key at "
                    "https://aistudio.google.com/apikey and add it to .env."
                )
            self._client = genai.Client(api_key=key)
        return self._client

    def _config(self, prompt: Prompt, schema: type[BaseModel] | None) -> Any:
        from google.genai import types

        kwargs: dict[str, Any] = {
            "system_instruction": prompt.system,
            "max_output_tokens": self.config.max_output_tokens,
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = schema
        return types.GenerateContentConfig(**kwargs)

    def complete_structured(self, prompt: Prompt, schema: type[ModelT]) -> StructuredResult[ModelT]:
        started = time.monotonic()
        try:
            response = self.client.models.generate_content(
                model=self.model_id, contents=prompt.user, config=self._config(prompt, schema)
            )
        except Exception as error:
            raise _normalise(error) from error
        elapsed = time.monotonic() - started

        text = response.text or ""
        # The SDK can hand back an already-parsed object; prefer it, fall back to text.
        candidate = getattr(response, "parsed", None)
        if isinstance(candidate, schema):
            parsed = candidate
        else:
            try:
                parsed = schema.model_validate_json(text)
            except ValueError as error:
                raise ProviderError(f"response failed schema validation: {error}") from error

        return StructuredResult[schema](  # type: ignore[valid-type]
            parsed=parsed, usage=_usage_from_gemini(response, elapsed), raw_text=text
        )

    def complete_text(self, prompt: Prompt) -> StructuredResult[TextResponse]:
        started = time.monotonic()
        try:
            response = self.client.models.generate_content(
                model=self.model_id, contents=prompt.user, config=self._config(prompt, None)
            )
        except Exception as error:
            raise _normalise(error) from error
        elapsed = time.monotonic() - started
        text = response.text or ""
        return StructuredResult[TextResponse](
            parsed=TextResponse(text=text),
            usage=_usage_from_gemini(response, elapsed),
            raw_text=text,
        )


def _usage_from_gemini(response: Any, elapsed: float) -> TokenUsage:
    usage = getattr(response, "usage_metadata", None)
    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        latency_seconds=round(elapsed, 3),
    )


# ---------------------------------------------------------------------- Anthropic
class AnthropicProvider:
    """The paid path, kept working so it can be swapped back in unchanged."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: Any = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def context_limit(self) -> int:
        return self.config.context_limit

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ProviderError(
                    "ANTHROPIC_API_KEY is not set. This provider is NOT free tier -- "
                    "every call is billed. Use --provider groq for the free path."
                )
            self._client = anthropic.Anthropic()
        return self._client

    def _system(self, prompt: Prompt) -> list[dict[str, Any]]:
        return [
            {
                "type": "text",
                "text": prompt.system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def complete_structured(self, prompt: Prompt, schema: type[ModelT]) -> StructuredResult[ModelT]:
        started = time.monotonic()
        try:
            response = self.client.messages.parse(
                model=self.model_id,
                max_tokens=self.config.max_output_tokens,
                output_config={"effort": self.config.reasoning_effort or "medium"},
                system=self._system(prompt),
                messages=[{"role": "user", "content": prompt.user}],
                output_format=schema,
            )
        except Exception as error:
            raise _normalise(error) from error
        elapsed = time.monotonic() - started

        if response.stop_reason == "refusal":
            raise ProviderError(f"policy refusal (details={response.stop_details})")
        parsed = response.parsed_output
        if parsed is None:
            raise ProviderError(f"no structured output (stop_reason={response.stop_reason})")

        return StructuredResult[schema](  # type: ignore[valid-type]
            parsed=parsed,
            usage=_usage_from_anthropic(response, elapsed),
            raw_text=parsed.model_dump_json(),
        )

    def complete_text(self, prompt: Prompt) -> StructuredResult[TextResponse]:
        started = time.monotonic()
        try:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=self.config.max_output_tokens,
                output_config={"effort": self.config.reasoning_effort or "medium"},
                system=self._system(prompt),
                messages=[{"role": "user", "content": prompt.user}],
            )
        except Exception as error:
            raise _normalise(error) from error
        elapsed = time.monotonic() - started
        text = "".join(block.text for block in response.content if block.type == "text")
        return StructuredResult[TextResponse](
            parsed=TextResponse(text=text),
            usage=_usage_from_anthropic(response, elapsed),
            raw_text=text,
        )


def _usage_from_anthropic(response: Any, elapsed: float) -> TokenUsage:
    usage = response.usage
    return TokenUsage(
        prompt_tokens=(usage.input_tokens or 0)
        + (getattr(usage, "cache_read_input_tokens", 0) or 0)
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0),
        completion_tokens=usage.output_tokens or 0,
        latency_seconds=round(elapsed, 3),
    )


# ----------------------------------------------------------------------- shared
_RETRYABLE_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",  # Groq's code is rate_limit_exceeded, with an underscore
    "overloaded",
    "503",
    "502",
    "timeout",
    "500",
    "request too large",  # Groq returns 413 for a per-minute token overage
)
"""Substrings that mean "try again later".

``rate_limit`` and ``request too large`` are here because of a real loss: Groq refused
question am-08 with HTTP 413 and code ``rate_limit_exceeded`` -- a transient per-minute
overage that a retry would have cleared. Neither 413 nor the underscored code matched the
original list, so it was classified permanent and the question errored out for good.
"""

_RETRYABLE_STATUSES = frozenset({408, 409, 413, 429, 500, 502, 503, 504})
"""413 is included because Groq uses it for a rate-limit condition rather than for a
genuinely oversized payload."""


def _normalise(error: Exception) -> ProviderError:
    """Map an SDK exception onto ``ProviderError``, marking whether a retry could help.

    Deliberately string-based rather than per-SDK exception types: three SDKs with three
    hierarchies, and the runner only needs to know retryable-or-not. Getting that wrong
    in the safe direction (not retrying) wastes no quota.
    """
    if isinstance(error, ProviderError):
        return error
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    text = f"{type(error).__name__}: {error}"
    retryable = any(marker in text.lower() for marker in _RETRYABLE_MARKERS)
    if isinstance(status, int) and status in _RETRYABLE_STATUSES:
        retryable = True
    return ProviderError(
        text, retryable=retryable, status=status if isinstance(status, int) else None
    )


def retry_after_seconds(error: ProviderError) -> float | None:
    """Parse a provider's suggested wait out of an error message, if it offered one."""
    text = str(error).lower()
    for marker in ("retry-after", "try again in", "retry after"):
        if marker in text:
            tail = text.split(marker, 1)[1]
            digits = ""
            for char in tail:
                if char.isdigit() or char == ".":
                    digits += char
                elif digits:
                    break
            if digits:
                try:
                    return float(digits)
                except ValueError:
                    return None
    return None


def usage_as_json(usage: TokenUsage) -> str:
    return json.dumps(usage.model_dump(), sort_keys=True)
