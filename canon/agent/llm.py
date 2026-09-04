"""The Anthropic boundary. The only module in the project that touches the network.

Three things matter here for the benchmark to mean anything:

**The model is pinned.** ``MODEL_ID`` is a constant, recorded in every run's provenance.
Server-side refusal fallbacks are deliberately *not* enabled: a fallback would answer on
a different model than the one the report names, which would silently corrupt the
provenance the eval doc requires. A policy refusal is instead surfaced as an error.

**Responses are cached to disk.** Keyed by a hash of everything that could change the
answer — model, system prompt, question, schema, effort. Re-running the benchmark
tomorrow costs nothing and returns the same answers, so a scoring change can be
attributed to the scoring change. The cache is a correctness tool as much as a cost one.

**Sampling parameters are not set.** Current models reject ``temperature``, so
reproducibility comes from the cache, not from temperature=0. Worth stating plainly
because the usual trick is unavailable here.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Final, TypeVar

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

# Reads .env if present, without overriding a key already exported in the shell. Called
# at import so every entry point -- harness, Streamlit, a REPL -- picks the key up the
# same way, and none of them has to remember to.
load_dotenv(override=False)

MODEL_ID: Final[str] = "claude-opus-5"
"""Pinned. Changing this invalidates every cached response, by design."""

EFFORT: Final[str] = "medium"
"""Metric selection is a reading-comprehension task over a catalogue in the prompt, not
a long-horizon reasoning problem. Medium keeps 80 questions x 2 systems affordable; the
level is recorded in provenance so a rerun at another level is a comparable experiment
rather than an unexplained difference."""

MAX_TOKENS: Final[int] = 8192

CACHE_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent / "data" / "llm_cache"

_FROZEN = ConfigDict(frozen=True, extra="forbid")

ModelT = TypeVar("ModelT", bound=BaseModel)


class Usage(BaseModel):
    """Tokens and cost for one call. Zeroed for a cache hit, which is the honest
    accounting: a cached answer cost nothing on this run."""

    model_config = _FROZEN

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cached: bool = False
    latency_seconds: float = 0.0

    @property
    def cost_usd(self) -> float:
        """Cost at Claude Opus 5 list rates: $5/MTok in, $25/MTok out.

        Cache reads bill at roughly a tenth of input and writes at roughly 1.25x; both
        are applied here so the reported cost is not flattered by pretending cached
        input is free at the API level.
        """
        return (
            self.input_tokens * 5.0
            + self.cache_read_tokens * 0.5
            + self.cache_write_tokens * 6.25
            + self.output_tokens * 25.0
        ) / 1_000_000


class LlmError(RuntimeError):
    """The call did not produce a usable structured object.

    Distinct from a refusal by the *analytics* system: this is the API layer failing, and
    the harness records it as an error rather than scoring it as an answer.
    """


class CallContext(BaseModel):
    """Identifies *which* call this is, not just what was asked.

    Without this the cache is keyed on content alone, so asking the same question three
    times returns the same stored answer three times. That would make runs 2 and 3
    replays of run 1 and any reported variance identically zero -- a number that looks
    like a measurement and is an artefact of the cache.

    With ``run_index`` in the key, each run is a fresh sample. The cache still does its
    job across *sessions*: re-running run 2 tomorrow replays run 2, so a scoring change
    can be attributed to the scoring change.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    system: str
    """Which system under test -- 'canon' or 'baseline'. Keeps their caches disjoint
    even where the question text is identical."""

    question_id: str
    run_index: int = Field(ge=0)


def _cache_key(
    system: str,
    question: str,
    schema_name: str,
    schema: str,
    effort: str,
    model: str,
    context: CallContext | None,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "effort": effort,
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


class StructuredResponse(BaseModel):
    """A parsed object plus what it cost to get it."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    raw_json: str
    usage: Usage


class LlmClient:
    """Makes one structured call per question, with a disk cache in front."""

    def __init__(
        self,
        *,
        model: str = MODEL_ID,
        effort: str = EFFORT,
        cache_dir: Path = CACHE_DIR,
        use_cache: bool = True,
    ) -> None:
        self.model = model
        self.effort = effort
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self._client: anthropic.Anthropic | None = None
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def client(self) -> anthropic.Anthropic:
        """Constructed lazily so importing this module never requires a key."""
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise LlmError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add a "
                    "key, or export it. Cached questions still work without one."
                )
            self._client = anthropic.Anthropic()
        return self._client

    def structured(
        self,
        *,
        system: str,
        question: str,
        output_format: type[ModelT],
        context: CallContext | None = None,
    ) -> tuple[ModelT, Usage]:
        """One call, one validated object.

        No retry loop. A retry would be a second call hidden inside what the report
        counts as one, and a schema-invalid response is information about the system
        under test rather than noise to be smoothed away.
        """
        schema = json.dumps(output_format.model_json_schema(), sort_keys=True)
        key = _cache_key(
            system, question, output_format.__name__, schema, self.effort, self.model, context
        )
        path = self.cache_dir / f"{key}.json"

        if self.use_cache and path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            return (
                output_format.model_validate_json(cached["raw_json"]),
                Usage(cached=True),
            )

        started = time.monotonic()
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=MAX_TOKENS,
            output_config={"effort": self.effort},
            system=[
                {
                    "type": "text",
                    "text": system,
                    # The whole semantic layer is the stable prefix; the question follows
                    # it in messages. Without this breakpoint every question would re-pay
                    # for the full catalogue.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": question}],
            output_format=output_format,
        )
        elapsed = time.monotonic() - started

        if response.stop_reason == "refusal":
            raise LlmError(
                "the API declined this request at the policy layer "
                f"(stop_reason=refusal, details={response.stop_details})"
            )

        parsed = response.parsed_output
        if parsed is None:
            raise LlmError(
                f"no structured output returned (stop_reason={response.stop_reason}); "
                "the response may have hit max_tokens"
            )

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_input_tokens or 0,
            cache_write_tokens=response.usage.cache_creation_input_tokens or 0,
            cached=False,
            latency_seconds=elapsed,
        )

        if self.use_cache:
            path.write_text(
                json.dumps(
                    {
                        "raw_json": parsed.model_dump_json(),
                        "model": self.model,
                        "effort": self.effort,
                        "usage": usage.model_dump(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return parsed, usage

    def freeform(
        self, *, system: str, question: str, context: CallContext | None = None
    ) -> tuple[str, Usage]:
        """A plain text response. Used only by the naive baseline, which writes SQL.

        Cached identically, so the two systems are compared under the same conditions.
        """
        key = _cache_key(system, question, "freeform", "", self.effort, self.model, context)
        path = self.cache_dir / f"{key}.json"

        if self.use_cache and path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            return str(cached["raw_json"]), Usage(cached=True)

        started = time.monotonic()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            output_config={"effort": self.effort},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": question}],
        )
        elapsed = time.monotonic() - started

        if response.stop_reason == "refusal":
            raise LlmError(f"API policy refusal (details={response.stop_details})")

        text = "".join(block.text for block in response.content if block.type == "text")
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_input_tokens or 0,
            cache_write_tokens=response.usage.cache_creation_input_tokens or 0,
            cached=False,
            latency_seconds=elapsed,
        )
        if self.use_cache:
            path.write_text(
                json.dumps({"raw_json": text, "usage": usage.model_dump()}, indent=2),
                encoding="utf-8",
            )
        return text, usage
