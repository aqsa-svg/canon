"""Configuration for the public demo, separated from the page that renders it.

`app.py` is a Streamlit script: importing it executes the page. That makes its constants
untestable, which matters here because the most dangerous failure in the deployed demo is
silent -- a recording whose cache key has drifted, so an example advertised as free
quietly becomes a live call. These values live here so a test can read them without
starting a Streamlit runtime.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent
DB_PATH: Final[Path] = REPO_ROOT / "data" / "canon.duckdb"

DEMO_CACHE: Final[Path] = REPO_ROOT / "demo_cache"
"""Recorded responses for the four examples, committed to the repository.

The working cache in ``data/llm_cache/`` is gitignored and large, so a fresh clone -- or a
Streamlit Cloud container -- has nothing to replay from. These four files are the whole
public demo's token budget: about 4.5 KB, which buys every visitor the examples for free.
Kept in their own directory rather than un-ignoring four opaque hash filenames, so that a
prompt change invalidating the keys is visible rather than silent.
"""

RECORDED_RUN: Final[str] = "sweep-groq-r1"
"""The benchmark run whose stored responses the examples replay."""

SESSION_LIVE_CALL_CAP: Final[int] = 3
"""Live calls one browser session may make. The examples are unlimited because they cost
nothing; this bounds only what a stranger can spend."""

DEMO_DAILY_TOKEN_CEILING: Final[int] = 40_000
"""Tokens per UTC day this demo may spend, well under the provider's 200,000 free-tier
budget. The gap is deliberate: a demo that can exhaust the day's budget would make the
benchmark unreproducible for whoever reads the repository next."""

DEMO_LEDGER_KEY: Final[str] = "demo"
"""The demo's spend is metered under its own key in the shared ledger.

Metering it against the provider's total would mean a benchmark sweep -- which routinely
spends most of a day's budget -- silently disabling the public demo. The two draw on the
same free-tier allowance but they are not the same budget.
"""


class Example(BaseModel):
    """One demo example, and the eval-set question whose recording it replays."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    question_id: str
    """Eval-set id. Also the cache context, which is what makes the replay resolve."""

    question: str
    why: str


EXAMPLES: Final[tuple[Example, ...]] = (
    Example(
        label="Clean answer",
        question_id="t1-01",
        question="What was net revenue in 2018?",
        why=(
            "One metric named outright. Watch the card's default status filter get "
            "applied without being asked for, and show up as a caveat."
        ),
    ),
    Example(
        label="Ambiguous — it asks",
        question_id="am-01",
        question="How much revenue did we make in 2018?",
        why=(
            "Three revenue definitions exist and none is the default. A number here "
            "would be a guess dressed as an answer. This is the moment the project is "
            "about."
        ),
    ),
    Example(
        label="Definition changes mid-series",
        question_id="t3-02",
        question="Show me reported revenue by quarter across 2017 and 2018.",
        why=(
            "reported_revenue changes its freight treatment on 2018-01-01 and prior "
            "periods were never restated. The numbers are right; growth through the "
            "boundary is not, and the caveat has to say so."
        ),
    ),
    Example(
        label="Verification gate trips",
        question_id="am-20",
        question="What was net revenue in October 2018?",
        why=(
            "Legal intent, correct SQL, and no rows: all four orders placed that month "
            "were canceled and carry no line items. The gate returns a flag rather than "
            "a figure."
        ),
    ),
)

BY_QUESTION: Final[dict[str, str]] = {item.question: item.question_id for item in EXAMPLES}


PROVIDER_KEY_NAMES: Final[tuple[str, ...]] = (
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
)
"""Every provider key the app may need, in the names the provider layer looks up."""


def _secret(name: str) -> str | None:
    """One value from Streamlit's secrets store, or None if there is nothing to read.

    Guarded so this module stays importable without Streamlit: the test suite, the
    harness and the scripts all need these constants and none of them runs in a
    Streamlit runtime. Every caught exception is a specific, expected "no secret here"
    condition rather than a blanket swallow -- ImportError when Streamlit is absent,
    and the three ways its secrets store reports a missing file or key.
    """
    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        return None
    try:
        value = st.secrets[name]
    except (KeyError, FileNotFoundError, StreamlitSecretNotFoundError):
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def _load_dotenv_once() -> None:
    """Load ``.env`` into the environment if it has not been loaded already.

    ``canon.agent.llm`` does this at import, so in the app it has usually happened
    before anything here runs. Doing it again (idempotently, never overriding) removes
    the dependency on import order, which is the kind of thing that works in the app and
    fails in a test for no visible reason.
    """
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)


def resolve_provider_key(name: str) -> str | None:
    """A provider key from Streamlit's secrets, else the environment, else None.

    Secrets are consulted first because that is the only place a Streamlit Cloud
    deployment can put one -- there is no filesystem to hold a ``.env``. The environment
    is the fallback, which is how a stranger running this locally from ``.env`` gets the
    same behaviour without knowing Streamlit exists.

    The ordering matters in exactly one case: both present and disagreeing. Secrets win
    there, on the grounds that ``.streamlit/secrets.toml`` is the more deliberate of the
    two -- a ``.env`` is often left over from something else.
    """
    secret = _secret(name)
    if secret:
        return secret
    _load_dotenv_once()
    return (os.environ.get(name) or "").strip() or None


def install_provider_keys() -> tuple[str, ...]:
    """Publish resolved keys into the environment. Returns the names that resolved.

    The provider layer reads ``os.environ`` because it must also work from the harness
    and the scripts, where Streamlit does not exist. Rather than teach it about secrets,
    the resolved value is placed where it already looks.
    """
    installed: list[str] = []
    for name in PROVIDER_KEY_NAMES:
        value = resolve_provider_key(name)
        if value:
            os.environ[name] = value
            installed.append(name)
    return tuple(installed)


REPO_URL: Final[str] = "https://github.com/aqsa-svg/canon"
RESULTS_URL: Final[str] = f"{REPO_URL}/blob/main/benchmark/RESULTS.md"

# Figures quoted on the page. Every one comes from the run recorded in
# benchmark/RESULTS.md (tag sweep-groq-r1, groq/openai-gpt-oss-120b, free tier) and is
# asserted against that document by a test, because benchmark/runs/ is gitignored and
# therefore absent on Streamlit Cloud -- the page cannot read the report at runtime and
# has to carry the numbers. Carrying them is the risk the test exists to contain.
BENCHMARK_QUESTIONS: Final[int] = 80
UNDER_REFUSAL_CANON: Final[str] = "0.0%"
UNDER_REFUSAL_BASELINE: Final[str] = "72.0%"
DETERMINISM_FLIPS: Final[str] = "2 of 80"
