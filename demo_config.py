"""Configuration for the public demo, separated from the page that renders it.

`app.py` is a Streamlit script: importing it executes the page. That makes its constants
untestable, which matters here because the most dangerous failure in the deployed demo is
silent -- a recording whose cache key has drifted, so an example advertised as free
quietly becomes a live call. These values live here so a test can read them without
starting a Streamlit runtime.
"""

from __future__ import annotations

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
