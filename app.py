"""Canon demo: a question in, and the whole pipeline visible on the way out.

The point of the page is not that it answers questions. It is that you can see *why* the
answer is what it is: the typed intent the model produced, the SQL a compiler derived from
it, the canonical definition that governed it, and the caveats that travel with the
number. And when the question is ambiguous, that the system asks instead of guessing —
which is the moment worth showing.

Run:  uv run streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from canon.agent.llm import CacheMiss, CallContext, LlmClient, LlmError
from canon.agent.pipeline import CanonPipeline, PipelineResult
from canon.agent.providers import (
    DEFAULT_PROVIDER,
    QuotaExhausted,
    QuotaLedger,
    available_providers,
    is_free_tier,
)
from canon.semantic.loader import load_semantic_layer, semantic_layer_hash
from demo_config import (
    BY_QUESTION,
    DB_PATH,
    DEMO_CACHE,
    DEMO_DAILY_TOKEN_CEILING,
    DEMO_LEDGER_KEY,
    EXAMPLES,
    RECORDED_RUN,
    SESSION_LIVE_CALL_CAP,
)

st.set_page_config(page_title="Canon", layout="wide", page_icon="📐")


@st.cache_resource
def _layer():  # noqa: ANN202 - streamlit cache wrapper
    return load_semantic_layer()


@st.cache_resource
def _pipeline(provider: str) -> CanonPipeline:
    return CanonPipeline(db_path=DB_PATH, layer=_layer(), client=LlmClient(provider=provider))


@st.cache_resource
def _replay_pipeline(provider: str) -> CanonPipeline:
    """Cache-only pipeline: raises ``CacheMiss`` rather than ever reaching the network.

    Enforced by the client, not by this page remembering to check. A demo that *intends*
    to be free but can still make a call is one bug away from draining the day's quota.
    """
    return CanonPipeline(
        db_path=DB_PATH,
        layer=_layer(),
        client=LlmClient(provider=provider, cache_only=True, cache_dir=DEMO_CACHE),
        decision_log=None,
    )


def _demo_spent_today() -> int:
    """Tokens this demo has spent today, excluding anything the benchmark spent."""
    try:
        return QuotaLedger().usage(DEMO_LEDGER_KEY).tokens
    except Exception:  # pragma: no cover - a missing ledger must not break the page
        return 0


def _record_demo_spend(tokens: int) -> None:
    """Meter a live call against the demo's daily share.

    A failure here is not swallowed. ``CLAUDE.md`` forbids silent fallbacks, and this is
    exactly the case the rule is for: if metering fails quietly the ceiling stops being
    enforced while still appearing to be, which is worse than a visible warning.
    """
    try:
        QuotaLedger().record(DEMO_LEDGER_KEY, tokens=tokens)
    except OSError as error:
        st.warning(
            f"Could not record this call against the demo's daily budget ({error}). "
            "The per-session cap still applies, but the daily ceiling is not being "
            "enforced right now."
        )


def _live_calls_left(provider: str) -> tuple[int, str | None]:
    """Remaining live calls for this session, and why none are left if so.

    Two independent ceilings, checked in order of severity: the provider's own daily
    allowance, which nothing can be done about, then the demo's self-imposed share of it,
    then this session's.
    """
    try:
        _, tokens_left = LlmClient(provider=provider).remaining_today()
    except Exception:  # pragma: no cover - never block on a metering failure
        tokens_left = None
    if tokens_left is not None and tokens_left <= 0:
        return 0, "provider"
    if _demo_spent_today() >= DEMO_DAILY_TOKEN_CEILING:
        return 0, "daily"
    used = st.session_state.get("live_calls", 0)
    if used >= SESSION_LIVE_CALL_CAP:
        return 0, "session"
    return SESSION_LIVE_CALL_CAP - used, None


layer = _layer()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Canon")
    st.caption(
        f"{len(layer.metrics)} canonical metrics · metrics.yaml @ `{semantic_layer_hash()[:12]}`"
    )

    provider_name = st.selectbox(
        "provider",
        list(available_providers()),
        index=list(available_providers()).index(DEFAULT_PROVIDER),
        help="Free tiers are the default; the benchmark reproduces for $0.",
    )
    if not is_free_tier(provider_name):
        st.warning(f"`{provider_name}` is **not** free tier — calls are billed.")

    client = LlmClient(provider=provider_name)
    if not client.has_credentials():
        st.error(
            f"No API key for `{provider_name}`. Copy `.env.example` to `.env` and add one. "
            "Never put a real key in `.env.example` — it is tracked by git."
        )

    requests_left, tokens_left = client.remaining_today()
    st.metric(
        "free tokens left today",
        f"{tokens_left:,}" if tokens_left is not None else "no daily cap",
        help="Read from the on-disk quota ledger, which persists across days.",
    )
    if requests_left is not None:
        st.caption(f"{requests_left} requests left today")

    st.divider()
    st.caption(
        "**How to read this page.** The model only ever fills a typed `QueryIntent`. "
        "Every identifier in the SQL comes from `metrics.yaml`, not from the model — so a "
        "metric the semantic layer does not define cannot be returned at all."
    )
    st.caption(
        "The first run of an example costs free-tier tokens; after that it is served from "
        "the response cache and costs nothing."
    )

# ---------------------------------------------------------------- header
st.title("Canon")
st.markdown(
    "A verified number, the SQL that produced it, the canonical definition it used — "
    "**or a question back, when the question is genuinely ambiguous.**"
)

if "question" not in st.session_state:
    st.session_state.question = EXAMPLES[0].question

st.markdown("##### Try one — these replay recorded runs and cost nothing")
columns = st.columns(len(EXAMPLES))
for column, example in zip(columns, EXAMPLES, strict=True):
    with column:
        if st.button(example.label, use_container_width=True, help=example.why):
            st.session_state.question = example.question
        st.caption(example.why)

question = st.text_input(
    "Ask a business question",
    key="question",
    label_visibility="collapsed",
)


@st.cache_resource(show_spinner=False)
def _ensure_warehouse() -> bool:
    """Build the warehouse if this checkout does not have one.

    ``data/*.duckdb`` is gitignored -- it is 58 MB and reproducible -- so a fresh clone
    and a Streamlit Cloud container both start without it. The loader pulls public CSVs
    over plain HTTPS with no credentials, so it can run here; cached so it happens once
    per container rather than once per page view.
    """
    if DB_PATH.exists():
        return True
    from scripts.load_data import main as load_data

    load_data()
    return DB_PATH.exists()


if not DB_PATH.exists():
    with st.status("First run: building the warehouse from public CSVs…", expanded=True) as status:
        st.write(
            "`data/canon.duckdb` is 58 MB and reproducible, so it is not committed. "
            "Downloading the Olist dataset and loading it — about a minute, once per "
            "container."
        )
        try:
            built = _ensure_warehouse()
        except Exception as error:
            status.update(label="Could not build the warehouse", state="error")
            st.error(
                "The dataset could not be downloaded, so there is nothing to query. "
                "This is a network or upstream-mirror problem rather than a fault in "
                "Canon. Locally, run `python scripts/load_data.py` and reload."
            )
            st.caption(f"{type(error).__name__}: {str(error)[:300]}")
            st.stop()
        if not built:
            status.update(label="Could not build the warehouse", state="error")
            st.error("The loader finished but no database appeared. Check the logs.")
            st.stop()
        status.update(label="Warehouse ready", state="complete")

go = st.button("Ask", type="primary")
if not go and "last_result" not in st.session_state:
    st.info("Pick an example or type a question, then press **Ask**.")
    st.stop()


def _examples_reminder() -> None:
    """Point back at the four free examples. Used wherever a live call is refused."""
    st.markdown(
        "**The four examples above still work and cost nothing** — they replay recorded "
        f"runs from `{RECORDED_RUN}`. They are also the part worth seeing: a clean "
        "answer, a refusal, a definition change mid-series, and the verification gate "
        "catching an empty result."
    )


if go:
    recorded_id = BY_QUESTION.get(question.strip())

    if recorded_id is not None:
        # A known example: replay it. The cache-only client cannot reach the network, so
        # this path is free by construction rather than by intention.
        with st.spinner("replaying the recorded run…"):
            try:
                st.session_state.last_result = _replay_pipeline(provider_name).answer(
                    question,
                    context=CallContext(system="canon", question_id=recorded_id, run_index=0),
                )
                st.session_state.last_was_replay = recorded_id
            except CacheMiss:
                st.warning(
                    f"No recorded response for `{recorded_id}` in this checkout, so this "
                    "example cannot be replayed. Type the question and press Ask to run "
                    "it live, which spends free-tier tokens."
                )
                st.stop()
    else:
        remaining, blocked = _live_calls_left(provider_name)
        if blocked == "provider":
            st.error(
                "**The provider's free-tier budget for today is spent.** This is the "
                "constraint the project runs under rather than a fault: the whole "
                "benchmark is reproducible for $0, and the price of that is a daily "
                "ceiling. It resets at 00:00 UTC."
            )
            _examples_reminder()
            st.stop()
        if blocked == "daily":
            st.error(
                f"**This demo has spent its daily budget.** It is capped at "
                f"{DEMO_DAILY_TOKEN_CEILING:,} tokens a day, well under the provider's "
                "free-tier allowance, so that a busy day here cannot make the benchmark "
                "unreproducible for whoever reads it next. It resets at 00:00 UTC."
            )
            _examples_reminder()
            st.stop()
        if blocked == "session":
            st.warning(
                f"**You have used this session's {SESSION_LIVE_CALL_CAP} live questions.** "
                "The cap exists because this page runs on a shared free-tier quota that "
                "any visitor can drain. Reload for a new session, or clone the repo and "
                "run it against your own key — it costs nothing."
            )
            _examples_reminder()
            st.stop()

        with st.spinner("one structured call, then a deterministic compiler…"):
            try:
                answered = _pipeline(provider_name).answer(question)
                st.session_state.last_result = answered
                st.session_state.last_was_replay = None
                st.session_state.live_calls = st.session_state.get("live_calls", 0) + 1
                _record_demo_spend(answered.usage.total_tokens)
            except QuotaExhausted:
                st.error(
                    "**The provider's free-tier budget for today is spent.** This is the "
                    "constraint the project runs under rather than a fault: the whole "
                    "benchmark is reproducible for $0, and the price of that is a daily "
                    "ceiling. It resets at 00:00 UTC."
                )
                _examples_reminder()
                st.stop()
            except LlmError as error:
                st.error(
                    "**The provider could not answer that just now.** This is usually a "
                    "rate limit or a transient outage rather than anything about the "
                    "question; trying again in a minute often works."
                )
                st.caption(f"Reported as: {type(error).__name__}: {str(error)[:200]}")
                _examples_reminder()
                st.stop()
            except Exception as error:  # pragma: no cover - last resort, never a traceback
                st.error(
                    "**Something went wrong producing that answer.** The error is shown "
                    "below rather than as a stack trace."
                )
                st.caption(f"{type(error).__name__}: {str(error)[:200]}")
                _examples_reminder()
                st.stop()

result: PipelineResult = st.session_state.last_result
answer = result.answer

# The interface must not be capable of implying something the data does not support: a
# replayed response and a live one look identical downstream, so the difference is stated
# here rather than left to the visitor to assume.
_replayed = st.session_state.get("last_was_replay")
if _replayed is not None:
    st.caption(
        f"↻ **Replay of a recorded run** — question `{_replayed}` from the benchmark run "
        f"`{RECORDED_RUN}`, served from the response cache. No model call was made and no "
        "tokens were spent. The pipeline below is real: the intent, the compiled SQL and "
        "the verification all re-ran against the warehouse just now; only the model's "
        "reply was stored."
    )
else:
    _left, _ = _live_calls_left(provider_name)
    st.caption(
        f"● **Live call** — answered just now by `{provider_name}`. "
        f"{_left} of {SESSION_LIVE_CALL_CAP} live questions left this session."
    )

if result.error or answer is None:
    st.error(f"Could not produce an answer: {result.error}")
    st.stop()

# ---------------------------------------------------------------- the clarify path
# Deliberately the most prominent state on the page. A system that asks a good question
# instead of guessing is the product, not an error path.
if answer.is_refusal:
    st.markdown("### ✋ Canon is asking, not answering")
    st.info(f"**{answer.clarifying_question}**")

    decision = result.decision
    if decision is not None:
        gate_tripped = decision.rule.value.startswith("compiler") or (
            result.verification is not None and not result.verification.passed
        )
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Why it asked**")
            st.markdown(f"- rule: `{decision.rule.value}`")
            st.markdown(
                f"- best metric `{decision.top_metric}` scored **{decision.top_score:.2f}**"
            )
            if decision.runner_up:
                st.markdown(
                    f"- runner-up `{decision.runner_up}` scored "
                    f"**{decision.runner_up_score:.2f}** — gap **{decision.score_gap:.2f}**, "
                    f"threshold **{decision.thresholds.metric_score_margin}**"
                )
            for gap in decision.missing_information:
                st.markdown(f"- missing: {gap}")
            st.caption(
                "The decision is deterministic Python over the model's scores, not a "
                "model judgement — so the threshold can be re-tuned without another call."
            )
        with right:
            st.markdown("**The model's reading**")
            st.write(decision.reading)
            st.metric("confidence in asking", f"{answer.confidence:.2f}")

        if result.verification is not None and not result.verification.passed:
            st.warning(
                "The verification gate rejected the result: "
                + ", ".join(failure.value for failure in result.verification.failures)
            )
            for note in result.verification.notes:
                st.caption(note)
            if result.sql:
                with st.expander("The SQL it ran, which returned nothing usable"):
                    st.code(result.sql, language="sql")
        elif gate_tripped and result.intent is not None:
            with st.expander("The intent the compiler rejected"):
                st.code(result.intent.model_dump_json(indent=2), language="json")
    st.stop()

# ---------------------------------------------------------------- the answered path
card = answer.metric_definition_used
value = answer.answer_value
# card and value are non-null here: Answer's exclusivity validator guarantees that an
# answered response carries both, and the refusal branch returned above.

st.markdown("### Answer")
top_left, top_right = st.columns([2, 3])

with top_left:
    if value.scalar is not None:
        display = (
            f"{float(value.scalar):,.2f}"
            if isinstance(value.scalar, int | float) or hasattr(value.scalar, "as_tuple")
            else str(value.scalar)
        )
        st.metric(card.name, display)
    else:
        st.metric(card.name, f"{len(value.rows)} rows")
    st.caption(f"confidence {answer.confidence:.2f} · owned by `{card.owner}`")

with top_right:
    frame = pd.DataFrame(list(value.rows), columns=[item.name for item in value.columns])
    if len(frame) > 1:
        st.dataframe(frame, width="stretch", hide_index=True)
    spec = answer.chart_spec
    if spec is not None and len(frame) > 1 and spec.x in frame.columns:
        plotted = frame.set_index(spec.x)[spec.y]
        (st.line_chart if spec.kind.value == "line" else st.bar_chart)(plotted)

st.divider()

# The pipeline, in the order it ran.
st.markdown("### How it got there")
stage_intent, stage_sql = st.columns(2)

with stage_intent:
    st.markdown("**1 · The model's only output — a typed intent**")
    st.code(result.intent.model_dump_json(indent=2) if result.intent else "{}", language="json")
    decision = result.decision
    if decision is not None:
        st.caption(
            f"chose `{decision.top_metric}` at {decision.top_score:.2f}"
            + (
                f", over `{decision.runner_up}` at {decision.runner_up_score:.2f} "
                f"(gap {decision.score_gap:.2f})"
                if decision.runner_up
                else " with nothing close to it"
            )
        )

with stage_sql:
    st.markdown("**2 · SQL a compiler derived from it**")
    st.code(answer.sql or "", language="sql")
    st.caption(
        "No identifier here came from the model. Filter values enter as typed literal "
        "nodes in a syntax tree, never as spliced text."
    )

st.markdown(f"**3 · The canonical definition — `{card.name}`**")
definition_left, definition_right = st.columns([3, 2])
with definition_left:
    st.write(card.description)
    st.markdown(f"- **grain**: {card.grain}")
    st.markdown(f"- **expression**: `{card.expression}`")
    if card.default_filters:
        applied = "; ".join(
            f"`{item.column}` {item.op.value}"
            + (f" ({', '.join(str(v) for v in item.values)})" if item.values else "")
            for item in card.default_filters
        )
        st.markdown(f"- **always filtered by**: {applied}")
    if card.definition_change is not None:
        change = card.definition_change
        st.error(
            f"**This definition changes on {change.effective_date}**: "
            f"{change.before} before, {change.after} from that date. "
            "Prior periods were never restated."
        )
with definition_right:
    st.markdown(f"**4 · Caveats ({len(answer.caveats)})**")
    if not answer.caveats:
        st.caption("None apply to this answer.")
    for caveat in answer.caveats:
        icon = {"from_metric_card": "📋", "from_compiler": "⚙", "from_data": "🗄"}.get(
            caveat.source.value, "•"
        )
        st.markdown(f"{icon} {caveat.text}")

report = result.verification
if report is not None:
    with st.expander(f"Verification gate: {'passed' if report.passed else 'FAILED'}"):
        st.write(
            f"{report.row_count} rows returned"
            + (
                f", at most {report.expected_max_rows} plausible for this grain"
                if report.expected_max_rows is not None
                else ""
            )
        )
        st.caption(
            "Checks shape, cell types, a numeric and not-all-null metric column, "
            "unexpected emptiness, and a row count plausible for the requested grain. "
            "On failure the pipeline returns a flag rather than a number."
        )

usage = result.usage
st.caption(
    f"{usage.provider} · `{usage.model_id}` · config `{usage.config_hash}` · "
    f"{'cache hit' if usage.cached else f'{usage.total_tokens:,} tokens'} · "
    f"${usage.cost_usd:.4f}"
)
