"""Canon demo: a question in, and the whole pipeline visible on the way out.

The point of the page is not that it answers questions. It is that you can see *why* the
answer is what it is: the typed intent the model produced, the SQL a compiler derived from
it, the canonical definition that governed it, and the caveats that travel with the
number. And when the question is ambiguous, that the system asks instead of guessing —
which is the moment worth showing.

Run:  uv run streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from canon.agent.llm import LlmClient, LlmError
from canon.agent.pipeline import CanonPipeline, PipelineResult
from canon.agent.providers import DEFAULT_PROVIDER, available_providers, is_free_tier
from canon.semantic.loader import load_semantic_layer, semantic_layer_hash

DB_PATH = Path(__file__).resolve().parent / "data" / "canon.duckdb"

EXAMPLES: tuple[tuple[str, str, str], ...] = (
    (
        "Clean answer",
        "What was net revenue in 2018?",
        "One metric named outright. Watch the card's default status filter get applied "
        "without being asked for, and show up as a caveat.",
    ),
    (
        "Ambiguous — it asks",
        "How much revenue did we make in 2018?",
        "Three revenue definitions exist and none is the default. A number here would be "
        "a guess dressed as an answer. This is the moment the project is about.",
    ),
    (
        "Definition changes mid-series",
        "Show me reported revenue by quarter across 2017 and 2018.",
        "reported_revenue changes its freight treatment on 2018-01-01 and prior periods "
        "were never restated. The numbers are right; growth through the boundary is not, "
        "and the caveat has to say so.",
    ),
    (
        "Verification gate trips",
        "What was net revenue in October 2018?",
        "Legal intent, correct SQL, and no rows: all four orders placed that month were "
        "canceled and carry no line items. The gate returns a flag rather than a figure.",
    ),
)


st.set_page_config(page_title="Canon", layout="wide", page_icon="📐")


@st.cache_resource
def _layer():  # noqa: ANN202 - streamlit cache wrapper
    return load_semantic_layer()


@st.cache_resource
def _pipeline(provider: str) -> CanonPipeline:
    return CanonPipeline(db_path=DB_PATH, layer=_layer(), client=LlmClient(provider=provider))


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
    st.session_state.question = EXAMPLES[0][1]

st.markdown("##### Try one")
columns = st.columns(len(EXAMPLES))
for column, (label, text, why) in zip(columns, EXAMPLES, strict=True):
    with column:
        if st.button(label, use_container_width=True, help=why):
            st.session_state.question = text
        st.caption(why)

question = st.text_input(
    "Ask a business question",
    key="question",
    label_visibility="collapsed",
)

if not DB_PATH.exists():
    st.error(f"{DB_PATH.name} not built. Run `uv run python scripts/load_data.py`.")
    st.stop()

go = st.button("Ask", type="primary")
if not go and "last_result" not in st.session_state:
    st.info("Pick an example or type a question, then press **Ask**.")
    st.stop()

if go:
    pipeline = _pipeline(provider_name)
    with st.spinner("one structured call, then a deterministic compiler…"):
        try:
            st.session_state.last_result = pipeline.answer(question)
        except LlmError as error:
            st.error(f"Provider error: {error}")
            st.stop()
        except Exception as error:
            st.error(f"{type(error).__name__}: {error}")
            st.stop()

result: PipelineResult = st.session_state.last_result
answer = result.answer

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
