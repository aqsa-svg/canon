"""Canon demo -- Day 1 state: the compiler, with no LLM anywhere.

You build a ``QueryIntent`` with the controls instead of asking a question in English;
the natural-language front end lands on Day 2. What this page already shows is the part
that makes the project trustworthy: the exact SQL, the canonical definition that produced
it, the caveats that travel with the number, and a typed refusal when the
metric/dimension pair is illegal.

Run:  uv run streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from canon.compiler import compile_intent
from canon.schemas.errors import InvalidIntent
from canon.schemas.intent import QueryIntent, SortSpec, TimeRange
from canon.semantic.enums import DimensionName, MetricName, SortDirection, SortTarget, TimeGrain
from canon.semantic.loader import load_semantic_layer, semantic_layer_hash

DB_PATH = Path(__file__).resolve().parent / "data" / "canon.duckdb"

st.set_page_config(page_title="Canon", layout="wide")
layer = load_semantic_layer()

st.title("Canon")
st.caption(
    f"Deterministic compiler over {len(layer.metrics)} canonical metrics. "
    f"No LLM in this page. metrics.yaml @ {semantic_layer_hash()[:12]}"
)

with st.sidebar:
    st.header("Intent")
    st.caption("This is exactly the object the LLM will fill on Day 2 -- nothing more.")

    metric_name = st.selectbox("metric", sorted(str(name) for name in MetricName))
    card = layer.metrics[metric_name]

    # Only the dimensions this metric actually permits are offered. The wrong-grain
    # refusal is still reachable below, because the compiler is the authority, not the UI.
    dimension_names = st.multiselect(
        "dimensions", sorted(str(name) for name in DimensionName), default=[]
    )
    grain_choice = st.selectbox("time_grain", ["(none)", *(grain.value for grain in TimeGrain)])
    start = st.text_input("time_range.start (inclusive)", value="")
    end = st.text_input("time_range.end (exclusive)", value="")
    sort_metric_desc = st.checkbox("sort by metric, descending", value=True)
    limit_value = st.number_input("limit (0 for none)", min_value=0, max_value=1000, value=0)

try:
    intent = QueryIntent(
        metric=MetricName(metric_name),
        dimensions=tuple(DimensionName(name) for name in dimension_names),
        time_grain=None if grain_choice == "(none)" else TimeGrain(grain_choice),
        time_range=(TimeRange(start=start or None, end=end or None) if (start or end) else None),
        sort=(
            SortSpec(target=SortTarget.METRIC, direction=SortDirection.DESC)
            if sort_metric_desc and (dimension_names or grain_choice != "(none)")
            else None
        ),
        limit=int(limit_value) or None,
    )
except ValueError as error:
    st.error(f"That is not a valid intent:\n\n{error}")
    st.stop()

try:
    compiled = compile_intent(intent, layer)
except InvalidIntent as invalid:
    detail = invalid.detail
    st.warning(f"**Refused before any SQL was built** — `{detail.reason.value}`")
    st.write(detail.message)
    if detail.suggestions:
        st.write("**Metrics that can answer this instead:** " + ", ".join(detail.suggestions))
    st.caption(
        "This is the point of the design: an illegal metric/dimension pair fails in "
        "validation, so a plausible-looking wrong number is never produced."
    )
    st.stop()

left, right = st.columns([3, 2])

with left:
    st.subheader("Result")
    if not DB_PATH.exists():
        st.info(f"{DB_PATH.name} not built yet — run `uv run python scripts/load_data.py`.")
    else:
        connection = duckdb.connect(str(DB_PATH), read_only=True)
        try:
            rows = connection.execute(compiled.sql).fetchall()
        finally:
            connection.close()

        frame = pd.DataFrame(rows, columns=list(compiled.columns))
        if len(frame) == 1 and len(frame.columns) == 1:
            st.metric(compiled.columns[0], f"{frame.iat[0, 0]:,}")
        else:
            st.dataframe(frame, width="stretch")
            spec = compiled.chart_spec
            if spec is not None and len(frame) > 1:
                plotted = frame.set_index(spec.x)[spec.y]
                (st.line_chart if spec.kind.value == "line" else st.bar_chart)(plotted)

    st.subheader("SQL")
    st.caption("Compiled, not written by a model. Runnable as-is.")
    st.code(compiled.sql, language="sql")

with right:
    st.subheader("Metric definition used")
    st.markdown(f"**{compiled.metric.name}** — owned by `{compiled.metric.owner}`")
    st.write(compiled.metric.description)
    st.caption(f"Grain: {compiled.metric.grain}")
    st.caption(f"Expression: `{compiled.metric.expression}`")

    st.subheader(f"Caveats ({len(compiled.caveats)})")
    for caveat in compiled.caveats:
        st.markdown(f"- _{caveat.source.value}_ — {caveat.text}")

    with st.expander("The intent object"):
        st.code(intent.model_dump_json(indent=2), language="json")
