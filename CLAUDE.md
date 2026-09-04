# Canon — working agreement

Canon is a **trustworthy analytics agent**. It answers plain-English business questions
over a warehouse and returns a *verified* number, the exact SQL that produced it, the
canonical metric definition it used, and — when the question is genuinely ambiguous — a
refusal in the form of a clarifying question instead of an answer.

The name refers to the **canonical metric definition**: the single authoritative source
for what "revenue" means. Canon *enforces* that definition. It does not guess at it.

---

## The two ideas. Protect these above all else.

### 1. The LLM never writes SQL.

The model's only job is to fill in a typed `QueryIntent` object. A **deterministic
compiler** turns `QueryIntent` + canonical metric definitions into SQL.

- Metric and dimension names are **enums generated from YAML**. A hallucinated metric is
  not "caught" — it is *unrepresentable*. The structured-output call cannot emit it.
- No string interpolation of model output into SQL. Ever. Not for a filter value, not
  for a column, not for a date literal. Values are bound; identifiers come from the
  semantic layer.
- If you ever find yourself writing `f"SELECT {something_from_the_model}"`, stop. That
  line is the project's failure mode.
- The compiler is pure: `(QueryIntent, MetricCard[]) -> SQL`. Same inputs, same SQL,
  byte for byte. No clock, no randomness, no network.

### 2. Refusal is a scored metric.

Ambiguous questions are first-class test cases. For them, **asking a clarifying question
is the correct answer and confidently guessing is scored a failure**.

- "What was revenue last month?" is ambiguous when both `gross_revenue` and
  `net_revenue` exist and neither is a defensible default. The right output is a
  `clarifying_question`, not a number.
- A confident wrong answer is worse than a refusal. The benchmark reports
  **confident-wrong rate** separately, and it is the number we most want near zero.
- Refusal must never become a hedge-everything reflex. Answering a clear question with a
  clarifying question is *also* a scored failure. See `docs/01-evaluation.md`.

---

## Out of scope. Do not build. Do not suggest.

This is a ~25 hour, 3-day build. The following are deliberately excluded. Proposing them
is not helpful; it is scope creep with extra steps.

- **Vector DB / embedding retrieval over metrics.** With ~20 metrics the entire semantic
  layer fits in the prompt. This is a *deliberate architectural decision to document*,
  not an omission to apologise for. Retrieval would add a failure mode (wrong metric
  card retrieved) to buy nothing.
- Driver analysis / "why did X change" decomposition
- Forecasting or any time-series projection
- Dual-query reconciliation (running two SQL variants and comparing)
- Neo4j or any graph store
- Slack bot / any chat surface other than the Streamlit demo
- CI eval gate (the harness runs on demand, by hand)
- Adversarial / red-team eval set (the honest eval set is the deliverable)
- Any JavaScript frontend. Streamlit only.

---

## Directory layout

```
canon/
  semantic/      # canonical metric layer: metrics.yaml is the moat
    metrics.yaml #   24 metric cards + join graph + dimensions. The source of truth.
    enums.py     #   MetricName / DimensionName, GENERATED from the YAML at import
    cards.py     #   MetricCard, SemanticLayer, and the load-time consistency proof
    loader.py    #   cached, validated SemanticLayer
  compiler/      # QueryIntent -> SQL. Deterministic. No LLM, no network, no clock.
    validate.py  #   legality checks; raises InvalidIntent BEFORE any SQL exists
    plan.py      #   QueryPlan: a validated intent, so the builder decides nothing
    build.py     #   SQL construction (joins, grain, fan-out, filter order, caveats)
  agent/         # the ONLY place the LLM lives. Fills QueryIntent, nothing else.
  schemas/       # boundary models: QueryIntent, Answer, InvalidIntent
benchmark/
  questions/     # the eval set: questions + gold answers + refusal cases
  harness/       # runs the eval, computes the three metrics + confident-wrong rate
data/
  raw/           # downloaded CSVs. gitignored.
  canon.duckdb   # built by scripts/load_data.py. gitignored, reproducible.
docs/
  01-evaluation.md  # written BEFORE the code it evaluates. Read it first.
scripts/         # one-shot operational scripts (data loading)
app.py           # Streamlit demo
```

Dependency direction is one-way, deepest first:

```
semantic  ->  schemas  ->  compiler  ->  agent  ->  app
```

`semantic` is the deepest module and imports nothing from `canon`. This is the opposite
of where the metric card models would normally live, and the reason is the first core
idea: `MetricName` is *generated* from `metrics.yaml`, so anything that mentions a metric
type must sit above the semantic layer. Putting the cards in `schemas` instead would
force the schema layer to read YAML, which is worse. Nothing lower ever imports something
higher; `compiler` must never import `agent`.

---

## Conventions

**Typed everything.**
- No bare `dict` or `list` crosses a module boundary. If two modules exchange data, that
  data is a Pydantic model in `canon/schemas/`. Dicts are fine *inside* a function.
- Every public function has full annotations. `Any` requires a comment explaining why.
- Pydantic v2 at every boundary: file I/O (YAML, JSON), LLM output, SQL results, HTTP.
  Parse at the edge; the interior deals only in validated models.
- `model_config = ConfigDict(frozen=True, extra="forbid")` on everything that is data
  rather than state. `extra="forbid"` is what turns a typo in `metrics.yaml` into a
  loud error instead of a silently ignored key.
- Errors are typed and specific: `InvalidIntent`, not `ValueError`. An error carries
  enough structure for the agent layer to turn it into a clarifying question.

**Fail early, fail loudly.**
- Illegal metric/dimension pairs are rejected during *validation*, before SQL is built
  and long before anything is executed. "Compiles but returns nonsense" is the outcome
  this whole design exists to prevent.
- No silent fallbacks. No `except Exception: pass`. No defaulting a missing time grain
  to "month" because it seemed reasonable. Missing information is either a validation
  error or a clarifying question — never an assumption.

**Every generated SQL string is parsed by sqlglot before it is executed.** If sqlglot
can't parse it, that's a compiler bug and it raises, loudly.

**Tests.**
- The compiler is pure, so test it exhaustively and without mocks: intent in, SQL and
  result out.
- Day 1 has **zero LLM calls in the test suite**, and that stays true — LLM-dependent
  tests live behind an explicit marker and are never required to pass for the compiler
  to be considered correct.

**Tooling.**
- `uv` for everything. `uv run <cmd>`, never a bare `python`. Python is pinned to 3.12.
- `uv run ruff check --fix . && uv run ruff format .` before every commit.
- `uv run pytest` before every commit.
- Commit after each completed step, with a message that says what changed and why.

---

## The number rule

**No number appears in any document, README, docstring, comment, or commit message
unless it came from a real benchmark run on this machine.**

Write `TBD` instead. Every time. `TBD` is honest and costs nothing; an invented "94%
semantic accuracy" is the exact failure Canon exists to argue against, and putting one in
our own docs would be self-refuting.

This applies to:
- accuracy / refusal / confident-wrong figures — `TBD` until `benchmark/harness` runs
- latency and cost figures — `TBD` until measured
- row counts and dataset shapes — allowed **only** when printed by the loader against
  the real file, and cited as such

When a real run produces a number, record *which run* produced it (date, commit, eval
set version) next to the number. A number without provenance gets deleted, not trusted.
