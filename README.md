# Canon

A trustworthy analytics agent. Ask a business question in plain English; get back a
verified number, the exact SQL that produced it, the canonical metric definition used,
and — when the question is genuinely ambiguous — a clarifying question instead of a
guess.

Two ideas carry the project:

1. **The LLM never writes SQL.** It fills a typed `QueryIntent`; a deterministic compiler
   turns intent + canonical metric definitions into SQL. Metric and dimension names are
   enums generated from `canon/semantic/metrics.yaml`, so a hallucinated metric is not
   caught — it is unrepresentable.
2. **Refusal is a scored metric.** Ambiguous questions are test cases where asking is the
   correct answer and confidently guessing is a scored failure. See
   [docs/01-evaluation.md](docs/01-evaluation.md).

Results: **TBD** — no number goes in this README until `benchmark/harness` has run. See
the number rule in [CLAUDE.md](CLAUDE.md).

## Setup

```bash
uv sync                        # Python 3.12 is pinned via .python-version
uv run python scripts/load_data.py    # downloads Olist -> data/canon.duckdb (idempotent)
uv run pytest                  # compiler tests; zero LLM calls
```

For the agent (Day 2+), copy `.env.example` to `.env` and add an Anthropic API key. The
loader, semantic layer, and compiler need no key.

## Layout

See the directory layout section of [CLAUDE.md](CLAUDE.md). Dependency direction is
one-way: `schemas` <- `semantic` <- `compiler` <- `agent` <- `app`.
