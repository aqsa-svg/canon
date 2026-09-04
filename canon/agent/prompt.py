"""The system prompt: the entire canonical layer, rendered as text.

All 24 metric cards go in the prompt. That is the architectural decision CLAUDE.md
records as deliberate: with a semantic layer this size, retrieval would add a failure
mode — the wrong card retrieved, invisibly — to buy nothing. The whole layer fits, so
the whole layer goes.

The rendering is **deterministic and stable**: sorted keys, no timestamps, no run ids.
That is a caching requirement rather than a stylistic preference. Prompt caching is a
prefix match, so a single varying byte anywhere in this text would invalidate the cache
on every request and quietly multiply the cost of the benchmark.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict

from canon.semantic.cards import MetricCard, SemanticLayer

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class DataWindow(BaseModel):
    """The span the warehouse actually covers.

    In the prompt because the alternative is worse: without it the model anchors
    relative time on its own idea of "now", which is not in the data, and silently
    returns an empty result for "last month". With it, an unanchored relative reference
    is visibly unanswerable and becomes a clarifying question.
    """

    model_config = _FROZEN

    first_order: date
    last_order: date


def _render_metric(card: MetricCard) -> str:
    lines = [
        f"### {card.name}",
        f"  means: {card.description.strip()}",
        f"  grain: {card.grain}",
        f"  owner: {card.owner}",
    ]
    if card.synonyms:
        lines.append(f"  phrasings that map here: {', '.join(sorted(card.synonyms))}")
    lines.append(
        "  can be broken down by: "
        + (", ".join(sorted(card.valid_dimensions)) if card.valid_dimensions else "nothing")
    )
    if card.default_filters:
        applied = "; ".join(
            f"{item.column} {item.op.value}"
            + (f" ({', '.join(str(value) for value in item.values)})" if item.values else "")
            for item in card.default_filters
        )
        lines.append(f"  always filtered by: {applied}")
    if card.definition_change is not None:
        change = card.definition_change
        lines.append(
            f"  DEFINITION CHANGES on {change.effective_date}: "
            f"{change.before} before, {change.after} from that date. Prior periods were "
            "not restated."
        )
    for caveat in card.caveats:
        lines.append(f"  caveat: {caveat.strip()}")
    return "\n".join(lines)


def _render_dimensions(layer: SemanticLayer) -> str:
    return "\n".join(
        f"- {name}: {layer.dimensions[name].description.strip()}"
        for name in sorted(layer.dimensions)
    )


def _render_ambiguous(layer: SemanticLayer) -> str:
    blocks = []
    for term in sorted(layer.ambiguous_terms, key=lambda item: item.term):
        blocks.append(
            f'- "{term.term}" could mean: {", ".join(term.candidates)}\n'
            f"    why there is no default: {term.why.strip()}"
        )
    return "\n".join(blocks)


INSTRUCTIONS = """\
You translate business questions into a typed QueryIntent over a fixed catalogue of
metrics. You never write SQL. A deterministic compiler turns your QueryIntent into SQL
using the definitions below, so your only job is to choose correctly among them.

For every question, do both of these:

1. Score every metric that could plausibly be what the question means, in `candidates`.
   Score on fit to the question's *wording and intent*, not on which is more common.
   - One clear match: score it high and score nothing else close to it.
   - Two genuinely plausible readings: score both close together. Close scores are how
     you say "this question is ambiguous" — that is a useful, correct answer here, not a
     failure. Do not manufacture a gap you do not believe in.

2. Either fill `intent`, or say what is missing.
   - Fill `intent` when the question determines a metric, a time range, and any
     breakdown it asks for.
   - Leave `intent` null and list `missing_information` when it does not. Each entry
     names one specific gap ("which revenue definition", "which date range").

Rules that matter:

- A term listed under AMBIGUOUS TERMS below has no default. If a question turns on one
  and offers nothing else to settle it, that is missing information — do not pick the
  one that seems more likely.
- Only break a metric down by a dimension listed under that metric's "can be broken down
  by". The lists differ between metrics on purpose: where a breakdown is excluded, the
  SQL would run and the number would be wrong.
- If a question needs a metric that is not in the catalogue (margin, profit, CAC, LTV,
  inventory, ad spend, anything the definitions below do not cover), do not substitute
  the nearest available one. Leave `intent` null and say the metric is not defined.
- Time ranges are half-open: start inclusive, end exclusive. A single month is
  start=2018-03-01, end=2018-04-01. A full year is start=2018-01-01, end=2019-01-01.
- There is no "today". The data ends on a fixed date given below. A question that says
  "recently", "last month", or "this quarter" without anchoring it to a date is missing
  information, because any anchor you chose would be your invention, not the question's.
- For period-over-period questions set `comparison` and a bounded `time_range`. The
  compiler derives the earlier range; do not compute it yourself and do not pass a
  `limit` with a comparison.
- `reading` is your one-sentence paraphrase of the question. Write it before deciding.
"""


def build_system_prompt(layer: SemanticLayer, window: DataWindow) -> str:
    """Render the whole canonical layer plus instructions.

    Deterministic for a given layer and window: identical bytes on every call, which is
    what lets the prompt cache actually hit.
    """
    metrics = "\n\n".join(_render_metric(layer.metrics[name]) for name in sorted(layer.metrics))
    return f"""{INSTRUCTIONS}

## DATA WINDOW

Orders in this warehouse span {window.first_order} to {window.last_order} inclusive.
Nothing exists outside that range. Note that the last weeks before the end are very
sparse, so a question about the final period may be answerable but unrepresentative.

## AMBIGUOUS TERMS — no default exists for these

{_render_ambiguous(layer)}

## DIMENSIONS

{_render_dimensions(layer)}

## METRIC CATALOGUE — {len(layer.metrics)} metrics, this is the complete list

{metrics}
"""
