"""Tests on metrics.yaml itself.

The YAML is the moat, so it gets tested like code. These checks are about the integrity
of the definitions rather than the compiler: a metric with no owner, two metrics claiming
the same synonym, or a declared ambiguity that some card quietly resolves anyway are all
ways the semantic layer could stop being authoritative while everything still ran.
"""

from __future__ import annotations

from datetime import date

from canon.semantic.cards import SemanticLayer
from canon.semantic.enums import Cardinality, DimensionName, MetricName

DATA_FIRST_ORDER = date(2016, 9, 4)
DATA_LAST_ORDER = date(2018, 10, 17)
"""The warehouse's real span, printed by scripts/load_data.py."""


class TestEveryCardIsComplete:
    def test_every_metric_is_documented_and_owned(self, layer: SemanticLayer) -> None:
        for name, card in layer.metrics.items():
            assert len(card.description) > 40, f"{name}: description is too thin to arbitrate"
            assert card.owner, f"{name}: no owner means no one to ask"
            assert card.synonyms, f"{name}: no synonyms means the term cannot be recognised"
            assert card.grain, f"{name}: grain unstated"

    def test_every_metric_has_at_least_one_caveat(self, layer: SemanticLayer) -> None:
        """Every real metric has something a reader needs to know.

        A card with no caveats is far more likely to be unfinished than to be perfect.
        """
        for name, card in layer.metrics.items():
            assert card.caveats, f"{name}: no caveats -- suspicious rather than clean"

    def test_metric_count_matches_the_generated_enum(self, layer: SemanticLayer) -> None:
        assert len(layer.metrics) == len(list(MetricName))
        assert len(layer.dimensions) == len(list(DimensionName))

    def test_every_dimension_is_reachable_by_some_metric(self, layer: SemanticLayer) -> None:
        """A dimension no metric accepts is dead weight the LLM can still choose."""
        used = {dimension for card in layer.metrics.values() for dimension in card.valid_dimensions}
        assert set(layer.dimensions) - used == set()


class TestSynonymsDoNotCollide:
    def test_no_synonym_maps_to_two_metrics(self, layer: SemanticLayer) -> None:
        """A phrase that means two things is an ambiguity, not a synonym.

        If two cards claimed the same phrase, the resolution would silently depend on
        iteration order. Such a phrase belongs in ambiguous_terms instead.
        """
        seen: dict[str, str] = {}
        for name, card in layer.metrics.items():
            for synonym in card.synonyms:
                key = synonym.lower().strip()
                assert key not in seen, (
                    f"{synonym!r} is claimed by both {seen.get(key)} and {name}; "
                    "declare it in ambiguous_terms instead"
                )
                seen[key] = name

    def test_no_ambiguous_term_is_also_a_synonym(self, layer: SemanticLayer) -> None:
        """A declared fork must not be quietly resolved by one card claiming the word.

        This is the check that keeps refusal honest: if "revenue" were a synonym of
        gross_revenue, the system would have a default and the ambiguity would be
        decoration rather than a real fork.
        """
        synonyms = {
            synonym.lower().strip() for card in layer.metrics.values() for synonym in card.synonyms
        }
        for term in layer.ambiguous_terms:
            assert term.term.lower().strip() not in synonyms, (
                f"{term.term!r} is declared ambiguous but is also a synonym of a single "
                "metric, which would give it a de facto default"
            )


class TestTheThreeTraps:
    """The traps are load-bearing for the eval, so their properties are asserted."""

    def test_trap_one_similar_names_different_meaning(self, layer: SemanticLayer) -> None:
        gross = layer.metrics["gross_revenue"]
        net = layer.metrics["net_revenue"]

        # They differ on freight...
        assert "freight_value" in gross.expression
        assert "freight_value" not in net.expression
        # ...and on which orders count.
        assert gross.default_filters == ()
        assert net.default_filters != ()
        # So confusing them is consequential, not cosmetic.
        assert gross.expression != net.expression

    def test_the_bare_word_revenue_has_no_default(self, layer: SemanticLayer) -> None:
        terms = {term.term: term for term in layer.ambiguous_terms}
        assert "revenue" in terms
        candidates = set(terms["revenue"].candidates)
        assert {"gross_revenue", "net_revenue"} <= candidates
        assert terms["revenue"].why

    def test_trap_two_exactly_one_metric_changes_definition(self, layer: SemanticLayer) -> None:
        changing = [card for card in layer.metrics.values() if card.definition_change is not None]
        assert len(changing) == 1, "the eval assumes a single definition-change trap"

        change = changing[0].definition_change
        assert change is not None
        # The change must fall inside the data, or nothing would ever span it.
        assert DATA_FIRST_ORDER < change.effective_date < DATA_LAST_ORDER
        assert change.before != change.after
        assert len(change.caveat) > 80, "the caveat has to actually explain the discontinuity"
        # The expression must genuinely branch on the date.
        assert "CASE" in changing[0].expression.upper()
        assert str(change.effective_date) in changing[0].expression

    def test_trap_three_a_plausible_dimension_is_deliberately_excluded(
        self, layer: SemanticLayer
    ) -> None:
        """avg_order_value by product_category is joinable but excluded on purpose.

        This is what proves the two protections are independent. If the exclusion were
        only about fan-out, the join graph would already have caught it.
        """
        card = layer.metrics["avg_order_value"]
        assert "product_category" not in card.valid_dimensions
        assert layer.join_path(card.base, "products") is not None
        # And the card explains itself, so a refusal can quote a reason.
        assert any("categor" in caveat for caveat in card.caveats)

    def test_the_excluded_dimension_is_legal_elsewhere(self, layer: SemanticLayer) -> None:
        """The exclusion is about grain, not about the dimension being unusable."""
        assert "product_category" in layer.metrics["avg_item_price"].valid_dimensions


class TestJoinGraph:
    def test_no_one_to_many_edge_is_declared(self, layer: SemanticLayer) -> None:
        """One-to-many edges are omitted rather than declared and skipped.

        Leaving them out means a future change cannot make them traversable by
        accident; the graph contains only joins that are safe to walk.
        """
        for name, table in layer.tables.items():
            for edge in table.joins:
                assert edge.cardinality is Cardinality.MANY_TO_ONE, (
                    f"{name} -> {edge.to} is declared one_to_many; omit it instead"
                )

    def test_the_fan_out_join_is_absent(self, layer: SemanticLayer) -> None:
        """orders -> order_payments and orders -> order_items must not be reachable."""
        assert layer.join_path("orders", "order_payments") is None
        assert layer.join_path("orders", "order_items") is None
        assert layer.join_path("orders", "order_reviews") is None

    def test_geolocation_is_intentionally_unreachable(self, layer: SemanticLayer) -> None:
        """Zip prefix is not unique in geolocation, so any join through it fans out."""
        assert "geolocation" in layer.tables
        for base in ("orders", "order_items", "customers"):
            assert layer.join_path(base, "geolocation") is None

    def test_every_fact_table_reaches_time(self, layer: SemanticLayer) -> None:
        """Every metric can be put on a timeline, or it cannot answer a business question."""
        for name, card in layer.metrics.items():
            assert layer.join_path(card.base, card.time_table) is not None, name

    def test_join_paths_are_stable(self, layer: SemanticLayer) -> None:
        """Repeated resolution gives the identical path, so SQL cannot drift."""
        first = layer.join_path("order_items", "product_category_translation")
        for _ in range(5):
            assert layer.join_path("order_items", "product_category_translation") == first
