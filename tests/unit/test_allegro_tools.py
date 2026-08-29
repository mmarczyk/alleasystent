"""Unit tests for agents/allegro/allegro_tools.py."""
from __future__ import annotations

import pytest


class TestAllegroTools:
    @pytest.fixture(autouse=True)
    def load_tools(self):
        from agents.allegro.allegro_tools import ALLEGRO_TOOLS
        self.tools = ALLEGRO_TOOLS

    def test_is_list(self):
        assert isinstance(self.tools, list)

    def test_has_at_least_10_tools(self):
        assert len(self.tools) >= 10

    def test_all_have_type_function(self):
        for tool in self.tools:
            assert tool.get("type") == "function", f"Tool missing type=function: {tool}"

    def test_all_have_name(self):
        for tool in self.tools:
            assert "name" in tool["function"], f"Tool missing name: {tool}"
            assert isinstance(tool["function"]["name"], str)
            assert tool["function"]["name"] != ""

    def test_all_have_description(self):
        for tool in self.tools:
            assert "description" in tool["function"]
            assert len(tool["function"]["description"]) > 10

    def test_all_have_parameters(self):
        for tool in self.tools:
            assert "parameters" in tool["function"]
            params = tool["function"]["parameters"]
            assert params.get("type") == "object"
            assert "properties" in params

    def test_known_tool_names_present(self):
        names = {t["function"]["name"] for t in self.tools}
        expected = {
            "get_new_orders",
            "get_orders",
            "get_order_details",
            "get_active_offers",
            "update_offer_price",
            "update_offer_stock",
            "send_message_to_buyer",
            "get_message_threads",
            "get_account_info",
        }
        for name in expected:
            assert name in names, f"Expected tool '{name}' not found"

    def test_names_are_unique(self):
        names = [t["function"]["name"] for t in self.tools]
        assert len(names) == len(set(names)), "Duplicate tool names found"

    def test_get_orders_has_status_enum(self):
        tool = next(t for t in self.tools if t["function"]["name"] == "get_orders")
        props = tool["function"]["parameters"]["properties"]
        assert "status" in props
        assert "enum" in props["status"]


class TestToolLabelCoverage:
    """Every tool must be reachable through the label filter — an orphaned
    tool (no label, or a label with no stems) would silently vanish from
    every tool-select call regardless of what the user asks."""

    # ask_clarifying_question is deliberately label-less — tools_for_labels()
    # injects it into every filtered subset regardless of topic (see its
    # docstring), since it's the escape hatch for a missing/ambiguous
    # parameter WITHIN whatever domain got matched, not a domain itself.
    _UNLABELED_TOOLS = frozenset({"ask_clarifying_question"})

    def test_every_tool_has_a_label(self):
        from agents.allegro.allegro_tools import ALLEGRO_TOOLS, _TOOL_LABELS
        names = {t["function"]["name"] for t in ALLEGRO_TOOLS} - self._UNLABELED_TOOLS
        assert names == set(_TOOL_LABELS.keys())

    def test_every_label_has_stems(self):
        from agents.allegro.allegro_tools import _LABEL_STEMS, _TOOL_LABELS
        used_labels = set(_TOOL_LABELS.values())
        assert used_labels <= set(_LABEL_STEMS.keys())
        for label in used_labels:
            assert _LABEL_STEMS[label], f"Label '{label}' has no stems"


class TestMatchedLabels:
    def test_finds_label_from_stem_prefix(self):
        from agents.allegro.allegro_tools import matched_labels
        assert "zamowienia" in matched_labels("jakie mam nowe zamówienia")

    def test_diacritic_free_query_still_matches(self):
        from agents.allegro.allegro_tools import matched_labels
        assert "zamowienia" in matched_labels("ile mam zamowien")

    def test_handles_inflected_forms(self):
        from agents.allegro.allegro_tools import matched_labels
        for word in ("zamówienie", "zamówienia", "zamówień", "zamówieniem", "zamówieniu"):
            assert "zamowienia" in matched_labels(word), word

    def test_unrelated_text_matches_nothing(self):
        from agents.allegro.allegro_tools import matched_labels
        assert matched_labels("jaka jest dzisiaj pogoda") == set()

    def test_multiple_labels_in_one_query(self):
        from agents.allegro.allegro_tools import matched_labels
        found = matched_labels("wystaw fakturę dla ostatniego zamówienia")
        assert {"zamowienia", "faktury"} <= found

    def test_english_query_matches(self):
        from agents.allegro.allegro_tools import matched_labels
        assert "zamowienia" in matched_labels("show me my new orders")


class TestOrderListingConsistency:
    """The three order tools are three intent presets over ONE implementation
    (AllegroAgent._ORDERS_PRESETS) — these pin the parts of that contract that
    live in the schemas."""

    _ORDER_TOOLS = ("get_new_orders", "get_orders", "get_orders_delivery")

    @pytest.fixture(autouse=True)
    def load_tools(self):
        from agents.allegro.allegro_tools import ALLEGRO_TOOLS
        self.tools = ALLEGRO_TOOLS

    def test_all_order_listings_reply_as_plain_text(self):
        from agents.allegro.allegro_tools import TOOL_OUTPUT_FORMAT
        for name in self._ORDER_TOOLS:
            assert TOOL_OUTPUT_FORMAT[name] == "chat", f"{name} would render as a document"

    def test_shared_arguments_have_one_definition(self):
        """Same argument name ⇒ same schema (bar per-tool description/default
        overrides), so a filter can't quietly mean two things."""
        params = {
            name: next(t for t in self.tools if t["function"]["name"] == name)["function"]["parameters"]["properties"]
            for name in self._ORDER_TOOLS
        }
        for shared in ("status", "fulfillment_status", "buyer_login", "count_only",
                       "dispatch_before_local"):
            definitions = [p[shared] for p in params.values() if shared in p]
            assert len(definitions) > 1, f"{shared} is not shared by any two order tools"
            for definition in definitions[1:]:
                assert definition.get("type") == definitions[0].get("type")
                assert definition.get("enum") == definitions[0].get("enum")

    def test_general_listing_carries_every_filter(self):
        props = next(
            t for t in self.tools if t["function"]["name"] == "get_orders"
        )["function"]["parameters"]["properties"]
        for expected in ("status", "fulfillment_status", "buyer_login", "line_items_sent",
                         "bought_after_local", "bought_before_local",
                         "paid_after_local", "paid_before_local",
                         "dispatch_after_local", "dispatch_before_local",
                         "include_delivery", "count_only", "limit"):
            assert expected in props, f"get_orders lost the {expected} filter"


class TestSelectToolsForContext:
    def test_no_label_match_returns_none(self):
        from agents.allegro.allegro_tools import select_tools_for_context
        assert select_tools_for_context("jaka jest dzisiaj pogoda") is None

    def test_domain_query_excludes_unrelated_tools(self):
        from agents.allegro.allegro_tools import select_tools_for_context
        names = {t["function"]["name"] for t in select_tools_for_context("jakie mam nowe zamówienia")}
        assert "get_new_orders" in names
        assert "issue_invoice_for_order" not in names
        assert "get_message_threads" not in names

    def test_multi_topic_query_includes_both_domains(self):
        from agents.allegro.allegro_tools import select_tools_for_context
        names = {t["function"]["name"] for t in select_tools_for_context("nowe zamówienia i moje konto")}
        assert "get_new_orders" in names
        assert "get_account_info" in names

    def test_returned_tools_are_well_formed(self):
        from agents.allegro.allegro_tools import select_tools_for_context
        for tool in select_tools_for_context("jakie mam oferty"):
            assert tool["type"] == "function"
            assert "name" in tool["function"]


class TestLabelPhraseCoverage:
    """Sample realistic Polish (and one English) phrasing per tool — mostly
    lifted straight from that tool's own description — and confirm it
    survives the label filter. A tool missing here would be unreachable for
    a completely ordinary query."""

    _CASES = [
        ("jakie mam nowe zamówienia", "get_new_orders"),
        ("pokaż listę zamówień z tego miesiąca", "get_orders"),
        ("jaki jest status tego zamówienia", "get_order_details"),
        ("jakie kurierzy w zamówieniach do wysyłki", "get_orders_delivery"),
        ("pokaż moje oferty", "get_active_offers"),
        ("podsumowanie moich ofert", "get_offers_summary"),
        ("oferty z niskim stanem magazynowym", "query_offers_by_stock"),
        ("oferty poniżej 50 zł", "query_offers_by_price"),
        ("przygotuj zamówienie uzupełniające do dostawcy", "get_products_to_reorder"),
        ("szczegóły tej oferty", "get_offer_details"),
        ("zmień cenę oferty", "update_offer_price"),
        ("zmień stan magazynowy oferty", "update_offer_stock"),
        ("odpisz kupującemu na wiadomość", "send_message_to_buyer"),
        ("czy mam nowe wiadomości", "get_message_threads"),
        ("przeczytaj treść wiadomości od Jana", "get_thread_messages"),
        ("moje konto allegro", "get_account_info"),
        ("jakie opłaty miałem w tym miesiącu", "get_billing_summary"),
        ("ile zarobiłem w tym tygodniu", "get_sales_summary"),
        ("jakie zamówienia czekają na fakturę", "get_orders_pending_invoice"),
        ("dane do faktury dla tego zamówienia", "get_order_invoice_data"),
        ("wystaw brakujące faktury za ten miesiąc", "preview_pending_invoices"),
        ("wystaw fakturę dla tego zamówienia", "issue_invoice_for_order"),
        ("załącz fakturę do zamówienia w allegro", "attach_invoice_to_allegro_order"),
        ("wyślij fakturę do ksef", "send_invoice_to_ksef"),
        ("czy mam jakieś zwroty", "get_new_returns"),
        ("zwroty do obsłużenia", "get_returns_to_process"),
        ("jakie mam reklamacje", "get_new_complaints"),
        ("włącz monitoring zamówień", "suggest_order_monitoring"),
        ("wyłącz monitoring zamówień", "disable_order_monitoring"),
        ("włącz powiadomienia o fakturach", "suggest_invoice_monitoring"),
        ("wyłącz powiadomienia o fakturach", "disable_invoice_monitoring"),
        ("chcę przypomnienia o niewystawionych fakturach", "suggest_invoice_reminder"),
        ("wyłącz przypomnienia o fakturach", "disable_invoice_reminder"),
        ("powiadamiaj mnie o nowych wiadomościach", "suggest_message_monitoring"),
        ("wyłącz monitoring wiadomości", "disable_message_monitoring"),
        ("powiadamiaj mnie o zwrotach i reklamacjach", "suggest_returns_monitoring"),
        ("wyłącz monitoring zwrotów", "disable_returns_monitoring"),
    ]

    @pytest.mark.parametrize("query,tool_name", _CASES)
    def test_query_surfaces_expected_tool(self, query, tool_name):
        from agents.allegro.allegro_tools import select_tools_for_context
        tools = select_tools_for_context(query)
        assert tools is not None, f"No label matched for: {query!r}"
        names = {t["function"]["name"] for t in tools}
        assert tool_name in names, f"{tool_name!r} missing for query {query!r} (got {sorted(names)})"
