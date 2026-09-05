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

    def test_buyer_words_that_also_describe_orders_stay_out_of_the_label(self):
        """'klient' is a buyer word a seller uses just as often ABOUT an order
        ('co klient odebrał'). Labeling it would put two topics on those queries
        and switch off the deterministic order dispatch — see _LABEL_STEMS."""
        from agents.allegro.allegro_tools import matched_labels
        assert matched_labels("co klient odebrał") == {"zamowienia"}

    def test_english_query_matches(self):
        from agents.allegro.allegro_tools import matched_labels
        assert "zamowienia" in matched_labels("show me my new orders")


class TestNamedBuyerLogin:
    """"z konta np1988" names SOMEONE ELSE'S account — the buyer's login, which
    only the order listing can filter by. Before this, "czy w tym roku kupował
    ode mnie ktoś z konta np1988" matched {konto, kupujacy}: get_orders was not
    even among the schemas the model saw, and get_buyers answered with the whole
    year's customer list."""

    @pytest.mark.parametrize("query,login", [
        ("Czy w tym roku kupował ode mnie ktoś z konta np1988", "np1988"),
        ("czy z konta 'np1988' coś kupiono?", "np1988"),
        ("co kupił użytkownik anna.kowalska88", "anna.kowalska88"),
        ("zamówienia od kupującego marek_zielinski", "marek_zielinski"),
        ("login: kasia.w — jakie ma zamówienia", "kasia.w"),
        ("pokaż zamówienia z konta o nazwie sklep-abc", "sklep-abc"),
    ])
    def test_finds_the_login(self, query, login):
        from agents.allegro.allegro_tools import named_buyer_login
        assert named_buyer_login(query) == login

    @pytest.mark.parametrize("query", [
        "moje konto allegro",
        "moje konto allegro jest zawieszone",
        "jakie mam dane konta",
        "ile zamówień z konta firmowego",
        "pokaż listę kupujących z tego roku",
        "ile mam nowych zamówień",
    ])
    def test_does_not_invent_one(self, query):
        """A name that doesn't LOOK like a login (no digit, no ._- separator) is
        left alone — a miss just keeps the old behaviour, a false positive would
        put a made-up login into a tool call."""
        from agents.allegro.allegro_tools import named_buyer_login
        assert named_buyer_login(query) is None

    def test_a_named_account_makes_it_an_order_question(self):
        from agents.allegro.allegro_tools import matched_labels
        labels = matched_labels("Czy w tym roku kupował ode mnie ktoś z konta np1988")
        # Added, not substituted: get_buyers stays a candidate too.
        assert "zamowienia" in labels
        assert "kupujacy" in labels

    def test_the_tool_that_can_filter_by_it_is_offered(self):
        from agents.allegro.allegro_tools import select_tools_for_context
        names = {
            t["function"]["name"]
            for t in select_tools_for_context("Czy w tym roku kupował ode mnie ktoś z konta np1988")
        }
        assert "get_orders" in names
        assert "buyer_login" in next(
            t["function"]["parameters"]["properties"]
            for t in select_tools_for_context("czy kupował ode mnie ktoś z konta np1988")
            if t["function"]["name"] == "get_orders"
        )


class TestNamedPhoneNumber:
    """A pasted phone number is the only routing signal in "sprawdź 880 197
    834" — no stem in _LABEL_STEMS touches it — so the number itself has to
    put the customer lookup on the table. The danger is everything else in a
    store message that is also a long digit run: offer IDs, NIPs, REGONs,
    tracking codes."""

    @pytest.mark.parametrize("query,expected", [
        ("Czy mam klienta z takim nr telefonu +48 880 197 834", "+48 880 197 834"),
        ("czy mam klienta z takim nr telefonu +48 880 197 834?", "+48 880 197 834"),
        ("kto to jest 880 197 834", "880 197 834"),
        ("czy ten numer telefonu 880-197-834 coś u mnie kupował", "880-197-834"),
        ("sprawdź numer 880197834", "880197834"),
        ("klient dzwonił z 0048880197834", "0048880197834"),
    ])
    def test_finds_the_number(self, query, expected):
        from agents.allegro.allegro_tools import named_phone_number
        assert named_phone_number(query) == expected

    @pytest.mark.parametrize("query", [
        "zmień cenę oferty 14587236901",             # offer ID, 11 digits
        "sprawdź przesyłkę 620012345678901234567890",  # tracking code
        "wystaw fakturę dla NIP 7792445588",         # NIP, 10 digits
        "REGON 123456789",                            # 9 digits, but not a phone
        "zamówienie 0c4854a0-9646-11f1-8028-338c43adc37a",
        "ile zamówień miałem w 2026 roku",
        "jakie mam nowe zamówienia",
    ])
    def test_does_not_invent_one(self, query):
        """A false positive would answer a question nobody asked ("nie masz
        takiego klienta") about a number that was never a phone."""
        from agents.allegro.allegro_tools import named_phone_number
        assert named_phone_number(query) is None

    @pytest.mark.parametrize("raw,digits", [
        ("+48 880 197 834", "880197834"),
        ("0048880197834", "880197834"),
        ("880-197-834", "880197834"),
        ("880197834", "880197834"),
        ("48880197834", "880197834"),
        ("+49 151 12345678", "4915112345678"),  # foreign: nothing to strip
    ])
    def test_phone_digits_normalizes_every_spelling_to_one(self, raw, digits):
        from agents.allegro.allegro_tools import phone_digits
        assert phone_digits(raw) == digits

    def test_a_phone_number_makes_it_a_customer_question(self):
        from agents.allegro.allegro_tools import matched_labels
        assert matched_labels("sprawdź 880 197 834") == {"kupujacy"}

    def test_the_tool_that_can_filter_by_it_is_offered(self):
        from agents.allegro.allegro_tools import select_tools_for_context
        names = {
            t["function"]["name"]
            for t in select_tools_for_context("Czy mam klienta z takim nr telefonu +48 880 197 834")
        }
        assert "find_buyer_by_contact" in names

    def test_order_questions_keep_their_own_label(self):
        """The contact stems must not drag every order question into the buyer
        topic — that would cost those queries the deterministic layer."""
        from agents.allegro.allegro_tools import matched_labels
        assert matched_labels("co klient odebrał") == {"zamowienia"}
        assert matched_labels("jakie mam nowe zamówienia") == {"zamowienia"}


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
        ("pokaż listę kupujących z tego roku", "get_buyers"),
        ("jakie firmy u mnie kupowały", "get_buyers"),
        ("ilu miałem kupujących w tym roku", "get_buyers"),
        ("czy mam klienta z takim nr telefonu +48 880 197 834", "find_buyer_by_contact"),
        ("czy kupował ode mnie ktoś z adresu jan@example.com", "find_buyer_by_contact"),
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
        ("włącz powiadomienia o fakturach", "suggest_invoice_reminder"),
        ("wyłącz powiadomienia o fakturach", "disable_invoice_reminder"),
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
