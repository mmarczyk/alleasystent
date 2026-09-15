"""Unit tests for agents/allegro/deterministic_dispatch.py — layer 2/3 of the
tool-select pipeline (see that module's docstring for the design)."""
from __future__ import annotations

import pytest

from agents.allegro.allegro_tools import matched_labels
from agents.allegro.deterministic_dispatch import (
    extract_buyer_scope,
    extract_min_orders,
    extract_value_bounds,
    resolve_deterministic,
    wants_latest_order_details,
)


def _resolve(query: str):
    return resolve_deterministic(query, matched_labels(query))


class TestGetNewOrders:
    def test_bare_new_orders(self):
        assert _resolve("jakie mam nowe zamówienia") == ("get_new_orders", {})

    def test_english_new_orders(self):
        assert _resolve("show me my new orders") == ("get_new_orders", {})

    def test_count_only(self):
        assert _resolve("ile mam nowych zamówień") == ("get_new_orders", {"count_only": True})

    def test_singular_last_order(self):
        assert _resolve("ostatnie nowe zamówienie") == ("get_new_orders", {"limit": 1})

    def test_bails_on_detail_signal(self):
        assert _resolve("szczegóły ostatniego nowego zamówienia") is None

    def test_bails_on_date_range(self):
        assert _resolve("lista zamówień z tego miesiąca") is None

    def test_bails_without_new_signal(self):
        """Ambiguous vs get_orders — no 'nowe'/count/date signal at all."""
        assert _resolve("jakie zamówienia mam") is None

    def test_bails_on_courier_signal(self):
        assert _resolve("jacy kurierzy w nowych zamówieniach") is None


class TestOrderStageVocabulary:
    """The five order stages and the three registers a seller asks each of them
    in — formal, colloquial and count (see deterministic_dispatch's
    _ORDER_STAGE_SIGNALS)."""

    @pytest.mark.parametrize("query", [
        # formalne
        "nowe zamówienia",
        "świeże zamówienia",
        "zamówienia do obsłużenia",
        "jakie zamówienia są złożone",
        "zamówienia zarejestrowane",
        "zamówienia oczekujące",
        # potoczne
        "co nowego wpadło w zamówieniach",
        "jakie zamówienia są jeszcze nietknięte",
        # do spakowania = wciąż NOWE, nie 'do wysłania'
        "które zamówienia mam spakować",
        "zamówienia do spakowania",
    ])
    def test_new_stage(self, query):
        assert _resolve(query) == ("get_new_orders", {})

    @pytest.mark.parametrize("query", [
        "które zamówienia są w trakcie realizacji",
        "zamówienia przetwarzane",
        "które zamówienia są w toku",
        "co teraz kompletuję",
        "co mam w robocie",
        "które zamówienia są nieskończone",
        "co zostało do dokończenia w zamówieniach",
    ])
    def test_in_progress_stage(self, query):
        assert _resolve(query) == ("get_orders", {"fulfillment_status": "PROCESSING"})

    @pytest.mark.parametrize("query", [
        "Które zamówienia czekają na wysyłkę?",
        "pokaż mi zamówienia do wysłania",
        "zamówienia gotowe do wysyłki",
        "zamówienia oczekujące na wysyłkę",
        # "niewysłane" is deliberately NOT here — see TestNegatedStage.
        "zamówienia przygotowane do nadania",
        "zapakowane zamówienia",
        "co czeka na kuriera",
        "paczki do nadania",
        "ile mam gotowych do wysyłki",
        "co jest gotowe do wywózki",
    ])
    def test_to_ship_stage(self, query):
        assert _resolve(query) == ("get_orders_delivery", {})

    @pytest.mark.parametrize("query", [
        "wysłane zamówienia",
        "które zamówienia są w transporcie",
        "zamówienia przekazane przewoźnikowi",
        "co już poszło",
        "co odebrał kurier",
        "które paczki już wyjechały",
    ])
    def test_shipped_stage(self, query):
        assert _resolve(query) == ("get_orders_delivery", {"fulfillment_status": "SENT"})

    @pytest.mark.parametrize("query", [
        "odebrane zamówienia",
        "które zamówienia są dostarczone",
        "które zamówienia są zrealizowane",
        "które paczki dotarły",
        "co klient odebrał",
    ])
    def test_delivered_stage(self, query):
        assert _resolve(query) == ("get_orders", {"fulfillment_status": "PICKED_UP"})

    def test_shipping_plan_wins_over_shipped_wording(self):
        """'do wysłania' shares its stem with 'wysłane' — the plan sense must
        win, or every DO WYSŁANIA question would report already-sent parcels."""
        assert _resolve("zamówienia do wysłania") == ("get_orders_delivery", {})

    @pytest.mark.parametrize("query", [
        "które zamówienia są spakowane, a które już wysłane",   # two stages
        "zamówienia w realizacji i te odebrane",                 # two stages
    ])
    def test_bails_on_mixed_stages(self, query):
        assert _resolve(query) is None

    @pytest.mark.parametrize("query", [
        "ile zamówień wysłałem dzisiaj",          # period — this layer has no clock
        "zamówienia wysłane w tym miesiącu",      # period
        "status zamówień gotowych do wysyłki",    # detail intent
        "faktury do wysłanych zamówień",          # invoices, not a plain listing
    ])
    def test_bails_like_the_rest_of_the_layer(self, query):
        assert _resolve(query) is None

    @pytest.mark.parametrize("query,expected", [
        ("ile zamówień jest w trakcie", ("get_orders", {"fulfillment_status": "PROCESSING", "count_only": True})),
        ("ile zamówień zrealizowanych", ("get_orders", {"fulfillment_status": "PICKED_UP", "count_only": True})),
        ("ile paczek do nadania", ("get_orders_delivery", {"count_only": True})),
        ("ile przesyłek już wyjechało", ("get_orders_delivery", {"fulfillment_status": "SENT", "count_only": True})),
    ])
    def test_counting_a_stage_counts_that_stage(self, query, expected):
        """'ile ...' + 'zamówień' used to mean get_new_orders(count_only) no
        matter what stage was named — 'ile zamówień jest w trakcie' answered
        with the NEW count. The count question also names what the seller
        handles ('ile paczek do nadania'), not always the order itself."""
        assert _resolve(query) == expected

    def test_stageless_listing_stays_the_llm_fallback(self):
        assert _resolve("pokaż zamówienia") is None


class TestGetOrdersDelivery:
    """'Do wysłania' is the ready-to-ship preset of the same listing, so it
    resolves without an LLM round-trip too."""

    def test_orders_to_send(self):
        assert _resolve("które zamówienia są do wysłania") == ("get_orders_delivery", {})

    def test_count_only(self):
        assert _resolve("ile zamówień mam do wysłania") == (
            "get_orders_delivery", {"count_only": True},
        )

    def test_bails_on_date_range(self):
        assert _resolve("zamówienia do wysłania z tego tygodnia") is None

    def test_bails_on_singular(self):
        assert _resolve("ostatnie zamówienie do wysłania") is None

    def test_deadline_question_is_not_a_ready_to_ship_question(self):
        """'Do kiedy wysłać' asks about the dispatch deadline — a filter this
        layer can't compute, so it must fall through to the LLM."""
        assert _resolve("do kiedy mam wysłać zamówienia") is None


class TestNegatedStage:
    """A negated stage is an EXCLUSION, not one positive status: "niewysłane"
    means every order that has not left yet — packed, unpacked, in progress —
    so it resolves to get_orders with the shipped statuses excluded.

    Before this, the compact spelling answered with READY_FOR_SHIPMENT alone
    (hiding everything nobody had packed) and the spaced spelling 'nie wysłane'
    fell through to the WYSŁANE pattern and answered with the exact opposite
    listing: orders that had already gone out.
    """

    UNSENT = ["SENT", "IN_TRANSIT", "READY_FOR_PICKUP", "PICKED_UP"]

    @pytest.mark.parametrize("query", [
        "pokaż zamówienia niewysłane",
        "pokaż nie wysłane zamówienia",
        "które zamówienia nie zostały wysłane",
        "jakie zamówienia jeszcze nie wysłałem",
        "zamówienia które nie są wysłane",
    ])
    def test_unsent_is_everything_but_sent(self, query):
        assert _resolve(query) == ("get_orders", {"exclude_fulfillment_status": self.UNSENT})

    def test_count_only(self):
        assert _resolve("ile mam niewysłanych zamówień") == (
            "get_orders", {"exclude_fulfillment_status": self.UNSENT, "count_only": True},
        )

    @pytest.mark.parametrize("query,excluded", [
        ("nieodebrane zamówienia", ["PICKED_UP"]),
        ("niespakowane zamówienia",
         ["READY_FOR_SHIPMENT", "SENT", "IN_TRANSIT", "READY_FOR_PICKUP", "PICKED_UP"]),
    ])
    def test_other_stages_negate_the_same_way(self, query, excluded):
        assert _resolve(query) == ("get_orders", {"exclude_fulfillment_status": excluded})

    @pytest.mark.parametrize("query", [
        # A negation-shaped word that is one stage's own vocabulary must not be
        # torn into "nie" + stem (see _split_compact_negations).
        ("jakie zamówienia są jeszcze nietknięte"),
        ("które zamówienia są nieskończone"),
    ])
    def test_lexicalised_words_are_not_negations(self, query):
        tool, _ = _resolve(query)
        assert tool in ("get_new_orders", "get_orders")

    def test_positive_and_negated_together_go_to_the_llm(self):
        """"Spakowane, ale jeszcze nie wysłane" is two stages at once — the
        ambiguity this layer always hands over rather than guessing at."""
        assert _resolve("które są spakowane, a które jeszcze nie wysłane") is None

    def test_negation_of_the_verb_is_not_a_negated_stage(self):
        assert _resolve("nie mam nic do wysłania") == ("get_orders_delivery", {})


class TestNegationAndAmountTogether:
    """The seller's own sentence, which used to come back as a list of orders
    that had ALREADY been sent, with the amount silently dropped. The stage
    resolves here; the amount is read out of the same sentence and put on the
    call by AllegroAgent._with_value_bounds (see extract_value_bounds)."""

    QUERY = "Pokaż mi zamówienie jeszcze nie wysłane o wartości powyżej 400zl"

    def test_the_stage_resolves_to_an_exclusion(self):
        assert _resolve(self.QUERY) == (
            "get_orders",
            {"exclude_fulfillment_status": ["SENT", "IN_TRANSIT", "READY_FOR_PICKUP", "PICKED_UP"]},
        )

    def test_the_amount_is_read_from_the_same_sentence(self):
        assert extract_value_bounds(self.QUERY) == {"min_value": 400.0}


class TestOrdersDueToday:
    """The deadline question is the one order question where a date word is not
    a reason to bail — 'dzisiaj' is the tool's own default cut-off."""

    @pytest.mark.parametrize("query", [
        "co muszę wysłać dzisiaj",
        "co muszę dziś wysłać",
        "co mam dzisiaj do wysłania",
        "co mam dziś do wysyłki",
        "jakie mam dzisiaj terminy wysyłki",
        "co mam dziś do nadania",
    ])
    def test_today_deadline_questions(self, query):
        assert _resolve(query) == ("get_orders_due_today", {})

    @pytest.mark.parametrize("query", [
        "ile paczek muszę dziś nadać",
        "ile zamówień mam dziś do wysłania",
    ])
    def test_count_only(self, query):
        assert _resolve(query) == ("get_orders_due_today", {"count_only": True})

    def test_already_shipped_today_is_not_a_deadline_question(self):
        """'ile dziś wysłałem' is today + shipping words, but asks about
        parcels that already left — a period question for the LLM."""
        assert _resolve("ile dziś wysłałem") is None

    def test_mixed_past_and_future_shipping_bails(self):
        assert _resolve("co dziś wysłałem i co jeszcze muszę wysłać") is None

    @pytest.mark.parametrize("query", [
        "co muszę wysłać do jutra",            # another horizon — needs a computed date
        "co muszę wysłać w tym tygodniu",      # ditto
        "co muszę wysłać do 15 września",      # ditto
        "co jest po terminie",                 # needs the current time as the cut-off
        "do kiedy mam wysłać zamówienia",      # asks about the deadline, not for a list
    ])
    def test_other_horizons_stay_with_the_llm(self, query):
        assert _resolve(query) is None

    def test_today_without_a_dispatch_word_is_not_this_tool(self):
        """'ile zamówień dzisiaj' is a placement-time question (get_orders with
        bought_after_local), not a deadline one."""
        assert _resolve("ile zamówień wpłynęło dzisiaj") is None

    def test_deadline_beats_the_ready_to_ship_stage(self):
        """'co mam dziś do wysłania' names the DO WYSŁANIA stage too, but the
        deadline is the narrower answer — and the stage matchers bail on a
        period word anyway."""
        assert _resolve("co mam dziś do wysłania") == ("get_orders_due_today", {})
        assert _resolve("co mam do wysłania") == ("get_orders_delivery", {})


class TestGetMessageThreads:
    def test_bare_list(self):
        assert _resolve("pokaż wiadomości") == ("get_message_threads", {})

    def test_count_only(self):
        assert _resolve("czy mam nowe wiadomości") == ("get_message_threads", {"count_only": True})

    def test_list_override_wins_over_question_word(self):
        assert _resolve("pokaż czy mam wiadomości") == ("get_message_threads", {})

    def test_bails_on_content_intent(self):
        assert _resolve("przeczytaj treść wiadomości od Jana") is None


class TestGetAccountInfo:
    @pytest.mark.parametrize("query", ["moje konto", "moje konto allegro", "pokaż konto", "jakie mam konto"])
    def test_canonical_phrasings(self, query):
        assert _resolve(query) == ("get_account_info", {})

    def test_bails_on_non_canonical_phrasing(self):
        assert _resolve("ile mam na koncie punktów lojalnościowych za konto premium") is None


class TestGetOffersSummary:
    def test_canonical_phrasing(self):
        assert _resolve("podsumowanie moich ofert") == ("get_offers_summary", {})

    def test_bails_on_plain_offers_list(self):
        # get_active_offers isn't covered by this layer (name-filter risk) —
        # must fall through to the LLM, not misfire as get_offers_summary.
        assert _resolve("pokaż moje oferty") is None


class TestReturnsAndComplaints:
    def test_bare_returns_count_only(self):
        assert _resolve("czy mam jakieś zwroty") == ("get_new_returns", {"count_only": True})

    def test_returns_to_process(self):
        assert _resolve("zwroty do obsłużenia") == ("get_returns_to_process", {})

    def test_returns_to_process_count_only(self):
        assert _resolve("ile zwrotów do obsłużenia") == ("get_returns_to_process", {"count_only": True})

    def test_complaints(self):
        assert _resolve("jakie mam reklamacje") == ("get_new_complaints", {})

    def test_bails_when_both_named(self):
        assert _resolve("czy mam zwroty i reklamacje") is None


class TestMonitoringToggles:
    @pytest.mark.parametrize("query,expected", [
        ("włącz monitoring zamówień", ("suggest_order_monitoring", {})),
        ("wyłącz monitoring zamówień", ("disable_order_monitoring", {})),
        ("włącz powiadomienia o fakturach", ("suggest_invoice_reminder", {})),
        ("wyłącz powiadomienia o fakturach", ("disable_invoice_reminder", {})),
        ("chcę przypomnienia o niewystawionych fakturach", ("suggest_invoice_reminder", {})),
        ("wyłącz przypomnienia o fakturach", ("disable_invoice_reminder", {})),
        ("powiadamiaj mnie o nowych wiadomościach", ("suggest_message_monitoring", {})),
        ("wyłącz monitoring wiadomości", ("disable_message_monitoring", {})),
        ("powiadamiaj mnie o zwrotach i reklamacjach", ("suggest_returns_monitoring", {})),
        ("wyłącz monitoring zwrotów", ("disable_returns_monitoring", {})),
        ("przypominaj mi o ewidencji sprzedaży bezrachunkowej", ("suggest_sales_record_reminder", {})),
        ("wyłącz przypomnienia o ewidencji", ("disable_sales_record_reminder", {})),
    ])
    def test_toggle_resolves(self, query, expected):
        assert _resolve(query) == expected

    def test_all_invoice_monitoring_wording_resolves_to_the_reminder(self):
        """The plain "new order needs an invoice" notifier was removed, so the
        reminder is the only invoice automation left — both the reminder
        wording and the generic monitoring wording must reach it."""
        for query in (
            "włącz przypomnienia o fakturach",
            "włącz monitoring faktur",
            "chcę powiadomienia o fakturach",
        ):
            assert _resolve(query) == ("suggest_invoice_reminder", {}), query

    def test_an_ewidencja_question_is_not_a_toggle(self):
        """"Ewidencja" is also just something the seller asks about — answering
        that with a toggle button instead of an answer is the failure the
        reminder-word guard in _match_sales_record_reminder prevents."""
        for query in (
            "co to jest ewidencja sprzedaży bezrachunkowej",
            "do kiedy muszę wystawić ewidencję",
        ):
            assert _resolve(query) != ("suggest_sales_record_reminder", {}), query

    def test_the_ewidencja_outranks_the_invoice_reminder_when_both_are_named(self):
        """It is the more specific topic — "faktury" is half of how a seller
        describes the ewidencja in the first place."""
        assert _resolve("przypominaj mi o ewidencji zamiast o fakturach") == (
            "suggest_sales_record_reminder", {},
        )


class TestNamedBuyerAccount:
    """No matcher here extracts a buyer login — they resolve arguments from
    stage/period wording alone — so a query naming one bails to the LLM instead
    of answering about one buyer with the whole store's listing."""

    @pytest.mark.parametrize("query", [
        "co mam do wysłania dla konta np1988",
        "ile zamówień od kupującego marek_zielinski",
        "nowe zamówienia użytkownika anna.kowalska88",
    ])
    def test_named_login_bails_to_the_llm(self, query):
        assert _resolve(query) is None

    def test_the_same_query_without_a_login_still_dispatches(self):
        assert _resolve("co mam do wysłania") == ("get_orders_delivery", {})


class TestFindBuyerByContact:
    """"Czy mam klienta z takim nr telefonu +48 880 197 834" — the one argument
    the tool needs is written out in the query, and named_phone_number reads it
    off far more reliably than a model retypes nine digits."""

    @pytest.mark.parametrize("query,phone", [
        ("Czy mam klienta z takim nr telefonu +48 880 197 834", "+48 880 197 834"),
        ("czy mam klienta z takim nr telefonu +48 880 197 834?", "+48 880 197 834"),
        ("kto to jest 880 197 834", "880 197 834"),
        ("sprawdź numer 880197834", "880197834"),
        ("czy ten numer telefonu 880-197-834 coś u mnie kupował", "880-197-834"),
    ])
    def test_resolves_the_phone_lookup(self, query, phone):
        assert _resolve(query) == ("find_buyer_by_contact", {"phone": phone})

    @pytest.mark.parametrize("query", [
        # A period needs a clock this layer doesn't have.
        "czy klient 880 197 834 kupował coś w tym miesiącu",
        # The whole customer population, not one contact.
        "lista klientów z numerami telefonów",
        # Another tool's job entirely.
        "napisz do klienta 880 197 834",
        "wystaw fakturę dla klienta 880 197 834",
    ])
    def test_bails_to_the_llm(self, query):
        assert _resolve(query) is None

    @pytest.mark.parametrize("query", [
        "zmień cenę oferty 14587236901",
        "sprawdź przesyłkę 620012345678901234567890",
        "wystaw fakturę dla NIP 7792445588",
    ])
    def test_other_long_numbers_are_never_read_as_a_phone(self, query):
        resolved = _resolve(query)
        assert resolved is None or resolved[0] != "find_buyer_by_contact"


class TestBuyerProducts:
    """"Dla tego kupującego „P.P.H.U. Gadżet z Jajem. Monika Sornat” pokaż mi
    zestawienie, jakie produkty kupował" — the customer's name is quoted, which
    is the one thing that says where a multi-word company name begins and ends,
    so the whole call resolves here."""

    @pytest.mark.parametrize("query,name", [
        ("Dla tego kupującego „P.P.H.U. Gadżet z Jajem. Monika Sornat” pokaż mi "
         "zestawienie jakie produkty kupował", "P.P.H.U. Gadżet z Jajem. Monika Sornat"),
        ('co kupował klient "Kawa i Spółka"', "Kawa i Spółka"),
        ("zestawienie zakupów firmy „Biuro Serwis sp. z o.o.”", "Biuro Serwis sp. z o.o."),
        ("jaki asortyment bierze klient „Kawa i Spółka”", "Kawa i Spółka"),
    ])
    def test_resolves_the_product_summary_for_that_customer(self, query, name):
        assert _resolve(query) == ("get_buyer_products", {"name": name})

    @pytest.mark.parametrize("query", [
        # A period needs a clock this layer doesn't have.
        "co kupował klient „Kawa i Spółka” w tym roku",
        # Another tool's job entirely.
        "napisz do klienta „Kawa i Spółka” co kupował",
        "wystaw fakturę klientowi „Kawa i Spółka” za produkty, które kupował",
        # Who they are, not what they buy — find_buyer_by_contact's question.
        "kim jest klient „Kawa i Spółka” i co kupował",
        "co kupował klient „Kawa i Spółka”, telefon 880 197 834",
        # No name to pass: the tool would answer about nobody.
        "jakie produkty kupowali moi klienci",
        "co kupowali klienci w tym miesiącu",
    ])
    def test_bails_to_the_llm(self, query):
        resolved = _resolve(query)
        assert resolved is None or resolved[0] != "get_buyer_products"

    def test_the_product_word_does_not_make_it_a_two_topic_query(self):
        """"produkty" pulls in the "oferty" label and "sprzedaży" the "finanse"
        one, but neither is a second question — no tool under them can say what
        ONE customer took."""
        assert matched_labels(
            "co kupował klient „Kawa i Spółka” — zestawienie sprzedaży, jakie produkty"
        ) >= {"kupujacy", "oferty", "finanse"}
        assert _resolve(
            "co kupował klient „Kawa i Spółka” — zestawienie sprzedaży, jakie produkty"
        ) == ("get_buyer_products", {"name": "Kawa i Spółka"})


class TestFollowUpAboutOneKnownOrder:
    """A question about the CONTENTS of one order the assistant just showed
    reads exactly like a listing count to a stem matcher ('ile' + 'zamów'),
    and used to be answered with the number of NEW ORDERS — passed straight
    through to the seller as the finished answer. Both signals now bail."""

    @pytest.mark.parametrize("query", [
        # anafora — wskazanie na jedno, już pokazane zamówienie
        "ile w tym zamówieniu jest sztuk",
        "W tym ostatnio zamówieniu powyżej 2000 ile jest sztuk",
        "ile pozycji w tym zamówieniu",
        "ile motków w tamtym zamówieniu",
        "co jest w tym zamówieniu",
        "ile produktów zawiera to zamówienie",
        # jednostka liczona = zawartość zamówienia, nie liczba zamówień
        "ile sztuk mam w nowych zamówieniach",
        "ile szt. jest w zamówieniach do wysłania",
    ])
    def test_bails_to_the_llm(self, query):
        assert _resolve(query) is None

    @pytest.mark.parametrize("query", [
        # zwykłe pytania o LISTĘ zamówień — guard nie może ich dotknąć;
        # 'paczek'/'przesyłek' to jednostki wysyłkowe, nie zawartość zamówienia
        "ile mam nowych zamówień",
        "jakie mam nowe zamówienia",
        "ile paczek do nadania",
        "ile przesyłek czeka na kuriera",
        "ile zamówień muszę wysłać dzisiaj",
        "zamówienia do wysłania",
    ])
    def test_listing_questions_still_resolve(self, query):
        assert _resolve(query) is not None


class TestExtractMinOrders:
    """"Tylko ci, którzy zrobili więcej niż 3 zamówienia" — a narrowing the
    model drops as readily as an amount, and just as invisibly: the answer comes
    back as every customer of the period. See extract_min_orders."""

    def test_the_production_question(self):
        assert extract_min_orders(
            "Podaj mi listę kupujących z ostatnich 3 miesięcy, uwzględnij tylko tych "
            "którzy zrobili więcej niż 3 zamówienia"
        ) == 4

    @pytest.mark.parametrize("query,expected", [
        # Strict: "more than 3" starts at 4. One word apart from the inclusive
        # wordings below, and a silent off-by-one adds a whole row of customers.
        ("kupujący, którzy zrobili więcej niż 3 zamówienia", 4),
        ("klienci powyżej 3 zamówień", 4),
        ("klienci ponad 10 zakupów", 11),
        # Inclusive: the number itself.
        ("kupujący z co najmniej 3 zamówieniami", 3),
        ("klienci przynajmniej 2 transakcje", 2),
        ("klienci z minimum 4 zamówieniami", 4),
        ("klienci z 5 lub więcej zamówień", 5),
        # The count IS the noun.
        ("kto kupił u mnie więcej niż raz", 2),
        ("klienci, którzy zamówili więcej niż jeden raz", 2),
        # Spelled-out numerals.
        ("kupujący z więcej niż dwa zamówienia", 3),
    ])
    def test_polish_wordings(self, query, expected):
        assert extract_min_orders(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("buyers with more than 3 orders", 4),
        ("customers with at least 5 purchases", 5),
        ("who bought more than once", 2),
    ])
    def test_english_wording_too(self, query, expected):
        assert extract_min_orders(query) == expected

    @pytest.mark.parametrize("query", [
        # The period is a count too, and it is not this one.
        "lista kupujących z ostatnich 3 miesięcy",
        "lista kupujących z tego roku",
        # An amount, not an order count — the noun is what tells them apart.
        "zamówienia powyżej 500 zł",
        "faktury do wysłania powyżej 500 zł",
        # A piece count, still not an order count.
        "zamówienia z ponad 5 sztukami",
        # No number stated: "stali klienci" alone is the model's judgement call,
        # not a bound this layer may invent.
        "stali klienci",
        "moi najlepsi klienci",
    ])
    def test_reads_nothing_when_no_order_count_is_stated(self, query):
        assert extract_min_orders(query) is None

    def test_more_than_one_order_is_two_not_one(self):
        """"więcej niż 1" must not come back as 1 — that keeps everybody."""
        assert extract_min_orders("klienci z więcej niż 1 zamówieniem") == 2


class TestExtractBuyerScope:
    """What a buyer question narrows and orders by — companies, the invoice
    state, an order count, the ordering. All four have a get_buyers parameter,
    so reading them here only stops them evaporating between the seller's
    sentence and the call. See extract_buyer_scope."""

    @pytest.mark.parametrize("query,expected", [
        # Sorting: "największe zamówienia" is about the size of ONE order, and
        # the default (total spend) answers it with whoever ordered most often.
        ("Którzy klienci robią największe zamówienia?", {"sort_by": "avg_value"}),
        ("kto składa u mnie duże zamówienia", {"sort_by": "avg_value"}),
        ("klienci z najwyższą średnią wartością zamówienia", {"sort_by": "avg_value"}),
        ("kto kupuje u mnie hurtowo", {"sort_by": "avg_items"}),
        ("którzy klienci biorą najwięcej sztuk na raz", {"sort_by": "avg_items"}),
        # Who the buyer is.
        ("którzy klienci firmowi kupują u mnie najczęściej", {"buyer_type": "company"}),
        ("jakie firmy u mnie kupowały w tym roku", {"buyer_type": "company"}),
        ("lista kontrahentów z tego roku", {"buyer_type": "company"}),
        ("kupujący z NIP", {"buyer_type": "company"}),
        ("pokaż klientów prywatnych", {"buyer_type": "person"}),
        # The invoice state.
        ("klienci zamawiający z fakturą", {"invoice_status": "requested"}),
        ("którzy klienci proszą o fakturę VAT", {"invoice_status": "requested"}),
        ("kupujący, dla których wystawiłem faktury VAT", {"invoice_status": "issued"}),
        ("komu jeszcze nie wystawiłem faktury", {"invoice_status": "missing"}),
        ("klienci, którzy czekają na fakturę", {"invoice_status": "missing"}),
    ])
    def test_one_narrowing_at_a_time(self, query, expected):
        assert extract_buyer_scope(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("klienci, którzy wydali u mnie powyżej 5000 zł", {"min_value": 5000.0}),
        ("kto zostawił u mnie ponad 1000 zł", {"min_value": 1000.0}),
        ("klienci, którzy wydali mniej niż 200 zł", {"max_value": 200.0}),
        ("kupujący, którzy wydali między 500 a 1000 zł", {"min_value": 500.0,
                                                          "max_value": 1000.0}),
    ])
    def test_what_the_customer_spent(self, query, expected):
        """The same amount wording an order listing uses — on a buyer question it
        bounds the customer's total, which is what the reply then says
        ("łącznie od …")."""
        assert extract_buyer_scope(query) == expected

    def test_the_amount_travels_with_the_other_narrowings(self):
        assert extract_buyer_scope(
            "które firmy wydały u mnie powyżej 5000 zł i zrobiły więcej niż 3 zamówienia"
        ) == {"buyer_type": "company", "min_value": 5000.0, "min_orders": 4}

    def test_a_question_carrying_several_of_them(self):
        assert extract_buyer_scope(
            "Lista kupujących z tego roku, dla których wystawiałem faktury VAT — tylko firmy"
        ) == {"buyer_type": "company", "invoice_status": "issued"}

    def test_the_order_count_travels_with_the_rest(self):
        assert extract_buyer_scope(
            "pokaż osoby prywatne, które zrobiły więcej niż 3 zamówienia"
        ) == {"buyer_type": "person", "min_orders": 4}

    @pytest.mark.parametrize("query", [
        "lista kupujących z tego roku",
        "pokaż listę klientów z ostatnich 3 miesięcy",
        # "Najlepsi klienci" IS the default ordering — nothing to override, and
        # reading it as anything else would answer a question nobody asked.
        "moi najlepsi klienci",
        "kto u mnie kupował",
    ])
    def test_a_plain_buyer_list_narrows_nothing(self, query):
        assert extract_buyer_scope(query) == {}

    def test_a_negated_invoice_beats_the_word_it_contains(self):
        """"nie wystawiłem" contains "wystawiłem" — read as 'issued' it would
        answer with exactly the customers the seller is NOT asking about."""
        assert extract_buyer_scope("którym klientom nie wystawiłem faktury") == {
            "invoice_status": "missing",
        }


class TestExtractValueBounds:
    """The order amount is the filter a model drops most readily, and dropping
    it is invisible — the listing comes back full and reads like an answer. It
    is stated plainly enough in Polish to read in Python, so it is."""

    def test_the_production_question(self):
        assert extract_value_bounds(
            "Ile kosztowała dostawa zamówienia z ostatnich dni które było na kwotę ponad 2000zl"
        ) == {"min_value": 2000.0}

    @pytest.mark.parametrize("query,expected", [
        ("zamówienia powyżej 500 zł", {"min_value": 500.0}),
        ("coś za więcej niż 2 000 zł", {"min_value": 2000.0}),
        ("zamówienia co najmniej 300 zł", {"min_value": 300.0}),
        ("zamówienia poniżej 100 zł", {"max_value": 100.0}),
        ("zamówienia mniej niż 50 zł", {"max_value": 50.0}),
        ("zamówienia do 80 zł", {"max_value": 80.0}),
        ("zamówienia między 500 a 1000 zł", {"min_value": 500.0, "max_value": 1000.0}),
        ("zamówienia od 500 do 1000 zł", {"min_value": 500.0, "max_value": 1000.0}),
    ])
    def test_direction_words(self, query, expected):
        assert extract_value_bounds(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("zamówienia ponad 2000zl", 2000.0),
        ("zamówienia ponad 2 000 zł", 2000.0),
        ("zamówienia ponad 2000 PLN", 2000.0),
        ("zamówienia ponad 2000 złotych", 2000.0),
        ("zamówienia ponad 1.500,50 zł", 1500.5),
        ("zamówienia ponad 99,90 zł", 99.9),
        ("zamówienia ponad 1.5 zł", 1.5),
    ])
    def test_polish_amount_spellings(self, query, expected):
        assert extract_value_bounds(query) == {"min_value": expected}

    @pytest.mark.parametrize("query", [
        "ile mam nowych zamówień",
        # No currency — not an amount.
        "zamówienia z ponad 5 sztukami",
        "co muszę wysłać do jutra",
        "zamówienia od 5 sierpnia",
        # An exact amount, no direction: no order matches a value to the grosz,
        # so a bound here would answer with an empty listing.
        "zamówienie na kwotę 2000 zł",
    ])
    def test_reads_nothing_when_nothing_is_stated(self, query):
        assert extract_value_bounds(query) == {}

    @pytest.mark.parametrize("query", [
        # "min" lives inside "termin" and "od" inside "przychód" — without a
        # word boundary on the direction word, both grew a filter out of a
        # number that had nothing to do with an order value.
        "jaki mam termin 500 zł",
        "przychód 500 zł w tym tygodniu",
        "dochód 2000 zł",
    ])
    def test_a_direction_word_inside_another_word_is_not_a_direction(self, query):
        assert extract_value_bounds(query) == {}

    @pytest.mark.parametrize("query,expected", [
        ("orders above 500 zł", {"min_value": 500.0}),
        ("orders under 100 zł", {"max_value": 100.0}),
        ("orders between 500 and 1000 zł", {"min_value": 500.0, "max_value": 1000.0}),
    ])
    def test_english_wording_too(self, query, expected):
        """The assistant answers English questions as well, and the amount is
        dropped just as readily there."""
        assert extract_value_bounds(query) == expected

    def test_a_backwards_explicit_range_is_read_as_the_range_it_describes(self):
        """One range, one typo, one obvious meaning."""
        assert extract_value_bounds("zamówienia od 100 zł do 50 zł") == {
            "min_value": 50.0, "max_value": 100.0,
        }

    @pytest.mark.parametrize("query", [
        "dla tego zamówienia policz zysk zakładając koszt 1 szt. na poziomie 8,10 zł",
        "ile zarobiłem, jeśli kupiłem po 8 zł za sztukę",
        "jaka marża przy koszcie zakupu 12 zł",
    ])
    def test_a_per_unit_purchase_cost_is_not_an_order_value(self, query):
        """Same digits, entirely different number — filtering the listing by
        the cost of one item would answer a question nobody asked."""
        assert extract_value_bounds(query) == {}


class TestPendingInvoicesScopedToAStage:
    """"Jakie mam faktury do wysłania w zamówieniach nie nowych" — one question
    with two halves, of which only the first used to be answered: the invoice
    listing came back for the whole month with the stage silently dropped."""

    DISPATCHED = ["SENT", "IN_TRANSIT", "READY_FOR_PICKUP", "PICKED_UP"]

    @pytest.mark.parametrize("query", [
        "jakie mam faktury do wysłania w zamówieniach nie nowych",
        "jakie mam faktury do wysłania w zamówieniach nienowych",
        "jakie faktury mam do wystawienia w nie nowych zamówieniach",
    ])
    def test_the_seller_question(self, query):
        assert _resolve(query) == (
            "get_orders_pending_invoice", {"exclude_fulfillment_status": ["NEW"]},
        )

    @pytest.mark.parametrize("query,statuses", [
        ("jakie faktury mam do wystawienia w wysłanych zamówieniach", DISPATCHED),
        ("faktury do wysłania w zamówieniach spakowanych", ["READY_FOR_SHIPMENT"]),
        ("jakie mam faktury do wysłania w zamówieniach do wysłania", ["READY_FOR_SHIPMENT"]),
        ("jakie faktury mam do wysłania w zamówieniach w realizacji", ["PROCESSING"]),
        ("jakie mam faktury do wystawienia w nowych zamówieniach", ["NEW"]),
        ("faktury do wysłania w zamówieniach już odebranych", ["PICKED_UP"]),
    ])
    def test_a_positive_stage_is_kept(self, query, statuses):
        assert _resolve(query) == (
            "get_orders_pending_invoice", {"fulfillment_status": statuses},
        )

    def test_shipped_covers_the_whole_dispatched_family(self):
        """An order IN_TRANSIT or PICKED_UP is every bit as "wysłane" as a SENT
        one, and its invoice is the most overdue of all — answering with SENT
        alone would hide exactly those."""
        _, args = _resolve("jakie faktury mam do wysłania w wysłanych zamówieniach")
        assert args == {"fulfillment_status": self.DISPATCHED}

    @pytest.mark.parametrize("query,excluded", [
        ("które faktury są do wysłania w zamówieniach nie odebranych", ["PICKED_UP"]),
        ("jakie faktury mam do wystawienia w niewysłanych zamówieniach", DISPATCHED),
    ])
    def test_other_negations_exclude_the_same_way(self, query, excluded):
        assert _resolve(query) == (
            "get_orders_pending_invoice", {"exclude_fulfillment_status": excluded},
        )

    def test_without_a_stage_nothing_is_resolved_here(self):
        """The half the model does not drop — it goes to the LLM exactly as it
        always did."""
        assert _resolve("jakie mam faktury do wystawienia") is None

    @pytest.mark.parametrize("query", [
        # An issuance command, not a question about what is pending.
        "wystaw faktury do wysłanych zamówień",
        "wygeneruj brakujące faktury w nowych zamówieniach",
        # One named order, and the billing-address lookup — other tools.
        "dane do faktury dla tego zamówienia",
        "dołącz fakturę do wysłanego zamówienia",
        # A month other than the current one needs a clock this layer lacks.
        "jakie mam faktury do wysłania w zamówieniach nie nowych w zeszłym miesiącu",
        # Two stages at once — the ambiguity this layer hands over.
        "faktury do wystawienia w zamówieniach spakowanych, ale jeszcze nie wysłanych",
    ])
    def test_bails(self, query):
        assert _resolve(query) is None

    def test_a_stage_word_not_said_about_orders_is_not_a_stage(self):
        """"Nowych" describes the CUSTOMERS here — filtering orders by it would
        answer a question nobody asked."""
        assert _resolve("jakie mam faktury do wysłania dla nowych klientów") is None

    def test_the_order_stage_listings_are_untouched(self):
        """The pairing of labels this matcher needs must not swallow a plain
        order question that happens to mention neither invoices nor a stage."""
        assert _resolve("co mam do wysłania") == ("get_orders_delivery", {})


class TestMultiTopicAndUnrelatedQueries:
    def test_multi_topic_query_never_dispatches(self):
        assert _resolve("nowe zamówienia i moje konto") is None

    def test_unrelated_query_returns_none(self):
        assert _resolve("jaka jest dzisiaj pogoda") is None

    def test_empty_labels_returns_none(self):
        assert resolve_deterministic("cokolwiek", set()) is None


class TestWantsLatestOrderDetails:
    @pytest.mark.parametrize("query", [
        "szczegóły ostatniego nowego zamówienia",
        "jaki jest status ostatniego zamówienia",
        "adres ostatniego zamówienia",
        "dane do faktury ostatniego zamówienia",
        "kiedy wysłane było ostatnie zamówienie",
        "co się dzieje z ostatnim zamówieniem",
        "koszty tego ostatniego zamówienia",
        "faktura ostatniego zamówienia",
        "najnowsze zamówienie — szczegóły",
    ])
    def test_matches(self, query):
        assert wants_latest_order_details(query) is True

    @pytest.mark.parametrize("query", [
        "jakie mam nowe zamówienia",           # no detail intent, no singular
        "szczegóły zamówienia z 15 czerwca",   # detail intent, not "latest"
        "ostatnie nowe zamówienia",             # singular signal, no detail intent
        "status mojego konta",                  # detail word, wrong topic
    ])
    def test_does_not_match(self, query):
        assert wants_latest_order_details(query) is False


class TestOrderQuestionNamingAProduct:
    """A product inside an order question is a filter this layer cannot
    extract (it has no way to tell where a model name ends — "yarnart jeans"
    carries no stem), so every order matcher hands the turn to the LLM, which
    has get_orders' product_names. Serving the preset instead would answer
    with the whole unfiltered listing, product dropped."""

    @pytest.mark.parametrize("query", [
        "pokaż nowe zamówienia z włóczką jeans",
        "jakie mam nowe zamówienia z yarnart jeans, które miały tylko ją",
        "ile mam nowych zamówień z przędzą merino",
    ])
    def test_new_orders_bails(self, query):
        assert _resolve(query) is None

    @pytest.mark.parametrize("query", [
        "co mam do wysłania z włóczką jeans plus",
        "które wysłane zamówienia zawierały kordonek",
    ])
    def test_stage_listing_bails(self, query):
        assert _resolve(query) is None

    def test_negated_stage_bails(self):
        assert _resolve("niewysłane zamówienia z włóczką jeans") is None

    def test_due_today_bails(self):
        assert _resolve("co muszę dzisiaj wysłać z włóczką jeans") is None

    def test_an_ordinary_order_question_is_unaffected(self):
        """The bail is scoped to contents wording — the everyday listings stay
        on the LLM-free path."""
        assert _resolve("jakie mam nowe zamówienia") == ("get_new_orders", {})
        assert _resolve("ile paczek mam do nadania")[0] == "get_orders_delivery"


class TestDeliverInvoices:
    """The delivery commands from the thread that produced this matcher.

    "Ok dodaj te faktury do Allegro a firmową wyślij również do ksef" — sent
    right after four invoices were issued and linked — came back as a sentence
    about there being no table data to copy. The follow-up, which spelled the
    inFakt id out in full, came back as the details of an unrelated order.
    Nothing was attached, and nothing said so."""

    ORDER = "a076f250-af8f-11f1-b272-c766ff73256d"
    INVOICE = "69bb3d97-1024-4d6c-b11a-813a38ce62e9"

    def test_the_whole_batch_with_ksef(self):
        assert _resolve("Ok dodaj te faktury do Allegro a firmowa wyślij również do ksef") == (
            "deliver_invoices", {"attach": True, "ksef": True},
        )

    def test_one_invoice_named_by_its_infakt_id(self):
        """Typos included: 'kser' is what the f/r slip produces, and the id is
        the INVOICE's — the order it belongs to is the ledger's to supply."""
        assert _resolve(
            f"Te fakturę dodaj do Allegro u wyślij do kser ID faktury w inFakt: {self.INVOICE}"
        ) == (
            "deliver_invoices",
            {"attach": True, "ksef": True, "invoice_uuids": [self.INVOICE]},
        )

    def test_one_order_named_by_its_allegro_id(self):
        assert _resolve(f"dołącz fakturę do zamówienia {self.ORDER}") == (
            "deliver_invoices", {"attach": True, "ksef": False, "order_ids": [self.ORDER]},
        )

    def test_ksef_alone_does_not_attach(self):
        tool, args = _resolve(f"wyślij fakturę do KSeF dla zamówienia {self.ORDER}")
        assert tool == "deliver_invoices"
        assert args["ksef"] is True and args["attach"] is False

    def test_several_orders_in_one_sentence(self):
        other = "d1d48e50-af87-11f1-acdd-5126b12f3316"
        tool, args = _resolve(
            f"dołącz faktury do zamówienia {self.ORDER} i do zamówienia {other}"
        )
        assert tool == "deliver_invoices"
        assert args["order_ids"] == [self.ORDER, other]

    def test_a_single_invoice_with_nothing_to_point_at_is_left_to_the_llm(self):
        """The ledger may hold several waiting invoices; "dołącz fakturę" did
        not say to deliver all of them."""
        assert _resolve("dołącz fakturę") is None

    def test_a_question_about_delivery_is_not_a_command(self):
        assert _resolve("czy dołączyłeś już te faktury do Allegro?") is None
        assert _resolve("kiedy dołączysz te faktury") is None

    def test_a_uuid_nobody_labelled_resolves_nothing(self):
        """Order id and invoice id are the same 36 characters, and swapping them
        addresses somebody else's document — so an id with no noun in front of
        it is the LLM's to read from context, not this layer's to guess."""
        assert _resolve(f"wyślij do ksef {self.INVOICE}") is None

    def test_the_nearest_noun_decides_which_id_it_is(self):
        """"faktury <id>" is the invoice; only a nearer "zamówienia" makes it an
        order. A mislabelled id is recovered behind the tool (the ledger knows
        both directions), never guessed here."""
        _, args = _resolve(f"dołącz te faktury {self.INVOICE}")
        assert args["invoice_uuids"] == [self.INVOICE]

    def test_one_of_each_is_a_pair_and_needs_no_lookup(self):
        assert _resolve(f"dołącz fakturę {self.INVOICE} do zamówienia {self.ORDER}") == (
            "deliver_invoices",
            {
                "attach": True, "ksef": False,
                "order_ids": [self.ORDER], "invoice_uuids": [self.INVOICE],
            },
        )

    def test_the_genitive_plural_of_zamowienie_still_marks_an_order(self):
        """"do zamówień" ends in "ń" — the form a stem looking for "zamówieni"
        misses, and with it every id in the sentence would read as an invoice."""
        other = "d1d48e50-af87-11f1-acdd-5126b12f3316"
        _, args = _resolve(f"dołącz faktury do zamówień {self.ORDER} i {other}")
        assert args["order_ids"] == [self.ORDER, other]

    def test_several_of_each_are_left_to_the_llm(self):
        """Which invoice belongs to which order is then an ordering guess, and
        a wrong guess puts one buyer's invoice on another buyer's order."""
        other_order = "d1d48e50-af87-11f1-acdd-5126b12f3316"
        other_invoice = "9362a39f-37f1-4ce7-8d98-74bcf48ff00c"
        assert _resolve(
            f"dołącz faktury {self.INVOICE} i {other_invoice} do zamówień "
            f"{self.ORDER} i {other_order}"
        ) is None

    def test_an_issuance_is_not_a_delivery(self):
        """Issuing stays one order per call, on its own path — and delivery may
        not ride along on the turn that issued anything."""
        assert _resolve(f"wystaw fakturę dla zamówienia {self.ORDER} i dołącz ją") is None

    def test_the_pending_listing_still_wins_its_own_questions(self):
        assert _resolve("jakie mam faktury do wysłania w zamówieniach nie nowych") == (
            "get_orders_pending_invoice", {"exclude_fulfillment_status": ["NEW"]},
        )
        assert _resolve("jakie mam faktury do wystawienia") is None

    def test_kserokopia_is_not_ksef(self):
        assert _resolve("wyślij kserokopię faktury") is None

    @pytest.mark.parametrize("query", [
        "pokaż wszystkie dołączone faktury",
        "które faktury dołączyłem do Allegro",
        "ile faktur wysłałem do ksef",
        "lista faktur",
        "sprawdź czy te faktury są dołączone",
    ])
    def test_a_listing_question_never_becomes_a_delivery(self, query):
        """The failure that would cost the most here: "pokaż wszystkie dołączone
        faktury" carries an attach stem and a set word, and served as a command
        it would attach the whole backlog to answer a question."""
        result = _resolve(query)
        assert result is None or result[0] != "deliver_invoices"
