"""Unit tests for agents/allegro/deterministic_dispatch.py — layer 2/3 of the
tool-select pipeline (see that module's docstring for the design)."""
from __future__ import annotations

import pytest

from agents.allegro.allegro_tools import matched_labels
from agents.allegro.deterministic_dispatch import resolve_deterministic, wants_latest_order_details


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
        "niespakowane zamówienia",
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
        "niewysłane zamówienia",
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

    def test_unsent_orders(self):
        assert _resolve("pokaż zamówienia niewysłane") == ("get_orders_delivery", {})

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
