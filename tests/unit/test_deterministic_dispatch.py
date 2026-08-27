"""Unit tests for agents/allegro/deterministic_dispatch.py — layer 2/3 of the
tool-select pipeline (see that module's docstring for the design)."""
from __future__ import annotations

import pytest

from agents.allegro.allegro_tools import matched_labels
from agents.allegro.deterministic_dispatch import resolve_deterministic


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
        ("włącz powiadomienia o fakturach", ("suggest_invoice_monitoring", {})),
        ("wyłącz powiadomienia o fakturach", ("disable_invoice_monitoring", {})),
        ("chcę przypomnienia o niewystawionych fakturach", ("suggest_invoice_reminder", {})),
        ("wyłącz przypomnienia o fakturach", ("disable_invoice_reminder", {})),
        ("powiadamiaj mnie o nowych wiadomościach", ("suggest_message_monitoring", {})),
        ("wyłącz monitoring wiadomości", ("disable_message_monitoring", {})),
        ("powiadamiaj mnie o zwrotach i reklamacjach", ("suggest_returns_monitoring", {})),
        ("wyłącz monitoring zwrotów", ("disable_returns_monitoring", {})),
    ])
    def test_toggle_resolves(self, query, expected):
        assert _resolve(query) == expected

    def test_reminder_takes_priority_over_generic_invoice_monitoring(self):
        """suggest_invoice_reminder and suggest_invoice_monitoring are
        different features (see their tool descriptions) — 'przypomnienia'
        wording must never resolve to the plain notifier."""
        result = _resolve("włącz przypomnienia o fakturach")
        assert result == ("suggest_invoice_reminder", {})


class TestMultiTopicAndUnrelatedQueries:
    def test_multi_topic_query_never_dispatches(self):
        assert _resolve("nowe zamówienia i moje konto") is None

    def test_unrelated_query_returns_none(self):
        assert _resolve("jaka jest dzisiaj pogoda") is None

    def test_empty_labels_returns_none(self):
        assert resolve_deterministic("cokolwiek", set()) is None
