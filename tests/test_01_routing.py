"""
Moduł 1 — Routing i klasyfikacja intencji

Weryfikuje, że orkiestrator kieruje zapytania do właściwego agenta
bez względu na to, czy klasyfikacja idzie przez słowa kluczowe czy LLM.

Kryterium sukcesu: pole `agent` w odpowiedzi odpowiada oczekiwanemu agentowi.
"""

import pytest
from conftest import query, new_session


class TestKeywordRouting:
    """Keyword pre-routing — nie wymaga wywołania LLM."""

    def test_order_keywords_route_to_allegro(self):
        """
        Scenariusz: Pytanie zawiera słowo kluczowe 'zamówieni'
        Oczekiwany agent: allegro
        """
        session = new_session()
        result = query("Pokaż moje zamówienia", session)
        assert result["agent"] == "allegro", (
            f"Pytanie o zamówienia powinno trafiać do agenta 'allegro', "
            f"dostało: '{result['agent']}'. Odpowiedź: {result['response'][:200]}"
        )

    def test_offer_keywords_route_to_allegro(self):
        """
        Scenariusz: Pytanie o oferty
        Oczekiwany agent: allegro
        """
        result = query("Jakie mam aktywne oferty?", new_session())
        assert result["agent"] == "allegro"

    def test_messaging_keywords_route_to_allegro(self):
        """
        Scenariusz: Pytanie o wiadomości od kupujących
        Oczekiwany agent: allegro
        """
        result = query("Sprawdź wiadomości od kupujących", new_session())
        assert result["agent"] == "allegro"

    def test_account_keywords_route_to_allegro(self):
        """
        Scenariusz: Pytanie o opłaty / konto
        Oczekiwany agent: allegro
        """
        result = query("Jakie mam ostatnie opłaty na Allegro?", new_session())
        assert result["agent"] == "allegro"

    def test_capabilities_question_routes_to_chitchat(self):
        """
        Scenariusz: Pytanie o możliwości asystenta
        Oczekiwany agent: chitchat (NIE allegro — nie może triggerować auth flow)
        """
        result = query("Jakie funkcje obsługujesz?", new_session())
        assert result["agent"] == "chitchat", (
            f"Pytanie o możliwości nie powinno trafiać do agenta Allegro. "
            f"Agent: '{result['agent']}'. Odpowiedź: {result['response'][:300]}"
        )

    def test_greeting_routes_to_chitchat(self):
        """
        Scenariusz: Powitanie
        Oczekiwany agent: chitchat
        """
        result = query("Cześć!", new_session())
        assert result["agent"] == "chitchat"


class TestLLMRouting:
    """Klasyfikacja przez Gemini — zapytania niejednoznaczne lub bez słów kluczowych."""

    def test_order_inquiry_without_keyword(self):
        """
        Scenariusz: Pytanie o zamówienia bez dosłownych słów kluczowych
        Oczekiwany agent: allegro
        """
        result = query("Co kupiło się u mnie ostatnio?", new_session())
        assert result["agent"] == "allegro"

    def test_price_change_request(self):
        """
        Scenariusz: Prośba o zmianę ceny produktu
        Oczekiwany agent: allegro
        """
        result = query("Chcę zmienić cenę produktu w moim sklepie", new_session())
        assert result["agent"] == "allegro"

    def test_unknown_query_fallback(self):
        """
        Scenariusz: Kompletnie niezwiązane pytanie
        Oczekiwany agent: NIE allegro (nie może triggerować auth flow)
        """
        result = query("Jaka jest stolica Francji?", new_session())
        assert result["agent"] != "allegro", (
            "Pytanie niezwiązane ze sklepem nie powinno trafiać do Allegro."
        )

    def test_english_order_query_routes_to_allegro(self):
        """
        Scenariusz: Pytanie o zamówienia po angielsku
        Oczekiwany agent: allegro
        """
        result = query("Show me my recent orders", new_session())
        assert result["agent"] == "allegro"
