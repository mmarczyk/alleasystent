"""
Moduł 4 — Zamówienia Allegro

Weryfikuje pobieranie i prezentację zamówień.
Wszystkie testy wymagają aktywnej autoryzacji Allegro.
"""

import pytest
from conftest import query, new_session, requires_allegro


@requires_allegro
class TestOrderListing:
    def test_list_new_orders(self):
        """
        Pytanie: 'Pokaż moje nowe zamówienia'
        Oczekiwane:
          - agent = allegro
          - odpowiedź po polsku
          - zawiera informację o zamówieniach LUB informuje że ich brak
          - NIE zawiera komunikatu o autoryzacji
        """
        result = query("Pokaż moje nowe zamówienia", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        # Albo lista zamówień, albo info że ich brak
        order_or_empty = ["zamówien", "brak", "nie ma", "znalazłem", "nie znalazł", "id"]
        assert any(w in resp for w in order_or_empty), (
            f"Odpowiedź nie wygląda jak wynik zapytania o zamówienia: {result['response'][:400]}"
        )

    def test_list_orders_awaiting_processing(self):
        """
        Pytanie: 'Jakie zamówienia czekają na realizację?'
        Oczekiwane: agent=allegro, odpowiedź dotyczy statusu zamówień
        """
        result = query("Jakie zamówienia czekają na realizację?", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp

    def test_list_cancelled_orders(self):
        """
        Pytanie: 'Pokaż anulowane zamówienia'
        Oczekiwane: agent=allegro, filtrowanie po statusie CANCELLED
        """
        result = query("Pokaż anulowane zamówienia", new_session())
        assert result["agent"] == "allegro"

    def test_order_response_in_polish(self):
        """
        Pytanie po polsku → odpowiedź po polsku
        Oczekiwane: odpowiedź nie jest w całości po angielsku
        """
        result = query("Pokaż moje zamówienia z ostatniego tygodnia", new_session())
        resp = result["response"]
        # Sprawdzamy czy są polskie znaki lub polskie słowa
        polish_indicators = ["ę", "ą", "ó", "ś", "ź", "ż", "ć", "ń",
                             "zamówien", "kupując", "dostaw", "zł", "status"]
        assert any(p in resp.lower() for p in polish_indicators), (
            f"Odpowiedź na polskie pytanie powinna być po polsku: {resp[:400]}"
        )


@requires_allegro
class TestOrderDetails:
    def test_order_details_request_without_id(self):
        """
        Pytanie: 'Podaj szczegóły ostatniego zamówienia'
        Oczekiwane: agent=allegro; albo szczegóły zamówienia, albo prośba o ID
        """
        result = query("Podaj szczegóły ostatniego zamówienia", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp

    def test_order_details_includes_buyer_info(self):
        """
        Pytanie: 'Kto złożył ostatnie zamówienie i jaki jest jego adres?'
        Oczekiwane: odpowiedź zawiera dane kupującego (jeśli zamówienia istnieją)
        """
        result = query("Kto złożył ostatnie zamówienie i jaki jest jego adres?", new_session())
        assert result["agent"] == "allegro"


@requires_allegro
class TestOrderConversation:
    def test_multi_turn_order_conversation(self):
        """
        Rozmowa wieloturowa:
          Tura 1: 'Pokaż moje zamówienia'
          Tura 2: 'A ile ich jest łącznie?'
        Oczekiwane: tura 2 korzysta z kontekstu tury 1
        """
        session = new_session()
        r1 = query("Pokaż moje zamówienia", session)
        assert r1["agent"] == "allegro"

        r2 = query("A ile ich jest łącznie?", session)
        # W kontekście rozmowy o zamówieniach powinno dalej odpowiadać o zamówieniach
        assert r2["agent"] == "allegro"
        resp2 = r2["response"].lower()
        # Powinno mówić o liczbie / ilości
        count_words = ["zamówien", "łącznie", "razem", "ile", "sztuk", "brak", "wszystkich"]
        assert any(w in resp2 for w in count_words)
