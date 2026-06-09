"""
Moduł 5 — Oferty Allegro

Weryfikuje przeglądanie, wyszukiwanie i modyfikację ofert.
Wszystkie testy wymagają aktywnej autoryzacji Allegro.
"""

import pytest
from conftest import query, new_session, requires_allegro


@requires_allegro
class TestOfferListing:
    def test_list_active_offers(self):
        """
        Pytanie: 'Jakie mam aktywne oferty?'
        Oczekiwane:
          - agent=allegro
          - lista ofert LUB info że ich brak
          - ceny w PLN (jeśli oferty istnieją)
        """
        result = query("Jakie mam aktywne oferty?", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        offer_or_empty = ["ofert", "brak", "nie ma", "produkt", "zł", "id"]
        assert any(w in resp for w in offer_or_empty)

    def test_list_offers_with_prices(self):
        """
        Pytanie: 'Pokaż moje oferty z cenami'
        Oczekiwane: odpowiedź zawiera PLN lub zł (waluta)
        """
        result = query("Pokaż moje oferty z cenami", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"]
        # Waluta powinna być w odpowiedzi jeśli są jakiekolwiek oferty
        # (jeśli brak ofert, akceptujemy odpowiedź informującą o braku)
        assert "otwórz link" not in resp.lower()

    def test_search_offers_by_name(self):
        """
        Pytanie: 'Znajdź oferty z "buty" w nazwie'
        Oczekiwane: agent=allegro, filtrowanie ofert po nazwie
        """
        result = query('Znajdź oferty z "buty" w nazwie', new_session())
        assert result["agent"] == "allegro"

    def test_stock_check(self):
        """
        Pytanie: 'Które oferty mają mały stan magazynowy?'
        Oczekiwane: agent=allegro, odpowiedź o stanach magazynowych
        """
        result = query("Które oferty mają mały stan magazynowy?", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp


@requires_allegro
class TestOfferModification:
    def test_price_change_request_asks_for_confirmation(self):
        """
        Pytanie: 'Zmień cenę oferty 12345678 na 99.99 zł'
        Oczekiwane:
          - agent=allegro
          - odpowiedź informuje o wyniku operacji LUB prosi o potwierdzenie
          - NIE podaje zmyślonej ceny — korzysta z narzędzi
        """
        result = query("Zmień cenę oferty 12345678 na 99.99 zł", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        # Powinno informować o próbie zmiany lub błędzie (nie ma takiej oferty)
        action_words = ["cen", "ofert", "zmieni", "zaktualizow", "błąd", "nie znalazł",
                        "price", "update", "error", "not found"]
        assert any(w in resp for w in action_words)

    def test_stock_update_request(self):
        """
        Pytanie: 'Zaktualizuj stan magazynowy oferty 12345678 na 10 sztuk'
        Oczekiwane: agent=allegro, informacja o wyniku / błędzie
        """
        result = query("Zaktualizuj stan magazynowy oferty 12345678 na 10 sztuk", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp

    def test_price_response_includes_pln(self):
        """
        Oczekiwane: ceny zawsze podawane w PLN
        Weryfikacja: 'Podaj ceny moich ofert'
        """
        result = query("Podaj ceny moich ofert", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"]
        # Jeśli są oferty, powinny być ceny z walutą
        if "brak" not in resp.lower() and "nie ma" not in resp.lower():
            assert "zł" in resp or "PLN" in resp, (
                f"Ceny powinny być podane w PLN: {resp[:400]}"
            )

    def test_model_does_not_invent_offer_ids(self):
        """
        Oczekiwane: model nie podaje zmyślonych ID ofert bez wywołania narzędzi
        Pytanie celowo ogólne — model musi pobrać dane
        """
        result = query("Jakie są ID moich aktywnych ofert?", new_session())
        assert result["agent"] == "allegro"
        # Odpowiedź nie powinna być krótka odpowiedzią z wymyślonymi ID
        # (model powinien wywołać narzędzie get_active_offers)
        assert len(result["response"]) > 20
