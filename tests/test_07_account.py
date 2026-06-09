"""
Moduł 7 — Konto sprzedawcy i rozliczenia

Weryfikuje pobieranie informacji o koncie i opłatach.
Wszystkie testy wymagają aktywnej autoryzacji Allegro.
"""

import pytest
from conftest import query, new_session, requires_allegro


@requires_allegro
class TestAccountInfo:
    def test_get_account_info(self):
        """
        Pytanie: 'Pokaż informacje o moim koncie Allegro'
        Oczekiwane:
          - agent=allegro
          - odpowiedź zawiera dane konta: login, email lub firma
        """
        result = query("Pokaż informacje o moim koncie Allegro", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        account_fields = ["login", "email", "firma", "konto", "sprzedawc",
                          "company", "account", "registered"]
        assert any(w in resp for w in account_fields), (
            f"Odpowiedź powinna zawierać dane konta: {result['response'][:400]}"
        )

    def test_account_info_in_polish(self):
        """
        Pytanie po polsku → odpowiedź po polsku
        """
        result = query("Jakie są dane mojego konta?", new_session())
        resp = result["response"]
        polish_indicators = ["ę", "ą", "ó", "ś", "ź", "ż", "ć", "ń",
                             "login", "konto", "data", "email"]
        assert any(p in resp.lower() for p in polish_indicators)


@requires_allegro
class TestBilling:
    def test_get_billing_summary(self):
        """
        Pytanie: 'Jakie mam ostatnie rozliczenia na Allegro?'
        Oczekiwane:
          - agent=allegro
          - lista opłat LUB info że ich brak
          - jeśli są — kwoty w PLN
        """
        result = query("Jakie mam ostatnie rozliczenia na Allegro?", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        billing_or_empty = ["opłat", "prowizj", "rozliczen", "zł", "brak",
                            "fee", "billing", "pln", "charge"]
        assert any(w in resp for w in billing_or_empty)

    def test_get_fees_information(self):
        """
        Pytanie: 'Ile zapłaciłem prowizji w ostatnim czasie?'
        Oczekiwane: agent=allegro, informacja o prowizjach
        """
        result = query("Ile zapłaciłem prowizji w ostatnim czasie?", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp

    def test_billing_amounts_in_pln(self):
        """
        Oczekiwane: kwoty w rozliczeniach podawane w PLN
        """
        result = query("Pokaż moje ostatnie opłaty Allegro", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"]
        if "brak" not in resp.lower() and "nie ma" not in resp.lower():
            assert "zł" in resp or "PLN" in resp, (
                f"Kwoty rozliczeń powinny być w PLN: {resp[:400]}"
            )
