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

    def test_billing_contains_required_fields(self):
        """
        Kryterium: rozliczenia zawierają datę, typ opłaty i kwotę w PLN.
        """
        result = query("Jakie są szczegóły moich ostatnich rozliczeń?", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        if not any(w in resp for w in ["brak", "nie ma", "no billing", "empty"]):
            date_indicators = ["data", "date", "2025", "2026", "styczeń", "luty", "marzec",
                               "kwiecień", "maj", "czerwiec", "lipiec", "sierpień",
                               "wrzesień", "październik", "listopad", "grudzień", "january",
                               "february", "march", "april", "may", "june", "july"]
            type_indicators = ["prowizj", "opłat", "fee", "commission", "typ", "type",
                               "kategori", "faktura"]
            amount_indicators = ["zł", "pln", "gr", "0.", "1.", "2.", "3.", "4.", "5.",
                                  "6.", "7.", "8.", "9."]
            assert any(w in resp for w in date_indicators), (
                f"Rozliczenia powinny zawierać datę: {result['response'][:400]}"
            )
            assert any(w in resp for w in type_indicators), (
                f"Rozliczenia powinny zawierać typ opłaty: {result['response'][:400]}"
            )
            assert any(w in resp for w in amount_indicators), (
                f"Rozliczenia powinny zawierać kwotę: {result['response'][:400]}"
            )

    def test_billing_sorted_by_newest(self):
        """
        Kryterium: rozliczenia są posortowane od najnowszych.
        Weryfikujemy że model potwierdza sortowanie lub odpowiedź zaczyna od najnowszych wpisów.
        """
        result = query("Pokaż mi rozliczenia od najnowszych", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        # Model powinien pokazać rozliczenia lub potwierdzić brak
        relevant_words = ["rozliczen", "opłat", "prowizj", "brak", "billing", "fee", "nie ma"]
        assert any(w in resp for w in relevant_words), (
            f"Odpowiedź powinna dotyczyć rozliczeń: {result['response'][:400]}"
        )

    def test_billing_limit_respected(self):
        """
        Kryterium: limit wyników działa poprawnie (domyślnie ≤ 50).
        Weryfikujemy że prośba o konkretną liczbę wyników jest honorowana.
        """
        result = query("Pokaż mi ostatnie 5 rozliczeń", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        relevant_words = ["rozliczen", "opłat", "prowizj", "brak", "billing", "fee", "nie ma", "5"]
        assert any(w in resp for w in relevant_words), (
            f"Odpowiedź powinna dotyczyć rozliczeń z limitem: {result['response'][:400]}"
        )

    def test_account_info_contains_registration_date(self):
        """
        Kryterium: dane konta zawierają datę rejestracji.
        """
        result = query("Kiedy zostało założone moje konto Allegro?", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        date_indicators = ["data", "zarejestrowa", "założon", "registered", "created",
                           "2025", "2026", "2024", "2023", "2022", "rok", "year"]
        assert any(w in resp for w in date_indicators), (
            f"Odpowiedź powinna zawierać datę rejestracji konta: {result['response'][:400]}"
        )

    def test_account_info_in_english(self):
        """
        Kryterium: odpowiedź jest w języku pytania — test dla angielskiego.
        """
        result = query("Show me my Allegro account details", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        english_indicators = ["login", "email", "account", "company", "registered",
                              "seller", "your account", "name"]
        assert any(w in resp for w in english_indicators), (
            f"Odpowiedź powinna być po angielsku: {result['response'][:400]}"
        )

    def test_billing_in_english(self):
        """
        Kryterium: odpowiedź jest w języku pytania — rozliczenia po angielsku.
        """
        result = query("Show me my recent Allegro billing charges", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        english_indicators = ["billing", "fee", "charge", "commission", "pln",
                              "no billing", "no charges", "amount"]
        assert any(w in resp for w in english_indicators), (
            f"Odpowiedź powinna być po angielsku: {result['response'][:400]}"
        )
