"""
Moduł 6 — Wiadomości do kupujących

Weryfikuje listowanie wątków i wysyłanie odpowiedzi.
Wszystkie testy wymagają aktywnej autoryzacji Allegro.
"""

import pytest
from conftest import query, new_session, requires_allegro


@requires_allegro
class TestMessageListing:
    def test_list_message_threads(self):
        """
        Pytanie: 'Sprawdź moje wiadomości od kupujących'
        Oczekiwane:
          - agent=allegro
          - lista wątków LUB info że ich brak
        """
        result = query("Sprawdź moje wiadomości od kupujących", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        message_or_empty = ["wiadomoś", "wątek", "temat", "kupując", "brak", "nie ma",
                            "thread", "message", "unread"]
        assert any(w in resp for w in message_or_empty)

    def test_check_unread_messages(self):
        """
        Pytanie: 'Mam jakieś nieprzeczytane wiadomości?'
        Oczekiwane: agent=allegro, informacja o nieprzeczytanych
        """
        result = query("Mam jakieś nieprzeczytane wiadomości?", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp

    def test_message_response_in_polish(self):
        """
        Pytanie po polsku → odpowiedź po polsku
        """
        result = query("Pokaż wiadomości od kupujących z ostatnich dni", new_session())
        resp = result["response"]
        polish_indicators = ["ę", "ą", "ó", "ś", "ź", "ż", "ć", "ń",
                             "wiadomoś", "kupując", "wątek", "brak"]
        assert any(p in resp.lower() for p in polish_indicators), (
            f"Odpowiedź powinna być po polsku: {resp[:400]}"
        )


@requires_allegro
class TestSendMessage:
    def test_send_message_to_buyer_requires_thread_id(self):
        """
        Pytanie: 'Napisz do kupującego w wątku 99999999 że zamówienie jest w drodze'
        Oczekiwane: agent=allegro, próba wysłania lub informacja o błędzie (nieistniejący wątek)
        """
        result = query(
            "Napisz do kupującego w wątku 99999999 że zamówienie jest w drodze",
            new_session(),
        )
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        send_or_error = ["wysłan", "wiadomoś", "błąd", "nie znalazł", "wątek",
                         "sent", "error", "not found", "thread"]
        assert any(w in resp for w in send_or_error)

    def test_send_message_without_thread_id(self):
        """
        Pytanie: 'Napisz do kupującego że zamówienie idzie jutro' (bez ID wątku)
        Oczekiwane: agent=allegro, prosi o podanie ID wątku lub pobiera listę wątków
        """
        result = query("Napisz do kupującego że zamówienie idzie jutro", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        # Powinien poprosić o wątek lub pokazać listę wątków
        clarification_words = ["wątek", "id", "który", "podaj", "thread", "which",
                               "wiadomoś", "thread_id"]
        assert any(w in resp for w in clarification_words)

    def test_send_message_success_confirmation(self):
        """
        Kryterium: wysłanie wiadomości w istniejącym wątku kończy się potwierdzeniem.
        Używamy fikcyjnego ID — oczekujemy albo potwierdzenia albo czytelnego błędu (nie ciszy).
        """
        result = query(
            "Wyślij wiadomość w wątku 00000001 z treścią: Dziękuję za zakup, paczka już w drodze!",
            new_session(),
        )
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        # Odpowiedź musi coś powiedzieć — sukces LUB błąd, ale nie milczenie
        informative_words = ["wysłan", "wiadomoś", "potwierdz", "błąd", "nie znalazł",
                             "sent", "message", "confirm", "error", "not found", "failed"]
        assert any(w in resp for w in informative_words), (
            f"Odpowiedź powinna potwierdzać wysłanie lub informować o błędzie: {result['response'][:400]}"
        )

    def test_send_message_content_not_truncated(self):
        """
        Kryterium: treść wysyłanej wiadomości nie jest obcinana.
        Weryfikujemy że agent przyjmuje długą treść bez błędu walidacji.
        """
        long_message = (
            "Wyślij wiadomość w wątku 00000002 z treścią: "
            "Szanowny Kliencie, dziękujemy za złożone zamówienie. "
            "Informujemy, że Pana/Pani paczka została nadana dzisiaj i powinna dotrzeć "
            "w ciągu 2-3 dni roboczych. Numer przesyłki: PL123456789PL. "
            "W razie pytań prosimy o kontakt."
        )
        result = query(long_message, new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        # Nie może być błędu o za długiej treści
        truncation_errors = ["za długi", "too long", "zbyt długi", "przekroczono", "limit znaków"]
        assert not any(e in resp for e in truncation_errors), (
            f"Treść wiadomości nie powinna być odrzucana jako za długa: {result['response'][:400]}"
        )

    def test_thread_list_contains_required_fields(self):
        """
        Kryterium: lista wątków zawiera ID wątku, temat, status odczytania, datę.
        """
        result = query("Pokaż listę moich wątków wiadomości", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        # Jeśli są wątki, muszą zawierać wymagane pola
        if not any(w in resp for w in ["brak", "nie ma", "no messages", "empty"]):
            field_indicators = ["id", "wątek", "temat", "thread", "subject",
                                "przeczyta", "unread", "data", "date"]
            assert any(w in resp for w in field_indicators), (
                f"Lista wątków powinna zawierać wymagane pola (ID, temat, status, data): {result['response'][:400]}"
            )

    def test_unread_messages_clearly_marked(self):
        """
        Kryterium: nieprzeczytane wiadomości są wyraźnie oznaczone w odpowiedzi.
        """
        result = query("Które wiadomości od kupujących są nieprzeczytane?", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        # Odpowiedź musi jasno informować o statusie odczytania
        read_status_words = ["nieprzeczyta", "przeczyta", "unread", "read", "nowe", "brak", "nie ma"]
        assert any(w in resp for w in read_status_words), (
            f"Odpowiedź powinna jasno informować o statusie odczytania: {result['response'][:400]}"
        )

    def test_message_response_in_english(self):
        """
        Kryterium: odpowiedź jest w języku pytania — test dla angielskiego.
        """
        result = query("Do I have any unread messages from buyers?", new_session())
        assert result["agent"] == "allegro"
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        english_indicators = ["message", "thread", "unread", "buyer", "you have", "no messages",
                              "inbox", "conversation"]
        assert any(w in resp for w in english_indicators), (
            f"Odpowiedź powinna być po angielsku: {result['response'][:400]}"
        )
