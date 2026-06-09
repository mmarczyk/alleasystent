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
