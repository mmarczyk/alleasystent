"""
Moduł 2 — Chitchat i możliwości asystenta

Weryfikuje, że asystent prawidłowo odpowiada na powitania
i pytania o własne możliwości, wymieniając funkcje Allegro.
"""

import pytest
from conftest import query, new_session

ALLEGRO_FEATURES = [
    "zamówieni",   # orders
    "ofert",       # offers
    "wiadomoś",    # messages
    "konto",       # account
]


class TestGreetings:
    def test_polish_greeting_returns_polish(self):
        """
        Scenariusz: 'Cześć'
        Oczekiwane: odpowiedź po polsku, agent=chitchat
        """
        result = query("Cześć!", new_session())
        assert result["agent"] == "chitchat"
        resp = result["response"].lower()
        # Powinna być jakaś odpowiedź powitalna po polsku
        polish_greeting_words = ["cześć", "witaj", "dzień dobry", "hej", "pomoc", "pomog", "jak mogę"]
        assert any(w in resp for w in polish_greeting_words), (
            f"Oczekiwano polskiego powitania, dostałem: {result['response'][:300]}"
        )

    def test_english_greeting_returns_english(self):
        """
        Scenariusz: 'Hello!'
        Oczekiwane: odpowiedź po angielsku
        """
        result = query("Hello!", new_session())
        assert result["agent"] == "chitchat"
        resp = result["response"].lower()
        english_words = ["hello", "hi", "help", "how can", "assist"]
        assert any(w in resp for w in english_words), (
            f"Oczekiwano angielskiego powitania, dostałem: {result['response'][:300]}"
        )


class TestCapabilities:
    def test_capability_question_lists_features(self):
        """
        Scenariusz: 'Jakie funkcje obsługujesz?'
        Oczekiwane: wymienienie przynajmniej 2 z głównych funkcji Allegro
        """
        result = query("Jakie funkcje obsługujesz?", new_session())
        assert result["agent"] == "chitchat", (
            f"Pytanie o możliwości powinno trafiać do chitchat, nie do '{result['agent']}'"
        )
        resp = result["response"].lower()
        matched = [f for f in ALLEGRO_FEATURES if f in resp]
        assert len(matched) >= 2, (
            f"Odpowiedź powinna wymieniać przynajmniej 2 funkcje Allegro. "
            f"Dopasowane: {matched}. Odpowiedź: {result['response'][:400]}"
        )

    def test_capability_question_in_polish(self):
        """
        Scenariusz: 'Co potrafisz?'
        Oczekiwane: odpowiedź po polsku z opisem możliwości
        """
        result = query("Co potrafisz?", new_session())
        resp = result["response"].lower()
        # Odpowiedź powinna być po polsku
        polish_words = ["mogę", "potrafię", "obsługuję", "pomagam", "umiem"]
        assert any(w in resp for w in polish_words), (
            f"Oczekiwano polskiej odpowiedzi o możliwościach: {result['response'][:400]}"
        )

    def test_capability_not_triggering_allegro_auth(self):
        """
        Scenariusz: 'Powiedz jakie funkcje obsługujesz'
        Oczekiwane: NIE pojawia się prośba o autoryzację Allegro
        """
        result = query("Powiedz jakie funkcje obsługujesz", new_session())
        resp = result["response"].lower()
        auth_trigger_phrases = [
            "autoryzacj", "authorization", "otwórz link", "open this link",
            "device", "zatwierdź dostęp",
        ]
        assert not any(p in resp for p in auth_trigger_phrases), (
            f"Pytanie o możliwości wywołało flow autoryzacji Allegro: {result['response'][:400]}"
        )

    def test_english_capability_question(self):
        """
        Scenariusz: 'What can you do?'
        Oczekiwane: odpowiedź po angielsku z opisem funkcji
        """
        result = query("What can you do?", new_session())
        resp = result["response"].lower()
        english_capability_words = ["order", "offer", "message", "account", "allegro", "can"]
        assert any(w in resp for w in english_capability_words), (
            f"Oczekiwano angielskiej odpowiedzi o możliwościach: {result['response'][:400]}"
        )
