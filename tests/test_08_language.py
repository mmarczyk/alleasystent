"""
Moduł 8 — Wielojęzyczność

Weryfikuje, że asystent odpowiada w języku pytania
niezależnie od tego, który agent obsługuje zapytanie.
"""

import pytest
from conftest import query, new_session


POLISH_CHARS = set("ęąóśźżćńĘĄÓŚŹŻĆŃ")
POLISH_WORDS = ["zamówien", "ofert", "konto", "sprzedawc", "kupując",
                "mogę", "jest", "nie", "tak", "mam", "moje", "twoje",
                "sprawdz", "pokaż", "brak", "znalazłem"]


def is_likely_polish(text: str) -> bool:
    """Heurystyka: czy tekst wygląda jak polski."""
    low = text.lower()
    has_polish_chars = any(c in POLISH_CHARS for c in text)
    has_polish_words = any(w in low for w in POLISH_WORDS)
    return has_polish_chars or has_polish_words


class TestPolishResponses:
    def test_polish_greeting_answered_in_polish(self):
        """Cześć → odpowiedź po polsku"""
        result = query("Cześć, jak się masz?", new_session())
        assert is_likely_polish(result["response"]), (
            f"Polskie pytanie powinno dawać polską odpowiedź: {result['response'][:300]}"
        )

    def test_polish_order_question_answered_in_polish(self):
        """Pytanie o zamówienia po polsku → odpowiedź po polsku"""
        result = query("Pokaż moje nowe zamówienia", new_session())
        assert is_likely_polish(result["response"]), (
            f"Polskie pytanie powinno dawać polską odpowiedź: {result['response'][:300]}"
        )

    def test_polish_capability_answered_in_polish(self):
        """Pytanie o możliwości po polsku → odpowiedź po polsku"""
        result = query("Co potrafisz zrobić?", new_session())
        assert is_likely_polish(result["response"]), (
            f"Polskie pytanie o możliwości → polska odpowiedź: {result['response'][:300]}"
        )

    def test_no_english_only_response_to_polish_question(self):
        """
        Odpowiedź na polskie pytanie nie może być wyłącznie po angielsku.
        Sprawdzamy typowe angielskie frazy które wskazują na złą wersję językową.
        """
        result = query("Jakie mam aktywne oferty?", new_session())
        resp = result["response"]
        fully_english_phrases = [
            "Here are your active offers",
            "You have no active offers",
            "I need authorization first",
            "To access your Allegro store",
        ]
        for phrase in fully_english_phrases:
            assert phrase not in resp, (
                f"Odpowiedź zawiera angielski tekst na polskie pytanie: '{phrase}'. "
                f"Pełna odpowiedź: {resp[:400]}"
            )


class TestEnglishResponses:
    def test_english_greeting_answered_in_english(self):
        """Hello → odpowiedź po angielsku"""
        result = query("Hello, how are you?", new_session())
        resp = result["response"].lower()
        english_words = ["hello", "hi", "help", "how", "assist", "can", "store", "order"]
        assert any(w in resp for w in english_words), (
            f"Angielskie pytanie powinno dawać angielską odpowiedź: {result['response'][:300]}"
        )

    def test_english_order_query_answered_in_english(self):
        """Show me my orders → odpowiedź po angielsku"""
        result = query("Show me my recent orders", new_session())
        resp = result["response"].lower()
        english_words = ["order", "allegro", "authoriz", "link", "access", "recent"]
        assert any(w in resp for w in english_words)

    def test_english_capability_answered_in_english(self):
        """What can you do? → odpowiedź po angielsku"""
        result = query("What can you do?", new_session())
        resp = result["response"].lower()
        english_words = ["order", "offer", "message", "account", "can", "help", "allegro"]
        assert any(w in resp for w in english_words)


class TestLanguageConsistency:
    def test_language_stable_in_multi_turn_conversation(self):
        """
        Rozmowa po polsku powinna pozostać po polsku w kolejnych turach.
        Tura 1: 'Cześć' → PL
        Tura 2: 'Co potrafisz?' → PL
        """
        session = new_session()
        r1 = query("Cześć!", session)
        r2 = query("Co potrafisz?", session)
        assert is_likely_polish(r1["response"]), f"Tura 1 powinna być po polsku: {r1['response'][:200]}"
        assert is_likely_polish(r2["response"]), f"Tura 2 powinna być po polsku: {r2['response'][:200]}"
