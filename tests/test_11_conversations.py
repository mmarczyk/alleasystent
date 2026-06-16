"""
Moduł 11 — Kontekst rozmowy wieloturowej

Weryfikuje że historia konwersacji jest przekazywana między turami
i że każda sesja jest izolowana od innych.
"""

import pytest
from conftest import query, new_session


class TestSessionIsolation:
    def test_two_sessions_are_independent(self):
        """
        Dwie niezależne sesje nie dzielą kontekstu.
        """
        s1 = new_session()
        s2 = new_session()
        query("Cześć, jestem Zygmunt Krupczyński.", s1)
        r = query("Jak mam na imię?", s2)
        resp = r["response"].lower()
        # Sesja 2 nie powinna wiedzieć o Zygmuncie Krupczyńskim
        assert "krupczyń" not in resp and "zygmunt" not in resp, (
            f"Sesje powinny być izolowane. s2 zwróciła: {r['response'][:300]}"
        )

    def test_same_session_retains_context(self):
        """
        Ta sama sesja pamięta kontekst poprzedniej tury.
        """
        session = new_session()
        query("Interesują mnie tylko zamówienia z ostatniego tygodnia.", session)
        r = query("Jakich zamówień szukam?", session)
        resp = r["response"].lower()
        # Model powinien pamiętać kontekst
        context_words = ["ostatn", "tydzień", "tygodnia", "week", "recent"]
        assert any(w in resp for w in context_words), (
            f"Model powinien pamiętać kontekst: {r['response'][:300]}"
        )


class TestConversationFlow:
    def test_follow_up_question_after_order_list(self):
        """
        Rozmowa:
          T1: 'Pokaż moje zamówienia' → lista zamówień
          T2: 'Ile to razem?' → odpowiedź w kontekście T1
        """
        session = new_session()
        r1 = query("Pokaż moje zamówienia", session)
        assert r1["agent"] == "allegro"

        r2 = query("Ile ich jest?", session)
        # T2 powinna być nadal o zamówieniach
        assert r2["agent"] == "allegro"

    def test_topic_switch_in_conversation(self):
        """
        Rozmowa:
          T1: pytanie o zamówienia
          T2: pytanie o oferty
        Oczekiwane: T2 trafia do allegro (nie gubi się po zmianie tematu)
        """
        session = new_session()
        query("Pokaż moje zamówienia", session)
        r2 = query("A jakie mam aktywne oferty?", session)
        assert r2["agent"] == "allegro"

    def test_clarification_follow_up(self):
        """
        Rozmowa:
          T1: 'Mam problem' (niejasne)
          T2: 'Chodzi mi o zamówienia'
        Oczekiwane: T2 = allegro
        """
        session = new_session()
        query("Mam problem", session)
        r2 = query("Chodzi mi o zamówienia — chcę sprawdzić ostatnie", session)
        assert r2["agent"] == "allegro"
