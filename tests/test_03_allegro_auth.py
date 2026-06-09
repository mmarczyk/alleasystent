"""
Moduł 3 — Autoryzacja Allegro

Weryfikuje zachowanie systemu przy braku autoryzacji
oraz poprawność mechanizmu device flow.
"""

import pytest
import httpx
from conftest import query, allegro_auth_status, new_session, BASE_URL, requires_allegro


class TestAuthStatus:
    def test_auth_status_endpoint_reachable(self):
        """
        Scenariusz: Wywołanie GET /allegro/auth/status
        Oczekiwane: odpowiedź 200 z polem 'status'
        """
        status = allegro_auth_status()
        assert "status" in status
        assert status["status"] in {"idle", "pending", "authorized", "expired", "error"}

    def test_auth_start_endpoint_reachable(self):
        """
        Scenariusz: Wywołanie GET /allegro/auth
        Oczekiwane: redirect 302 na stronę Allegro lub błąd konfiguracji (503)
        Nie sprawdzamy powodzenia auth — tylko że endpoint reaguje
        """
        resp = httpx.get(f"{BASE_URL}/allegro/auth", timeout=15, follow_redirects=False)
        # 302 = redirect do Allegro (Allegro credentials skonfigurowane)
        # 503 = brak credentials
        assert resp.status_code in {302, 503}, (
            f"Nieoczekiwany status: {resp.status_code}"
        )
        if resp.status_code == 302:
            assert "allegro.pl" in resp.headers.get("location", ""), (
                "Redirect powinien prowadzić na allegro.pl"
            )


class TestUnauthenticatedBehavior:
    """Zachowanie gdy brak tokenów Allegro."""

    def test_order_query_without_auth_returns_auth_request(self):
        """
        Scenariusz: Pytanie o zamówienia bez autoryzacji
        Oczekiwane: asystent prosi o autoryzację — zawiera link lub instrukcję
        Odpowiedź powinna być po POLSKU
        """
        # Ten test ma sens tylko gdy system nie jest autoryzowany
        status = allegro_auth_status()
        if status["status"] == "authorized":
            pytest.skip("Allegro jest już autoryzowane — pomiń test braku auth")

        result = query("Pokaż moje nowe zamówienia", new_session())
        resp = result["response"].lower()

        auth_phrases = [
            "autoryzacj", "otwórz link", "link", "zatwierdź",
            "authorization", "open", "approve",
        ]
        assert any(p in resp for p in auth_phrases), (
            f"Bez autoryzacji odpowiedź powinna zawierać prośbę o auth: {result['response'][:400]}"
        )

    def test_auth_request_is_in_polish(self):
        """
        Scenariusz: Pytanie o zamówienia po polsku bez autoryzacji
        Oczekiwane: komunikat autoryzacyjny w języku polskim
        """
        status = allegro_auth_status()
        if status["status"] == "authorized":
            pytest.skip("Allegro jest już autoryzowane")

        result = query("Pokaż moje zamówienia", new_session())
        resp = result["response"]

        english_only_phrases = [
            "To access your Allegro",
            "I need authorization first",
            "Please open this link",
            "I wasn't able to confirm",
        ]
        for phrase in english_only_phrases:
            assert phrase not in resp, (
                f"Komunikat autoryzacyjny zawiera angielski tekst: '{phrase}'. "
                f"Pełna odpowiedź: {resp[:400]}"
            )

    def test_capabilities_question_never_triggers_auth(self):
        """
        Scenariusz: Pytanie o możliwości (niezależnie od stanu autoryzacji)
        Oczekiwane: NIE wywołuje flow autoryzacji
        """
        result = query("Jakie masz możliwości?", new_session())
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        assert "open this link" not in resp
        assert result["agent"] == "chitchat"


class TestAuthenticatedBehavior:
    """Testy wymagające aktywnej autoryzacji Allegro."""

    @requires_allegro
    def test_auth_status_shows_authorized(self):
        """
        Scenariusz: Po pomyślnej autoryzacji status = 'authorized'
        """
        status = allegro_auth_status()
        assert status["status"] == "authorized"
        assert status.get("authenticated") is True

    @requires_allegro
    def test_order_query_after_auth_does_not_ask_for_auth(self):
        """
        Scenariusz: Pytanie o zamówienia po autoryzacji
        Oczekiwane: odpowiedź zawiera dane zamówień (nie prośbę o auth)
        """
        result = query("Pokaż moje nowe zamówienia", new_session())
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        assert "open this link" not in resp
        assert result["agent"] == "allegro"
