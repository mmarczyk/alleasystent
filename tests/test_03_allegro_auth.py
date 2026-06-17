"""
Moduł 3 — Autoryzacja Allegro

Weryfikuje zachowanie systemu przy braku autoryzacji
oraz poprawność mechanizmu OAuth2 (browser login flow).
"""

import pytest
import httpx
from conftest import query, allegro_auth_status, new_session, BASE_URL, requires_allegro, _SESSION_COOKIE


class TestAuthEndpoints:
    def test_login_endpoint_redirects_to_allegro(self):
        """
        GET /allegro/login → redirect 302 na stronę Allegro lub 503 gdy brak credentials.
        Endpoint zastąpił stary device flow.
        """
        resp = httpx.get(f"{BASE_URL}/allegro/login", timeout=15, follow_redirects=False)
        assert resp.status_code in {302, 503}, (
            f"Nieoczekiwany status: {resp.status_code}"
        )
        if resp.status_code == 302:
            location = resp.headers.get("location", "")
            assert "allegro.pl" in location, (
                f"Redirect powinien prowadzić na allegro.pl, dostałem: {location}"
            )

    def test_auth_me_requires_session(self):
        """
        GET /auth/me bez sesji → 401 Unauthorized.
        """
        resp = httpx.get(f"{BASE_URL}/auth/me", timeout=10)
        assert resp.status_code == 401

    def test_auth_me_with_valid_session(self):
        """
        GET /auth/me z ważną sesją JWT → 200 z polem sub.
        Wymaga JWT_SECRET ustawionego w środowisku testowym.
        """
        if not _SESSION_COOKIE:
            pytest.skip("JWT_SECRET nie ustawiony — pomiń test sesji")
        cookies = {"session": _SESSION_COOKIE}
        resp = httpx.get(f"{BASE_URL}/auth/me", cookies=cookies, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "sub" in data


class TestUnauthenticatedBehavior:
    """Zachowanie gdy brak tokenów Allegro."""

    def test_order_query_without_auth_returns_login_link(self):
        """
        Scenariusz: Pytanie o zamówienia bez autoryzacji Allegro
        Oczekiwane: asystent zwraca link do /allegro/login (OAuth2 browser flow)
        """
        status = allegro_auth_status()
        if status["status"] == "authorized":
            pytest.skip("Allegro jest już autoryzowane — pomiń test braku auth")

        result = query("Pokaż moje nowe zamówienia", new_session())
        resp = result["response"]

        # Nowy flow: link do /allegro/login zamiast kodu device flow
        auth_phrases = [
            "/allegro/login", "allegro/login", "zaloguj", "autoryzacj",
            "authorization", "login", "link",
        ]
        assert any(p in resp.lower() for p in auth_phrases), (
            f"Bez autoryzacji odpowiedź powinna zawierać link do logowania: {resp[:400]}"
        )

    def test_auth_message_no_device_flow_code(self):
        """
        Nowy flow OAuth2 nie używa device flow — odpowiedź nie może zawierać
        kodu do wpisania ani URL weryfikacyjnego device flow Allegro.
        """
        status = allegro_auth_status()
        if status["status"] == "authorized":
            pytest.skip("Allegro jest już autoryzowane")

        result = query("Pokaż moje zamówienia", new_session())
        resp = result["response"]

        device_flow_phrases = [
            "allegro.pl/device", "kod weryfikacyjny", "wpisz kod",
            "enter code", "user_code", "verification_uri",
        ]
        for phrase in device_flow_phrases:
            assert phrase not in resp.lower(), (
                f"Odpowiedź nie powinna zawierać elementów device flow: '{phrase}'. "
                f"Pełna odpowiedź: {resp[:400]}"
            )

    def test_auth_request_is_in_polish(self):
        """
        Pytanie po polsku bez autoryzacji → komunikat autoryzacyjny po polsku.
        """
        status = allegro_auth_status()
        if status["status"] == "authorized":
            pytest.skip("Allegro jest już autoryzowane")

        result = query("Pokaż moje zamówienia", new_session())
        resp = result["response"]

        english_only_phrases = [
            "I need authorization first",
            "Please open this link",
            "I wasn't able to confirm",
            "To access your Allegro",
        ]
        for phrase in english_only_phrases:
            assert phrase not in resp, (
                f"Komunikat autoryzacyjny zawiera angielski tekst: '{phrase}'. "
                f"Pełna odpowiedź: {resp[:400]}"
            )

    def test_capabilities_question_never_triggers_auth(self):
        """
        Pytanie o możliwości asystenta nie wywołuje flow autoryzacji.
        """
        result = query("Jakie masz możliwości?", new_session())
        resp = result["response"].lower()
        assert "allegro/login" not in resp or result["agent"] == "chitchat"
        assert result["agent"] == "chitchat"


class TestAuthenticatedBehavior:
    """Testy wymagające aktywnej autoryzacji Allegro."""

    @requires_allegro
    def test_auth_status_shows_authorized(self):
        """
        Po pomyślnej autoryzacji /auth/me zwraca dane użytkownika.
        """
        status = allegro_auth_status()
        assert status["status"] == "authorized"
        assert status.get("authenticated") is True
        assert "sub" in status

    @requires_allegro
    def test_order_query_after_auth_does_not_ask_for_login(self):
        """
        Pytanie o zamówienia po autoryzacji nie zawiera prośby o logowanie.
        """
        result = query("Pokaż moje nowe zamówienia", new_session())
        resp = result["response"].lower()
        assert "/allegro/login" not in resp
        assert result["agent"] == "allegro"
