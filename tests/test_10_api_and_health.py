"""
Moduł 10 — API, health check i obsługa błędów

Weryfikuje infrastrukturę API: dostępność, kształt odpowiedzi,
zachowanie przy nieprawidłowych danych wejściowych.
"""

import pytest
import httpx
from conftest import query, new_session, BASE_URL, _SESSION_COOKIE


class TestHealthCheck:
    def test_health_endpoint_returns_200(self):
        """
        GET /health → 200 z polem status=ok
        """
        resp = httpx.get(f"{BASE_URL}/health", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "env" in data

    def test_health_env_field_valid(self):
        """
        Pole env powinno być 'development' lub 'production'
        """
        resp = httpx.get(f"{BASE_URL}/health", timeout=10)
        data = resp.json()
        assert data["env"] in {"development", "production"}


class TestQueryEndpointShape:
    def test_query_returns_required_fields(self):
        """
        POST /query zwraca: response (str), agent (str), sources (list)
        """
        result = query("Cześć", new_session())
        assert "response" in result
        assert "agent" in result
        assert "sources" in result
        assert isinstance(result["response"], str)
        assert isinstance(result["agent"], str)
        assert isinstance(result["sources"], list)

    def test_query_response_is_not_empty(self):
        """
        Odpowiedź nie może być pustym stringiem
        """
        result = query("Cześć", new_session())
        assert len(result["response"].strip()) > 0

    def test_query_agent_field_is_known_value(self):
        """
        Pole agent powinno być jedną z znanych wartości
        """
        result = query("Jakie mam zamówienia?", new_session())
        known_agents = {"allegro", "rag", "chitchat", "base"}
        assert result["agent"] in known_agents, (
            f"Nieznana wartość pola agent: '{result['agent']}'"
        )

    def test_query_with_custom_session_id(self):
        """
        session_id przekazany przez klienta jest respektowany (rozmowa jest izolowana)
        """
        session = "test_custom_session_12345"
        r1 = query("Cześć", session)
        r2 = query("Co powiedziałem poprzednio?", session)
        assert r1["response"]
        assert r2["response"]

    def test_query_without_session_id_uses_default(self):
        """
        POST /query bez session_id działa (używa domyślnego 'api_session')
        """
        resp = httpx.post(
            f"{BASE_URL}/query",
            json={"message": "Cześć"},
            timeout=60,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data


class TestEdgeCases:
    def test_very_short_message(self):
        """
        Wiadomość jednowyrazowa nie powoduje błędu
        """
        result = query("?", new_session())
        assert result["response"]

    def test_long_message_does_not_crash(self):
        """
        Długa wiadomość (1000 znaków) nie powoduje błędu 500
        """
        long_message = "Pokaż mi wszystkie zamówienia. " * 35
        result = query(long_message, new_session())
        assert result["response"]

    def test_special_characters_in_message(self):
        """
        Znaki specjalne i emoji nie powodują błędu
        """
        result = query("Cześć 😊 mam pytanie o zamówieni@ nr #1234!", new_session())
        assert result["response"]

    def test_missing_message_field_returns_422(self):
        """
        Brak wymaganego pola 'message' → 422 Unprocessable Entity
        """
        resp = httpx.post(
            f"{BASE_URL}/query",
            json={"session_id": "test"},
            timeout=10,
        )
        assert resp.status_code == 422

    def test_off_topic_question_does_not_trigger_allegro_auth(self):
        """
        Pytanie zupełnie niezwiązane ze sklepem nie może wywoływać auth flow Allegro
        """
        result = query("Jaka pogoda będzie jutro w Krakowie?", new_session())
        resp = result["response"].lower()
        assert "otwórz link" not in resp
        assert "open this link" not in resp
        assert "authorization" not in resp
        assert "autoryzacj" not in resp


class TestStaticUI:
    def test_frontend_served_at_root(self):
        """
        GET / serwuje interfejs webowy (HTML)
        """
        resp = httpx.get(f"{BASE_URL}/", timeout=10)
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_frontend_contains_app_title(self):
        """
        Strona główna zawiera tytuł aplikacji
        """
        resp = httpx.get(f"{BASE_URL}/", timeout=10)
        assert "AllEasystent" in resp.text


class TestPushEndpoints:
    """Testy endpointów Web Push (dodanych po wdrożeniu powiadomień cross-device)."""

    def test_push_pending_requires_session(self):
        """
        GET /push/pending bez sesji → 401.
        """
        resp = httpx.get(f"{BASE_URL}/push/pending", timeout=10)
        assert resp.status_code == 401

    def test_push_status_requires_session(self):
        """
        GET /push/status bez sesji → 401.
        """
        resp = httpx.get(f"{BASE_URL}/push/status", timeout=10)
        assert resp.status_code == 401

    def test_push_pending_with_session_returns_chat_message_field(self):
        """
        GET /push/pending z sesją → 200 z polem chatMessage (str lub null).
        """
        if not _SESSION_COOKIE:
            pytest.skip("JWT_SECRET nie ustawiony — pomiń test sesji")
        cookies = {"session": _SESSION_COOKIE}
        resp = httpx.get(f"{BASE_URL}/push/pending", cookies=cookies, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "chatMessage" in data
        assert data["chatMessage"] is None or isinstance(data["chatMessage"], str)

    def test_push_status_with_session_returns_config_fields(self):
        """
        GET /push/status z sesją → 200 z polami o konfiguracji VAPID i subskrypcjach.
        """
        if not _SESSION_COOKIE:
            pytest.skip("JWT_SECRET nie ustawiony — pomiń test sesji")
        cookies = {"session": _SESSION_COOKIE}
        resp = httpx.get(f"{BASE_URL}/push/status", cookies=cookies, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "vapid_public_key_set" in data
        assert "vapid_private_key_set" in data
        assert "subscriptions_count" in data
        assert isinstance(data["vapid_public_key_set"], bool)
        assert isinstance(data["subscriptions_count"], int)

    def test_push_vapid_public_key_endpoint(self):
        """
        GET /push/vapid-public-key → 200 z kluczem lub 503 gdy brak konfiguracji VAPID.
        """
        resp = httpx.get(f"{BASE_URL}/push/vapid-public-key", timeout=10)
        assert resp.status_code in {200, 503}
        if resp.status_code == 200:
            data = resp.json()
            assert "publicKey" in data
            assert isinstance(data["publicKey"], str)
