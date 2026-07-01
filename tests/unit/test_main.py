"""Unit tests for main.py FastAPI endpoints."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("JWT_SECRET", "test-secret")


@pytest.fixture(scope="module")
def app():
    """Build the FastAPI app with all startup hooks mocked out."""
    with patch("main.asyncio.create_task"), \
         patch("agents.orchestrator.AsyncOpenAI"), \
         patch("agents.orchestrator.FirestoreService"), \
         patch("agents.rag.retriever.ChromaRetriever._init"), \
         patch("agents.rag.retriever.build_retriever"), \
         patch("webhooks.facebook_webhook.FacebookCommunicationAgent"):
        import main as main_module
        return main_module.app


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "env" in data

    def test_health_returns_dev_env_by_default(self, client):
        resp = client.get("/health")
        assert resp.json()["env"] == "development"


class TestPushVapidKey:
    def test_returns_503_when_not_configured(self, client):
        resp = client.get("/push/vapid-public-key")
        assert resp.status_code == 503

    def test_returns_key_when_configured(self, client, monkeypatch):
        monkeypatch.setenv("VAPID_PUBLIC_KEY", "BPublicKeyHere")
        from config.settings import get_settings
        get_settings.cache_clear()
        resp = client.get("/push/vapid-public-key")
        assert resp.status_code == 200
        assert resp.json()["publicKey"] == "BPublicKeyHere"
        get_settings.cache_clear()


class TestAllegroLogin:
    def test_login_redirects_to_allegro(self, client):
        resp = client.get("/allegro/login", follow_redirects=False)
        assert resp.status_code == 302
        assert "allegro.pl" in resp.headers.get("location", "")

    def test_auth_alias_redirects(self, client):
        resp = client.get("/allegro/auth", follow_redirects=False)
        assert resp.status_code == 302


class TestAllegroAuthStatus:
    def test_returns_idle_when_no_session(self, client):
        resp = client.get("/allegro/auth/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "idle"
        assert resp.json()["authenticated"] is False


class TestAuthLogout:
    def test_logout_redirects(self, client):
        resp = client.get("/auth/logout", follow_redirects=False)
        assert resp.status_code == 302

    def test_logout_clears_session_cookie(self, client):
        resp = client.get("/auth/logout", follow_redirects=False)
        # The response should delete the session cookie
        set_cookie = resp.headers.get("set-cookie", "")
        assert "session" in set_cookie or resp.status_code == 302


class TestAuthMe:
    def test_returns_401_without_session(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code == 401

    def test_returns_user_with_valid_session(self, client):
        from services.auth_service import create_session_token
        token = create_session_token({"sub": "myuser", "name": "My User"})
        resp = client.get("/auth/me", cookies={"session": token})
        assert resp.status_code == 200
        assert resp.json()["sub"] == "myuser"


class TestQueryEndpoint:
    def test_query_without_auth_uses_sender_id(self, client):
        mock_resp = MagicMock()
        mock_resp.text = "Hello!"
        mock_resp.agent_type = "chitchat"
        mock_resp.sources = []
        with patch("main._orchestrator") as mock_orc:
            mock_orc.handle = AsyncMock(return_value=mock_resp)
            resp = client.post("/query", json={
                "message": "hi",
                "session_id": "test",
                "sender_id": "anon",
            })
        assert resp.status_code == 200
        assert resp.json()["response"] == "Hello!"

    def test_query_with_valid_session(self, client):
        from services.auth_service import create_session_token
        token = create_session_token({"sub": "seller1", "name": "Seller"})
        mock_resp = MagicMock()
        mock_resp.text = "Order found."
        mock_resp.agent_type = "allegro"
        mock_resp.sources = []
        with patch("main._orchestrator") as mock_orc:
            mock_orc.handle = AsyncMock(return_value=mock_resp)
            resp = client.post("/query",
                json={"message": "my orders"},
                cookies={"session": token},
            )
        assert resp.status_code == 200
        assert resp.json()["agent"] == "allegro"


class TestPushSubscribe:
    def test_subscribe_requires_auth(self, client):
        resp = client.post("/push/subscribe", json={"endpoint": "https://push.example.com"})
        assert resp.status_code == 401

    def test_subscribe_with_auth(self, client):
        from services.auth_service import create_session_token
        token = create_session_token({"sub": "user1", "name": "User"})
        with patch("services.push_service.save_subscription", new_callable=AsyncMock):
            resp = client.post(
                "/push/subscribe",
                json={"endpoint": "https://push.example.com"},
                cookies={"session": token},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "subscribed"


class TestPushPending:
    def test_pending_requires_auth(self, client):
        resp = client.get("/push/pending")
        assert resp.status_code == 401

    def test_pending_returns_null_when_no_messages(self, client):
        from services.auth_service import create_session_token
        token = create_session_token({"sub": "user1", "name": "User"})
        with patch("services.push_service.pop_pending_chat", new_callable=AsyncMock, return_value=None):
            resp = client.get("/push/pending", cookies={"session": token})
        assert resp.status_code == 200
        assert resp.json()["chatMessage"] is None
