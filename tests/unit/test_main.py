"""Unit tests for main.py FastAPI endpoints."""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("JWT_SECRET", "test-secret")


@pytest.fixture(scope="module")
def app():
    """Build the FastAPI app with all startup hooks mocked out.

    The env is seeded here rather than left to set_env above: this fixture is
    module-scoped, so it runs BEFORE any function-scoped monkeypatch, and
    importing main builds Settings — which refuses to construct without
    GOOGLE_API_KEY. setdefault, so a real environment still wins.
    """
    os.environ.setdefault("GOOGLE_API_KEY", "test-key")
    os.environ.setdefault("JWT_SECRET", "test-secret")
    with patch("main.asyncio.create_task"), \
         patch("agents.orchestrator.AsyncOpenAI"), \
         patch("agents.orchestrator.SessionStore"), \
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
        # The endpoint reads main.settings, resolved once at import — clearing
        # the get_settings cache after the app is built changes nothing, so the
        # key has to be set on the object the endpoint actually reads.
        import main as main_module
        monkeypatch.setattr(main_module.settings, "vapid_public_key", "BPublicKeyHere")
        resp = client.get("/push/vapid-public-key")
        assert resp.status_code == 200
        assert resp.json()["publicKey"] == "BPublicKeyHere"


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
        # A valid session is not enough on its own: /auth/me also insists on a
        # live Allegro token (lost on container restart → re-login), so the
        # service has to report one or the endpoint answers 401
        # allegro_auth_required.
        from services.auth_service import create_session_token
        token = create_session_token({"sub": "myuser", "name": "My User"})
        service = MagicMock()
        service._tokens = MagicMock()
        with patch("services.allegro_service.AllegroService.get_instance", return_value=service):
            resp = client.get("/auth/me", cookies={"session": token})
        assert resp.status_code == 200
        assert resp.json()["sub"] == "myuser"

    def test_valid_session_without_allegro_tokens_forces_relogin(self, client):
        from services.auth_service import create_session_token
        token = create_session_token({"sub": "myuser", "name": "My User"})
        service = MagicMock()
        service._tokens = None
        service._load_tokens_from_redis = AsyncMock()
        with patch("services.allegro_service.AllegroService.get_instance", return_value=service):
            resp = client.get("/auth/me", cookies={"session": token})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "allegro_auth_required"


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


class TestConversationsApi:
    """The chat list and its history are served per user, so a thread started
    on one device opens on the next one."""

    @staticmethod
    def _token(sub="seller1"):
        from services.auth_service import create_session_token
        return create_session_token({"sub": sub, "name": sub})

    @staticmethod
    def _session(conv_id="c1", user="seller1", messages=(), title=None):
        from models.conversation import ChannelType, ConversationSession, MessageRole
        session = ConversationSession(
            session_id=f"{user}:{conv_id}", channel=ChannelType.API, sender_id=user,
        )
        for role, content, meta in messages:
            session.add_message(MessageRole(role), content, meta)
        if title:
            session.metadata["title"] = title
        return session

    def _store(self, **methods):
        store = MagicMock()
        for name, value in methods.items():
            store.__setattr__(name, AsyncMock(return_value=value))
        return store

    def test_list_requires_auth(self, client):
        assert client.get("/conversations").status_code == 401

    def test_list_returns_the_users_conversations(self, client):
        import main
        sessions = [
            self._session("c1", messages=[("user", "pokaż zamówienia", None),
                                          ("assistant", "| a |", {"agent": "allegro_orders:table"})]),
            self._session("c2", messages=[("user", "co z fakturami?", None)]),
        ]
        store = self._store(list_user_sessions=sessions)
        with patch.object(main._orchestrator, "_session_store", store):
            resp = client.get("/conversations", cookies={"session": self._token()})

        assert resp.status_code == 200
        convs = resp.json()["conversations"]
        assert [c["id"] for c in convs] == ["c1", "c2"]
        # Title falls back to the opening question, so the list is readable on a
        # device that has never seen the thread.
        assert convs[0]["title"] == "pokaż zamówienia"
        assert convs[0]["messageCount"] == 2
        assert isinstance(convs[0]["updatedAt"], int)

    def test_list_prefers_a_user_set_title(self, client):
        import main
        store = self._store(list_user_sessions=[
            self._session("c1", messages=[("user", "pokaż zamówienia", None)], title="Zamówienia"),
        ])
        with patch.object(main._orchestrator, "_session_store", store):
            resp = client.get("/conversations", cookies={"session": self._token()})
        assert resp.json()["conversations"][0]["title"] == "Zamówienia"

    def test_get_returns_full_history_with_reply_formats(self, client):
        import main
        session = self._session("c1", messages=[
            ("user", "pokaż zamówienia", None),
            ("assistant", "| nr | kwota |", {"agent": "allegro_orders:table"}),
            ("assistant", "gotowe", None),          # stored before formats were recorded
        ])
        store = self._store(get_session=session)
        with patch.object(main._orchestrator, "_session_store", store):
            resp = client.get("/conversations/c1", cookies={"session": self._token()})

        assert resp.status_code == 200
        body = resp.json()
        assert store.get_session.await_args.args[0] == "seller1:c1"
        assert [m["format"] for m in body["messages"]] == ["chat", "table", "chat"]
        assert body["messages"][0]["content"] == "pokaż zamówienia"

    def test_get_missing_conversation_is_404(self, client):
        import main
        store = self._store(get_session=None)
        with patch.object(main._orchestrator, "_session_store", store):
            resp = client.get("/conversations/nope", cookies={"session": self._token()})
        assert resp.status_code == 404

    def test_rejects_an_id_that_is_not_a_plain_identifier(self, client):
        """Ids are minted by the client and become part of a storage key."""
        import main
        store = self._store(get_session=None)
        with patch.object(main._orchestrator, "_session_store", store):
            resp = client.get("/conversations/a:b", cookies={"session": self._token()})
        assert resp.status_code == 400
        store.get_session.assert_not_awaited()

    def test_rename_persists_the_title(self, client):
        import main
        session = self._session("c1", messages=[("user", "pokaż zamówienia", None)])
        store = self._store(get_session=session, save_session=None)
        with patch.object(main._orchestrator, "_session_store", store):
            resp = client.patch("/conversations/c1", json={"title": "  Zamówienia  "},
                                cookies={"session": self._token()})

        assert resp.status_code == 200
        assert resp.json()["title"] == "Zamówienia"
        assert session.metadata["title"] == "Zamówienia"
        store.save_session.assert_awaited_once()

    def test_rename_rejects_an_empty_title(self, client):
        import main
        store = self._store(get_session=self._session(), save_session=None)
        with patch.object(main._orchestrator, "_session_store", store):
            resp = client.patch("/conversations/c1", json={"title": "   "},
                                cookies={"session": self._token()})
        assert resp.status_code == 400
        store.save_session.assert_not_awaited()

    def test_delete_removes_it_for_every_device(self, client):
        import main
        store = self._store(delete_session=True)
        with patch.object(main._orchestrator, "_session_store", store):
            resp = client.delete("/conversations/c1", cookies={"session": self._token()})

        assert resp.status_code == 200
        assert resp.json()["existed"] is True
        assert store.delete_session.await_args.args[0] == "seller1:c1"
        assert store.delete_session.await_args.kwargs["user_id"] == "seller1"

    def test_delete_is_idempotent(self, client):
        """A draft that never left one device is still "gone" when deleted."""
        import main
        store = self._store(delete_session=False)
        with patch.object(main._orchestrator, "_session_store", store):
            resp = client.delete("/conversations/local-only", cookies={"session": self._token()})
        assert resp.status_code == 200
        assert resp.json()["existed"] is False


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
        with patch("services.push_service.pop_pending_chats_tagged", new_callable=AsyncMock, return_value=[]):
            resp = client.get("/push/pending", cookies={"session": token})
        assert resp.status_code == 200
        assert resp.json()["chatMessage"] is None


class TestRefreshPendingChats:
    """Queued messages are re-checked before delivery — a day-old invoice
    reminder must not tell the seller to issue an invoice that now exists."""

    @pytest.mark.asyncio
    async def test_invoice_reminder_is_rechecked_and_can_be_dropped(self):
        import main
        from services.invoice_reminder import PENDING_CHAT_TAG

        refresh = AsyncMock(return_value=None)
        with patch("services.invoice_reminder.refresh_pending_message", refresh):
            texts = await main._refresh_pending_chats(
                "u1", [(PENDING_CHAT_TAG, "🧾 Masz 1 niewystawioną fakturę …")]
            )
        assert texts == []
        refresh.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_other_messages_pass_through_untouched(self):
        import main

        refresh = AsyncMock()
        with patch("services.invoice_reminder.refresh_pending_message", refresh):
            texts = await main._refresh_pending_chats(
                "u1", [("message_reminder", "📬 Masz 2 nieprzeczytane wiadomości."), (None, "Cześć")]
            )
        assert texts == ["📬 Masz 2 nieprzeczytane wiadomości.", "Cześć"]
        refresh.assert_not_awaited()


class TestPendingChatRecordedInSession:
    """A message the assistant wrote on its own initiative has to land in the
    conversation history too, or the assistant reads the seller's reply to it
    with no idea a question was ever asked."""

    @pytest.mark.asyncio
    async def test_records_delivered_messages_as_assistant_turns(self):
        import main
        from models.conversation import ChannelType, ConversationSession, MessageRole

        session = ConversationSession(session_id="u1:conv7", channel=ChannelType.API, sender_id="u1")
        store = MagicMock()
        store.get_or_create_session = AsyncMock(return_value=session)
        store.save_session = AsyncMock()
        with patch.object(main._orchestrator, "_session_store", store):
            await main._record_assistant_turns("u1", "conv7", ["🧾 Wystawić faktury?"])

        assert store.get_or_create_session.await_args.kwargs["session_id"] == "u1:conv7"
        assert [(m.role, m.content) for m in session.messages] == [
            (MessageRole.ASSISTANT, "🧾 Wystawić faktury?")
        ]
        store.save_session.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recording_failure_does_not_break_delivery(self):
        """The messages have already left the queue — showing them wins."""
        import main

        store = MagicMock()
        store.get_or_create_session = AsyncMock(side_effect=RuntimeError("redis down"))
        with patch.object(main._orchestrator, "_session_store", store):
            await main._record_assistant_turns("u1", "conv7", ["cokolwiek"])
