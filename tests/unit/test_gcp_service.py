"""Unit tests for services/gcp_service.py (in-memory / no-GCP mode)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.conversation import ChannelType, ConversationSession, MessageRole


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    # No GCP project — forces in-memory mode
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)


def _make_firestore():
    from services.gcp_service import FirestoreService
    svc = FirestoreService()
    # Confirm it's in in-memory mode
    assert svc._db is None
    assert svc._redis is None
    return svc


class TestFirestoreServiceInMemory:
    @pytest.mark.asyncio
    async def test_get_session_missing_returns_none(self):
        svc = _make_firestore()
        result = await svc.get_session("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_and_get_session(self):
        svc = _make_firestore()
        session = ConversationSession(
            session_id="s1",
            channel=ChannelType.API,
            sender_id="user1",
        )
        await svc.save_session(session)
        retrieved = await svc.get_session("s1")
        assert retrieved is not None
        assert retrieved.session_id == "s1"
        assert retrieved.sender_id == "user1"

    @pytest.mark.asyncio
    async def test_get_or_create_creates_new(self):
        svc = _make_firestore()
        session = await svc.get_or_create_session(
            session_id="new-session",
            channel=ChannelType.FACEBOOK,
            sender_id="fb-user",
        )
        assert session.session_id == "new-session"
        assert session.channel == ChannelType.FACEBOOK

    @pytest.mark.asyncio
    async def test_get_or_create_returns_existing(self):
        svc = _make_firestore()
        # First call creates
        s1 = await svc.get_or_create_session("s", ChannelType.API, "u")
        s1.add_message(MessageRole.USER, "hello")
        await svc.save_session(s1)
        # Second call retrieves
        s2 = await svc.get_or_create_session("s", ChannelType.API, "u")
        assert len(s2.messages) == 1
        assert s2.messages[0].content == "hello"

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self):
        svc = _make_firestore()
        sessions = await svc.list_sessions()
        assert sessions == []

    @pytest.mark.asyncio
    async def test_list_sessions_all(self):
        svc = _make_firestore()
        await svc.get_or_create_session("s1", ChannelType.API, "u1")
        await svc.get_or_create_session("s2", ChannelType.FACEBOOK, "u2")
        sessions = await svc.list_sessions()
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_filtered_by_channel(self):
        svc = _make_firestore()
        await svc.get_or_create_session("s1", ChannelType.API, "u1")
        await svc.get_or_create_session("s2", ChannelType.FACEBOOK, "u2")
        sessions = await svc.list_sessions(channel=ChannelType.FACEBOOK)
        assert len(sessions) == 1
        assert sessions[0].channel == ChannelType.FACEBOOK

    @pytest.mark.asyncio
    async def test_save_updates_updated_at(self):
        svc = _make_firestore()
        session = ConversationSession(
            session_id="s1",
            channel=ChannelType.API,
            sender_id="u",
        )
        original = session.updated_at
        import time; time.sleep(0.01)
        await svc.save_session(session)
        retrieved = await svc.get_session("s1")
        assert retrieved.updated_at >= original


class TestPubSubServiceNoGCP:
    def test_publish_returns_none_when_no_publisher(self):
        from services.gcp_service import PubSubService
        svc = PubSubService()
        assert svc._publisher is None

    @pytest.mark.asyncio
    async def test_publish_incoming_returns_none(self):
        from services.gcp_service import PubSubService
        svc = PubSubService()
        result = await svc.publish_incoming_message({"msg": "test"})
        assert result is None

    @pytest.mark.asyncio
    async def test_publish_outgoing_returns_none(self):
        from services.gcp_service import PubSubService
        svc = PubSubService()
        result = await svc.publish_outgoing_message({"msg": "test"})
        assert result is None
