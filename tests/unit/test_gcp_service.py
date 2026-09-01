"""Unit tests for services/gcp_service.py (in-memory / no-Redis mode)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.conversation import ChannelType, ConversationSession, MessageRole


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    # No REDIS_URL — forces in-memory mode
    monkeypatch.delenv("REDIS_URL", raising=False)


def _make_store():
    from services.gcp_service import SessionStore
    svc = SessionStore()
    # Confirm it's in in-memory mode
    assert svc._redis is None
    return svc


class TestSessionStoreInMemory:
    @pytest.mark.asyncio
    async def test_get_session_missing_returns_none(self):
        svc = _make_store()
        result = await svc.get_session("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_and_get_session(self):
        svc = _make_store()
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
        svc = _make_store()
        session = await svc.get_or_create_session(
            session_id="new-session",
            channel=ChannelType.FACEBOOK,
            sender_id="fb-user",
        )
        assert session.session_id == "new-session"
        assert session.channel == ChannelType.FACEBOOK

    @pytest.mark.asyncio
    async def test_get_or_create_returns_existing(self):
        svc = _make_store()
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
        svc = _make_store()
        sessions = await svc.list_sessions()
        assert sessions == []

    @pytest.mark.asyncio
    async def test_list_sessions_all(self):
        svc = _make_store()
        await svc.get_or_create_session("s1", ChannelType.API, "u1")
        await svc.get_or_create_session("s2", ChannelType.FACEBOOK, "u2")
        sessions = await svc.list_sessions()
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_filtered_by_channel(self):
        svc = _make_store()
        await svc.get_or_create_session("s1", ChannelType.API, "u1")
        await svc.get_or_create_session("s2", ChannelType.FACEBOOK, "u2")
        sessions = await svc.list_sessions(channel=ChannelType.FACEBOOK)
        assert len(sessions) == 1
        assert sessions[0].channel == ChannelType.FACEBOOK

    @pytest.mark.asyncio
    async def test_save_updates_updated_at(self):
        svc = _make_store()
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


class TestUserConversationList:
    """The seller's chat list is per user, not per device — that is what makes
    a conversation started on the desktop openable from the phone."""

    @pytest.mark.asyncio
    async def test_lists_only_that_users_conversations(self):
        svc = _make_store()
        await svc.get_or_create_session("seller1:a", ChannelType.API, "seller1")
        await svc.get_or_create_session("seller1:b", ChannelType.API, "seller1")
        await svc.get_or_create_session("seller2:a", ChannelType.API, "seller2")

        sessions = await svc.list_user_sessions("seller1")
        assert {s.session_id for s in sessions} == {"seller1:a", "seller1:b"}

    @pytest.mark.asyncio
    async def test_orders_most_recently_updated_first(self):
        svc = _make_store()
        old = await svc.get_or_create_session("u:old", ChannelType.API, "u")
        await svc.get_or_create_session("u:new", ChannelType.API, "u")
        import time; time.sleep(0.01)
        old.add_message(MessageRole.USER, "jeszcze jedno pytanie")
        await svc.save_session(old)

        sessions = await svc.list_user_sessions("u")
        assert [s.session_id for s in sessions] == ["u:old", "u:new"]

    @pytest.mark.asyncio
    async def test_limit_caps_the_list(self):
        svc = _make_store()
        for i in range(5):
            await svc.get_or_create_session(f"u:{i}", ChannelType.API, "u")
        assert len(await svc.list_user_sessions("u", limit=3)) == 3

    @pytest.mark.asyncio
    async def test_empty_for_unknown_user(self):
        svc = _make_store()
        await svc.get_or_create_session("u:a", ChannelType.API, "u")
        assert await svc.list_user_sessions("someone-else") == []

    @pytest.mark.asyncio
    async def test_delete_removes_session(self):
        svc = _make_store()
        await svc.get_or_create_session("u:a", ChannelType.API, "u")
        assert await svc.delete_session("u:a", user_id="u") is True
        assert await svc.get_session("u:a") is None
        assert await svc.list_user_sessions("u") == []

    @pytest.mark.asyncio
    async def test_delete_missing_session_reports_false(self):
        svc = _make_store()
        assert await svc.delete_session("u:nope", user_id="u") is False


class TestRedisUserIndex:
    """With Redis, the list comes from a per-user sorted set — a KEYS scan over
    a shared instance would be O(keyspace) on every chat-list load."""

    @pytest.mark.asyncio
    async def test_save_indexes_the_session_for_its_owner(self):
        svc = _make_store()
        svc._redis = MagicMock()
        svc._redis.set = AsyncMock()
        svc._redis.zadd = AsyncMock()
        svc._redis.expire = AsyncMock()

        session = ConversationSession(session_id="u:a", channel=ChannelType.API, sender_id="u")
        await svc.save_session(session)

        key, mapping = svc._redis.zadd.await_args.args
        assert key == "convidx:u"
        assert set(mapping) == {"u:a"}

    @pytest.mark.asyncio
    async def test_indexing_failure_does_not_lose_the_turn(self):
        """The session itself is already written — an index write that fails
        must not surface as a failed save."""
        svc = _make_store()
        svc._redis = MagicMock()
        svc._redis.set = AsyncMock()
        svc._redis.zadd = AsyncMock(side_effect=RuntimeError("redis down"))

        session = ConversationSession(session_id="u:a", channel=ChannelType.API, sender_id="u")
        await svc.save_session(session)   # must not raise
        svc._redis.set.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_index_is_recovered_by_one_scan_only(self):
        """An empty index is also what a brand-new seller looks like, and their
        app polls this list — so the recovery scan must not repeat."""
        svc = _make_store()
        svc._redis = MagicMock()
        svc._redis.zrevrange = AsyncMock(return_value=[])
        svc._redis.zadd = AsyncMock()
        svc._redis.expire = AsyncMock()
        old = ConversationSession(session_id="u:old", channel=ChannelType.API, sender_id="u")
        svc._redis.get = AsyncMock(return_value=old.model_dump_json())

        def _scan(match=None, count=None):
            async def _gen():
                yield "conv:u:old"
            return _gen()

        svc._redis.scan_iter = _scan

        first = await svc.list_user_sessions("u")
        assert [s.session_id for s in first] == ["u:old"]
        svc._redis.zadd.assert_awaited()   # recovered session written to the index

        assert await svc.list_user_sessions("u") == []   # no second scan

    @pytest.mark.asyncio
    async def test_expired_session_is_dropped_from_the_index(self):
        svc = _make_store()
        svc._redis = MagicMock()
        svc._redis.zrevrange = AsyncMock(return_value=["u:gone", "u:here"])
        svc._redis.zrem = AsyncMock()
        alive = ConversationSession(session_id="u:here", channel=ChannelType.API, sender_id="u")
        svc._redis.get = AsyncMock(
            side_effect=[None, alive.model_dump_json()],
        )

        sessions = await svc.list_user_sessions("u")
        assert [s.session_id for s in sessions] == ["u:here"]
        svc._redis.zrem.assert_awaited_once_with("convidx:u", "u:gone")


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
