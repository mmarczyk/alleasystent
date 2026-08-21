"""Unit tests for services/push_service.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("REDIS_URL", raising=False)


class TestNoRedisEarlyReturns:
    @pytest.mark.asyncio
    async def test_save_subscription_no_redis_returns_early(self):
        from services.push_service import save_subscription
        # Should not raise, just log warning
        await save_subscription("user1", {"endpoint": "https://example.com/push"})

    @pytest.mark.asyncio
    async def test_store_pending_chat_no_redis_returns_early(self):
        from services.push_service import store_pending_chat
        await store_pending_chat("user1", "hello")

    @pytest.mark.asyncio
    async def test_pop_pending_chats_no_redis_returns_empty(self):
        from services.push_service import pop_pending_chats
        result = await pop_pending_chats("user1")
        assert result == []

    @pytest.mark.asyncio
    async def test_remove_subscription_no_redis_returns_early(self):
        from services.push_service import remove_subscription
        await remove_subscription("user1", "https://example.com/push")

    @pytest.mark.asyncio
    async def test_get_subscriptions_no_redis_returns_empty(self):
        from services.push_service import _get_subscriptions
        result = await _get_subscriptions("user1")
        assert result == []

    @pytest.mark.asyncio
    async def test_send_push_no_vapid_keys_returns_early(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        monkeypatch.delenv("VAPID_PRIVATE_KEY", raising=False)
        monkeypatch.delenv("VAPID_PUBLIC_KEY", raising=False)
        from services.push_service import send_push
        # Should return early without error
        await send_push("user1", "Title", "Body")


class TestWithMockedRedis:
    @pytest.mark.asyncio
    async def test_save_subscription_calls_redis_set(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock()
        mock_redis.aclose = AsyncMock()
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            from services.push_service import save_subscription
            await save_subscription("user1", {"endpoint": "https://push.example.com"})
        mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_pending_chat_calls_rpush(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        mock_redis = AsyncMock()
        mock_redis.rpush = AsyncMock()
        mock_redis.expire = AsyncMock()
        mock_redis.aclose = AsyncMock()
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            from services.push_service import store_pending_chat
            await store_pending_chat("user1", "test message")
        mock_redis.rpush.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_pending_chat_replaces_same_tag(self, monkeypatch):
        """A recurring reminder states the current situation — an unanswered one
        is replaced, not stacked, so a seller away for a day doesn't come back
        to a dozen identical messages."""
        import json
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        queued = json.dumps({"tag": "invoice_reminder", "text": "stara wersja"})
        mock_redis = AsyncMock()
        mock_redis.lrange = AsyncMock(return_value=[queued, json.dumps({"tag": None, "text": "coś innego"})])
        mock_redis.lrem = AsyncMock()
        mock_redis.rpush = AsyncMock()
        mock_redis.expire = AsyncMock()
        mock_redis.aclose = AsyncMock()
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            from services.push_service import store_pending_chat
            await store_pending_chat("user1", "nowa wersja", dedupe_tag="invoice_reminder")
        mock_redis.lrem.assert_called_once_with("push:chat:user1", 0, queued)
        assert json.loads(mock_redis.rpush.call_args[0][1])["text"] == "nowa wersja"

    def _mock_redis_with_queue(self, entries):
        """Redis mock whose MULTI/EXEC pipeline returns `entries` then deletes."""
        pipe = MagicMock()
        pipe.lrange = MagicMock()
        pipe.delete = MagicMock()
        pipe.execute = AsyncMock(return_value=[entries, 1])
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=pipe)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_redis = AsyncMock()
        mock_redis.pipeline = MagicMock(return_value=ctx)
        mock_redis.aclose = AsyncMock()
        return mock_redis

    @pytest.mark.asyncio
    async def test_pop_pending_chats_drains_whole_queue(self, monkeypatch):
        import json
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        entries = [
            json.dumps({"tag": "invoice_reminder", "text": "pierwsza"}),
            json.dumps({"tag": None, "text": "druga"}),
        ]
        mock_redis = self._mock_redis_with_queue(entries)
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            from services.push_service import pop_pending_chats
            result = await pop_pending_chats("user1")
        assert result == ["pierwsza", "druga"]
        mock_redis.pipeline.assert_called_once_with(transaction=True)

    @pytest.mark.asyncio
    async def test_pop_pending_chats_reads_legacy_plain_entries(self, monkeypatch):
        """Messages queued before entries carried a tag are bare strings."""
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        mock_redis = self._mock_redis_with_queue(["zwykły tekst"])
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            from services.push_service import pop_pending_chats
            result = await pop_pending_chats("user1")
        assert result == ["zwykły tekst"]

    @pytest.mark.asyncio
    async def test_pop_pending_chats_empty_queue(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        mock_redis = self._mock_redis_with_queue([])
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            from services.push_service import pop_pending_chats
            result = await pop_pending_chats("user1")
        assert result == []

    @pytest.mark.asyncio
    async def test_remove_subscription_calls_delete(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock()
        mock_redis.aclose = AsyncMock()
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            from services.push_service import remove_subscription
            await remove_subscription("user1", "https://push.example.com")
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_subscriptions_returns_subscriptions(self, monkeypatch):
        import json
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        sub = {"endpoint": "https://push.example.com", "keys": {}}
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(return_value=["push:sub:user1:abc"])
        mock_redis.mget = AsyncMock(return_value=[json.dumps(sub)])
        mock_redis.aclose = AsyncMock()
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            from services.push_service import _get_subscriptions
            result = await _get_subscriptions("user1")
        assert len(result) == 1
        assert result[0]["endpoint"] == "https://push.example.com"
