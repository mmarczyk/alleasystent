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
    async def test_pop_pending_chat_no_redis_returns_none(self):
        from services.push_service import pop_pending_chat
        result = await pop_pending_chat("user1")
        assert result is None

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
    async def test_pop_pending_chat_calls_lpop(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        mock_redis = AsyncMock()
        mock_redis.lpop = AsyncMock(return_value="pending message")
        mock_redis.aclose = AsyncMock()
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            from services.push_service import pop_pending_chat
            result = await pop_pending_chat("user1")
        assert result == "pending message"

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
