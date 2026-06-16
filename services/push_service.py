from __future__ import annotations

"""Web Push notification service (VAPID).

Subscriptions are stored in Redis: push:sub:{user_id}:{md5(endpoint)}
Each subscription is the full JSON object from the browser's PushSubscription.toJSON().
"""

import asyncio
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


async def save_subscription(user_id: str, subscription: dict) -> None:
    """Persist a push subscription for a user (upsert by endpoint)."""
    from config.settings import get_settings
    settings = get_settings()
    if not settings.redis_url:
        logger.warning("No Redis URL — push subscriptions are not persisted")
        return
    import redis.asyncio as aioredis
    endpoint = subscription.get("endpoint", "")
    key = f"push:sub:{user_id}:{hashlib.md5(endpoint.encode()).hexdigest()}"
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        await r.set(key, json.dumps(subscription), ex=60 * 60 * 24 * 365)
    finally:
        await r.aclose()


async def store_pending_chat(user_id: str, text: str, ttl: int = 1800) -> None:
    """Store a chat message in Redis to be delivered when the user opens the app.

    Multiple messages accumulate as a list (FIFO). TTL 30 min by default.
    This lets devices that weren't open during polling still see the message.
    """
    from config.settings import get_settings
    settings = get_settings()
    if not settings.redis_url:
        return
    import redis.asyncio as aioredis
    key = f"push:chat:{user_id}"
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        await r.rpush(key, text)
        await r.expire(key, ttl)
    finally:
        await r.aclose()


async def pop_pending_chat(user_id: str) -> str | None:
    """Pop and return the oldest pending chat message, or None if queue is empty."""
    from config.settings import get_settings
    settings = get_settings()
    if not settings.redis_url:
        return None
    import redis.asyncio as aioredis
    key = f"push:chat:{user_id}"
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        return await r.lpop(key)
    finally:
        await r.aclose()


async def remove_subscription(user_id: str, endpoint: str) -> None:
    """Remove a specific push subscription by endpoint URL."""
    from config.settings import get_settings
    settings = get_settings()
    if not settings.redis_url:
        return
    import redis.asyncio as aioredis
    key = f"push:sub:{user_id}:{hashlib.md5(endpoint.encode()).hexdigest()}"
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        await r.delete(key)
    finally:
        await r.aclose()


async def _get_subscriptions(user_id: str) -> list[dict]:
    from config.settings import get_settings
    settings = get_settings()
    if not settings.redis_url:
        return []
    import redis.asyncio as aioredis
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        keys = await r.keys(f"push:sub:{user_id}:*")
        if not keys:
            return []
        vals = await r.mget(*keys)
        return [json.loads(v) for v in vals if v]
    finally:
        await r.aclose()


async def send_push(user_id: str, title: str, body: str, url: str = "/") -> None:
    """Send a Web Push notification to all subscribed devices for a user."""
    from config.settings import get_settings
    settings = get_settings()
    if not settings.vapid_private_key or not settings.vapid_public_key:
        logger.debug("VAPID keys not configured — skipping push notification")
        return

    subscriptions = await _get_subscriptions(user_id)
    if not subscriptions:
        logger.debug("No push subscriptions for user %s", user_id)
        return

    try:
        from pywebpush import webpush, WebPushException
    except ImportError:
        logger.error("pywebpush not installed — cannot send push notifications")
        return

    payload = json.dumps({"title": title, "body": body, "url": url}).encode()
    loop = asyncio.get_event_loop()
    stale: list[str] = []

    for sub in subscriptions:
        try:
            await loop.run_in_executor(
                None,
                lambda s=sub: webpush(
                    subscription_info=s,
                    data=payload,
                    vapid_private_key=settings.vapid_private_key,
                    vapid_claims={"sub": settings.vapid_email},
                ),
            )
            logger.debug("Push sent → %s", sub.get("endpoint", "")[:60])
        except WebPushException as exc:
            logger.warning("Push delivery failed: %s", exc)
            # 404/410 = subscription no longer valid, clean it up
            if exc.response is not None and exc.response.status_code in (404, 410):
                stale.append(sub.get("endpoint", ""))
        except Exception as exc:
            logger.error("Unexpected push error: %s", exc)

    for endpoint in stale:
        logger.info("Removing stale push subscription for user %s", user_id)
        await remove_subscription(user_id, endpoint)
