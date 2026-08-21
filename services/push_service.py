from __future__ import annotations

"""Web Push notification service (VAPID) and in-app Notifications inbox.

Subscriptions are stored in Redis: push:sub:{user_id}:{md5(endpoint)}
Each subscription is the full JSON object from the browser's PushSubscription.toJSON().

In-app notifications (bell icon panel) are stored in Redis: notif:list:{user_id}
— a capped list of {id, title, body, url, created_at, read} entries, newest first.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_VALID_SCHEMES = ('redis://', 'rediss://', 'unix://')
_NOTIF_TTL = 60 * 60 * 24 * 30  # 30 days
_NOTIF_MAX = 50  # keep at most this many entries per user


def _valid_redis_url(url: str | None) -> bool:
    return bool(url and url.startswith(_VALID_SCHEMES))


async def save_subscription(user_id: str, subscription: dict) -> None:
    """Persist a push subscription for a user (upsert by endpoint)."""
    from config.settings import get_settings
    settings = get_settings()
    if not _valid_redis_url(settings.redis_url):
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


# A queued chat message is the assistant speaking to the seller on its own
# initiative, so it has to survive until the seller actually opens the app —
# the old 30-minute TTL silently dropped every message written outside that
# window, which is most of them for a reminder that only fires every 2 hours.
_PENDING_CHAT_TTL = 60 * 60 * 24  # 24h


async def store_pending_chat(
    user_id: str,
    text: str,
    ttl: int = _PENDING_CHAT_TTL,
    dedupe_tag: str | None = None,
) -> None:
    """Queue a chat message to be written into the seller's chat when they next
    look at the app.

    Messages accumulate as a FIFO list. `dedupe_tag` drops the queued messages
    carrying that same tag first: a recurring reminder states the CURRENT
    situation rather than logging an event, so an unread one must be replaced,
    not stacked — otherwise a seller away for a day comes back to a dozen
    identical "wystawić faktury?" messages in a row.
    """
    from config.settings import get_settings
    settings = get_settings()
    if not _valid_redis_url(settings.redis_url):
        return
    import redis.asyncio as aioredis
    key = f"push:chat:{user_id}"
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        if dedupe_tag:
            for raw in await r.lrange(key, 0, -1):
                if _decode_pending_chat(raw)[0] == dedupe_tag:
                    await r.lrem(key, 0, raw)
        await r.rpush(key, json.dumps({"tag": dedupe_tag, "text": text}))
        await r.expire(key, ttl)
    finally:
        await r.aclose()


def _decode_pending_chat(raw: str) -> tuple[str | None, str]:
    """(tag, text) for a queued entry. Entries written before tagging existed
    are bare strings, so anything that isn't the tagged JSON shape is the text."""
    try:
        entry = json.loads(raw)
    except (TypeError, ValueError):
        return None, raw
    if isinstance(entry, dict) and "text" in entry:
        return entry.get("tag"), entry.get("text") or ""
    return None, raw


async def pop_pending_chats(user_id: str) -> list[str]:
    """Drain and return every queued chat message, oldest first.

    Drained in one MULTI/EXEC so two devices (or a poll racing a page load)
    can't each take half of the queue and show the seller a partial thread.
    """
    from config.settings import get_settings
    settings = get_settings()
    if not _valid_redis_url(settings.redis_url):
        return []
    import redis.asyncio as aioredis
    key = f"push:chat:{user_id}"
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        async with r.pipeline(transaction=True) as pipe:
            pipe.lrange(key, 0, -1)
            pipe.delete(key)
            entries, _ = await pipe.execute()
        return [text for text in (_decode_pending_chat(e)[1] for e in entries or []) if text]
    finally:
        await r.aclose()


async def add_notification(user_id: str, title: str, body: str, url: str = "/", prompt: str | None = None) -> dict | None:
    """Append an entry to the user's in-app Notifications list (bell icon panel).

    Stored as a capped Redis list (newest first) so the frontend can render a
    persistent inbox instead of the automatic monitors writing into the chat.

    `prompt`, if set, is a ready-made chat question the frontend fires
    automatically when the user taps the notification (e.g. "Podaj mi
    szczegóły 2 ostatnich zamówień."), instead of just navigating somewhere.
    """
    from config.settings import get_settings
    settings = get_settings()
    if not _valid_redis_url(settings.redis_url):
        return None
    import redis.asyncio as aioredis
    entry = {
        "id": uuid.uuid4().hex,
        "title": title,
        "body": body,
        "url": url,
        "prompt": prompt,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "read": False,
    }
    key = f"notif:list:{user_id}"
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        await r.lpush(key, json.dumps(entry))
        await r.ltrim(key, 0, _NOTIF_MAX - 1)
        await r.expire(key, _NOTIF_TTL)
    finally:
        await r.aclose()
    return entry


async def list_notifications(user_id: str) -> list[dict]:
    """Return the user's notifications, newest first."""
    from config.settings import get_settings
    settings = get_settings()
    if not _valid_redis_url(settings.redis_url):
        return []
    import redis.asyncio as aioredis
    key = f"notif:list:{user_id}"
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        raw = await r.lrange(key, 0, -1)
        return [json.loads(v) for v in raw]
    finally:
        await r.aclose()


async def mark_notifications_read(user_id: str) -> None:
    """Mark every stored notification for this user as read."""
    from config.settings import get_settings
    settings = get_settings()
    if not _valid_redis_url(settings.redis_url):
        return
    import redis.asyncio as aioredis
    key = f"notif:list:{user_id}"
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        raw = await r.lrange(key, 0, -1)
        pipe = r.pipeline()
        dirty = False
        for idx, v in enumerate(raw):
            entry = json.loads(v)
            if not entry.get("read"):
                entry["read"] = True
                pipe.lset(key, idx, json.dumps(entry))
                dirty = True
        if dirty:
            await pipe.execute()
    finally:
        await r.aclose()


async def remove_subscription(user_id: str, endpoint: str) -> None:
    """Remove a specific push subscription by endpoint URL."""
    from config.settings import get_settings
    settings = get_settings()
    if not _valid_redis_url(settings.redis_url):
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
    if not _valid_redis_url(settings.redis_url):
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


async def send_push(
    user_id: str,
    title: str,
    body: str,
    url: str = "/",
    prompt: str | None = None,
    notif_id: str | None = None,
    created_at: str | None = None,
) -> None:
    """Send a Web Push notification to all subscribed devices for a user.

    `prompt`, if set, travels in the push payload so the service worker can
    attach it to the notification's click target — tapping the OS notification
    then fires that chat question automatically instead of just opening the app.

    `notif_id`/`created_at`, if set, should be the `id`/`created_at` of the
    matching entry already written to the Notifications inbox (see
    `add_notification`). The service worker forwards them to the app on launch
    so it can paint the notification into the inbox instantly — before its
    background refresh() has a chance to round-trip to the server.
    """
    from config.settings import get_settings
    settings = get_settings()
    if not settings.vapid_private_key or not settings.vapid_public_key:
        logger.warning("VAPID keys not configured (set VAPID_PRIVATE_KEY + VAPID_PUBLIC_KEY in Railway) — skipping push")
        return

    subscriptions = await _get_subscriptions(user_id)
    if not subscriptions:
        logger.info("No push subscriptions for user %s — nothing to send", user_id)
        return

    try:
        from pywebpush import webpush, WebPushException
        from py_vapid import Vapid01
    except ImportError:
        logger.error("pywebpush not installed — cannot send push notifications")
        return

    try:
        # Pass a pre-built Vapid01 object, not the raw PEM string: webpush()'s
        # string path (Vapid.from_string) doesn't strip the "-----BEGIN/END-----"
        # header lines before base64-decoding, so it always fails on a real PEM
        # key with "ASN.1 parsing error: invalid length" — Vapid01.from_pem()
        # (used here) handles the full PEM correctly.
        vapid_key = Vapid01.from_pem(settings.vapid_private_key.encode())
    except Exception as exc:
        logger.error("Invalid VAPID private key, cannot send push: %s", exc)
        return

    payload = json.dumps({
        "title": title, "body": body, "url": url, "prompt": prompt,
        "id": notif_id, "created_at": created_at,
    }).encode()
    loop = asyncio.get_event_loop()
    stale: list[str] = []

    for sub in subscriptions:
        try:
            await loop.run_in_executor(
                None,
                lambda s=sub: webpush(
                    subscription_info=s,
                    data=payload,
                    vapid_private_key=vapid_key,
                    vapid_claims={"sub": settings.vapid_email},
                    # "high" tells the push service (incl. Apple's, for iOS PWA push)
                    # to prioritize immediate delivery instead of batching/delaying it
                    # for battery savings — iOS otherwise can sit on "normal"-urgency
                    # pushes for tens of minutes before surfacing them.
                    headers={"Urgency": "high"},
                    ttl=1800,  # keep retrying delivery for 30 min if the device is briefly unreachable
                ),
            )
            logger.debug("Push sent → %s", sub.get("endpoint", "")[:60])
        except WebPushException as exc:
            logger.warning("Push delivery failed: %s", exc)
            is_stale = False
            if exc.response is not None:
                # 404/410 = subscription no longer valid.
                if exc.response.status_code in (404, 410):
                    is_stale = True
                # VapidPkHashMismatch = subscription was created under a VAPID public
                # key that no longer matches ours (e.g. after a key rotation, or a
                # local browser subscription orphaned by clearing site data before it
                # could be unsubscribed) — permanently unrecoverable, same as 410.
                elif exc.response.status_code == 400:
                    try:
                        is_stale = exc.response.json().get("reason") == "VapidPkHashMismatch"
                    except Exception:
                        pass
            if is_stale:
                stale.append(sub.get("endpoint", ""))
        except Exception as exc:
            logger.error("Unexpected push error: %s", exc)

    for endpoint in stale:
        logger.info("Removing stale push subscription for user %s", user_id)
        await remove_subscription(user_id, endpoint)
