from __future__ import annotations

"""
Background order monitor — polls Allegro via event API and pushes notifications
to all subscribed devices when new READY_FOR_PROCESSING orders arrive.

This runs as an asyncio background task started in the FastAPI lifespan so that
iOS PWA users receive push notifications even when the app is backgrounded
(JavaScript is paused by iOS and cannot poll on its own).
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 90   # seconds between checks per user
_STARTUP_DELAY  = 25  # wait for app to fully start before first poll
_STATE_KEY      = "allegro:monitor:last_event:{user_id}"
_STATE_TTL      = 86400 * 30  # 30 days


async def run_order_monitor() -> None:
    """Entry point — call as asyncio.create_task() from FastAPI lifespan."""
    from config.settings import get_settings

    redis_url = get_settings().redis_url
    if not redis_url or not redis_url.startswith(('redis://', 'rediss://', 'unix://')):
        logger.info("Order monitor disabled: REDIS_URL not set or has invalid scheme")
        return

    await asyncio.sleep(_STARTUP_DELAY)
    logger.info("Order monitor started (poll every %ds)", _POLL_INTERVAL)

    while True:
        try:
            await _poll_all_users()
        except Exception:
            logger.exception("Order monitor cycle error")
        await asyncio.sleep(_POLL_INTERVAL)


async def _poll_all_users() -> None:
    import redis.asyncio as aioredis
    from config.settings import get_settings

    redis_url = get_settings().redis_url
    if not redis_url or not redis_url.startswith(('redis://', 'rediss://', 'unix://')):
        return
    r = aioredis.from_url(redis_url, decode_responses=True)
    try:
        # Collect user IDs that have at least one push subscription
        sub_keys = await r.keys("push:sub:*")
        user_ids = {k.split(":")[2] for k in sub_keys if k.count(":") >= 3}

        for user_id in user_ids:
            try:
                await _poll_user(r, user_id)
            except Exception as exc:
                logger.warning("Order monitor: user=%s skipped: %s", user_id, exc)
    finally:
        await r.aclose()


async def _poll_user(r, user_id: str) -> None:
    from services.allegro_service import AllegroService, AllegroAuthError, AllegroAPIError
    from services.push_service import send_push, store_pending_chat

    # Skip if user has no Allegro auth tokens
    if not await r.exists(f"allegro:tokens:{user_id}"):
        return

    allegro = AllegroService.get_instance(user_id)
    await allegro._load_tokens_from_redis()
    if not allegro._tokens:
        return

    state_key = _STATE_KEY.format(user_id=user_id)
    last_event_id = await r.get(state_key)

    try:
        if last_event_id is None:
            # First run: record current position as baseline; don't notify
            stats = await allegro.get_order_event_stats()
            baseline = stats.get("latest_event_id")
            if baseline:
                await r.set(state_key, baseline, ex=_STATE_TTL)
                logger.debug("Order monitor: baseline event_id=%s user=%s", baseline, user_id)
            return

        result = await allegro.get_order_events(since_event_id=last_event_id)

    except (AllegroAuthError, AllegroAPIError) as exc:
        logger.warning("Order monitor: Allegro API error user=%s: %s", user_id, exc)
        return
    finally:
        await allegro.close()

    # Persist the new last_event_id regardless of whether there are new orders
    new_last = result.get("last_event_id")
    if new_last and new_last != last_event_id:
        await r.set(state_key, new_last, ex=_STATE_TTL)

    new_orders = result.get("new_orders") or []
    count = len(new_orders)
    if not count:
        return

    logger.info("Order monitor: %d new order(s) for user=%s", count, user_id)

    title = "Nowe zamówienie na Allegro" if count == 1 else f"{count} nowych zamówień na Allegro"
    body  = "Zamówienie czeka na realizację." if count == 1 else f"{count} zamówień czeka na realizację."

    # store_pending_chat ensures the message appears in chat when iOS app reopens
    chat_text = f"🛒 {title}\n{body}"
    await store_pending_chat(user_id, chat_text)
    await send_push(user_id=user_id, title=title, body=body, url="/")
