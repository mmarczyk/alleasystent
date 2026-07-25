from __future__ import annotations

"""
Order monitor — polls Allegro via event API and pushes notifications to all
subscribed devices when new READY_FOR_PROCESSING orders arrive.

Runs as a one-shot pass (run_once) invoked by the alleasystent-order-monitor
Cloud Run Job on a Cloud Scheduler cron (see jobs/order_monitor_job.py) — NOT
as a long-lived task inside the web app. That used to run as an infinite
asyncio loop in the FastAPI process, but Cloud Run only allocates CPU while a
request is in flight and can scale instances to zero, so the loop could go
silently idle or die; a scheduled one-shot job is deterministic regardless of
web traffic and works even when iOS PWA is backgrounded (JS can't poll then).
"""

import logging

logger = logging.getLogger(__name__)

_STATE_KEY      = "allegro:monitor:last_event:{user_id}"
_STATE_TTL      = 86400 * 30  # 30 days
_ENABLED_KEY    = "allegro:monitor:enabled:{user_id}"


async def is_monitor_enabled(user_id: str) -> bool:
    """Whether automatic order checking is turned on for this user."""
    from config.settings import get_settings
    import redis.asyncio as aioredis

    redis_url = get_settings().redis_url
    if not redis_url or not redis_url.startswith(('redis://', 'rediss://', 'unix://')):
        return False
    r = aioredis.from_url(redis_url, decode_responses=True)
    try:
        return bool(await r.exists(_ENABLED_KEY.format(user_id=user_id)))
    finally:
        await r.aclose()


async def set_monitor_enabled(user_id: str, enabled: bool) -> None:
    """Turn automatic order checking on/off for this user."""
    from config.settings import get_settings
    import redis.asyncio as aioredis

    redis_url = get_settings().redis_url
    if not redis_url or not redis_url.startswith(('redis://', 'rediss://', 'unix://')):
        return
    r = aioredis.from_url(redis_url, decode_responses=True)
    key = _ENABLED_KEY.format(user_id=user_id)
    try:
        if enabled:
            await r.set(key, "1")
        else:
            await r.delete(key)
    finally:
        await r.aclose()


async def run_once() -> None:
    """Entry point for the Cloud Run Job — one polling pass over every
    user with automatic order checking enabled, then returns."""
    from config.settings import get_settings

    redis_url = get_settings().redis_url
    if not redis_url or not redis_url.startswith(('redis://', 'rediss://', 'unix://')):
        logger.info("Order monitor skipped: REDIS_URL not set or has invalid scheme")
        return

    await _poll_all_users()


async def _poll_all_users() -> None:
    import redis.asyncio as aioredis
    from config.settings import get_settings

    redis_url = get_settings().redis_url
    if not redis_url or not redis_url.startswith(('redis://', 'rediss://', 'unix://')):
        return
    r = aioredis.from_url(redis_url, decode_responses=True)
    try:
        # Collect user IDs that have explicitly turned on automatic order checking
        enabled_keys = await r.keys("allegro:monitor:enabled:*")
        user_ids = {k.split(":")[3] for k in enabled_keys if k.count(":") >= 3}

        for user_id in user_ids:
            try:
                await _poll_user(r, user_id)
            except Exception as exc:
                logger.warning("Order monitor: user=%s skipped: %s", user_id, exc)
    finally:
        await r.aclose()


async def _poll_user(r, user_id: str) -> None:
    from services.allegro_service import AllegroService, AllegroAuthError, AllegroAPIError
    from services.push_service import send_push, add_notification

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

    # Goes to the in-app Notifications inbox (bell icon), not the chat — the OS-level
    # push below is what actually alerts the user (browser + iOS PWA).
    await add_notification(user_id, title=title, body=body, url="/?open=notifications")
    await send_push(user_id=user_id, title=title, body=body, url="/?open=notifications")
