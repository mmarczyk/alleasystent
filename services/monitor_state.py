from __future__ import annotations

"""Per-user enabled/disabled flags for the client-side invoice and message
monitors (web/js/app.js InvoiceMonitor / MessageMonitor) — those poll from
the browser tab, not a Cloud Run Job like services/order_monitor.py, but the
chat agent still needs a source of truth to answer "is monitoring on?"
honestly instead of guessing (see AllegroAgent._invoice_monitoring_status_block
/ _message_monitoring_status_block, mirroring order_monitor.is_monitor_enabled).
"""

import logging

logger = logging.getLogger(__name__)

_ENABLED_KEY = "allegro:{kind}_monitor:enabled:{user_id}"


async def is_monitor_enabled(kind: str, user_id: str) -> bool:
    """Whether automatic checking is turned on for this user for the given kind
    ('invoice' or 'message')."""
    from config.settings import get_settings
    import redis.asyncio as aioredis

    redis_url = get_settings().redis_url
    if not redis_url or not redis_url.startswith(('redis://', 'rediss://', 'unix://')):
        return False
    r = aioredis.from_url(redis_url, decode_responses=True)
    try:
        return bool(await r.exists(_ENABLED_KEY.format(kind=kind, user_id=user_id)))
    finally:
        await r.aclose()


async def set_monitor_enabled(kind: str, user_id: str, enabled: bool) -> None:
    """Turn automatic checking on/off for this user for the given kind."""
    from config.settings import get_settings
    import redis.asyncio as aioredis

    redis_url = get_settings().redis_url
    if not redis_url or not redis_url.startswith(('redis://', 'rediss://', 'unix://')):
        return
    r = aioredis.from_url(redis_url, decode_responses=True)
    key = _ENABLED_KEY.format(kind=kind, user_id=user_id)
    try:
        if enabled:
            await r.set(key, "1")
        else:
            await r.delete(key)
    finally:
        await r.aclose()
