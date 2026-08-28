from __future__ import annotations

"""Per-user enabled/disabled flags for on/off monitor toggles that don't
warrant their own Redis module: the client-side invoice monitor (web/js/app.js
InvoiceMonitor, kind "invoice"), which polls from the browser tab rather than
a Cloud Run Job like services/order_monitor.py; the message monitor (kind
"message"), whose detection runs server-side in services/message_monitor.py
but whose flag has always lived here, so the flag users already toggled on
keeps working unchanged; and the server-side invoice REMINDER (kind
"invoice_reminder", see services/invoice_reminder.py), whose own detection
+ state live elsewhere but which reuses this same enabled-flag store. The
chat agent still needs a source of truth to answer "is this on?" honestly
instead of guessing (see AllegroAgent._invoice_monitoring_status_block /
_message_monitoring_status_block / _invoice_reminder_status_block,
mirroring order_monitor.is_monitor_enabled).
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
