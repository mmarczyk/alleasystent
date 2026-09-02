from __future__ import annotations

"""
Message monitor — polls Allegro for unread buyer messages and pushes
notifications to all subscribed devices, mirroring
services/return_complaint_monitor.py.

Runs as part of the same alleasystent-order-monitor Cloud Run service/Cloud
Scheduler cadence (see jobs/order_monitor_service.py). Detection used to live
ONLY in the browser tab (web/js/app.js MessageMonitor, a setInterval poll that
called WebPush.sendNotification itself), which meant a message arriving while
the PWA was closed or backgrounded — overnight, typically — produced no
notification at all: iOS suspends a backgrounded PWA's JS, so nothing polled
and nothing sent the push. A server-side pass notifies regardless of whether
any tab is open, exactly like the order monitor.

There is no "since X" cursor in Allegro's messaging API, so each pass fetches
the current threads and diffs them against the markers seen on the previous
pass, same approach as the returns/complaints monitor.
"""

import logging

logger = logging.getLogger(__name__)

_SEEN_KEY = "allegro:monitor:messages:seen:{user_id}"
# Kept in the seen-set alongside the real markers purely so the key exists from
# the very first pass, including passes that find nothing unread. A NUL byte
# can't occur in a real "<thread id>@<timestamp>" marker, so it never masks one.
_BASELINE_MEMBER = "\x00baselined"
_SEEN_TTL = 86400 * 30  # 30 days
_FETCH_LIMIT = 20  # Allegro's /messaging/threads caps `limit` at 20
_MONITOR_KIND = "message"


async def is_monitor_enabled(user_id: str) -> bool:
    """Whether automatic message checking is turned on for this user."""
    from services.monitor_state import is_monitor_enabled as _is_enabled
    return await _is_enabled(_MONITOR_KIND, user_id)


async def set_monitor_enabled(user_id: str, enabled: bool) -> None:
    """Turn automatic message checking on/off for this user."""
    from services.monitor_state import set_monitor_enabled as _set_enabled
    await _set_enabled(_MONITOR_KIND, user_id, enabled)


async def run_once() -> None:
    """Entry point invoked alongside services.order_monitor.run_once() — one
    polling pass over every user with automatic message checking enabled."""
    from config.settings import get_settings

    redis_url = get_settings().redis_url
    if not redis_url or not redis_url.startswith(('redis://', 'rediss://', 'unix://')):
        logger.info("Message monitor skipped: REDIS_URL not set or has invalid scheme")
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
        enabled_keys = await r.keys(f"allegro:{_MONITOR_KIND}_monitor:enabled:*")
        user_ids = {k.split(":")[3] for k in enabled_keys if k.count(":") >= 3}

        for user_id in user_ids:
            try:
                await _poll_user(r, user_id)
            except Exception as exc:
                logger.warning("Message monitor: user=%s skipped: %s", user_id, exc)
    finally:
        await r.aclose()


async def _poll_user(r, user_id: str) -> None:
    from services.allegro_service import (
        AllegroService, AllegroAuthError, AllegroAPIError, is_thread_unread,
    )

    if not await r.exists(f"allegro:tokens:{user_id}"):
        return

    allegro = AllegroService.get_instance(user_id)
    await allegro._load_tokens_from_redis()
    if not allegro._tokens:
        return

    try:
        threads = await allegro.get_message_threads(limit=_FETCH_LIMIT)
    except (AllegroAuthError, AllegroAPIError) as exc:
        logger.warning("Message monitor: Allegro API error user=%s: %s", user_id, exc)
        return

    markers = [_marker(t) for t in threads if is_thread_unread(t) and t.get("id")]
    new_markers = await _diff_and_record(r, _SEEN_KEY.format(user_id=user_id), markers)
    if not new_markers:
        return

    logger.info("Message monitor: %d new message(s) for user=%s", len(new_markers), user_id)
    await _notify(user_id, count=len(new_markers))


def _marker(thread: dict) -> str:
    """Identity of "this thread, as of its latest message".

    Keyed by thread id AND last-message timestamp rather than the thread id
    alone (which is all the browser-side monitor tracked) so a follow-up
    message in a thread already reported still counts as new — otherwise a
    buyer writing twice about the same order would be announced only once.

    Both fields come from services.allegro_service rather than being spelled
    out here; see is_thread_unread for why getting these names wrong fails
    silently instead of raising.
    """
    from services.allegro_service import thread_last_message_at
    return f"{thread.get('id')}@{thread_last_message_at(thread)}"


async def _diff_and_record(r, seen_key: str, current_markers: list[str]) -> list[str]:
    """Compare this pass's markers against last pass's and return which are new.

    The first pass FOR A USER records a baseline without reporting anything, so
    switching someone over to server-side detection doesn't announce every
    already-unread thread. "First pass" is decided by the key's existence, which
    is why every pass writes the key — including one that finds nothing unread.

    An earlier version returned early on an empty pass without writing anything.
    The key then only appeared on the first pass that found something, so the
    first unread message a user ever received was swallowed as the baseline: the
    pass fetched it, saw it, and recorded it as already known. The empty pass
    also never refreshed the TTL, so the same silent-swallow returned every time
    a user went 30 days without an unread thread.

    The seen-set is replaced whenever there is something to record (rather than
    accumulated) so it stays bounded to `_FETCH_LIMIT` entries; a thread that
    scrolls out of the fetched window and later comes back with the same last
    message would be reported again, an acceptable edge case at this volume.
    An empty pass keeps the markers already recorded, so a later message in one
    of those same threads still reads as new.
    """
    known = await r.exists(seen_key)
    seen = set(await r.smembers(seen_key)) if known else set()
    new_markers = [m for m in current_markers if m not in seen] if known else []

    pipe = r.pipeline()
    if current_markers:
        pipe.delete(seen_key)
        pipe.sadd(seen_key, _BASELINE_MEMBER, *current_markers)
    else:
        pipe.sadd(seen_key, _BASELINE_MEMBER)
    pipe.expire(seen_key, _SEEN_TTL)
    await pipe.execute()

    return new_markers


async def _notify(user_id: str, count: int) -> None:
    from services.push_service import send_push, add_notification

    title = "Nowa wiadomość na Allegro" if count == 1 else f"{count} nowych wiadomości na Allegro"
    body = (
        "Kupujący napisał nową wiadomość."
        if count == 1 else
        f"{count} nieprzeczytanych wiadomości od kupujących."
    )
    prompt = (
        "Pokaż mi tę nową wiadomość od kupującego."
        if count == 1 else
        f"Pokaż mi te {count} nowe wiadomości od kupujących."
    )

    entry = await add_notification(user_id, title=title, body=body, url="/?open=notifications", prompt=prompt)
    await send_push(
        user_id=user_id, title=title, body=body, url="/?open=notifications", prompt=prompt,
        notif_id=entry["id"] if entry else None,
        created_at=entry["created_at"] if entry else None,
    )
