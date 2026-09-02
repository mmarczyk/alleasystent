from __future__ import annotations

"""
Returns & complaints monitor — polls Allegro for new customer returns (zwroty)
and disputes/claims (reklamacje) and pushes notifications to all subscribed
devices, mirroring services/order_monitor.py.

Runs as part of the same alleasystent-order-monitor Cloud Run service/Cloud
Scheduler cadence (see jobs/order_monitor_service.py) rather than a separate
deployment — no event-stream API exists for returns/issues like the order
one, so each pass fetches the current list and diffs it against the IDs seen
on the previous pass, instead of following an event cursor.
"""

import logging

logger = logging.getLogger(__name__)

_SEEN_RETURNS_KEY = "allegro:monitor:returns:seen:{user_id}"
_SEEN_ISSUES_KEY = "allegro:monitor:issues:seen:{user_id}"
_SEEN_TTL = 86400 * 30  # 30 days
# Kept in each seen-set purely so the key exists from the very first pass,
# including passes that find nothing. A NUL byte can't occur in an Allegro ID.
_BASELINE_MEMBER = "\x00baselined"
_FETCH_LIMIT = 50
_MONITOR_KIND = "returns_complaints"


async def is_monitor_enabled(user_id: str) -> bool:
    """Whether automatic returns/complaints checking is turned on for this user."""
    from services.monitor_state import is_monitor_enabled as _is_enabled
    return await _is_enabled(_MONITOR_KIND, user_id)


async def set_monitor_enabled(user_id: str, enabled: bool) -> None:
    """Turn automatic returns/complaints checking on/off for this user."""
    from services.monitor_state import set_monitor_enabled as _set_enabled
    await _set_enabled(_MONITOR_KIND, user_id, enabled)


async def run_once() -> None:
    """Entry point invoked alongside services.order_monitor.run_once() — one
    polling pass over every user with returns/complaints checking enabled."""
    from config.settings import get_settings

    redis_url = get_settings().redis_url
    if not redis_url or not redis_url.startswith(('redis://', 'rediss://', 'unix://')):
        logger.info("Returns/complaints monitor skipped: REDIS_URL not set or has invalid scheme")
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
                logger.warning("Returns/complaints monitor: user=%s skipped: %s", user_id, exc)
    finally:
        await r.aclose()


async def _poll_user(r, user_id: str) -> None:
    from services.allegro_service import AllegroService

    if not await r.exists(f"allegro:tokens:{user_id}"):
        return

    allegro = AllegroService.get_instance(user_id)
    await allegro._load_tokens_from_redis()
    if not allegro._tokens:
        return

    new_return_ids = await _poll_returns(r, allegro, user_id)
    new_issue_ids = await _poll_issues(r, allegro, user_id)

    if new_return_ids:
        await _notify(user_id, kind="returns", count=len(new_return_ids))
    if new_issue_ids:
        await _notify(user_id, kind="issues", count=len(new_issue_ids))


async def _poll_returns(r, allegro, user_id: str) -> list[str]:
    from services.allegro_service import AllegroAuthError, AllegroAPIError

    try:
        returns = await allegro.get_customer_returns(limit=_FETCH_LIMIT)
    except (AllegroAuthError, AllegroAPIError) as exc:
        logger.warning("Returns monitor: Allegro API error user=%s: %s", user_id, exc)
        return []

    ids = [item["id"] for item in returns if item.get("id")]
    return await _diff_and_record(r, _SEEN_RETURNS_KEY.format(user_id=user_id), ids)


async def _poll_issues(r, allegro, user_id: str) -> list[str]:
    from services.allegro_service import AllegroAuthError, AllegroAPIError

    try:
        issues = await allegro.get_issues(limit=_FETCH_LIMIT)
    except (AllegroAuthError, AllegroAPIError) as exc:
        logger.warning("Complaints monitor: Allegro API error user=%s: %s", user_id, exc)
        return []

    ids = [item["id"] for item in issues if item.get("id")]
    return await _diff_and_record(r, _SEEN_ISSUES_KEY.format(user_id=user_id), ids)


async def _diff_and_record(r, seen_key: str, current_ids: list[str]) -> list[str]:
    """Compare the current fetch against last pass's seen-ID set and return
    which IDs are new.

    The first pass FOR A USER records a baseline without reporting anything,
    same as order_monitor's event baseline. "First pass" is decided by the key's
    existence, which is why every pass writes the key — including one that finds
    nothing. An earlier version returned early on an empty pass without writing,
    so the key only appeared on the first pass that found something and the
    user's first-ever return or complaint was swallowed as the baseline; see the
    same fix in services/message_monitor.py, where it cost a real notification.

    The seen-set is replaced whenever there is something to record (rather than
    accumulated) so it stays bounded to `_FETCH_LIMIT` entries instead of
    growing forever — the trade-off is that a return/issue which scrolls out of
    the API's recent-N window and later reappears would be reported again, an
    acceptable edge case for a low-volume category like this.
    """
    known = await r.exists(seen_key)
    seen_ids = set(await r.smembers(seen_key)) if known else set()
    new_ids = [cid for cid in current_ids if cid not in seen_ids] if known else []

    pipe = r.pipeline()
    if current_ids:
        pipe.delete(seen_key)
        pipe.sadd(seen_key, _BASELINE_MEMBER, *current_ids)
    else:
        pipe.sadd(seen_key, _BASELINE_MEMBER)
    pipe.expire(seen_key, _SEEN_TTL)
    await pipe.execute()

    return new_ids


async def _notify(user_id: str, kind: str, count: int) -> None:
    from services.push_service import send_push, add_notification

    if kind == "returns":
        title = "Nowy zwrot na Allegro" if count == 1 else f"{count} nowych zwrotów na Allegro"
        body = "Kupujący zgłosił zwrot produktu." if count == 1 else f"{count} zwrotów czeka na obsługę."
        prompt = (
            "Podaj mi szczegóły ostatniego zwrotu."
            if count == 1 else
            f"Podaj mi szczegóły {count} ostatnich zwrotów."
        )
    else:
        title = "Nowa reklamacja na Allegro" if count == 1 else f"{count} nowych reklamacji na Allegro"
        body = "Kupujący zgłosił reklamację lub spór." if count == 1 else f"{count} reklamacji/sporów czeka na obsługę."
        prompt = (
            "Podaj mi szczegóły ostatniej reklamacji."
            if count == 1 else
            f"Podaj mi szczegóły {count} ostatnich reklamacji."
        )

    entry = await add_notification(user_id, title=title, body=body, url="/?open=notifications", prompt=prompt)
    await send_push(
        user_id=user_id, title=title, body=body, url="/?open=notifications", prompt=prompt,
        notif_id=entry["id"] if entry else None,
        created_at=entry["created_at"] if entry else None,
    )
