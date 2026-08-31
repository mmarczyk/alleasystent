from __future__ import annotations

"""
Order monitor — polls Allegro via event API and pushes notifications to all
subscribed devices when new READY_FOR_PROCESSING orders arrive.

Runs as a one-shot pass (run_once) invoked over HTTP (POST /run) by the
alleasystent-order-monitor Cloud Run service on a Cloud Scheduler cron (see
jobs/order_monitor_service.py) — NOT as a long-lived task inside the web app.
That used to run as an infinite asyncio loop in the FastAPI process, but
Cloud Run only allocates CPU while a request is in flight and can scale
instances to zero, so the loop could go silently idle or die; a scheduled
one-shot pass is deterministic regardless of web traffic and works even when
iOS PWA is backgrounded (JS can't poll then).
"""

import logging

logger = logging.getLogger(__name__)

_STATE_KEY      = "allegro:monitor:last_event:{user_id}"
_STATE_TTL      = 86400 * 30  # 30 days
_ENABLED_KEY    = "allegro:monitor:enabled:{user_id}"
_MAX_DELIVERY_NAMES = 3  # distinct courier names printed in a multi-order push body


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


def _format_price(amount: float, currency: str = "PLN") -> str:
    """Same shape the agent prints prices in (AllegroAgent._format_price)."""
    return f"{amount:.2f}".replace(".", ",") + f" {currency}"


def _format_totals(orders: list[dict]) -> str:
    """Summed order value, per currency — orders in one pass are practically
    always PLN, but a mixed batch must not silently add złote to euro."""
    totals: dict[str, float] = {}
    for o in orders:
        currency = o.get("currency") or "PLN"
        totals[currency] = totals.get(currency, 0.0) + float(o.get("total_price") or 0.0)
    return " + ".join(_format_price(amount, currency) for currency, amount in totals.items())


def _format_delivery_methods(orders: list[dict]) -> str:
    """Distinct delivery methods, in first-seen order, capped so the push body
    stays readable when a batch spans many couriers."""
    names: list[str] = []
    for o in orders:
        name = (o.get("delivery_method") or "").strip()
        if name and name not in names:
            names.append(name)
    if not names:
        return ""
    if len(names) <= _MAX_DELIVERY_NAMES:
        return ", ".join(names)
    shown = names[:_MAX_DELIVERY_NAMES]
    return ", ".join(shown) + f" +{len(names) - len(shown)}"


def build_notification_body(new_orders: list[dict]) -> str:
    """The push/inbox body for a batch of new orders.

    Beyond "you have new orders", the seller immediately wants the three facts
    that decide whether to drop everything and pack now: what the order is
    worth, how it ships, and whether an invoice has to be issued. They ride
    along in the event payload (see AllegroService.order_event_details), so the
    notification can state them without the seller opening anything.

    Details are best-effort: an order whose checkout-form fetch failed carries
    none, and every line is dropped rather than printed empty — worst case the
    body degrades to the plain count sentence it was before.
    """
    count = len(new_orders)
    priced   = [o for o in new_orders if o.get("total_price") is not None]
    invoiced = [o for o in new_orders if "invoice_required" in o]
    delivery = _format_delivery_methods(new_orders)

    if count == 1:
        lines: list[str] = []
        if priced:
            lines.append(f"Wartość: {_format_totals(priced)}")
        if delivery:
            lines.append(f"Dostawa: {delivery}")
        if invoiced:
            lines.append("Faktura: tak" if invoiced[0].get("invoice_required") else "Faktura: nie")
        return "\n".join(lines) if lines else "Zamówienie czeka na realizację."

    lines = [f"{count} zamówień czeka na realizację."]
    if priced:
        lines.append(f"Łączna wartość: {_format_totals(priced)}")
    if delivery:
        lines.append(f"Dostawa: {delivery}")
    if invoiced:
        wanted = sum(1 for o in invoiced if o.get("invoice_required"))
        lines.append(f"Faktura: {wanted} z {count}")
    return "\n".join(lines)


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
    body  = build_notification_body(new_orders)
    prompt = (
        "Podaj mi szczegóły ostatniego nowego zamówienia."
        if count == 1 else
        f"Podaj mi szczegóły {count} ostatnich nowych zamówień."
    )

    # Goes to the in-app Notifications inbox (bell icon), not the chat — the OS-level
    # push below is what actually alerts the user (browser + iOS PWA). `prompt` is the
    # chat question fired automatically when the user taps the notification.
    entry = await add_notification(user_id, title=title, body=body, url="/?open=notifications", prompt=prompt)
    await send_push(
        user_id=user_id, title=title, body=body, url="/?open=notifications", prompt=prompt,
        notif_id=entry["id"] if entry else None,
        created_at=entry["created_at"] if entry else None,
    )
