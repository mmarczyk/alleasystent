from __future__ import annotations

"""
Per-order record of the VAT invoices this assistant has issued in inFakt.

NOT a cache of "does this order have an invoice" — that question is only ever
answered by asking Allegro (services/allegro_service.get_order_invoices), every
time, because an invoice can be attached to an order by anyone at any moment and
the seller is the one looking at it. Nothing here is allowed to stand in for
that answer or to silence the invoice reminder.

What it is for is the other direction: issuing and attaching are two calls
against two different APIs, and the second one fails on its own (a token without
allegro:api:orders:write comes back 403, a PDF can be over Allegro's size
limit). When that happens a real, numbered VAT invoice exists in inFakt while
Allegro still reports the order as uninvoiced — correctly. Without a memory of
the first call, the next "wystaw" would create a SECOND invoice for that order,
which cannot be undone. So an issuance is written here, and the issuing path
attaches the invoice it already has instead of making another one.
"""

import json
import logging
import time

logger = logging.getLogger(__name__)

_KEY = "allegro:invoice_issued:{user_id}:{order_id}"
# Comfortably longer than any invoicing deadline — the point is that an order
# invoiced months ago never comes back around as "not invoiced yet".
_TTL = 86400 * 180


def _valid_redis_url(url: str | None) -> bool:
    return bool(url and url.startswith(("redis://", "rediss://", "unix://")))


async def _with_redis(fn):
    import redis.asyncio as aioredis
    from config.settings import get_settings

    redis_url = get_settings().redis_url
    if not _valid_redis_url(redis_url):
        return None
    r = aioredis.from_url(redis_url, decode_responses=True)
    try:
        return await fn(r)
    finally:
        await r.aclose()


def user_id_of(allegro) -> str:
    """The user an AllegroService instance belongs to — the ledger is per seller,
    the same way tokens and reminder state are."""
    return getattr(allegro, "_user_id", None) or "default"


async def record_issued(
    user_id: str, order_id: str, *, invoice_uuid: str, number: str = "",
    attached: bool = False, note: str = "",
) -> None:
    """Write down that an invoice for this order exists in inFakt.

    Recorded even when the attachment to Allegro failed, and even when inFakt
    accepted the job without confirming it in time: in both cases a real invoice
    very probably exists, and the next "wystaw" must finish that one rather than
    create a second.
    """
    payload = {
        "invoice_uuid": invoice_uuid,
        "number": number,
        "attached": attached,
        "note": note,
        "at": time.time(),
    }

    async def _do(r):
        await r.set(_KEY.format(user_id=user_id, order_id=order_id), json.dumps(payload), ex=_TTL)

    await _with_redis(_do)
    logger.info(
        "Invoice ledger: user=%s order=%s recorded (attached=%s)", user_id, order_id, attached
    )


async def mark_attached(user_id: str, order_id: str, *, number: str = "") -> None:
    """Upgrade an existing record to "attached" after a later, successful attach."""
    existing = await get_record(user_id, order_id) or {}
    await record_issued(
        user_id, order_id,
        invoice_uuid=existing.get("invoice_uuid", ""),
        number=number or existing.get("number", ""),
        attached=True,
    )


async def get_record(user_id: str, order_id: str) -> dict | None:
    async def _do(r):
        return await r.get(_KEY.format(user_id=user_id, order_id=order_id))

    raw = await _with_redis(_do)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


async def get_records(user_id: str, order_ids: list[str]) -> dict[str, dict]:
    """order_id → record, for the orders that have one. One MGET rather than a
    lookup per order: the reminder and the pending-invoice listing both ask
    about a whole batch at once."""
    if not order_ids:
        return {}

    async def _do(r):
        return await r.mget([_KEY.format(user_id=user_id, order_id=oid) for oid in order_ids])

    raws = await _with_redis(_do)
    if not raws:
        return {}
    out: dict[str, dict] = {}
    for order_id, raw in zip(order_ids, raws):
        if not raw:
            continue
        try:
            out[order_id] = json.loads(raw)
        except (TypeError, ValueError):
            continue
    return out


async def forget(user_id: str, order_id: str) -> None:
    """Drop the record — for when the invoice turned out not to exist after all
    and the seller really does need to issue one."""
    async def _do(r):
        await r.delete(_KEY.format(user_id=user_id, order_id=order_id))

    await _with_redis(_do)
