from __future__ import annotations

"""
Per-order record of the VAT invoices this assistant has issued.

Allegro is the source of truth for "does this order have an invoice?" — but
only for invoices whose PDF actually reached the order
(GET /order/checkout-forms/{id}/invoices, see
services/allegro_service.get_order_invoices). An invoice that exists in inFakt
but never got attached is invisible there, and the invoice reminder
(services/invoice_reminder.py) reads that invisibility as "still to issue" and
nags about it every two hours, for ever.

That is not hypothetical: issuing and attaching are two separate steps against
two separate APIs, and the second one can fail on its own (Allegro 403 for a
token without allegro:api:orders:write, a PDF over Allegro's size limit, inFakt
not returning the file). Without a memory of the first step the seller is told
to issue an invoice that already exists — and issuing it again would create a
second, real, numbered VAT invoice for the same order, which cannot be undone.

So every issuance is written down here, attached or not, and the reminder
skips orders it finds in this ledger. Attachment failures are surfaced when
they happen (and by the pending-invoice listing) instead of being retold as
"you have an invoice to issue".
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


# Who says this order has an invoice: this assistant issued one, or the seller
# told us in the chat that one already exists. Both stop the reminder; they are
# worded differently everywhere the difference matters, because only the first
# one is something we can point at.
SOURCE_ASSISTANT = "assistant"
SOURCE_SELLER = "seller"


async def record_issued(
    user_id: str, order_id: str, *, invoice_uuid: str, number: str = "",
    attached: bool = False, note: str = "", source: str = SOURCE_ASSISTANT,
) -> None:
    """Write down that an invoice for this order exists in inFakt.

    Called even when the attachment to Allegro failed, and even when inFakt
    accepted the job without confirming it in time: in both cases an invoice
    very probably exists, and a reminder that keeps saying "not issued" is what
    pushes a seller into issuing it twice.
    """
    payload = {
        "invoice_uuid": invoice_uuid,
        "number": number,
        "attached": attached,
        "note": note,
        "source": source,
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


async def record_confirmed_by_seller(user_id: str, order_id: str) -> None:
    """Write down that the SELLER says this order already has its invoice.

    Their word, not Allegro's — but a seller looking at the order knows better
    than an API that only sees attached PDFs, and being told to issue an invoice
    they can see is what makes the reminder useless. The claim is recorded as
    theirs (source=seller) so nothing later presents it as an invoice we issued.
    """
    await record_issued(
        user_id, order_id, invoice_uuid="", attached=False, source=SOURCE_SELLER,
        note="sprzedawca potwierdził w czacie, że faktura już istnieje",
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
