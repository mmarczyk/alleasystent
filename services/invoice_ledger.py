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
# The same fact from the other side: which ORDER an inFakt invoice belongs to.
# Needed because the delivery steps are addressed by invoice, while everything
# that decides whether they are ALLOWED — above all whether the buyer is a
# company with a NIP — is a question only Allegro can answer, and Allegro is
# asked per order (see AllegroAgent._send_invoice_to_ksef).
_ORDER_KEY = "allegro:invoice_order:{user_id}:{invoice_uuid}"
# The orders whose invoice exists in inFakt but has NOT reached Allegro yet —
# a sorted set scored by issuance time, so the oldest debt comes first.
#
# The per-order keys above answer "does THIS order have an invoice we issued";
# nothing answered "which invoices are still waiting", and that is the set the
# seller means by "dodaj te faktury do Allegro" after a batch issuance. Read
# from a key pattern it would have to be a SCAN over the whole keyspace on a
# shared Redis; kept as an index it is one ZRANGE, and it is written by the
# only two functions that can change the answer.
_PENDING_KEY = "allegro:invoice_pending:{user_id}"
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
    await _store(user_id, order_id, {
        "invoice_uuid": invoice_uuid,
        "number": number,
        "attached": attached,
        "note": note,
        "at": time.time(),
    })
    logger.info(
        "Invoice ledger: user=%s order=%s recorded (attached=%s)", user_id, order_id, attached
    )


async def _store(user_id: str, order_id: str, payload: dict) -> None:
    """Write the record and keep both indexes in step with it."""
    invoice_uuid = payload.get("invoice_uuid") or ""

    async def _do(r):
        await r.set(_KEY.format(user_id=user_id, order_id=order_id), json.dumps(payload), ex=_TTL)
        if invoice_uuid:
            await r.set(
                _ORDER_KEY.format(user_id=user_id, invoice_uuid=invoice_uuid), order_id, ex=_TTL,
            )
        # An issuance that did not reach Allegro joins the waiting list; the
        # later attach (which comes back through here via mark_attached) takes
        # it off again. An issuance with no UUID at all is not put on the list:
        # there is no file to attach, and re-issuing is the seller's call.
        pending = _PENDING_KEY.format(user_id=user_id)
        if payload.get("attached") or not invoice_uuid:
            await r.zrem(pending, order_id)
        else:
            await r.zadd(pending, {order_id: payload.get("at") or time.time()})
            await r.expire(pending, _TTL)

    await _with_redis(_do)


async def _merge(user_id: str, order_id: str, **fields) -> None:
    """Update named fields of an existing record, leaving the rest as they are.

    A record accumulates what has been DONE with one invoice — issued, attached,
    filed with KSeF — and each of those is written by a different step, minutes
    or days apart. Rewriting the whole payload from the one field a step knows
    about is how the previous step's fact goes missing, and a lost "already
    filed" is a second KSeF submission that cannot be withdrawn.
    """
    existing = await get_record(user_id, order_id) or {}
    await _store(user_id, order_id, {
        "invoice_uuid": "", "number": "", "attached": False, "note": "",
        "at": time.time(), **existing, **fields,
    })


async def mark_attached(user_id: str, order_id: str, *, number: str = "") -> None:
    """Upgrade an existing record to "attached" after a later, successful attach."""
    fields: dict = {"attached": True}
    if number:
        fields["number"] = number
    await _merge(user_id, order_id, **fields)


async def mark_ksef_sent(user_id: str, order_id: str) -> None:
    """Write down that this order's invoice has been filed with KSeF.

    KSeF takes an invoice once. The submission is asynchronous, so nothing in
    the reply proves it landed, and a seller who was not told it went (see the
    action-report guard in AllegroAgent.run) will reasonably ask again — which
    is why this is remembered here rather than left to whoever reads the chat.
    """
    await _merge(user_id, order_id, ksef_sent=True, ksef_at=time.time())


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


async def order_of_invoice(user_id: str, invoice_uuid: str) -> str | None:
    """The order an inFakt invoice was issued for, or None if we never wrote it
    down. None is not "no order" — it is "we don't know", and a caller that
    needs the order to check whether something is allowed must treat it as a
    reason to refuse rather than to proceed."""
    if not invoice_uuid:
        return None

    async def _do(r):
        return await r.get(_ORDER_KEY.format(user_id=user_id, invoice_uuid=invoice_uuid))

    return await _with_redis(_do) or None


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


async def pending_delivery(user_id: str, limit: int = 50) -> list[tuple[str, dict]]:
    """(order_id, record) for every invoice we issued that never reached
    Allegro, oldest first.

    This is the list behind "dodaj te faktury do Allegro" — a seller who has
    just been shown four freshly issued invoices names the SET, not four UUIDs,
    and the alternative to reading it from here is scraping order ids out of
    the previous chat message, which is how the wrong invoice ends up on
    someone else's order.

    Records with no invoice_uuid are dropped rather than returned: they are the
    issuances that timed out before inFakt confirmed an id, and there is
    nothing to attach for them.
    """
    async def _do(r):
        return await r.zrange(_PENDING_KEY.format(user_id=user_id), 0, max(limit - 1, 0))

    order_ids = await _with_redis(_do) or []
    records = await get_records(user_id, list(order_ids))
    return [
        (order_id, records[order_id])
        for order_id in order_ids
        if records.get(order_id, {}).get("invoice_uuid")
        and not records[order_id].get("attached")
    ]


async def forget(user_id: str, order_id: str) -> None:
    """Drop the record — for when the invoice turned out not to exist after all
    and the seller really does need to issue one."""
    async def _do(r):
        await r.delete(_KEY.format(user_id=user_id, order_id=order_id))
        await r.zrem(_PENDING_KEY.format(user_id=user_id), order_id)

    await _with_redis(_do)
