from __future__ import annotations

"""
Per-order record of invoices already issued through inFakt — the duplicate guard.

Issuing a VAT invoice is irreversible, so "has this order already been
invoiced?" has to be answerable BEFORE the POST to inFakt. Until this module
existed the only answer available was Allegro's own per-order invoice list
(AllegroService.get_order_invoices), and that list only fills up once someone
runs the SEPARATE attach step (AllegroAgent._attach_invoice_to_allegro_order,
which uploads the PDF to the order). An invoice issued in inFakt and not
attached to Allegro was therefore invisible to every later check:

  * services/invoice_reminder.py polls get_orders_needing_invoice(), sees the
    order still without an Allegro invoice, and asks "wystawić?" again two
    hours later — a second "tak" issued a SECOND real invoice for that order;
  * the same happened in chat, where "wystaw fakturę dla zamówienia X" twice
    in a day produced two invoices.

So issuance writes here instead: a claim is taken atomically (Redis SET NX)
BEFORE inFakt is called and only ever released when inFakt is known NOT to
have created anything (validation error, HTTP error before creation). A
claim that we cannot rule out — the poll timing out, a status payload without
an invoice_uuid — deliberately STAYS, because a stale claim costs one manual
check in the inFakt panel while a wrong release costs a duplicate VAT invoice.

Redis is the store (shared by the chat process and the Cloud Run monitor job,
which are separate processes — an in-process guard would not see the other
one). Without REDIS_URL it degrades to a process-local dict: still enough to
stop a double-tap inside one process, and the only option left.
"""

import json
import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Timestamps are stamped in the seller's timezone, like every other time the
# assistant shows them (services/invoice_reminder.py works in Warsaw time too).
_TZ = ZoneInfo("Europe/Warsaw")

_KEY = "allegro:invoice_issued:{user_id}:{order_id}"
# Long enough that a duplicate can no longer plausibly be issued for the order:
# the reminder and get_orders_needing_invoice only ever look at one month at a
# time, so a year is far past any window in which the order can come back.
_TTL = 86400 * 365

STATUS_CLAIMED = "claimed"
STATUS_ISSUED = "issued"

# Fallback store used when REDIS_URL is not configured. Keyed exactly like the
# Redis keys. asyncio is single-threaded, so the read-modify-write in _claim
# below is atomic as long as it contains no await — it doesn't.
_MEMORY: dict[str, dict[str, Any]] = {}


class Claim:
    """Result of trying to reserve an order for issuance.

    `acquired` is True only for the caller that may actually call inFakt.
    Everyone else gets acquired=False plus `record`, the entry that blocked
    them (None only if it vanished between the SET and the GET).
    """

    __slots__ = ("acquired", "record")

    def __init__(self, acquired: bool, record: dict[str, Any] | None = None):
        self.acquired = acquired
        self.record = record


def _valid_redis_url(url: str | None) -> bool:
    return bool(url and url.startswith(("redis://", "rediss://", "unix://")))


def _redis_url() -> str | None:
    from config.settings import get_settings

    url = get_settings().redis_url
    return url if _valid_redis_url(url) else None


def _connect(url: str):
    import redis.asyncio as aioredis

    return aioredis.from_url(url, decode_responses=True)


def _key(user_id: str, order_id: str) -> str:
    return _KEY.format(user_id=user_id or "default", order_id=order_id)


async def claim(user_id: str, order_id: str) -> Claim:
    """Reserve `order_id` for issuance, atomically.

    Returns an acquired claim exactly once per order: every later caller —
    another process, another chat turn, the reminder firing again — is told
    what already happened instead of being allowed to issue a second invoice.
    """
    key = _key(user_id, order_id)
    record = {"status": STATUS_CLAIMED, "claimed_at": _now_iso(), "order_id": order_id}

    url = _redis_url()
    if not url:
        logger.warning(
            "invoice_ledger: REDIS_URL not configured — duplicate protection for order %s "
            "is limited to this process",
            order_id,
        )
        existing = _MEMORY.get(key)
        if existing is not None:
            return Claim(False, existing)
        _MEMORY[key] = record
        return Claim(True)

    r = _connect(url)
    try:
        acquired = await r.set(key, json.dumps(record), nx=True, ex=_TTL)
        if acquired:
            return Claim(True)
        raw = await r.get(key)
    finally:
        await r.aclose()

    return Claim(False, _decode(raw))


async def mark_issued(
    user_id: str, order_id: str, invoice_uuid: str, number: str | None = None
) -> None:
    """Upgrade the claim to a confirmed invoice. Never releases anything."""
    record = {
        "status": STATUS_ISSUED,
        "order_id": order_id,
        "invoice_uuid": invoice_uuid,
        "issued_at": _now_iso(),
    }
    if number:
        record["number"] = number

    key = _key(user_id, order_id)
    url = _redis_url()
    if not url:
        _MEMORY[key] = record
        return

    r = _connect(url)
    try:
        await r.set(key, json.dumps(record), ex=_TTL)
    finally:
        await r.aclose()


async def release(user_id: str, order_id: str) -> None:
    """Drop the claim — ONLY when inFakt is known not to have created anything.

    A released order becomes issuable again, so this is called for outright
    rejections (validation errors, HTTP failures before the invoice exists)
    and never for an outcome that merely couldn't be confirmed.
    """
    key = _key(user_id, order_id)
    url = _redis_url()
    if not url:
        _MEMORY.pop(key, None)
        return

    r = _connect(url)
    try:
        await r.delete(key)
    finally:
        await r.aclose()


async def get(user_id: str, order_id: str) -> dict[str, Any] | None:
    """The ledger entry for one order, or None if it was never issued here."""
    key = _key(user_id, order_id)
    url = _redis_url()
    if not url:
        return _MEMORY.get(key)

    r = _connect(url)
    try:
        return _decode(await r.get(key))
    finally:
        await r.aclose()


async def already_handled(user_id: str, order_ids: list[str]) -> set[str]:
    """Which of these orders this system has already issued (or is issuing).

    Used by the reminder so it stops nagging about an order whose invoice
    exists in inFakt but was never attached to Allegro — the exact gap that
    let the same order be invoiced twice.
    """
    if not order_ids:
        return set()

    url = _redis_url()
    if not url:
        return {oid for oid in order_ids if _key(user_id, oid) in _MEMORY}

    r = _connect(url)
    try:
        values = await r.mget([_key(user_id, oid) for oid in order_ids])
    finally:
        await r.aclose()

    return {oid for oid, raw in zip(order_ids, values) if raw}


def describe(order_id: str, record: dict[str, Any] | None) -> str:
    """The seller-facing explanation for a refused second issuance."""
    if record and record.get("status") == STATUS_ISSUED:
        uuid = record.get("invoice_uuid", "")
        number = record.get("number")
        issued_at = (record.get("issued_at") or "")[:16].replace("T", " ")
        what = f"faktura {number}" if number else "faktura"
        return (
            f"Zamówienie `{order_id}`: {what} została już wystawiona w inFakt"
            + (f" ({issued_at})" if issued_at else "")
            + " — nie wystawiam drugiej.\n"
            + (f"ID faktury w inFakt: `{uuid}`\n" if uuid else "")
            + "Jeśli chcesz, mogę ją dołączyć do zamówienia w Allegro albo wysłać do KSeF."
        )

    claimed_at = ((record or {}).get("claimed_at") or "")[:16].replace("T", " ")
    return (
        f"Zamówienie `{order_id}`: wystawienie faktury zostało już zlecone inFaktowi"
        + (f" ({claimed_at})" if claimed_at else "")
        + ", ale nie potwierdzono jego zakończenia — nie zlecam tego drugi raz, "
        "bo powstałby duplikat. Sprawdź fakturę w panelu inFakt."
    )


def _decode(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _now_iso() -> str:
    return datetime.now(_TZ).strftime("%Y-%m-%dT%H:%M:%S")
