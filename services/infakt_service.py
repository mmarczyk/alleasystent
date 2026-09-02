from __future__ import annotations

"""
inFakt REST API client — VAT invoice creation, KSeF submission, PDF export.

Ported from a previous Google Apps Script integration. inFakt's invoice
creation is asynchronous: POST kicks off a background task, then the task
status is polled until it resolves. KSeF submission is also asynchronous —
send_to_ksef() only confirms the request was accepted, final processing
must be checked in the inFakt panel.

NOTE: real submission (create_invoice/get_share_link) is only called from
allegro_agent._issue_invoice_for_order, which requires one concrete
order_id per call. Bulk issuance for a whole month/batch stays
preview-only (_preview_pending_invoices, build_invoice_payload only) —
that's the one that misfired on an ambiguous yes/no question earlier;
scoping real submission to a single named order at a time caps the
blast radius of any future misfire at one invoice.

That cap is per misfire, not per order: the same order asked about twice
used to produce two real invoices, because the only "already invoiced?"
check was Allegro's, which cannot see an inFakt invoice nobody attached to
the order. issue_invoice_for_order now claims each order in
services/invoice_ledger.py before calling inFakt at all.

API docs: https://github.com/infakt/API (KSeF specifics in ksef.md)
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from config.settings import get_settings
from models.allegro import AllegroOrder

logger = logging.getLogger(__name__)

# processing_code returned once the invoice was created successfully.
_PROCESSING_CODE_SUCCESS = 201
# inFakt uses the whole 1xx range for "not finished yet", with more than one
# code in it: 100 "Zlecenie przyjęte" right after the POST, then 140 "Zlecenie
# jest w trakcie przetwarzania" while the invoice is actually being built.
# Only 100 used to count as pending here, so a task that happened to be caught
# mid-processing was reported to the seller as a failed issuance — for an
# invoice inFakt then went on to create normally. Anything below this
# threshold means "keep polling", not "failed".
_PROCESSING_CODE_TERMINAL_FROM = 200


class InfaktAPIError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        super().__init__(f"inFakt API error {status_code}: {detail}")


class InfaktTaskError(Exception):
    """Raised when an async invoice-creation task finishes with a failure code."""

    def __init__(self, processing_code: int, description: str, errors: Any = None):
        self.processing_code = processing_code
        self.errors = errors
        super().__init__(f"inFakt invoice task failed ({processing_code}): {description}")


class InfaktService:
    """Thin async wrapper around inFakt API v3 invoice endpoints."""

    _instance: "InfaktService | None" = None

    @classmethod
    def get_instance(cls) -> "InfaktService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url=self._settings.infakt_api_url,
            timeout=30.0,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "X-inFakt-ApiKey": self._settings.infakt_api_key,
            "Content-Type": "application/json",
        }

    async def _get(self, path: str) -> dict[str, Any]:
        try:
            resp = await self._client.get(path, headers=self._headers())
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise InfaktAPIError(0, f"Network error: {exc}") from exc
        if resp.status_code >= 400:
            raise InfaktAPIError(resp.status_code, resp.text[:500])
        return resp.json()

    async def _post(self, path: str, body: dict | None = None) -> dict[str, Any]:
        try:
            resp = await self._client.post(path, headers=self._headers(), json=body)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise InfaktAPIError(0, f"Network error: {exc}") from exc
        if resp.status_code >= 400:
            raise InfaktAPIError(resp.status_code, resp.text[:500])
        return resp.json()

    # ── Async invoice creation ──────────────────────────────────────────────

    async def create_invoice_async(self, invoice_payload: dict) -> str:
        """Kick off invoice creation, return the task reference number."""
        data = await self._post("/async/invoices.json", {"invoice": invoice_payload})
        return data["invoice_task_reference_number"]

    async def get_task_status(self, task_ref: str) -> dict[str, Any]:
        return await self._get(f"/async/invoices/status/{task_ref}.json")

    async def create_invoice(
        self,
        invoice_payload: dict,
        poll_interval: float = 1.5,
        max_attempts: int = 40,
        on_task_ref: "Callable[[str], Awaitable[None]] | None" = None,
    ) -> dict[str, Any]:
        """
        Create an invoice and poll until the async task resolves.

        A pending task is checked repeatedly (up to max_attempts times, ~1 min
        at the default interval) — inFakt reports several distinct in-progress
        codes and needs a moment to build the invoice, so one look at the task
        is not enough to tell "still working" from "failed".

        `on_task_ref` is awaited with the task reference the moment the POST
        returns, before any polling. Every outcome below except a clean
        success leaves the caller unable to name the invoice, and the task
        reference is what makes that recoverable later (see
        unblock_invoice_for_order) — so it has to be handed over before the
        outcome is known, not returned alongside it.

        Returns the final status payload (includes invoice_uuid on success).
        Raises InfaktTaskError if inFakt rejects the invoice (validation errors
        etc.), InfaktAPIError on a network/HTTP-level failure, or TimeoutError
        if the task is still pending after max_attempts.
        """
        task_ref = await self.create_invoice_async(invoice_payload)
        if on_task_ref is not None:
            await on_task_ref(task_ref)
        last_transient: InfaktAPIError | None = None
        for attempt in range(max_attempts):
            # Check first, sleep only between checks — a small invoice is often
            # already done, and sleeping up front added poll_interval to every
            # issuance for nothing.
            try:
                status = await self.get_task_status(task_ref)
            except InfaktAPIError as exc:
                # The invoice may well be getting created while the status check
                # blips; a network error or a 5xx on one poll says nothing about
                # the task, so keep checking instead of reporting a failure.
                # A 4xx (bad key, unknown task ref) will not fix itself — raise.
                if exc.status_code != 0 and exc.status_code < 500:
                    raise
                logger.warning("inFakt task %s: status check failed, retrying: %s", task_ref, exc)
                last_transient = exc
                await asyncio.sleep(poll_interval)
                continue

            code = status.get("processing_code")
            if code == _PROCESSING_CODE_SUCCESS:
                return status
            if not isinstance(code, int) or code < _PROCESSING_CODE_TERMINAL_FROM:
                # Still queued or mid-processing (1xx), or a status payload we
                # can't read — either way inFakt hasn't said "done" or "failed".
                await asyncio.sleep(poll_interval)
                continue
            raise InfaktTaskError(
                code, status.get("processing_description", ""), status.get("invoice_errors")
            )

        pending_detail = f" (last status check failed: {last_transient})" if last_transient else ""
        raise TimeoutError(
            f"inFakt invoice task {task_ref} still pending after {max_attempts} polls{pending_detail}"
        )

    async def get_share_link(self, invoice_uuid: str) -> str:
        """Generate a shareable, no-login-required view link for an invoice."""
        data = await self._post(f"/invoices/{invoice_uuid}/share_links.json")
        return data["share_link"]

    async def get_invoice(self, invoice_uuid: str) -> dict[str, Any]:
        """Fetch full invoice details — includes "number" (human invoice number) and "ksef_number"."""
        return await self._get(f"/invoices/{invoice_uuid}.json")

    async def get_invoice_pdf(self, invoice_uuid: str) -> bytes:
        """Fetch the invoice's original PDF as raw bytes (for handing off to Allegro)."""
        try:
            resp = await self._client.get(
                f"/invoices/{invoice_uuid}/pdf.json",
                params={"document_type": "original"},
                headers=self._headers(),
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise InfaktAPIError(0, f"Network error: {exc}") from exc
        if resp.status_code >= 400:
            raise InfaktAPIError(resp.status_code, resp.text[:500])
        return resp.content

    async def send_to_ksef(self, invoice_uuid: str) -> dict[str, Any]:
        """Submit an issued invoice to KSeF (Krajowy System e-Faktur).

        Submission is asynchronous on inFakt's side — this call only confirms
        the request was accepted (status "sent"), not that KSeF finished
        processing it. Final status must be checked in the inFakt panel.
        """
        return await self._post(f"/invoices/{invoice_uuid}/send_to_ksef.json")

    async def aclose(self) -> None:
        await self._client.aclose()


# ── Invoice payload builder ──────────────────────────────────────────────────
# Ported from the previous Apps Script's API.buildInvoice/buildServiceList.

_VAT_TAX_SYMBOL = 23
_FLAT_RATE_TAX_SYMBOL = 5.5  # ryczałt — only applies outside sandbox/test mode


def build_invoice_payload(
    order: AllegroOrder,
    invoice_address: dict[str, Any],
    is_production: bool,
) -> dict[str, Any]:
    """Build an inFakt invoice payload from an Allegro order + its invoice address."""
    sale_date = (order.paid_at or order.created_at or "")[:10]
    invoice: dict[str, Any] = {
        "payment_method": "other",
        "sale_date": sale_date,
        "paid_date": sale_date,
        "paid_price": round(order.total_price * 100),
        "status": "paid",
        "client_street": invoice_address.get("street", ""),
        "client_city": invoice_address.get("city", ""),
        "client_post_code": invoice_address.get("zip_code", ""),
    }

    if invoice_address.get("company_name"):
        invoice["client_company_name"] = invoice_address["company_name"]
        invoice["client_tax_code"] = invoice_address.get("vat_id", "")
    else:
        invoice["client_first_name"] = invoice_address.get("first_name", "")
        invoice["client_last_name"] = invoice_address.get("last_name", "")
        invoice["client_business_activity_kind"] = "private_person"

    services = _build_services(order, is_production)
    delivery = order.delivery or {}
    delivery_cost = (delivery.get("cost") or {}).get("amount")
    if not delivery.get("smart") and delivery_cost:
        services.append(_delivery_service(float(delivery_cost), is_production))
    invoice["services"] = services

    return invoice


def _build_services(order: AllegroOrder, is_production: bool) -> list[dict[str, Any]]:
    # Merge line items that are the same product at the same unit price into
    # one invoice line, mirroring the previous Apps Script behavior.
    grouped: dict[tuple[str, float], dict[str, Any]] = {}
    for item in order.line_items:
        key = (item.offer_name, item.price)
        if key in grouped:
            grouped[key]["quantity"] += item.quantity
        else:
            grouped[key] = {"name": item.offer_name, "quantity": item.quantity, "unit_price": item.price}

    services = []
    for g in grouped.values():
        svc = {
            "name": g["name"],
            "quantity": g["quantity"],
            "gross_price": round(g["quantity"] * g["unit_price"] * 100),
            "tax_symbol": _VAT_TAX_SYMBOL,
        }
        if is_production:
            svc["flat_rate_tax_symbol"] = _FLAT_RATE_TAX_SYMBOL
        services.append(svc)
    return services


def _delivery_service(cost: float, is_production: bool) -> dict[str, Any]:
    svc = {
        "name": "Koszty wysyłki",
        "quantity": 1,
        "gross_price": round(cost * 100),
        "tax_symbol": _VAT_TAX_SYMBOL,
    }
    if is_production:
        svc["flat_rate_tax_symbol"] = _FLAT_RATE_TAX_SYMBOL
    return svc


# ── Single-order issuance ─────────────────────────────────────────────────────
# Shared by the chat tool (AllegroAgent._issue_invoice_for_order) and the
# invoice reminder's "issue now" action (services/invoice_reminder.py) —
# both issue one concrete order at a time through this same path, see the
# module docstring above for why bulk issuance stays preview-only.

async def issue_invoice_for_order(allegro, order_id: str, is_production: bool) -> str:
    """Create ONE real VAT invoice in inFakt for a single named Allegro order.

    Guarded on both sides, because either guard alone lets the same order be
    invoiced twice:
      * Allegro's own per-order invoice list — but it only knows about
        invoices someone already ATTACHED to the order (a separate step), so
        it says "no invoice" for one that exists in inFakt and was never
        uploaded;
      * services/invoice_ledger.py — every issuance this system performs,
        attached or not, claimed atomically BEFORE inFakt is called so a
        second caller (the reminder firing again, a repeated chat request,
        the monitor job racing the chat process) is refused rather than
        served.

    The claim is released only when inFakt is known NOT to have created
    anything. An outcome we cannot confirm keeps the claim — see
    services/invoice_ledger.py for why that trade is the right way round.
    """
    from services import invoice_ledger

    order = await allegro.get_order(order_id)
    if not order.invoice_required:
        return f"Zamówienie `{order_id}`: kupujący nie poprosił o fakturę VAT — nic nie wystawiono."

    existing = await allegro.get_order_invoices(order_id)
    if existing:
        return f"Zamówienie `{order_id}`: faktura już istnieje w Allegro — nie wystawiono kolejnej."

    user_id = getattr(allegro, "_user_id", "default")
    claim = await invoice_ledger.claim(user_id, order_id)
    if not claim.acquired:
        logger.info("issue_invoice_for_order: order %s already claimed — refusing duplicate", order_id)
        return invoice_ledger.describe(order_id, claim.record)

    try:
        address = await allegro.get_order_invoice_data(order_id)
        payload = build_invoice_payload(order, address, is_production)
    except Exception:
        # Still nothing sent to inFakt, so the order must stay issuable.
        await invoice_ledger.release(user_id, order_id)
        raise

    infakt = InfaktService.get_instance()

    async def _remember_task_ref(task_ref: str) -> None:
        # Stored before the outcome is known: if the polling below times out,
        # this reference is the only way to ever ask inFakt what happened to
        # this attempt (see unblock_invoice_for_order).
        await invoice_ledger.record_task_ref(user_id, order_id, task_ref)

    try:
        status = await infakt.create_invoice(payload, on_task_ref=_remember_task_ref)
    except TimeoutError as exc:
        # The task was accepted by inFakt and simply hasn't finished within the
        # polling window — the invoice may still appear a moment later, so this
        # must not read as "nothing happened, try again": a blind retry would
        # create a second invoice for the same order. The claim stays for the
        # same reason, so a later "wystaw" for this order is refused too.
        logger.error("issue_invoice_for_order: order %s still pending: %s", order_id, exc)
        return (
            f"⏳ inFakt przyjął zlecenie faktury dla zamówienia `{order_id}`, ale nie potwierdził "
            "jej wystawienia w czasie oczekiwania. Faktura prawdopodobnie i tak się utworzy — "
            "sprawdź panel inFakt za chwilę i NIE wystawiaj jej ponownie, dopóki tego nie "
            "zweryfikujesz (inaczej powstanie duplikat). Kolejne zlecenie dla tego zamówienia "
            "i tak odrzucę — a jeśli faktury w inFakt nie będzie, napisz o tym, sprawdzę status "
            "zlecenia w inFakt i odblokuję wystawianie.\n"
            f"Szczegóły: {exc}"
        )
    except (InfaktAPIError, InfaktTaskError) as exc:
        # inFakt rejected the invoice or never got the request — nothing was
        # created, so the order goes back to being issuable.
        await invoice_ledger.release(user_id, order_id)
        logger.error("issue_invoice_for_order: order %s failed: %s", order_id, exc)
        return f"❌ Nie udało się wystawić faktury dla zamówienia `{order_id}`: {exc}"

    invoice_uuid = status.get("invoice_uuid")
    if not invoice_uuid:
        # inFakt said 201 without telling us which invoice — it exists, we just
        # can't name it. Keep the claim: the seller must look it up, not reissue.
        logger.error("issue_invoice_for_order: order %s created without invoice_uuid: %s", order_id, status)
        return (
            f"⚠️ inFakt potwierdził utworzenie faktury dla zamówienia `{order_id}`, ale nie zwrócił "
            "jej identyfikatora. Znajdź ją w panelu inFakt — nie wystawiam jej ponownie, bo "
            "powstałby duplikat. Jeśli jej tam nie ma, napisz o tym, sprawdzę zlecenie w inFakt "
            "i odblokuję wystawianie."
        )

    await invoice_ledger.mark_issued(user_id, order_id, invoice_uuid)

    buyer_kind = "firma" if address.get("company_name") else "osoba prywatna"
    head = (
        f"✅ Faktura dla zamówienia `{order_id}` ({order.buyer_login}) wystawiona w inFakt"
    )
    try:
        link = await infakt.get_share_link(invoice_uuid)
    except InfaktAPIError as exc:
        # The invoice EXISTS — only the share link failed. Reporting this as a
        # failed issuance is what used to send sellers back to issue it again.
        logger.warning(
            "issue_invoice_for_order: order %s issued (%s) but share link failed: %s",
            order_id, invoice_uuid, exc,
        )
        return (
            f"{head}, ale nie udało się wygenerować linku do podglądu ({exc}).\n"
            f"ID faktury w inFakt: `{invoice_uuid}`\n"
            f"Nabywca: {buyer_kind}.\n"
            "Fakturę znajdziesz w panelu inFakt — jest wystawiona, nie wystawiaj jej ponownie."
        )

    return (
        f"{head}: {link}\n"
        f"ID faktury w inFakt: `{invoice_uuid}`\n"
        f"Nabywca: {buyer_kind}.\n"
        "Zweryfikuj fakturę pod linkiem, zanim pójdzie dalej (do Allegro / KSeF)."
    )


# ── Unblocking an order the ledger holds ─────────────────────────────────────

async def unblock_invoice_for_order(allegro, order_id: str, seller_confirmed: bool = False) -> str:
    """Lift the duplicate block on ONE order — but only after checking inFakt.

    The block exists because an unconfirmed issuance is more likely to have
    created an invoice than not (see issue_invoice_for_order). It is still a
    guess, and a wrong one strands the order: the reminder skips it and every
    "wystaw fakturę" for it is refused, forever.

    So this resolves the guess against inFakt rather than removing the block
    on request. Every branch below either has an authoritative answer or
    refuses:
      * the invoice is attached to the order in Allegro → it exists, refuse;
      * the ledger knows its uuid → GET the invoice: a 404 is inFakt saying
        it does not exist (unblock), anything else means it does (refuse);
      * the ledger knows the async task reference → re-read the task status,
        which is the same source create_invoice polled: success names the
        invoice (refuse, and record the uuid the timeout lost), a terminal
        failure means nothing was created (unblock), still-processing means
        ask again in a moment;
      * nothing to check against — the claim was taken but the POST never
        got far enough to produce a reference. Only here does the seller's
        own "I looked, it is not there" decide it, and it has to be an
        explicit statement, not an inference (`seller_confirmed`).
    """
    from services import invoice_ledger

    user_id = getattr(allegro, "_user_id", "default")
    record = await invoice_ledger.get(user_id, order_id)
    if not record:
        return (
            f"Zamówienie `{order_id}`: nic nie blokuje wystawienia faktury — nie mam zapisanego "
            "żadnego wystawienia dla tego zamówienia. Możesz je wystawić normalnie."
        )

    existing = await allegro.get_order_invoices(order_id)
    if existing:
        return (
            f"Zamówienie `{order_id}`: faktura jest dołączona do zamówienia w Allegro, więc na "
            "pewno istnieje — nie zdejmuję blokady."
        )

    infakt = InfaktService.get_instance()
    invoice_uuid = record.get("invoice_uuid")

    if invoice_uuid:
        try:
            invoice = await infakt.get_invoice(invoice_uuid)
        except InfaktAPIError as exc:
            if exc.status_code == 404:
                await invoice_ledger.release(user_id, order_id)
                logger.warning(
                    "unblock_invoice_for_order: order %s unblocked — inFakt has no invoice %s",
                    order_id, invoice_uuid,
                )
                return (
                    f"✅ Zamówienie `{order_id}`: inFakt nie zna faktury `{invoice_uuid}` (404), "
                    "więc jej tam nie ma. Zdjąłem blokadę — możesz wystawić fakturę dla tego "
                    "zamówienia."
                )
            return (
                f"❌ Zamówienie `{order_id}`: nie udało się sprawdzić faktury `{invoice_uuid}` "
                f"w inFakt ({exc}). Nie zdejmuję blokady, dopóki nie wiem, czy faktura istnieje — "
                "spróbuj ponownie za chwilę."
            )
        number = invoice.get("number") or invoice_uuid
        return (
            f"Zamówienie `{order_id}`: faktura {number} JEST w inFakt — nie zdejmuję blokady, bo "
            "wystawienie drugiej dałoby duplikat.\n"
            f"ID faktury w inFakt: `{invoice_uuid}`\n"
            "Jeśli ta faktura jest błędna, skoryguj lub anuluj ją w panelu inFakt."
        )

    task_ref = record.get("task_ref")
    if task_ref:
        try:
            status = await infakt.get_task_status(task_ref)
        except InfaktAPIError as exc:
            return (
                f"❌ Zamówienie `{order_id}`: nie udało się sprawdzić w inFakt statusu zlecenia "
                f"`{task_ref}` ({exc}). Nie zdejmuję blokady, dopóki nie wiem, czy faktura "
                "powstała — spróbuj ponownie za chwilę."
            )

        code = status.get("processing_code")
        if code == _PROCESSING_CODE_SUCCESS:
            uuid_from_task = status.get("invoice_uuid")
            if uuid_from_task:
                # The invoice the timeout couldn't name. Record it, so the
                # seller can now attach it or send it to KSeF like any other.
                await invoice_ledger.mark_issued(user_id, order_id, uuid_from_task)
                return (
                    f"Zamówienie `{order_id}`: faktura jednak powstała — inFakt zakończył zlecenie "
                    "sukcesem. Nie zdejmuję blokady.\n"
                    f"ID faktury w inFakt: `{uuid_from_task}`\n"
                    "Mogę ją dołączyć do zamówienia w Allegro albo wysłać do KSeF."
                )
            return (
                f"Zamówienie `{order_id}`: inFakt zakończył zlecenie sukcesem, więc faktura "
                "istnieje, ale nie zwrócił jej identyfikatora — nie zdejmuję blokady. Znajdź ją "
                "w panelu inFakt."
            )

        if not isinstance(code, int) or code < _PROCESSING_CODE_TERMINAL_FROM:
            return (
                f"Zamówienie `{order_id}`: inFakt nadal przetwarza to zlecenie "
                f"(status: {status.get('processing_description') or code}). Nie zdejmuję blokady — "
                "zapytaj ponownie za chwilę, wtedy będzie już wiadomo, czy faktura powstała."
            )

        await invoice_ledger.release(user_id, order_id)
        logger.warning(
            "unblock_invoice_for_order: order %s unblocked — inFakt task %s failed (%s)",
            order_id, task_ref, code,
        )
        return (
            f"✅ Zamówienie `{order_id}`: inFakt odrzucił tamto zlecenie "
            f"({status.get('processing_description') or code}), więc faktura nie powstała. "
            "Zdjąłem blokadę — możesz wystawić fakturę dla tego zamówienia."
        )

    if not seller_confirmed:
        return (
            f"⚠️ Zamówienie `{order_id}`: nie mam po czym sprawdzić tamtej próby w inFakt — nie "
            "znam ani numeru faktury, ani numeru zlecenia. Sprawdź w panelu inFakt, czy faktura "
            "dla tego zamówienia tam jest. Jeśli NIE MA jej tam, napisz mi to wprost, a zdejmę "
            "blokadę."
        )

    await invoice_ledger.release(user_id, order_id)
    logger.warning(
        "unblock_invoice_for_order: order %s unblocked on the seller's confirmation "
        "(no invoice_uuid, no task_ref to verify against)",
        order_id,
    )
    return (
        f"✅ Zamówienie `{order_id}`: zdjąłem blokadę na Twoje potwierdzenie, że faktury nie ma "
        "w inFakt. Możesz teraz wystawić ją normalnie."
    )
