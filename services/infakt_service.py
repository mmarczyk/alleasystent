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

API docs: https://github.com/infakt/API (KSeF specifics in ksef.md)
"""

import asyncio
import logging
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


class InvoiceTooLargeError(Exception):
    """The invoice PDF is bigger than Allegro accepts for an order attachment.

    Checked before the two-step upload starts rather than letting the PUT fail:
    the POST would already have registered an invoice record on the order, and a
    retry then has to work around that empty record.
    """

    def __init__(self, size_bytes: int, limit_bytes: int, number: str = ""):
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes
        self.number = number
        super().__init__(f"invoice PDF is {size_bytes} bytes, Allegro accepts {limit_bytes}")


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
    ) -> dict[str, Any]:
        """
        Create an invoice and poll until the async task resolves.

        A pending task is checked repeatedly (up to max_attempts times, ~1 min
        at the default interval) — inFakt reports several distinct in-progress
        codes and needs a moment to build the invoice, so one look at the task
        is not enough to tell "still working" from "failed".

        Returns the final status payload (includes invoice_uuid on success).
        Raises InfaktTaskError if inFakt rejects the invoice (validation errors
        etc.), InfaktAPIError on a network/HTTP-level failure, or TimeoutError
        if the task is still pending after max_attempts.
        """
        task_ref = await self.create_invoice_async(invoice_payload)
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
#
# Neither of them attaches anything: issuance ends in inFakt, and the invoice
# only reaches the buyer's Allegro order page once the seller has said the
# invoice is correct.

async def attach_invoice_to_order(allegro, order_id: str, invoice_uuid: str) -> str:
    """Fetch an inFakt invoice's PDF and attach it to the Allegro order.

    The mechanics of Allegro's two-step attachment in one place. The only
    caller is the chat tool (AllegroAgent._attach_invoice_to_allegro_order),
    which runs it after the seller has confirmed the invoice — issuance
    deliberately does not reach this function on its own (see
    issue_invoice_for_order below).
    Returns the human invoice number (may be empty). Raises InfaktAPIError,
    InvoiceTooLargeError or AllegroAPIError — callers word the failure.
    """
    from services.allegro_service import INVOICE_FILE_MAX_BYTES

    infakt = InfaktService.get_instance()
    invoice = await infakt.get_invoice(invoice_uuid)
    pdf_bytes = await infakt.get_invoice_pdf(invoice_uuid)

    number = invoice.get("number", "") or ""
    if len(pdf_bytes) > INVOICE_FILE_MAX_BYTES:
        raise InvoiceTooLargeError(len(pdf_bytes), INVOICE_FILE_MAX_BYTES, number)

    filename = f"faktura-{number or invoice_uuid}.pdf"
    allegro_invoice_id = await allegro.create_order_invoice_record(order_id, number, filename)
    await allegro.upload_order_invoice_file(order_id, allegro_invoice_id, pdf_bytes)
    return number


async def _share_link_suffix(infakt, invoice_uuid: str) -> str:
    """The seller cannot confirm an invoice they can't see, so the link is part
    of every "waiting for your OK" message. A missing link is not worth failing
    an issued invoice over — it degrades to a note, not an error."""
    try:
        return f": {await infakt.get_share_link(invoice_uuid)}"
    except InfaktAPIError as exc:
        logger.warning("issue_invoice_for_order: share link failed for %s: %s", invoice_uuid, exc)
        return " (nie udało się wygenerować linku podglądu)"


def _confirm_prompt(order_id: str) -> str:
    """The one sentence that turns an issued invoice into an attached one.

    Attaching is a one-way door — Allegro shows the PDF to the buyer the moment
    it lands, and the order takes only one — so it never happens off the back of
    the issuance itself. This is the ask that has to be answered first.
    """
    return (
        f"Sprawdź ją pod linkiem. Jeśli wszystko się zgadza, napisz „dołącz fakturę do "
        f"zamówienia `{order_id}`” — dopiero wtedy dołączę ją do zamówienia. "
        "Do KSeF też wysyłam wyłącznie na Twoje wyraźne polecenie."
    )


async def _report_earlier_issuance(order_id: str, known: dict) -> str:
    """Report an invoice we issued earlier instead of issuing a second one.

    Not an attachment: the earlier issuance was never confirmed either, so
    "wystaw" for the second time is still not permission to show the buyer a
    document nobody has checked.
    """
    invoice_uuid = known["invoice_uuid"]
    label = known.get("number") or invoice_uuid
    link_line = await _share_link_suffix(InfaktService.get_instance(), invoice_uuid) if invoice_uuid else ""
    return (
        f"📄 Faktura {label} dla zamówienia `{order_id}` była już wystawiona w inFakt — "
        f"nie wystawiam drugiej{link_line}\n"
        f"ID faktury w inFakt: `{invoice_uuid}`\n"
        f"⏸️ Nadal NIE jest dołączona do zamówienia w Allegro. {_confirm_prompt(order_id)}"
    )


async def issue_invoice_for_order(allegro, order_id: str, is_production: bool) -> str:
    """Create ONE real VAT invoice in inFakt for a single named Allegro order —
    and stop there, because the next step is the seller's to authorize.

    Issuing and attaching were briefly one step, so that Allegro (which calls an
    order uninvoiced until a PDF is on it) would stop feeding the reminder in
    services/invoice_reminder.py the same order every two hours. That fixed the
    loop by taking a decision away from the seller: the invoice reached the
    BUYER's order page before anyone had looked at it, and Allegro accepts one
    invoice per order, so a wrong NIP or a wrong amount could not be taken back.

    So the attachment is a separate, explicitly confirmed step again
    (AllegroAgent._attach_invoice_to_allegro_order), and the loop is kept shut
    by services/invoice_ledger.py instead: every issuance is written down, so
    the reminder and the next "wystaw" both say "this one is issued and waiting
    for your OK" rather than quietly creating a second invoice.
    """
    from services import invoice_ledger

    user_id = invoice_ledger.user_id_of(allegro)

    order = await allegro.get_order(order_id)
    if not order.invoice_required:
        return f"Zamówienie `{order_id}`: kupujący nie poprosił o fakturę VAT — nic nie wystawiono."

    existing = await allegro.get_order_invoices(order_id)
    if existing:
        return f"Zamówienie `{order_id}`: faktura już istnieje w Allegro — nie wystawiono kolejnej."

    # Allegro saying "no invoice" is not the same as "we never issued one": the
    # invoice may be sitting in inFakt waiting for the seller to confirm it, and
    # a second real VAT invoice for one order cannot be taken back.
    known = await invoice_ledger.get_record(user_id, order_id)
    if known and known.get("invoice_uuid"):
        return await _report_earlier_issuance(order_id, known)

    try:
        address = await allegro.get_order_invoice_data(order_id)
        payload = build_invoice_payload(order, address, is_production)
        infakt = InfaktService.get_instance()
        status = await infakt.create_invoice(payload)
        invoice_uuid = status["invoice_uuid"]
    except TimeoutError as exc:
        # The task was accepted by inFakt and simply hasn't finished within the
        # polling window — the invoice may still appear a moment later, so this
        # must not read as "nothing happened, try again": a blind retry would
        # create a second invoice for the same order. Recorded for the same
        # reason: the reminder must not be the thing that prompts that retry.
        logger.error("issue_invoice_for_order: order %s still pending: %s", order_id, exc)
        await invoice_ledger.record_issued(
            user_id, order_id, invoice_uuid="", attached=False,
            note="inFakt nie potwierdził wystawienia w czasie oczekiwania",
        )
        return (
            f"⏳ inFakt przyjął zlecenie faktury dla zamówienia `{order_id}`, ale nie potwierdził "
            "jej wystawienia w czasie oczekiwania. Faktura prawdopodobnie i tak się utworzy — "
            "sprawdź panel inFakt za chwilę i NIE wystawiaj jej ponownie, dopóki tego nie "
            "zweryfikujesz (inaczej powstanie duplikat).\n"
            f"Szczegóły: {exc}"
        )
    except (InfaktAPIError, InfaktTaskError) as exc:
        logger.error("issue_invoice_for_order: order %s failed: %s", order_id, exc)
        return f"❌ Nie udało się wystawić faktury dla zamówienia `{order_id}`: {exc}"

    buyer_kind = "firma" if address.get("company_name") else "osoba prywatna"
    link_line = await _share_link_suffix(infakt, invoice_uuid)

    await invoice_ledger.record_issued(
        user_id, order_id, invoice_uuid=invoice_uuid, attached=False,
        note="czeka na potwierdzenie sprzedawcy przed dołączeniem do Allegro",
    )
    return (
        f"✅ Faktura dla zamówienia `{order_id}` ({order.buyer_login}) wystawiona w inFakt{link_line}\n"
        f"ID faktury w inFakt: `{invoice_uuid}`\n"
        f"Nabywca: {buyer_kind}.\n"
        f"⏸️ NIE dołączyłem jej do zamówienia w Allegro — kupujący jej na razie nie widzi. "
        f"{_confirm_prompt(order_id)}"
    )
