from __future__ import annotations

"""
inFakt REST API client — VAT invoice creation.

Ported from a previous Google Apps Script integration. inFakt's invoice
creation is asynchronous: POST kicks off a background task, then the task
status is polled until it resolves. Only invoice issuance + a shareable
view link are implemented — no PDF export, no Google Drive storage (the
invoice still needs manual review in the inFakt web app either way).

API docs: https://github.com/infakt/API
"""

import asyncio
import logging
from typing import Any

import httpx

from config.settings import get_settings
from models.allegro import AllegroOrder

logger = logging.getLogger(__name__)

# processing_code returned while the async task is still being worked on.
_PROCESSING_CODE_PENDING = 100
# processing_code returned once the invoice was created successfully.
_PROCESSING_CODE_SUCCESS = 201


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
        max_attempts: int = 20,
    ) -> dict[str, Any]:
        """
        Create an invoice and poll until the async task resolves.

        Returns the final status payload (includes invoice_uuid on success).
        Raises InfaktTaskError if inFakt rejects the invoice (validation errors
        etc.), InfaktAPIError on a network/HTTP-level failure, or TimeoutError
        if the task is still pending after max_attempts.
        """
        task_ref = await self.create_invoice_async(invoice_payload)
        for attempt in range(max_attempts):
            await asyncio.sleep(poll_interval)
            status = await self.get_task_status(task_ref)
            code = status.get("processing_code")
            if code == _PROCESSING_CODE_PENDING:
                continue
            if code == _PROCESSING_CODE_SUCCESS:
                return status
            raise InfaktTaskError(
                code, status.get("processing_description", ""), status.get("invoice_errors")
            )
        raise TimeoutError(f"inFakt invoice task {task_ref} still pending after {max_attempts} polls")

    async def get_share_link(self, invoice_uuid: str) -> str:
        """Generate a shareable, no-login-required view link for an invoice."""
        data = await self._post(f"/invoices/{invoice_uuid}/share_links.json")
        return data["share_link"]

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
