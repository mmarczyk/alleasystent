from __future__ import annotations

"""In-process fakes for the Allegro and inFakt REST APIs.

Both are wired in as httpx MockTransports on the REAL service objects
(AllegroService / InfaktService), so every layer the app owns still runs for
real: OAuth headers, pagination, retries, response parsing, the TTL caches and
all client-side filtering. Only the network hop is replaced.
"""

import json
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from tests.tools import dataset as ds


class FakeAllegroAPI:
    """Routes Allegro REST calls against the synthetic dataset."""

    def __init__(self, empty: bool = False):
        self.empty = empty
        self.calls: list[tuple[str, str, dict[str, list[str]]]] = []
        # Mutated by POST/PUT handlers so a test can assert a write happened.
        self.writes: list[tuple[str, Any]] = []

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _json(payload: Any, status: int = 200) -> httpx.Response:
        return httpx.Response(
            status,
            content=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )

    @staticmethod
    def _one(q: dict[str, list[str]], key: str, default: Any = None) -> Any:
        vals = q.get(key)
        return vals[0] if vals else default

    def _page(self, items: list[dict], q: dict[str, list[str]]) -> tuple[list[dict], int]:
        limit = int(self._one(q, "limit", 100))
        offset = int(self._one(q, "offset", 0))
        return items[offset:offset + limit], len(items)

    # ── router ───────────────────────────────────────────────────────────────

    def handle(self, request: httpx.Request) -> httpx.Response:
        url = urlparse(str(request.url))
        path = url.path
        q = parse_qs(url.query)
        self.calls.append((request.method, path, q))
        method = request.method

        if method == "GET":
            return self._handle_get(path, q)
        if method == "POST":
            body = json.loads(request.content or b"{}")
            return self._handle_post(path, body)
        if method == "PUT":
            self.writes.append((path, request.content))
            return httpx.Response(204)
        return self._json({"error": f"unsupported method {method}"}, 405)

    def _handle_get(self, path: str, q: dict[str, list[str]]) -> httpx.Response:
        # ── Orders ───────────────────────────────────────────────────────────
        if path == "/order/checkout-forms":
            forms = [] if self.empty else list(ds.CHECKOUT_FORMS)
            status = self._one(q, "status")
            if status:
                forms = [f for f in forms if f["status"] == status]
            fulfillment = self._one(q, "fulfillment.status")
            if fulfillment:
                forms = [f for f in forms if (f.get("fulfillment") or {}).get("status") == fulfillment]
            buyer = self._one(q, "buyer.login")
            if buyer:
                forms = [f for f in forms if (f.get("buyer") or {}).get("login") == buyer]
            gte = self._one(q, "lineItems.boughtAt.gte")
            if gte:
                forms = [f for f in forms if f.get("boughtAt", "") >= gte]
            lte = self._one(q, "lineItems.boughtAt.lte")
            if lte:
                forms = [f for f in forms if f.get("boughtAt", "") <= lte]
            sent = q.get("fulfillment.shipmentSummary.lineItemsSent")
            if sent:
                want_sent = sent[0].upper() in ("ALL", "SOME", "TRUE")
                forms = [
                    f for f in forms
                    if ((f.get("fulfillment") or {}).get("status") in ("SENT", "PICKED_UP")) == want_sent
                ]
            page, total = self._page(forms, q)
            return self._json({"checkoutForms": page, "totalCount": total, "count": len(page)})

        if path.startswith("/order/checkout-forms/"):
            rest = path[len("/order/checkout-forms/"):]
            order_id, _, tail = rest.partition("/")
            form = next((f for f in ds.CHECKOUT_FORMS if f["id"] == order_id), None)
            if form is None:
                return self._json({"errors": [{"code": "NotFound"}]}, 404)
            if tail == "invoices":
                return self._json({"invoices": ds.ORDER_INVOICES.get(order_id, [])})
            if not tail:
                return self._json(form)
            return self._json({"errors": [{"code": "NotFound"}]}, 404)

        if path == "/order/carriers":
            return self._json({"carriers": ds.CARRIERS})

        if path == "/order/customer-returns":
            returns = [] if self.empty else list(ds.CUSTOMER_RETURNS)
            status = self._one(q, "status")
            if status:
                returns = [r for r in returns if r.get("status") == status]
            page, _ = self._page(returns, q)
            return self._json({"customerReturns": page, "count": len(page)})

        if path == "/sale/issues":
            issues = [] if self.empty else list(ds.ISSUES)
            page, _ = self._page(issues, q)
            return self._json({"issues": page, "count": len(page)})

        # ── Offers ───────────────────────────────────────────────────────────
        if path == "/sale/offers":
            status = self._one(q, "publication.status", "ACTIVE")
            offers = [] if self.empty else (
                list(ds.ACTIVE_OFFERS) if status == "ACTIVE" else list(ds.ENDED_OFFERS)
            )
            name = self._one(q, "name")
            if name:
                offers = [o for o in offers if name.lower() in o["name"].lower()]
            page, total = self._page(offers, q)
            return self._json({"offers": page, "totalCount": total, "count": len(page)})

        if path.startswith("/sale/offers/"):
            offer_id = path[len("/sale/offers/"):]
            offer = next(
                (o for o in ds.ACTIVE_OFFERS + ds.ENDED_OFFERS if o["id"] == offer_id), None
            )
            if offer is None:
                return self._json({"errors": [{"code": "NotFound"}]}, 404)
            detail = dict(offer)
            detail.update({
                "category": {"id": "165440", "name": "Ekspresy do kawy"},
                "delivery": {"shippingRates": {"id": "b9f2b1c0-standard"},
                             "handlingTime": "PT24H"},
                "payments": {"invoice": "VAT"},
                "location": {"countryCode": "PL", "province": "WIELKOPOLSKIE",
                             "city": "Poznań", "postCode": "60-734"},
                "afterSalesServices": {"warranty": {"name": "Gwarancja 24 miesiące"}},
            })
            return self._json(detail)

        # ── Messaging ────────────────────────────────────────────────────────
        if path == "/messaging/threads":
            threads = [] if self.empty else list(ds.THREADS)
            page, _ = self._page(threads, q)
            return self._json({"threads": page, "count": len(page)})

        if path.startswith("/messaging/threads/") and path.endswith("/messages"):
            thread_id = path[len("/messaging/threads/"):-len("/messages")]
            msgs = ds.THREAD_MESSAGES.get(thread_id, [])
            page, _ = self._page(msgs, q)
            return self._json({"messages": page, "count": len(page)})

        # ── Account & billing ────────────────────────────────────────────────
        if path == "/me":
            return self._json(ds.USER_INFO)

        if path == "/billing/billing-entries":
            entries = [] if self.empty else list(ds.BILLING_ENTRIES)
            order_id = self._one(q, "order.id")
            if order_id:
                entries = [e for e in entries if (e.get("order") or {}).get("id") == order_id]
            gte = self._one(q, "occurredAt.gte")
            if gte:
                entries = [e for e in entries if e.get("occurredAt", "") >= gte]
            lte = self._one(q, "occurredAt.lte")
            if lte:
                entries = [e for e in entries if e.get("occurredAt", "") <= lte]
            page, total = self._page(entries, q)
            return self._json({"billingEntries": page, "count": len(page), "totalCount": total})

        return self._json({"errors": [{"code": "NotFound", "path": path}]}, 404)

    def _handle_post(self, path: str, body: dict) -> httpx.Response:
        self.writes.append((path, body))

        if path.startswith("/sale/offers/"):
            offer_id = path[len("/sale/offers/"):]
            offer = next(
                (o for o in ds.ACTIVE_OFFERS + ds.ENDED_OFFERS if o["id"] == offer_id), None
            )
            if offer is None:
                return self._json({"errors": [{"code": "NotFound"}]}, 404)
            updated = json.loads(json.dumps(offer))
            for key, value in body.items():
                updated.setdefault(key, {})
                updated[key].update(value)
            return self._json(updated)

        if path.startswith("/messaging/threads/") and path.endswith("/messages"):
            return self._json({
                "id": "msg-sent-9001",
                "createdAt": ds.hours_ago(0),
                "type": body.get("type", "ANSWER"),
                "text": body.get("text", ""),
            })

        if path.startswith("/order/checkout-forms/") and path.endswith("/invoices"):
            return self._json({"id": "inv-allegro-new-1",
                               "invoiceNumber": body.get("invoiceNumber", ""),
                               "file": body.get("file", {})})

        return self._json({"errors": [{"code": "NotFound", "path": path}]}, 404)


class FakeInfaktAPI:
    """Routes inFakt API v3 calls: one happy-path invoice plus a 404 for any
    other UUID, which is what the tools' error branches are written against."""

    TASK_REF = "task-ref-77123"

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = urlparse(str(request.url)).path
        self.calls.append((request.method, path))
        uuid = ds.INFAKT_INVOICE_UUID

        if request.method == "POST" and path.endswith("/async/invoices.json"):
            return FakeAllegroAPI._json({"invoice_task_reference_number": self.TASK_REF})

        if request.method == "GET" and "/async/invoices/status/" in path:
            return FakeAllegroAPI._json({
                "processing_code": 201,
                "processing_description": "Invoice created",
                "invoice_uuid": uuid,
            })

        if request.method == "POST" and path.endswith(f"/invoices/{uuid}/share_links.json"):
            return FakeAllegroAPI._json({"share_link": ds.INFAKT_SHARE_LINK})

        if request.method == "POST" and path.endswith(f"/invoices/{uuid}/send_to_ksef.json"):
            return FakeAllegroAPI._json({"status": "sent", "ksef_reference_number": "20260825-SE-1A2B"})

        if request.method == "GET" and path.endswith(f"/invoices/{uuid}/pdf.json"):
            return httpx.Response(200, content=b"%PDF-1.4 fake invoice pdf\n%%EOF",
                                  headers={"Content-Type": "application/pdf"})

        if request.method == "GET" and path.endswith(f"/invoices/{uuid}.json"):
            return FakeAllegroAPI._json(ds.INFAKT_INVOICE)

        return FakeAllegroAPI._json({"error": "Not Found"}, 404)


def allegro_transport(api: FakeAllegroAPI) -> httpx.MockTransport:
    return httpx.MockTransport(api.handle)


def infakt_transport(api: FakeInfaktAPI) -> httpx.MockTransport:
    return httpx.MockTransport(api.handle)
