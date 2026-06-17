"""
Mock Allegro REST API server.

Implements the same endpoints and response schema as api.allegro.pl
so the main AllEasystent app can run fully offline with realistic fake data.

Usage:
  Set in the main app:
    ALLEGRO_API_URL   = https://<this-service>.railway.app
    ALLEGRO_MOCK_TOKEN = mock-token-12345   (any non-empty string)
    ALLEGRO_AUTH_URL  = https://<this-service>.railway.app/auth/oauth
"""

import os
import uvicorn
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Mock Allegro API")

# ── Fake data store ────────────────────────────────────────────────────────────

_NOW = datetime.now(timezone.utc)
_FMT = lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%SZ")

ORDERS: list[dict] = [
    {
        "id": "ORDER-MOCK-001",
        "status": "READY_FOR_PROCESSING",
        "fulfillment": {"status": "NEW", "shipmentSummary": {"lineItemsSent": "NONE"}},
        "payment": {
            "type": "ONLINE",
            "paidAmount": {"amount": "149.99", "currency": "PLN"},
            "finishedAt": _FMT(_NOW - timedelta(hours=2)),
        },
        "buyer": {
            "login": "jan_kowalski_99",
            "email": "jan.kowalski@example.com",
            "firstName": "Jan",
            "lastName": "Kowalski",
            "address": {
                "street": "ul. Marszałkowska 1",
                "city": "Warszawa",
                "zipCode": "00-001",
                "countryCode": "PL",
            },
        },
        "delivery": {
            "method": {"id": "INPOST_PACZKOMAT", "name": "InPost Paczkomat"},
            "pickupPoint": {"id": "WAW01A", "name": "Paczkomat WAW01A"},
            "address": None,
        },
        "lineItems": [
            {
                "id": "item-001",
                "offer": {"id": "OFFER-MOCK-001", "name": "Słuchawki bezprzewodowe XSound Pro"},
                "quantity": 1,
                "price": {"amount": "149.99", "currency": "PLN"},
                "boughtAt": _FMT(_NOW - timedelta(hours=3)),
            }
        ],
        "summary": {"totalToPay": {"amount": "149.99", "currency": "PLN"}},
        "boughtAt": _FMT(_NOW - timedelta(hours=3)),
        "invoice": {"required": False, "dontWant": False},
    },
    {
        "id": "ORDER-MOCK-002",
        "status": "READY_FOR_PROCESSING",
        "fulfillment": {"status": "PROCESSING", "shipmentSummary": {"lineItemsSent": "SOME"}},
        "payment": {
            "type": "ONLINE",
            "paidAmount": {"amount": "299.00", "currency": "PLN"},
            "finishedAt": _FMT(_NOW - timedelta(hours=24)),
        },
        "buyer": {
            "login": "anna_nowak_shop",
            "email": "anna.nowak@example.com",
            "firstName": "Anna",
            "lastName": "Nowak",
            "address": {
                "street": "ul. Floriańska 15",
                "city": "Kraków",
                "zipCode": "31-019",
                "countryCode": "PL",
            },
        },
        "delivery": {
            "method": {"id": "DPD", "name": "DPD Kurier"},
            "address": {
                "firstName": "Anna",
                "lastName": "Nowak",
                "street": "ul. Floriańska 15",
                "city": "Kraków",
                "zipCode": "31-019",
                "countryCode": "PL",
            },
        },
        "lineItems": [
            {
                "id": "item-002",
                "offer": {"id": "OFFER-MOCK-002", "name": "Klawiatura mechaniczna KeyMaster TKL"},
                "quantity": 1,
                "price": {"amount": "249.00", "currency": "PLN"},
                "boughtAt": _FMT(_NOW - timedelta(hours=25)),
            },
            {
                "id": "item-003",
                "offer": {"id": "OFFER-MOCK-003", "name": "Podkładka gamingowa XL 90x40"},
                "quantity": 1,
                "price": {"amount": "50.00", "currency": "PLN"},
                "boughtAt": _FMT(_NOW - timedelta(hours=25)),
            },
        ],
        "summary": {"totalToPay": {"amount": "299.00", "currency": "PLN"}},
        "boughtAt": _FMT(_NOW - timedelta(hours=25)),
        "invoice": {"required": True, "dontWant": False},
    },
    {
        "id": "ORDER-MOCK-003",
        "status": "READY_FOR_PROCESSING",
        "fulfillment": {"status": "SENT", "shipmentSummary": {"lineItemsSent": "ALL"}},
        "payment": {
            "type": "ONLINE",
            "paidAmount": {"amount": "89.99", "currency": "PLN"},
            "finishedAt": _FMT(_NOW - timedelta(days=3)),
        },
        "buyer": {
            "login": "piotr_wisnewski",
            "email": "p.wisnewski@example.com",
            "firstName": "Piotr",
            "lastName": "Wiśniewski",
            "address": {
                "street": "ul. Długa 8",
                "city": "Gdańsk",
                "zipCode": "80-001",
                "countryCode": "PL",
            },
        },
        "delivery": {
            "method": {"id": "INPOST_PACZKOMAT", "name": "InPost Paczkomat"},
            "tracking": "123456789012",
            "pickupPoint": {"id": "GDA01B", "name": "Paczkomat GDA01B"},
            "address": None,
        },
        "lineItems": [
            {
                "id": "item-004",
                "offer": {"id": "OFFER-MOCK-001", "name": "Słuchawki bezprzewodowe XSound Pro"},
                "quantity": 1,
                "price": {"amount": "89.99", "currency": "PLN"},
                "boughtAt": _FMT(_NOW - timedelta(days=3, hours=1)),
            }
        ],
        "summary": {"totalToPay": {"amount": "89.99", "currency": "PLN"}},
        "boughtAt": _FMT(_NOW - timedelta(days=3, hours=1)),
        "invoice": {"required": False, "dontWant": True},
    },
    {
        "id": "ORDER-MOCK-004",
        "status": "CANCELLED",
        "fulfillment": {"status": "CANCELLED", "shipmentSummary": {"lineItemsSent": "NONE"}},
        "payment": {
            "type": "ONLINE",
            "paidAmount": {"amount": "0.00", "currency": "PLN"},
            "finishedAt": None,
        },
        "buyer": {
            "login": "marta_kaczmarek",
            "email": "m.kaczmarek@example.com",
            "firstName": "Marta",
            "lastName": "Kaczmarek",
            "address": {"street": "ul. Poznańska 3", "city": "Poznań", "zipCode": "61-001", "countryCode": "PL"},
        },
        "delivery": {"method": {"id": "DHL", "name": "DHL Express"}, "address": None},
        "lineItems": [
            {
                "id": "item-005",
                "offer": {"id": "OFFER-MOCK-002", "name": "Klawiatura mechaniczna KeyMaster TKL"},
                "quantity": 1,
                "price": {"amount": "249.00", "currency": "PLN"},
                "boughtAt": _FMT(_NOW - timedelta(days=5)),
            }
        ],
        "summary": {"totalToPay": {"amount": "0.00", "currency": "PLN"}},
        "boughtAt": _FMT(_NOW - timedelta(days=5)),
        "invoice": {"required": False, "dontWant": False},
    },
]

OFFERS: list[dict] = [
    {
        "id": "OFFER-MOCK-001",
        "name": "Słuchawki bezprzewodowe XSound Pro",
        "status": "ACTIVE",
        "publication": {"status": "ACTIVE"},
        "sellingMode": {"price": {"amount": "149.99", "currency": "PLN"}, "format": "BUY_NOW"},
        "stock": {"available": 15, "unit": "UNIT"},
        "stats": {"watchersCount": 42, "visitsCount": 1234},
        "category": {"id": "257", "name": "Słuchawki"},
        "images": [],
    },
    {
        "id": "OFFER-MOCK-002",
        "name": "Klawiatura mechaniczna KeyMaster TKL",
        "status": "ACTIVE",
        "publication": {"status": "ACTIVE"},
        "sellingMode": {"price": {"amount": "249.00", "currency": "PLN"}, "format": "BUY_NOW"},
        "stock": {"available": 3, "unit": "UNIT"},
        "stats": {"watchersCount": 18, "visitsCount": 567},
        "category": {"id": "382", "name": "Klawiatury"},
        "images": [],
    },
    {
        "id": "OFFER-MOCK-003",
        "name": "Podkładka gamingowa XL 90x40",
        "status": "ACTIVE",
        "publication": {"status": "ACTIVE"},
        "sellingMode": {"price": {"amount": "50.00", "currency": "PLN"}, "format": "BUY_NOW"},
        "stock": {"available": 0, "unit": "UNIT"},
        "stats": {"watchersCount": 5, "visitsCount": 89},
        "category": {"id": "385", "name": "Podkładki pod mysz"},
        "images": [],
    },
]

THREADS: list[dict] = [
    {
        "id": "THREAD-MOCK-001",
        "subject": "Pytanie o słuchawki XSound Pro",
        "read": False,
        "messagesCount": 2,
        "lastMessageCreatedAt": _FMT(_NOW - timedelta(hours=1)),
        "buyer": {"login": "jan_kowalski_99"},
    },
    {
        "id": "THREAD-MOCK-002",
        "subject": "Kiedy wyślecie zamówienie?",
        "read": True,
        "messagesCount": 3,
        "lastMessageCreatedAt": _FMT(_NOW - timedelta(days=1)),
        "buyer": {"login": "anna_nowak_shop"},
    },
]

THREAD_MESSAGES: dict[str, list[dict]] = {
    "THREAD-MOCK-001": [
        {
            "id": "MSG-001",
            "text": "Dzień dobry, czy słuchawki XSound Pro mają aktywną redukcję szumów?",
            "type": "BUYER_MESSAGE",
            "createdAt": _FMT(_NOW - timedelta(hours=2)),
            "author": {"login": "jan_kowalski_99", "type": "BUYER"},
        },
        {
            "id": "MSG-002",
            "text": "Dzień dobry! Tak, słuchawki posiadają aktywną redukcję szumów ANC. Pozdrawiam.",
            "type": "SELLER_MESSAGE",
            "createdAt": _FMT(_NOW - timedelta(hours=1)),
            "author": {"login": "mock_seller", "type": "SELLER"},
        },
    ],
    "THREAD-MOCK-002": [
        {
            "id": "MSG-003",
            "text": "Kiedy zostanie wysłane moje zamówienie ORDER-MOCK-002?",
            "type": "BUYER_MESSAGE",
            "createdAt": _FMT(_NOW - timedelta(days=1, hours=3)),
            "author": {"login": "anna_nowak_shop", "type": "BUYER"},
        },
        {
            "id": "MSG-004",
            "text": "Dzień dobry, zamówienie jest w trakcie pakowania, wyślemy jutro.",
            "type": "SELLER_MESSAGE",
            "createdAt": _FMT(_NOW - timedelta(days=1, hours=2)),
            "author": {"login": "mock_seller", "type": "SELLER"},
        },
        {
            "id": "MSG-005",
            "text": "Dziękuję za informację!",
            "type": "BUYER_MESSAGE",
            "createdAt": _FMT(_NOW - timedelta(days=1)),
            "author": {"login": "anna_nowak_shop", "type": "BUYER"},
        },
    ],
}

BILLING: list[dict] = [
    {
        "id": "BILLING-001",
        "occurredAt": _FMT(_NOW - timedelta(days=1)),
        "type": {"id": "COMMISSION_FEE", "name": "Prowizja od sprzedaży"},
        "value": {"amount": "-14.99", "currency": "PLN"},
        "offer": {"id": "OFFER-MOCK-001", "name": "Słuchawki bezprzewodowe XSound Pro"},
        "order": {"id": "ORDER-MOCK-001"},
    },
    {
        "id": "BILLING-002",
        "occurredAt": _FMT(_NOW - timedelta(days=2)),
        "type": {"id": "COMMISSION_FEE", "name": "Prowizja od sprzedaży"},
        "value": {"amount": "-24.90", "currency": "PLN"},
        "offer": {"id": "OFFER-MOCK-002", "name": "Klawiatura mechaniczna KeyMaster TKL"},
        "order": {"id": "ORDER-MOCK-002"},
    },
    {
        "id": "BILLING-003",
        "occurredAt": _FMT(_NOW - timedelta(days=5)),
        "type": {"id": "LISTING_FEE", "name": "Opłata za wystawienie"},
        "value": {"amount": "-1.00", "currency": "PLN"},
        "offer": {"id": "OFFER-MOCK-003", "name": "Podkładka gamingowa XL 90x40"},
        "order": None,
    },
    {
        "id": "BILLING-004",
        "occurredAt": _FMT(_NOW - timedelta(days=7)),
        "type": {"id": "COMMISSION_FEE", "name": "Prowizja od sprzedaży"},
        "value": {"amount": "-8.99", "currency": "PLN"},
        "offer": {"id": "OFFER-MOCK-001", "name": "Słuchawki bezprzewodowe XSound Pro"},
        "order": {"id": "ORDER-MOCK-003"},
    },
]


# ── Auth endpoints ─────────────────────────────────────────────────────────────

@app.post("/auth/oauth/token")
async def oauth_token():
    """Accept any grant type and return a fake access token."""
    return {
        "access_token": "mock-access-token",
        "refresh_token": "mock-refresh-token",
        "token_type": "bearer",
        "expires_in": 43200,
        "scope": "allegro:api:orders:read allegro:api:offers:write allegro:api:billing:read",
    }


# ── Me endpoint ────────────────────────────────────────────────────────────────

@app.get("/me")
async def me():
    return {
        "id": "mock-seller-id",
        "login": "mock_seller_sklep",
        "email": "sprzedawca@example.com",
        "firstName": "Mock",
        "lastName": "Seller",
        "company": {"name": "Mock Shop Sp. z o.o."},
        "createdAt": "2020-03-15T10:00:00Z",
        "isActive": True,
    }


# ── Orders ─────────────────────────────────────────────────────────────────────

@app.get("/order/checkout-forms")
async def list_orders(request: Request):
    params = dict(request.query_params)
    result = list(ORDERS)

    status_filter = params.get("status")
    if status_filter:
        result = [o for o in result if o["status"] == status_filter]

    fulfillment_filter = params.get("fulfillment.status")
    if fulfillment_filter:
        result = [o for o in result if o["fulfillment"]["status"] == fulfillment_filter]

    buyer_filter = params.get("buyer.login")
    if buyer_filter:
        result = [o for o in result if o["buyer"]["login"] == buyer_filter]

    limit = int(params.get("limit", 100))
    offset = int(params.get("offset", 0))
    page = result[offset:offset + limit]

    return {"checkoutForms": page, "totalCount": len(result), "count": len(page)}


@app.get("/order/checkout-forms/{order_id}")
async def get_order(order_id: str):
    for o in ORDERS:
        if o["id"] == order_id:
            return o
    return JSONResponse(status_code=404, content={"errors": [{"code": "NOT_FOUND", "message": f"Order {order_id} not found"}]})


@app.get("/order/checkout-forms/{order_id}/invoices")
async def get_order_invoices(order_id: str):
    return {"invoices": []}


@app.get("/order/event-stats")
async def order_event_stats():
    return {"latestEvent": {"id": "mock-event-999", "occurredAt": _FMT(_NOW - timedelta(minutes=5))}}


@app.get("/order/events")
async def order_events(request: Request):
    params = dict(request.query_params)
    since = params.get("from")
    if since == "mock-event-999":
        return {"events": [], "count": 0}
    return {
        "events": [
            {
                "id": "mock-event-999",
                "type": "READY_FOR_PROCESSING",
                "occurredAt": _FMT(_NOW - timedelta(minutes=5)),
                "order": {"checkoutForm": {"id": "ORDER-MOCK-001"}},
            }
        ],
        "count": 1,
    }


@app.get("/order/carriers")
async def get_carriers():
    return {
        "carriers": [
            {"id": "INPOST_PACZKOMAT", "name": "InPost Paczkomat"},
            {"id": "DPD", "name": "DPD Kurier"},
            {"id": "DHL", "name": "DHL Express"},
            {"id": "GLS", "name": "GLS"},
        ]
    }


# ── Offers ─────────────────────────────────────────────────────────────────────

@app.get("/sale/offers")
async def list_offers(request: Request):
    params = dict(request.query_params)
    result = list(OFFERS)

    name_filter = params.get("name")
    if name_filter:
        result = [o for o in result if name_filter.lower() in o["name"].lower()]

    limit = int(params.get("limit", 100))
    offset = int(params.get("offset", 0))
    page = result[offset:offset + limit]

    return {"offers": page, "totalCount": len(result), "count": len(page)}


@app.get("/sale/offers/{offer_id}")
async def get_offer(offer_id: str):
    for o in OFFERS:
        if o["id"] == offer_id:
            return o
    return JSONResponse(status_code=404, content={"errors": [{"code": "NOT_FOUND"}]})


@app.post("/sale/offers/{offer_id}")
async def update_offer(offer_id: str, request: Request):
    body = await request.json()
    for o in OFFERS:
        if o["id"] == offer_id:
            if "sellingMode" in body:
                o["sellingMode"].update(body["sellingMode"])
            if "stock" in body:
                o["stock"].update(body["stock"])
            return o
    return JSONResponse(status_code=404, content={"errors": [{"code": "NOT_FOUND"}]})


# ── Messaging ──────────────────────────────────────────────────────────────────

@app.get("/messaging/threads")
async def list_threads(request: Request):
    params = dict(request.query_params)
    limit = int(params.get("limit", 50))
    return {"threads": THREADS[:limit], "totalCount": len(THREADS)}


@app.get("/messaging/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: str, request: Request):
    messages = THREAD_MESSAGES.get(thread_id, [])
    return {"messages": messages, "totalCount": len(messages)}


@app.post("/messaging/threads/{thread_id}/messages")
async def send_message(thread_id: str, request: Request):
    body = await request.json()
    msg_id = f"MSG-MOCK-{len(THREAD_MESSAGES.get(thread_id, [])) + 100}"
    new_msg = {
        "id": msg_id,
        "text": body.get("text", ""),
        "type": "SELLER_MESSAGE",
        "createdAt": _FMT(datetime.now(timezone.utc)),
        "author": {"login": "mock_seller", "type": "SELLER"},
    }
    THREAD_MESSAGES.setdefault(thread_id, []).append(new_msg)
    return new_msg


@app.post("/messaging/threads")
async def create_thread(request: Request):
    body = await request.json()
    return {"id": "THREAD-MOCK-NEW", "subject": body.get("subject", "Nowa wiadomość")}


# ── Billing ────────────────────────────────────────────────────────────────────

@app.get("/billing/billing-entries")
async def list_billing(request: Request):
    params = dict(request.query_params)
    limit = int(params.get("limit", 50))
    result = BILLING[:limit]
    return {"billingEntries": result, "totalCount": len(BILLING)}


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "mock": True}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
