from __future__ import annotations

"""Synthetic Allegro API dataset used by the tool test-matrix.

Everything here mimics the RAW shapes Allegro's REST API returns, not the
parsed models — the fake API in fake_allegro.py serves these dicts straight
from AllegroService._get/_post, so the real service code (pagination,
filtering, _parse_order, caching) runs untouched.

Dates are generated relative to "now" so a run always looks like a live store
(fresh orders, one dispatch deadline already blown, invoices inside the
current month for get_orders_needing_invoice's month filter).
"""

from datetime import datetime, timedelta, timezone

UTC = timezone.utc


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


NOW = datetime.now(UTC)


def hours_ago(h: float) -> str:
    return _iso(NOW - timedelta(hours=h))


def hours_ahead(h: float) -> str:
    return _iso(NOW + timedelta(hours=h))


def in_month(days_back: float) -> str:
    """`days_back` days ago, but never earlier than this month's 1st 08:00 UTC.

    get_orders_needing_invoice and get_sales_summary both work on the current
    calendar month, so fixtures that must land inside it are clamped instead of
    silently disappearing on the 1st–3rd of a month.
    """
    dt = NOW - timedelta(days=days_back)
    floor = NOW.replace(day=1, hour=8, minute=0, second=0, microsecond=0)
    return _iso(max(dt, floor))


SELLER_LOGIN = "elektrodom_pl"

# ── Orders (GET /order/checkout-forms) ───────────────────────────────────────
# ORD_1..ORD_3  new, unprocessed  (get_new_orders)
# ORD_4, ORD_5  packed, ready for the courier (get_orders_delivery)
# ORD_6         already sent, invoice requested and already issued

ORD_1 = "0c4854a0-9646-11f1-8028-338c43adc37a"
ORD_2 = "1d5965b1-a757-22e2-9139-449d54bed48b"
ORD_3 = "2e6a76c2-b868-33f3-a24a-55ae65cfe59c"
ORD_4 = "3f7b87d3-c979-44f4-b35b-66bf76d0f6ad"
ORD_5 = "4a8c98e4-da8a-55a5-c46c-77c087e1a7be"
ORD_6 = "5b9da9f5-eb9b-66b6-d57d-88d198f2b8cf"


def _price(amount: float, currency: str = "PLN") -> dict:
    return {"amount": f"{amount:.2f}", "currency": currency}


def _line_item(offer_id: str, name: str, qty: int, unit: float) -> dict:
    return {
        "id": f"li-{offer_id}",
        "offer": {"id": offer_id, "name": name},
        "quantity": qty,
        "price": _price(unit),
        "boughtAt": hours_ago(30),
    }


def _delivery(
    method_id: str,
    method_name: str,
    cost: float,
    dispatch_to: str,
    tracking: str | None = None,
    pickup_point: str | None = None,
    recipient: str = "",
) -> dict:
    """`recipient` is "Imię Nazwisko" (or a company name) on delivery.address —
    the name get_buyers falls back to for a buyer who never asked for an
    invoice, so it must be present on every order the way Allegro sends it."""
    first, _, last = recipient.partition(" ")
    d: dict = {
        "method": {"id": method_id, "name": method_name},
        "cost": _price(cost),
        "time": {"dispatch": {"from": hours_ago(24), "to": dispatch_to}},
        "address": {
            "firstName": first,
            "lastName": last,
            "street": "ul. Testowa 1",
            "city": "Warszawa",
            "zipCode": "00-001",
            "countryCode": "PL",
        },
    }
    if tracking:
        d["smart"] = {"trackingCode": tracking}
    if pickup_point:
        d["pickupPoint"] = {"id": "PP-1", "name": pickup_point}
    return d


CHECKOUT_FORMS: list[dict] = [
    {
        "id": ORD_1,
        "status": "READY_FOR_PROCESSING",
        "buyer": {"login": "anna.kowalska88", "email": "anna.kowalska88@allegromail.pl",
                  "firstName": "Anna", "lastName": "Kowalska"},
        "fulfillment": {"status": "NEW"},
        "payment": {"type": "ONLINE", "finishedAt": in_month(1.2)},
        "boughtAt": in_month(1.3),
        "summary": {"totalToPay": _price(429.98)},
        "delivery": _delivery("INPOST_LOCKER", "Allegro Paczkomaty InPost", 9.99,
                              hours_ahead(20), pickup_point="POZ01A — Poznań, ul. Głogowska 12",
                              recipient="Anna Kowalska"),
        "lineItems": [
            _line_item("14587236901", "Ekspres do kawy DeLonghi Magnifica S ECAM", 1, 399.99),
            _line_item("14587236955", "Filtr wody do ekspresu DLSC002", 1, 30.00),
        ],
        "invoice": {
            "required": True,
            "dontWant": False,
            "address": {
                "company": {"name": "Kawa i Spółka sp. z o.o.",
                            "ids": [{"type": "PL_NIP", "value": "7792445588"}]},
                "street": "ul. Głogowska 12/4",
                "city": "Poznań",
                "zipCode": "60-734",
                "countryCode": "PL",
            },
        },
    },
    {
        "id": ORD_2,
        "status": "READY_FOR_PROCESSING",
        "buyer": {"login": "marek_zielinski", "email": "marek.zielinski@allegromail.pl",
                  "firstName": "Marek", "lastName": "Zieliński"},
        "fulfillment": {"status": "NEW"},
        "payment": {"type": "ONLINE", "finishedAt": in_month(2.4)},
        "boughtAt": in_month(2.5),
        "summary": {"totalToPay": _price(899.00)},
        # Deadline already blown → "⚠️ po terminie" marker in every order view.
        "delivery": _delivery("DPD", "Kurier DPD", 14.99, hours_ago(6), recipient="Marek Zieliński"),
        "lineItems": [
            _line_item("14587301122", "Odkurzacz pionowy Dyson V12 Detect Slim", 1, 899.00),
        ],
        "invoice": {
            "required": True,
            "dontWant": False,
            "address": {
                "naturalPerson": {"firstName": "Marek", "lastName": "Zieliński"},
                "street": "ul. Piastowska 41",
                "city": "Wrocław",
                "zipCode": "50-361",
                "countryCode": "PL",
            },
        },
    },
    {
        "id": ORD_3,
        "status": "READY_FOR_PROCESSING",
        "buyer": {"login": "kasia.w", "email": "kasia.w@allegromail.pl",
                  "firstName": "Katarzyna", "lastName": "Wójcik"},
        "fulfillment": {"status": "NEW"},
        "payment": {"type": "ONLINE", "finishedAt": hours_ago(3)},
        "boughtAt": hours_ago(3.5),
        "summary": {"totalToPay": _price(137.70)},
        "delivery": _delivery("INPOST_COURIER", "Kurier InPost", 12.99, hours_ahead(44),
                              recipient="Katarzyna Wójcik"),
        "lineItems": [
            _line_item("14587355001", "Czajnik elektryczny Bosch TWK7203", 1, 109.00),
            _line_item("14587355044", "Zestaw filtrów Brita Maxtra+ 3 szt.", 1, 28.70),
        ],
        "invoice": {"required": False, "dontWant": True},
    },
    {
        "id": ORD_4,
        "status": "READY_FOR_PROCESSING",
        "buyer": {"login": "tomek.nowak", "email": "tomek.nowak@allegromail.pl"},
        "fulfillment": {"status": "READY_FOR_SHIPMENT"},
        "payment": {"type": "ONLINE", "finishedAt": in_month(3.1)},
        "boughtAt": in_month(3.2),
        "summary": {"totalToPay": _price(259.00)},
        "delivery": _delivery("DPD", "Kurier DPD", 14.99, hours_ahead(8),
                              tracking="1234567890PL", recipient="Tomasz Nowak"),
        "lineItems": [
            _line_item("14587377310", "Blender kielichowy Philips HR2291", 1, 259.00),
        ],
        "invoice": {"required": False, "dontWant": False},
    },
    {
        "id": ORD_5,
        "status": "READY_FOR_PROCESSING",
        "buyer": {"login": "agnieszka.maj", "email": "agnieszka.maj@allegromail.pl"},
        "fulfillment": {"status": "READY_FOR_SHIPMENT"},
        "payment": {"type": "ONLINE", "finishedAt": in_month(4.0)},
        "boughtAt": in_month(4.1),
        "summary": {"totalToPay": _price(74.98)},
        "delivery": _delivery("INPOST_LOCKER", "Allegro Paczkomaty InPost", 8.99,
                              hours_ahead(30), tracking="620012345678901234567890",
                              pickup_point="WRO24M — Wrocław, ul. Legnicka 58",
                              recipient="Agnieszka Maj"),
        "lineItems": [
            _line_item("14587355044", "Zestaw filtrów Brita Maxtra+ 3 szt.", 2, 28.70),
            _line_item("14587399881", "Ściereczki z mikrofibry 10 szt.", 1, 17.58),
        ],
        "invoice": {"required": False, "dontWant": False},
    },
    {
        "id": ORD_6,
        "status": "READY_FOR_PROCESSING",
        "buyer": {"login": "pawel.b", "email": "pawel.b@allegromail.pl"},
        "fulfillment": {"status": "SENT"},
        "payment": {"type": "ONLINE", "finishedAt": in_month(6.0)},
        "boughtAt": in_month(6.2),
        "summary": {"totalToPay": _price(1249.00)},
        "delivery": _delivery("DHL", "Kurier DHL", 16.99, in_month(5.0),
                              tracking="JJD000390007312345678", recipient="Paweł Bąk"),
        "lineItems": [
            _line_item("14587401234", "Ekspres do kawy DeLonghi Magnifica S ECAM", 1, 1249.00),
        ],
        # Invoice was requested AND already uploaded (see ORDER_INVOICES) —
        # this is the "faktura już wystawiona" path.
        "invoice": {
            "required": True,
            "dontWant": False,
            "address": {
                "company": {"name": "Biuro Serwis Paweł B.",
                            "ids": [{"type": "PL_NIP", "value": "5252445566"}]},
                "street": "al. Jerozolimskie 100",
                "city": "Warszawa",
                "zipCode": "00-807",
                "countryCode": "PL",
            },
        },
    },
]

# order_id → invoices already uploaded to Allegro
ORDER_INVOICES: dict[str, list[dict]] = {
    ORD_6: [{"id": "inv-allegro-1", "invoiceNumber": "FV/2026/08/41",
             "file": {"name": "faktura-FV-2026-08-41.pdf"}}],
}

# ── Offers (GET /sale/offers) ────────────────────────────────────────────────


def _offer(offer_id: str, name: str, price: float, stock: int, sold: int = 0) -> dict:
    return {
        "id": offer_id,
        "name": name,
        "sellingMode": {"format": "BUY_NOW", "price": _price(price)},
        "stock": {"available": stock, "sold": sold},
        "stats": {"watchersCount": max(0, sold // 2), "visitsCount": sold * 17},
        "publication": {"status": "ACTIVE", "startingAt": in_month(20)},
    }


ACTIVE_OFFERS: list[dict] = [
    # Same product name on two listings → exercises name aggregation.
    _offer("14587236901", "Ekspres do kawy DeLonghi Magnifica S ECAM", 399.99, 3, 61),
    _offer("14587401234", "Ekspres do kawy DeLonghi Magnifica S ECAM", 1249.00, 1, 12),
    _offer("14587301122", "Odkurzacz pionowy Dyson V12 Detect Slim", 899.00, 4, 23),
    _offer("14587355001", "Czajnik elektryczny Bosch TWK7203", 109.00, 27, 143),
    _offer("14587377310", "Blender kielichowy Philips HR2291", 259.00, 12, 55),
    _offer("14587355044", "Zestaw filtrów Brita Maxtra+ 3 szt.", 28.70, 210, 890),
    _offer("14587399881", "Ściereczki z mikrofibry 10 szt.", 17.58, 64, 320),
    _offer("14587236955", "Filtr wody do ekspresu DLSC002", 30.00, 8, 74),
    _offer("14587422003", "Toster Russell Hobbs Colours Plus", 179.00, 0, 38),
    _offer("14587433117", "Mikrofalówka Samsung MS23K3513", 519.00, 2, 9),
]

ENDED_OFFERS: list[dict] = [
    # Sold out to zero → Allegro auto-ended them; they still count as
    # "do zamówienia" in the reorder tools.
    dict(_offer("14587444208", "Żelazko parowe Philips Azur 8000", 329.00, 0, 47),
         publication={"status": "ENDED", "endingAt": in_month(4)}),
    dict(_offer("14587455319", "Waga kuchenna Xiaomi Smart Scale 2", 79.00, 0, 132),
         publication={"status": "ENDED", "endingAt": in_month(9)}),
    # Ended with stock left → deliberately withdrawn, must NOT show up.
    dict(_offer("14587466420", "Grill elektryczny Tefal OptiGrill", 649.00, 6, 4),
         publication={"status": "ENDED", "endingAt": in_month(14)}),
]

# ── Messaging (GET /messaging/threads) ───────────────────────────────────────

THREAD_1 = "5f9b3a10-1111-4a11-9b11-aaaaaaaaaaaa"
THREAD_2 = "5f9b3a10-2222-4a22-9b22-bbbbbbbbbbbb"
THREAD_3 = "5f9b3a10-3333-4a33-9b33-cccccccccccc"
THREAD_4 = "5f9b3a10-4444-4a44-9b44-dddddddddddd"

THREADS: list[dict] = [
    {"id": THREAD_1, "interlocutor": {"login": "anna.kowalska88", "avatarUrl": None},
     "read": False, "lastMessageDateTime": hours_ago(2)},
    {"id": THREAD_2, "interlocutor": {"login": "marek_zielinski", "avatarUrl": None},
     "read": False, "lastMessageDateTime": hours_ago(5)},
    {"id": THREAD_3, "interlocutor": {"login": "kasia.w", "avatarUrl": None},
     "read": True, "lastMessageDateTime": hours_ago(26)},
    {"id": THREAD_4, "interlocutor": {"login": "tomek.nowak", "avatarUrl": None},
     "read": True, "lastMessageDateTime": hours_ago(50)},
]

THREAD_MESSAGES: dict[str, list[dict]] = {
    THREAD_1: [
        {"id": "msg-1", "createdAt": hours_ago(3),
         "author": {"login": "anna.kowalska88", "isInterlocutor": True},
         "text": "Dzień dobry, czy ekspres jest fabrycznie nowy i objęty gwarancją 24 miesiące?"},
        {"id": "msg-2", "createdAt": hours_ago(2.6),
         "author": {"login": SELLER_LOGIN, "isInterlocutor": False},
         "text": "Dzień dobry, tak — sprzęt jest nowy, gwarancja producenta 24 miesiące."},
        {"id": "msg-3", "createdAt": hours_ago(2),
         "author": {"login": "anna.kowalska88", "isInterlocutor": True},
         "text": "Świetnie, proszę o wysyłkę na paczkomat POZ01A. Czy zdąży dziś wyjść?"},
    ],
    THREAD_2: [
        {"id": "msg-4", "createdAt": hours_ago(5),
         "author": {"login": "marek_zielinski", "isInterlocutor": True},
         "text": "Kiedy planowana jest wysyłka odkurzacza? Zależy mi na czasie."},
    ],
    THREAD_3: [
        {"id": "msg-5", "createdAt": hours_ago(26),
         "author": {"login": "kasia.w", "isInterlocutor": True},
         "text": "Dziękuję za szybką wysyłkę!"},
    ],
    THREAD_4: [
        {"id": "msg-6", "createdAt": hours_ago(50),
         "author": {"login": "tomek.nowak", "isInterlocutor": True},
         "text": "Proszę o fakturę na firmę do zamówienia."},
    ],
}

# ── Account (GET /me) ────────────────────────────────────────────────────────

USER_INFO: dict = {
    "id": "88123456",
    "login": SELLER_LOGIN,
    "email": "kontakt@elektrodom.pl",
    "company": {"name": "ElektroDom Sp. z o.o.", "taxId": "7811223344"},
    "registeredAt": "2019-03-14T09:21:00Z",
    "baseCurrency": "PLN",
}

# ── Billing (GET /billing/billing-entries) ───────────────────────────────────


def _entry(entry_id: str, type_id: str, desc: str, amount: float, occurred: str,
           order_id: str | None = None, offer_name: str | None = None) -> dict:
    e: dict = {
        "id": entry_id,
        "occurredAt": occurred,
        "type": {"id": type_id, "description": desc},
        "value": _price(amount),
        "balance": _price(1500.00),
    }
    if order_id:
        e["order"] = {"id": order_id}
    if offer_name:
        e["offer"] = {"id": "14587236901", "name": offer_name}
    return e


BILLING_ENTRIES: list[dict] = [
    _entry("be-1", "SUC", "Prowizja od sprzedaży", -47.30, in_month(1.1), ORD_1,
           "Ekspres do kawy DeLonghi Magnifica S ECAM"),
    _entry("be-2", "PRO", "Opłata za wyróżnienie oferty", -4.90, in_month(1.1), ORD_1),
    _entry("be-3", "SUC", "Prowizja od sprzedaży", -98.89, in_month(2.3), ORD_2,
           "Odkurzacz pionowy Dyson V12 Detect Slim"),
    _entry("be-4", "SUC", "Prowizja od sprzedaży", -15.15, hours_ago(2.8), ORD_3),
    _entry("be-5", "SUC", "Prowizja od sprzedaży", -28.49, in_month(3.0), ORD_4),
    _entry("be-6", "REF", "Zwrot prowizji", 12.40, in_month(3.0), ORD_4),
    _entry("be-7", "SUC", "Prowizja od sprzedaży", -8.24, in_month(3.9), ORD_5),
    _entry("be-8", "SUC", "Prowizja od sprzedaży", -137.39, in_month(5.9), ORD_6),
    # No order.id — an account-level cost (subscription).
    _entry("be-9", "SUB", "Abonament Allegro Firma", -49.00, in_month(7.0)),
    # PAD — internal transfer, must be shown but excluded from the totals.
    _entry("be-10", "PAD", "Pobranie opłat z wpływów", -220.00, in_month(2.0)),
]

# ── Carriers (GET /order/carriers) ───────────────────────────────────────────

CARRIERS: list[dict] = [
    {"id": "INPOST_LOCKER", "name": "Allegro Paczkomaty InPost"},
    {"id": "INPOST_COURIER", "name": "Kurier InPost"},
    {"id": "DPD", "name": "Kurier DPD"},
    {"id": "DHL", "name": "Kurier DHL"},
    {"id": "POCZTA_POLSKA", "name": "Poczta Polska"},
]

# ── Returns & complaints ─────────────────────────────────────────────────────

CUSTOMER_RETURNS: list[dict] = [
    {"id": "ret-1001", "status": "CREATED", "createdAt": hours_ago(4),
     "order": {"id": ORD_4}, "buyer": {"login": "tomek.nowak"},
     "items": [{"name": "Blender kielichowy Philips HR2291", "quantity": 1}],
     "reason": "Towar niezgodny z opisem"},
    {"id": "ret-1002", "status": "DELIVERED", "createdAt": hours_ago(28),
     "order": {"id": ORD_6}, "buyer": {"login": "pawel.b"},
     "items": [{"name": "Ekspres do kawy DeLonghi Magnifica S ECAM", "quantity": 1}],
     "reason": "Odstąpienie od umowy (14 dni)"},
    {"id": "ret-1003", "status": "DELIVERED", "createdAt": hours_ago(52),
     "order": {"id": ORD_5}, "buyer": {"login": "agnieszka.maj"},
     "items": [{"name": "Ściereczki z mikrofibry 10 szt.", "quantity": 1}],
     "reason": "Odstąpienie od umowy (14 dni)"},
    {"id": "ret-1004", "status": "IN_TRANSIT", "createdAt": hours_ago(70),
     "order": {"id": ORD_3}, "buyer": {"login": "kasia.w"},
     "items": [{"name": "Czajnik elektryczny Bosch TWK7203", "quantity": 1}],
     "reason": "Uszkodzenie w transporcie"},
]

ISSUES: list[dict] = [
    {"id": "iss-2001", "type": "DISPUTE", "subject": "Produkt niezgodny z opisem",
     "checkoutForm": {"id": ORD_2}, "openedDate": hours_ago(9),
     "decisionDueDate": hours_ahead(60), "status": "OPEN",
     "buyer": {"login": "marek_zielinski"}},
    {"id": "iss-2002", "type": "CLAIM", "subject": "Reklamacja gwarancyjna — nie włącza się",
     "checkoutForm": {"id": ORD_6}, "openedDate": hours_ago(31),
     "decisionDueDate": hours_ahead(20), "status": "OPEN",
     "buyer": {"login": "pawel.b"}},
    {"id": "iss-2003", "type": "DISPUTE", "subject": "Brak jednego elementu w zestawie",
     "checkoutForm": {"id": ORD_5}, "openedDate": hours_ago(60),
     "status": "OPEN", "buyer": {"login": "agnieszka.maj"}},
]

# ── inFakt ───────────────────────────────────────────────────────────────────

INFAKT_INVOICE_UUID = "b0f1c2d3-e4f5-6789-abcd-ef0123456789"
INFAKT_INVOICE = {
    "uuid": INFAKT_INVOICE_UUID,
    "number": "FV/2026/08/57",
    "gross_price": 42998,
    "currency": "PLN",
    "status": "paid",
    "ksef_number": None,
}
INFAKT_SHARE_LINK = f"https://app.infakt.pl/share/{INFAKT_INVOICE_UUID}"
