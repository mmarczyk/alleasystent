from __future__ import annotations

"""Allegro marketplace data models."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# What an absent value looks like per field type, for _AllegroPayloadModel below.
_EMPTY_BY_TYPE: dict[type, Any] = {str: "", int: 0, float: 0.0, bool: False}


class _AllegroPayloadModel(BaseModel):
    """Base for the models parsed straight out of an Allegro checkout form.

    Allegro states a missing value as an explicit `null`, not by leaving the
    key out: an order nobody has paid for yet carries `payment.finishedAt:
    null`, a checkout form the buyer has not completed carries `buyer.email:
    null`. A `.get("email", "")` hands that null straight through — the key IS
    there — and pydantic then rejects None for a `str` field, so ONE such order
    on the page raised a ValidationError that took the whole listing down with
    it ("An internal error occurred. Please try again." for "pokaż zamówienia
    jeszcze nie wysłane"). Nothing on the page could be shown because a single
    order had no payment date.

    Unfiltered listings are where this bites: every other order fetch pins
    `status=READY_FOR_PROCESSING`, whose orders are paid and complete, while a
    negated-stage question ("niewysłane") fetches every checkout form there is
    — BOUGHT and FILLED_IN ones included.

    So a null on a str/int/float/bool field becomes whatever that field already
    holds when Allegro omits the key entirely — its own default ("PLN" for a
    currency, "PL" for a country) or, for a field declared without one, the
    empty value of its type. Only nulls are touched: a value Allegro did send
    is passed to pydantic unchanged, so a genuinely wrong type still fails
    loudly.
    """

    @model_validator(mode="before")
    @classmethod
    def _nulls_become_empty(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        patched: dict[str, Any] | None = None
        for key, value in data.items():
            if value is not None:
                continue
            field = cls.model_fields.get(key)
            if field is None:
                continue
            if field.annotation not in _EMPTY_BY_TYPE:
                continue
            absent = (
                _EMPTY_BY_TYPE[field.annotation]  # type: ignore[index]
                if field.is_required()
                else field.get_default(call_default_factory=True)
            )
            if absent is None:
                continue
            if patched is None:
                patched = dict(data)
            patched[key] = absent
        return data if patched is None else patched


class AllegroTokens(BaseModel):
    access_token: str
    refresh_token: str
    expires_at: datetime
    token_type: str = "Bearer"

    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expires_at


class OrderStatus(str, Enum):
    BOUGHT = "BOUGHT"
    FILLED_IN = "FILLED_IN"
    READY_FOR_PROCESSING = "READY_FOR_PROCESSING"
    CANCELLED = "CANCELLED"


class DeliveryStatus(str, Enum):
    WAITING = "WAITING"
    IN_TRANSIT = "IN_TRANSIT"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"


class AllegroAddress(_AllegroPayloadModel):
    first_name: str = ""
    last_name: str = ""
    street: str = ""
    city: str = ""
    zip_code: str = ""
    country_code: str = "PL"
    phone_number: str = ""


class AllegroOfferSummary(BaseModel):
    id: str
    name: str
    selling_mode: dict[str, Any] = Field(default_factory=dict)
    stock: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
    publication: dict[str, Any] = Field(default_factory=dict)


class AllegroInvoiceBuyer(_AllegroPayloadModel):
    """Who the VAT invoice for an order is made out to, as Allegro states it in
    `invoice.address` on the checkout form.

    This is the ONLY place Allegro says whether the buyer is a company: the
    `company` block (name + NIP) is filled in for a business buyer, the
    `naturalPerson` block for a private one. Nothing else on the order carries
    that distinction — `buyer` is the Allegro account, which a company employee
    uses under their own name — so a buyer who never asked for an invoice simply
    cannot be classified (`required` is False and both name blocks are empty).
    """

    required: bool = False
    dont_want: bool = False
    company_name: str = ""
    vat_id: str = ""
    first_name: str = ""
    last_name: str = ""
    street: str = ""
    city: str = ""
    zip_code: str = ""
    country_code: str = ""

    @property
    def is_company(self) -> bool:
        return bool(self.company_name or self.vat_id)

    @property
    def display_name(self) -> str:
        """Company name, else the private person's full name, else "" — the
        caller decides what to fall back to (usually the Allegro login)."""
        if self.company_name:
            return self.company_name
        return f"{self.first_name} {self.last_name}".strip()


class AllegroOrderLine(_AllegroPayloadModel):
    offer_id: str
    offer_name: str
    quantity: int
    price: float
    currency: str = "PLN"


class AllegroOrder(_AllegroPayloadModel):
    order_id: str
    buyer_login: str
    buyer_email: str = ""
    # `buyer.phoneNumber` from the checkout form — the phone the buyer gave on
    # their Allegro account. The parcel's recipient can be someone else with a
    # different number (`delivery.address.phoneNumber`, kept inside `delivery`
    # below), so a lookup by phone has to try both — see
    # AllegroAgent._order_phones.
    buyer_phone: str = ""
    status: str
    payment_status: str = ""
    total_price: float = 0.0
    currency: str = "PLN"
    created_at: str = ""
    fulfillment_status: str = ""
    paid_at: str = ""  # payment.finishedAt from Allegro
    # delivery.time.dispatch.{from,to} from Allegro — the window the seller has
    # to hand the parcel over to the carrier. `dispatch_to` is the deadline the
    # store owner cares about ("do kiedy paczka ma zostać wysłana"); Allegro
    # returns it for every delivery method, but it can be missing on older
    # orders, so both default to "".
    dispatch_from: str = ""
    dispatch_to: str = ""
    delivery: dict[str, Any] = Field(default_factory=dict)
    line_items: list[AllegroOrderLine] = Field(default_factory=list)
    billing_address: AllegroAddress = Field(default_factory=AllegroAddress)
    invoice_required: bool = False
    # The VAT-invoice address block, parsed from the same checkout form the rest
    # of this model comes from — so anything asking "is this buyer a company?"
    # (get_buyers) reads it off the order it already has instead of re-fetching
    # the form once per order.
    invoice_buyer: AllegroInvoiceBuyer = Field(default_factory=AllegroInvoiceBuyer)

    @field_validator("delivery", mode="before")
    @classmethod
    def coerce_delivery(cls, v: Any) -> dict[str, Any]:
        return v if isinstance(v, dict) else {}


class ThreadOrderMatch(BaseModel):
    """Which order a buyer's message thread is about — see
    AllegroService.resolve_thread_order.

    A buyer asking "czy jest jeszcze możliwość wystawienia faktury do tej
    transakcji" never names the transaction, so the order number has to come
    from somewhere else: Allegro's own `relatedObject` tag on the message
    (`source="message"`, exact), or, failing that, the buyer's order history
    (`source="buyer_history"`, and only when they have exactly one order — one
    order is an identification, several are a question for the seller).

    `order_id` is empty whenever nothing could be pinned down; `candidates`
    then holds what the buyer did order, newest first, for the seller to pick
    from. Guessing is deliberately not an option — a wrong checkout-form id
    would attach an invoice to somebody else's purchase.
    """

    buyer_login: str = ""
    order_id: str = ""
    source: str = ""  # "message" | "buyer_history" | "" (unresolved)
    candidates: list[AllegroOrder] = Field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return bool(self.order_id)


class AllegroMessage(BaseModel):
    thread_id: str
    message_id: str = ""
    author_login: str = ""
    text: str
    created_at: str = ""
    type: str = "QUERY"  # QUERY | ANSWER | CUSTOM
