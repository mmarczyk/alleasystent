from __future__ import annotations

"""Allegro marketplace data models."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


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


class AllegroAddress(BaseModel):
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


class AllegroOrderLine(BaseModel):
    offer_id: str
    offer_name: str
    quantity: int
    price: float
    currency: str = "PLN"


class AllegroOrder(BaseModel):
    order_id: str
    buyer_login: str
    buyer_email: str = ""
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

    @field_validator("delivery", mode="before")
    @classmethod
    def coerce_delivery(cls, v: Any) -> dict[str, Any]:
        return v if isinstance(v, dict) else {}


class AllegroMessage(BaseModel):
    thread_id: str
    message_id: str = ""
    author_login: str = ""
    text: str
    created_at: str = ""
    type: str = "QUERY"  # QUERY | ANSWER | CUSTOM
