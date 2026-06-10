from __future__ import annotations

"""Allegro marketplace data models."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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
    delivery: dict[str, Any] = Field(default_factory=dict)
    line_items: list[AllegroOrderLine] = Field(default_factory=list)
    billing_address: AllegroAddress = Field(default_factory=AllegroAddress)
    invoice_required: bool = False


class AllegroMessage(BaseModel):
    thread_id: str
    message_id: str = ""
    author_login: str = ""
    text: str
    created_at: str = ""
    type: str = "QUERY"  # QUERY | ANSWER | CUSTOM
