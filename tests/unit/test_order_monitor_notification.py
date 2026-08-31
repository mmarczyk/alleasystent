"""Unit tests for the new-order push notification body.

The push is often the only thing the seller sees — they decide from the
notification alone whether to go pack a parcel — so it has to carry the
order's value, its delivery method and whether an invoice is required.
These tests pin both halves: the details riding along in the order event
payload, and the body text built from them.
"""
from __future__ import annotations

import pytest

from services.allegro_service import AllegroService
from services.order_monitor import build_notification_body


def _order(**overrides) -> dict:
    """One entry as `get_order_events` returns it, details included."""
    entry = {
        "event_id": "evt-1",
        "order_id": "order-1",
        "occurred_at": "2026-08-30T10:00:00Z",
        "total_price": 149.9,
        "currency": "PLN",
        "delivery_method": "InPost Paczkomaty",
        "invoice_required": False,
    }
    entry.update(overrides)
    return entry


class TestSingleOrderBody:
    def test_lists_value_delivery_and_invoice(self):
        body = build_notification_body([_order()])
        assert body == "Wartość: 149,90 PLN\nDostawa: InPost Paczkomaty\nFaktura: nie"

    def test_invoice_required_says_so(self):
        body = build_notification_body([_order(invoice_required=True)])
        assert body.endswith("Faktura: tak")

    def test_non_pln_currency_is_printed_as_is(self):
        body = build_notification_body([_order(total_price=20.0, currency="EUR")])
        assert "Wartość: 20,00 EUR" in body

    def test_missing_delivery_method_drops_that_line_only(self):
        body = build_notification_body([_order(delivery_method="")])
        assert body == "Wartość: 149,90 PLN\nFaktura: nie"

    def test_order_without_details_falls_back_to_plain_sentence(self):
        body = build_notification_body([{"event_id": "e", "order_id": "o"}])
        assert body == "Zamówienie czeka na realizację."


class TestMultiOrderBody:
    def test_sums_value_and_counts_invoices(self):
        body = build_notification_body([
            _order(order_id="a", total_price=100.0, invoice_required=True),
            _order(order_id="b", total_price=50.5, delivery_method="Kurier DPD"),
        ])
        assert body == (
            "2 zamówień czeka na realizację.\n"
            "Łączna wartość: 150,50 PLN\n"
            "Dostawa: InPost Paczkomaty, Kurier DPD\n"
            "Faktura: 1 z 2"
        )

    def test_mixed_currencies_are_summed_separately(self):
        body = build_notification_body([
            _order(order_id="a", total_price=100.0),
            _order(order_id="b", total_price=20.0, currency="EUR"),
        ])
        assert "Łączna wartość: 100,00 PLN + 20,00 EUR" in body

    def test_many_delivery_methods_are_capped(self):
        orders = [
            _order(order_id=str(i), delivery_method=f"Kurier {i}")
            for i in range(5)
        ]
        body = build_notification_body(orders)
        assert "Dostawa: Kurier 0, Kurier 1, Kurier 2 +2" in body

    def test_details_missing_everywhere_keeps_the_count_sentence(self):
        body = build_notification_body([
            {"event_id": "e1", "order_id": "a"},
            {"event_id": "e2", "order_id": "b"},
        ])
        assert body == "2 zamówień czeka na realizację."


class _StubOrder:
    """Just the AllegroOrder fields the notification details are read from."""

    def __init__(self, order_id, fulfillment_status="NEW", total_price=149.9,
                 currency="PLN", delivery=None, invoice_required=False):
        self.order_id = order_id
        self.fulfillment_status = fulfillment_status
        self.total_price = total_price
        self.currency = currency
        self.delivery = delivery if delivery is not None else {"method": {"id": "inpost", "name": "InPost Paczkomaty"}}
        self.invoice_required = invoice_required


class _StubService(AllegroService):
    """AllegroService without OAuth/Redis — only the two calls
    `get_order_events` makes are stubbed out."""

    def __init__(self, events, orders):
        self._events = events
        self._orders = orders

    async def _get(self, path, params=None, **kwargs):
        return {"events": self._events}

    async def get_order(self, order_id):
        order = self._orders.get(order_id)
        if order is None:
            raise RuntimeError("order fetch failed")
        return order


def _event(event_id: str, order_id: str) -> dict:
    return {
        "id": event_id,
        "type": "READY_FOR_PROCESSING",
        "occurredAt": "2026-08-30T10:00:00Z",
        "order": {"checkoutForm": {"id": order_id}},
    }


class TestOrderEventDetails:
    def test_details_read_off_the_order(self):
        details = AllegroService.order_event_details(
            _StubOrder("order-1", total_price=99.0, invoice_required=True)
        )
        assert details == {
            "total_price": 99.0,
            "currency": "PLN",
            "delivery_method": "InPost Paczkomaty",
            "invoice_required": True,
        }

    def test_delivery_without_method_gives_empty_name(self):
        details = AllegroService.order_event_details(_StubOrder("order-1", delivery={}))
        assert details["delivery_method"] == ""

    @pytest.mark.asyncio
    async def test_get_order_events_attaches_details(self):
        service = _StubService(
            [_event("evt-1", "order-1")],
            {"order-1": _StubOrder("order-1", total_price=20.0, invoice_required=True)},
        )
        result = await service.get_order_events(since_event_id="evt-0")
        assert result["count"] == 1
        assert result["new_orders"][0] == {
            "event_id": "evt-1",
            "order_id": "order-1",
            "occurred_at": "2026-08-30T10:00:00Z",
            "total_price": 20.0,
            "currency": "PLN",
            "delivery_method": "InPost Paczkomaty",
            "invoice_required": True,
        }

    @pytest.mark.asyncio
    async def test_already_processed_order_is_still_filtered_out(self):
        service = _StubService(
            [_event("evt-1", "order-1"), _event("evt-2", "order-2")],
            {
                "order-1": _StubOrder("order-1", fulfillment_status="SENT"),
                "order-2": _StubOrder("order-2"),
            },
        )
        result = await service.get_order_events(since_event_id="evt-0")
        assert [o["order_id"] for o in result["new_orders"]] == ["order-2"]

    @pytest.mark.asyncio
    async def test_failed_order_fetch_keeps_the_order_without_details(self):
        service = _StubService([_event("evt-1", "order-1")], {})
        result = await service.get_order_events(since_event_id="evt-0")
        assert result["new_orders"] == [
            {"event_id": "evt-1", "order_id": "order-1", "occurred_at": "2026-08-30T10:00:00Z"}
        ]
