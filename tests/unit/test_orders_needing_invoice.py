"""Unit tests for AllegroService.get_orders_needing_invoice's stage scope.

"Jakie mam faktury do wysłania w zamówieniach nie nowych" narrows the pending
invoice listing to part of the seller's orders. The filter runs here, on the
service, and BEFORE the per-order invoice lookups: each of those is an Allegro
request of its own, so an order the seller scoped out is one we must not pay
for.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from models.allegro import AllegroOrder
from services.allegro_service import AllegroService


def _order(order_id: str, fulfillment: str, invoice_required: bool = True) -> AllegroOrder:
    return AllegroOrder(
        order_id=order_id,
        buyer_login="jan",
        status="READY_FOR_PROCESSING",
        fulfillment_status=fulfillment,
        invoice_required=invoice_required,
    )


def _service(orders: list[AllegroOrder]) -> AllegroService:
    """Service whose month fetch returns `orders` and for which no order has an
    invoice uploaded yet — so what comes back is decided purely by the filters
    under test."""
    service = AllegroService.__new__(AllegroService)
    service.get_orders = AsyncMock(return_value=orders)  # type: ignore[method-assign]
    service.get_order_invoices = AsyncMock(return_value=[])  # type: ignore[method-assign]
    return service


def _ids(orders: list[AllegroOrder]) -> list[str]:
    return [o.order_id for o in orders]


ORDERS = [
    _order("nowe", "NEW"),
    _order("realizacja", "PROCESSING"),
    _order("spakowane", "READY_FOR_SHIPMENT"),
    _order("wyslane", "SENT"),
    _order("odebrane", "PICKED_UP"),
]


class TestStageScope:
    @pytest.mark.asyncio
    async def test_no_scope_returns_every_pending_order(self):
        service = _service(list(ORDERS))

        result = await service.get_orders_needing_invoice()

        assert _ids(result) == ["nowe", "realizacja", "spakowane", "wyslane", "odebrane"]

    @pytest.mark.asyncio
    async def test_a_negated_stage_keeps_everything_else(self):
        """"W zamówieniach nie nowych" — every stage but that one, not one
        other stage."""
        service = _service(list(ORDERS))

        result = await service.get_orders_needing_invoice(
            exclude_fulfillment_status=["NEW"],
        )

        assert _ids(result) == ["realizacja", "spakowane", "wyslane", "odebrane"]

    @pytest.mark.asyncio
    async def test_a_positive_stage_keeps_only_it(self):
        service = _service(list(ORDERS))

        result = await service.get_orders_needing_invoice(
            fulfillment_status=["READY_FOR_SHIPMENT"],
        )

        assert _ids(result) == ["spakowane"]

    @pytest.mark.asyncio
    async def test_a_stage_family_keeps_all_of_it(self):
        """"W wysłanych zamówieniach" covers the parcels already delivered too
        — their invoice is the most overdue of all."""
        service = _service(list(ORDERS))

        result = await service.get_orders_needing_invoice(
            fulfillment_status=["SENT", "IN_TRANSIT", "READY_FOR_PICKUP", "PICKED_UP"],
        )

        assert _ids(result) == ["wyslane", "odebrane"]

    @pytest.mark.asyncio
    async def test_statuses_are_matched_case_insensitively(self):
        service = _service(list(ORDERS))

        result = await service.get_orders_needing_invoice(fulfillment_status=["sent"])

        assert _ids(result) == ["wyslane"]

    @pytest.mark.asyncio
    async def test_the_buyer_still_has_to_have_asked_for_an_invoice(self):
        service = _service([
            _order("chce", "SENT"),
            _order("nie-chce", "SENT", invoice_required=False),
        ])

        result = await service.get_orders_needing_invoice(
            exclude_fulfillment_status=["NEW"],
        )

        assert _ids(result) == ["chce"]

    @pytest.mark.asyncio
    async def test_scoped_out_orders_are_never_asked_about(self):
        """The invoice lookup is one Allegro request per order — the whole
        reason the scope is applied before it."""
        service = _service(list(ORDERS))

        await service.get_orders_needing_invoice(fulfillment_status=["NEW"])

        assert service.get_order_invoices.await_count == 1


class TestCancelledOrdersAreNeverPending:
    """A cancelled order is nothing to invoice — there is no sale to document.
    It matters most for a NEGATED scope, whose whole meaning is "everything
    except X" and which would otherwise sweep them in."""

    @pytest.mark.asyncio
    async def test_dropped_from_an_unscoped_listing(self):
        service = _service([_order("zywe", "NEW"), _order("anulowane", "CANCELLED")])

        result = await service.get_orders_needing_invoice()

        assert _ids(result) == ["zywe"]

    @pytest.mark.asyncio
    async def test_a_negated_scope_does_not_sweep_them_in(self):
        service = _service([_order("wyslane", "SENT"), _order("anulowane", "CANCELLED")])

        result = await service.get_orders_needing_invoice(
            exclude_fulfillment_status=["NEW"],
        )

        assert _ids(result) == ["wyslane"]

    @pytest.mark.asyncio
    async def test_asking_for_them_explicitly_still_works(self):
        service = _service([_order("anulowane", "CANCELLED")])

        result = await service.get_orders_needing_invoice(
            fulfillment_status=["CANCELLED"],
        )

        assert _ids(result) == ["anulowane"]

    @pytest.mark.asyncio
    async def test_the_reminder_scope_is_unchanged(self):
        """services/invoice_reminder.py only nags about orders that already
        shipped — that must keep meaning exactly what it did."""
        service = _service(list(ORDERS))

        result = await service.get_orders_needing_invoice(shipped_only=True)

        assert _ids(result) == ["wyslane", "odebrane"]
