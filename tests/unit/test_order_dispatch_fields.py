"""Unit tests for the order status + dispatch-deadline fields.

Every order-related answer has to carry the order's current status and the
moment by which the parcel must be handed to the carrier (Allegro's
delivery.time.dispatch.to) — these tests pin both the parsing of that field
and its rendering.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agents.allegro.allegro_agent import AllegroAgent
from models.allegro import AllegroOrder
from services.allegro_service import AllegroService


def _checkout_form(dispatch: dict | None = None, **overrides) -> dict:
    """A minimal /order/checkout-forms entry, shaped like Allegro's response."""
    delivery: dict = {"method": {"id": "inpost", "name": "InPost Paczkomaty"}}
    if dispatch is not None:
        delivery["time"] = {
            "from": "2026-08-20T10:00:00Z",
            "to": "2026-08-22T10:00:00Z",
            "dispatch": dispatch,
        }
    form = {
        "id": "order-1",
        "status": "READY_FOR_PROCESSING",
        "buyer": {"login": "buyer1", "email": "b@example.com"},
        "fulfillment": {"status": "NEW"},
        "delivery": delivery,
    }
    form.update(overrides)
    return form


# `_parse_order` never touches `self`, so it can be exercised without building a
# real service (which would need OAuth tokens and Redis).
_parse = AllegroService._parse_order


class TestParseDispatchWindow:
    def test_dispatch_from_and_to_parsed(self):
        order = _parse(
            None,
            _checkout_form({"from": "2026-08-20T12:00:00Z", "to": "2026-08-21T18:00:00Z"}),
        )
        assert order.dispatch_from == "2026-08-20T12:00:00Z"
        assert order.dispatch_to == "2026-08-21T18:00:00Z"

    def test_missing_dispatch_block_gives_empty_strings(self):
        order = _parse(None, _checkout_form())
        assert order.dispatch_from == ""
        assert order.dispatch_to == ""

    def test_null_time_block_does_not_raise(self):
        form = _checkout_form()
        form["delivery"]["time"] = None
        order = _parse(None, form)
        assert order.dispatch_to == ""

    def test_null_dispatch_block_does_not_raise(self):
        form = _checkout_form({"from": "x", "to": "y"})
        form["delivery"]["time"]["dispatch"] = None
        assert _parse(None, form).dispatch_to == ""

    def test_non_dict_delivery_does_not_raise(self):
        order = _parse(None, _checkout_form(delivery=[]))
        assert order.dispatch_to == ""
        assert order.delivery == {}

    def test_delivery_still_parsed_alongside_dispatch(self):
        order = _parse(None, _checkout_form({"to": "2026-08-21T18:00:00Z"}))
        assert order.delivery["method"]["name"] == "InPost Paczkomaty"


def _order(dispatch_to: str = "", fulfillment_status: str = "NEW") -> AllegroOrder:
    return AllegroOrder(
        order_id="order-1",
        buyer_login="buyer1",
        status="READY_FOR_PROCESSING",
        fulfillment_status=fulfillment_status,
        dispatch_to=dispatch_to,
        total_price=99.0,
    )


def _order_with_delivery() -> AllegroOrder:
    order = _order("2099-08-21T10:00:00Z")
    order.delivery = {
        "method": {"id": "INPOST", "name": "InPost Paczkomaty"},
        "smart": {"trackingCode": "ABC123"},
    }
    return order


def _iso(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) + delta).strftime("%Y-%m-%dT%H:%M:%SZ")


class TestDispatchDeadlineRendering:
    def test_missing_deadline_renders_dash(self):
        assert AllegroAgent._dispatch_deadline_pl(_order()) == "—"

    def test_future_deadline_has_no_warning(self):
        rendered = AllegroAgent._dispatch_deadline_pl(_order(_iso(timedelta(days=1))))
        assert "po terminie" not in rendered
        assert rendered != "—"

    def test_past_deadline_on_unsent_order_is_flagged(self):
        rendered = AllegroAgent._dispatch_deadline_pl(_order(_iso(timedelta(hours=-2))))
        assert "po terminie" in rendered

    def test_past_deadline_on_sent_order_is_not_flagged(self):
        rendered = AllegroAgent._dispatch_deadline_pl(
            _order(_iso(timedelta(hours=-2)), fulfillment_status="SENT")
        )
        assert "po terminie" not in rendered

    def test_deadline_shown_in_warsaw_local_time(self):
        # 10:00 UTC in August = 12:00 in Warsaw (CEST, UTC+2).
        rendered = AllegroAgent._dispatch_deadline_pl(_order("2099-08-21T10:00:00Z"))
        assert rendered == "21.08.2099, 12:00"

    def test_unparsable_deadline_passed_through(self):
        assert AllegroAgent._dispatch_deadline_pl(_order("not-a-date")) == "not-a-date"


class TestOrderBlocksCarryBothFields:
    """_order_bullet is the ONLY order renderer — every listing tool goes
    through it (see AllegroAgent._orders_listing), so pinning it here pins the
    shape of every order answer."""

    def test_order_bullet_has_status_and_deadline(self):
        block = AllegroAgent._order_bullet(_order("2099-08-21T10:00:00Z"))
        assert "- Status: **Nowe**" in block
        assert "- Wysyłka do: **21.08.2099, 12:00**" in block

    def test_order_bullet_without_deadline_shows_dash(self):
        assert "- Wysyłka do: **—**" in AllegroAgent._order_bullet(_order())

    def test_order_bullet_delivery_details_are_opt_in(self):
        order = _order_with_delivery()
        assert "Numer śledzenia" not in AllegroAgent._order_bullet(order)
        with_delivery = AllegroAgent._order_bullet(order, include_delivery=True)
        assert "- Numer śledzenia: [ABC123]" in with_delivery

    def test_carrier_map_names_the_delivery_method(self):
        """The /order/carriers name wins over the order's own method.name —
        the same resolution the courier summary counts by."""
        block = AllegroAgent._order_bullet(
            _order_with_delivery(), carrier_map={"INPOST": "InPost Kurier"},
        )
        assert "- Rodzaj dostawy: InPost Kurier" in block

    def test_extra_lines_land_before_the_link(self):
        block = AllegroAgent._order_bullet(_order(), extra_lines=["E-mail: b@example.com"])
        assert block.index("- E-mail: b@example.com") < block.index("- Link: ")
