"""Unit tests for the Allegro billing fetch behind the sales summary.

Two bugs are pinned here. A yearly summary asked for ~14 months of billing in
one long chain of pages, so a single 429 or dropped connection anywhere in that
chain lost the whole cost section — and whatever the real failure was, the
report always blamed a missing billing permission and told the seller to log in
again, which never helped.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.allegro.allegro_agent import AllegroAgent
from models.allegro import AllegroOrder, AllegroOrderLine
from services.allegro_service import (
    _BILLING_WINDOW_DAYS,
    AllegroAPIError,
    AllegroService,
)


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("REDIS_URL", raising=False)


def _service(responses: list) -> tuple[AllegroService, MagicMock]:
    """Service whose GET yields each item of `responses` in turn.

    A dict is served as a 200 body; an exception is raised instead.
    """
    service = AllegroService.__new__(AllegroService)
    service._get_headers = AsyncMock(  # type: ignore[method-assign]
        return_value={"Authorization": "Bearer test"}
    )

    effects = []
    for item in responses:
        if isinstance(item, BaseException):
            effects.append(item)
            continue
        response = MagicMock(status_code=200)
        response.json.return_value = item
        effects.append(response)

    client = MagicMock()
    client.get = AsyncMock(side_effect=effects)
    service._client = client  # type: ignore[attr-defined]
    return service, client


def _entries(count: int, start_id: int = 0) -> list[dict]:
    return [
        {"id": f"b{start_id + i}", "occurredAt": "2026-01-10T10:00:00Z",
         "value": {"amount": "-1.00", "currency": "PLN"}}
        for i in range(count)
    ]


class TestPeriodSplitting:
    def test_a_year_is_split_into_month_sized_windows(self):
        windows = AllegroService._split_period(
            "2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z", _BILLING_WINDOW_DAYS
        )

        assert len(windows) == 12
        assert windows[0][0] == "2025-01-01T00:00:00Z"
        assert windows[-1][1] == "2026-01-01T00:00:00Z"

    def test_windows_are_contiguous_so_no_day_falls_between_them(self):
        windows = AllegroService._split_period(
            "2025-01-01T00:00:00Z", "2025-06-01T00:00:00Z", _BILLING_WINDOW_DAYS
        )

        assert all(a[1] == b[0] for a, b in zip(windows, windows[1:]))

    def test_a_short_period_stays_one_window(self):
        windows = AllegroService._split_period(
            "2026-03-01T00:00:00Z", "2026-03-31T22:59:59Z", _BILLING_WINDOW_DAYS
        )

        assert windows == [("2026-03-01T00:00:00Z", "2026-03-31T22:59:59Z")]


class TestBillingFetch:
    async def test_every_window_of_a_long_period_is_fetched(self):
        # 3 windows over ~3 months, one short page each.
        service, client = _service([{"billingEntries": _entries(2, start_id=i * 10)} for i in range(3)])

        entries = await service.get_billing_entries_in_period(
            "2026-01-01T00:00:00Z", "2026-03-25T00:00:00Z"
        )

        assert len(entries) == 6
        assert client.get.await_count == 3

    async def test_a_full_page_is_followed_by_the_next_one(self):
        service, client = _service([
            {"billingEntries": _entries(100)},
            {"billingEntries": _entries(7, start_id=100)},
        ])

        entries = await service.get_billing_entries_in_period(
            "2026-03-01T00:00:00Z", "2026-03-20T00:00:00Z"
        )

        assert len(entries) == 107
        assert client.get.await_args_list[1].kwargs["params"]["offset"] == 100

    async def test_an_entry_on_a_window_boundary_is_counted_once(self):
        """Both bounds are inclusive, so consecutive windows overlap by one
        instant — the entry there must not be double-counted as a second fee."""
        shared = {"id": "shared", "occurredAt": "2026-02-01T00:00:00Z",
                  "value": {"amount": "-9.99", "currency": "PLN"}}
        service, _ = _service([
            {"billingEntries": [shared]},
            {"billingEntries": [shared, *_entries(1, start_id=50)]},
        ])

        entries = await service.get_billing_entries_in_period(
            "2026-01-01T00:00:00Z", "2026-02-20T00:00:00Z"
        )

        assert [e["id"] for e in entries] == ["shared", "b50"]

    async def test_a_rate_limited_page_is_retried_instead_of_losing_the_period(self):
        service, client = _service([
            AllegroAPIError(429, "Too Many Requests"),
            {"billingEntries": _entries(3)},
        ])

        with patch("services.allegro_service.asyncio.sleep", new=AsyncMock()):
            entries = await service.get_billing_entries_in_period(
                "2026-03-01T00:00:00Z", "2026-03-10T00:00:00Z"
            )

        assert len(entries) == 3
        assert client.get.await_count == 2

    async def test_a_permission_error_is_not_retried(self):
        service, client = _service([AllegroAPIError(403, "Forbidden")] * 3)

        with patch("services.allegro_service.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(AllegroAPIError) as exc_info:
                await service.get_billing_entries_in_period(
                    "2026-03-01T00:00:00Z", "2026-03-10T00:00:00Z"
                )

        assert exc_info.value.status_code == 403
        assert client.get.await_count == 1


def _make_agent():
    with patch("agents.base_agent.AsyncOpenAI"), \
         patch("agents.allegro.allegro_agent.AllegroService") as MockService:
        mock_service = MagicMock()
        mock_service._tokens = MagicMock()
        mock_service._tokens.is_expired.return_value = False
        MockService.get_instance.return_value = mock_service
        return AllegroAgent()


def _order() -> AllegroOrder:
    return AllegroOrder(
        order_id="a",
        buyer_login="jan",
        status="READY_FOR_PROCESSING",
        fulfillment_status="SENT",
        total_price=100.0,
        currency="PLN",
        created_at="2026-01-10T10:00:00Z",
        paid_at="2026-01-10T10:00:00Z",
        line_items=[AllegroOrderLine(offer_id="1", offer_name="Włóczka", quantity=1, price=100.0)],
    )


def _agent_failing_billing(exc: BaseException) -> AllegroAgent:
    agent = _make_agent()
    agent._allegro.get_all_paid_orders_in_period = AsyncMock(return_value=[_order()])
    agent._allegro.get_billing_entries_in_period = AsyncMock(side_effect=exc)
    return agent


class TestBillingErrorMessage:
    async def test_a_403_asks_the_seller_to_log_in_again(self):
        agent = _agent_failing_billing(AllegroAPIError(403, "Forbidden"))

        result = await agent._dispatch(
            "get_sales_summary", {"date_from_local": "2026-01-01", "date_to_local": "2026-12-31"}
        )

        assert "allegro:api:billing:read" in result
        assert "/allegro/login" in result

    async def test_a_rate_limit_says_so_instead_of_blaming_permissions(self):
        """The old text blamed a missing permission for every failure — advice
        that can't work when the token is fine and Allegro is just throttling."""
        agent = _agent_failing_billing(AllegroAPIError(429, "Too Many Requests"))

        result = await agent._dispatch(
            "get_sales_summary", {"date_from_local": "2026-01-01", "date_to_local": "2026-12-31"}
        )

        assert "429" in result
        assert "uprawnie" not in result
        assert "/allegro/login" not in result

    async def test_a_dropped_connection_says_so(self):
        agent = _agent_failing_billing(AllegroAPIError(0, "Network error: timeout"))

        result = await agent._dispatch(
            "get_sales_summary", {"date_from_local": "2026-01-01", "date_to_local": "2026-12-31"}
        )

        assert "połączyć" in result
        assert "/allegro/login" not in result

    async def test_the_rest_of_the_summary_survives_a_billing_failure(self):
        agent = _agent_failing_billing(AllegroAPIError(500, "Internal Server Error"))

        result = await agent._dispatch(
            "get_sales_summary", {"date_from_local": "2026-01-01", "date_to_local": "2026-12-31"}
        )

        assert "łączny przychód: **100,00 PLN**" in result
        assert "Top produkty wg przychodu" in result

    async def test_billing_summary_reports_a_server_error_instead_of_raising(self):
        agent = _make_agent()
        agent._allegro.get_billing_entries_in_period = AsyncMock(
            side_effect=AllegroAPIError(503, "Service Unavailable")
        )

        result = await agent._dispatch(
            "get_billing_summary", {"date_from_local": "2026-01-01", "date_to_local": "2026-01-31"}
        )

        assert "503" in result
