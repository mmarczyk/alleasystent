"""Unit tests for the period filter on returns/complaints queries.

"ile miałem zwrotów w tym miesiącu" used to answer with the unfiltered page
size (50) because nothing along the path — deterministic dispatch, the tool
schema, the service call — carried a date window. These tests pin each layer.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.allegro.allegro_agent import AllegroAgent
from agents.allegro.allegro_tools import ALLEGRO_TOOLS, matched_labels
from agents.allegro.deterministic_dispatch import resolve_deterministic
from services.allegro_service import AllegroAPIError, AllegroService

BETA = "application/vnd.allegro.beta.v1+json"


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("REDIS_URL", raising=False)


def _service(pages: list[dict] | dict) -> tuple[AllegroService, MagicMock]:
    """Build a service whose GET returns each payload in `pages` in turn."""
    payloads = pages if isinstance(pages, list) else [pages]
    service = AllegroService.__new__(AllegroService)
    service._get_headers = AsyncMock(  # type: ignore[method-assign]
        return_value={"Authorization": "Bearer test", "Accept": "application/vnd.allegro.public.v1+json"}
    )

    responses = []
    for payload in payloads:
        response = MagicMock(status_code=200)
        response.json.return_value = payload
        responses.append(response)

    client = MagicMock()
    client.get = AsyncMock(side_effect=responses)
    service._client = client  # type: ignore[attr-defined]
    return service, client


def _returns(count: int, month: str, start_id: int = 0) -> list[dict]:
    """`count` returns created on consecutive days of `month` ('YYYY-MM')."""
    return [
        {"id": f"r{start_id + i}", "createdAt": f"{month}-{(i % 28) + 1:02d}T10:00:00Z"}
        for i in range(count)
    ]


class TestServicePeriodFilter:
    async def test_period_pages_through_and_counts_every_page(self):
        """The bug: one 50-row page was reported as the total. A period query
        must exhaust the listing, so the count is the real one."""
        service, client = _service([
            {"customerReturns": _returns(50, "2026-08")},
            {"customerReturns": _returns(12, "2026-08", start_id=50)},
        ])

        result = await service.get_customer_returns(
            date_from="2026-08-01T00:00:00Z", date_to="2026-08-28T21:59:59Z"
        )

        assert len(result) == 62
        assert client.get.call_count == 2
        first, second = client.get.call_args_list
        assert first.kwargs["params"]["offset"] == 0
        assert second.kwargs["params"]["offset"] == 50

    async def test_period_is_sent_server_side(self):
        service, client = _service({"customerReturns": []})

        await service.get_customer_returns(
            date_from="2026-08-01T00:00:00Z", date_to="2026-08-28T21:59:59Z"
        )

        params = client.get.call_args.kwargs["params"]
        assert params["createdAt.gte"] == "2026-08-01T00:00:00Z"
        assert params["createdAt.lte"] == "2026-08-28T21:59:59Z"
        assert client.get.call_args.kwargs["headers"]["Accept"] == BETA

    async def test_out_of_period_returns_are_dropped_client_side(self):
        """March returns answered an August question — the exact symptom the
        user reported. The client-side filter is the backstop for a server
        that ignores (or doesn't support) the createdAt params."""
        service, _ = _service({
            "customerReturns": [
                {"id": "march", "createdAt": "2026-03-07T10:43:00Z"},
                {"id": "august", "createdAt": "2026-08-14T09:00:00Z"},
            ]
        })

        result = await service.get_customer_returns(
            date_from="2026-08-01T00:00:00Z", date_to="2026-08-28T21:59:59Z"
        )

        assert [item["id"] for item in result] == ["august"]

    async def test_non_utc_offsets_compare_as_instants(self):
        service, _ = _service({
            "customerReturns": [{"id": "local", "createdAt": "2026-08-14T11:00:00+02:00"}]
        })

        result = await service.get_customer_returns(
            date_from="2026-08-01T00:00:00Z", date_to="2026-08-28T21:59:59Z"
        )

        assert [item["id"] for item in result] == ["local"]

    async def test_undated_returns_are_excluded_from_a_period(self):
        service, _ = _service({"customerReturns": [{"id": "no-date"}]})

        result = await service.get_customer_returns(
            date_from="2026-08-01T00:00:00Z", date_to="2026-08-28T21:59:59Z"
        )

        assert result == []

    async def test_reception_created_at_is_honoured(self):
        """createdAt isn't guaranteed on this beta resource — the same fallback
        keys _return_bullet displays are the ones the filter reads."""
        service, _ = _service({
            "customerReturns": [{"id": "r1", "reception": {"createdAt": "2026-08-14T09:00:00Z"}}]
        })

        result = await service.get_customer_returns(
            date_from="2026-08-01T00:00:00Z", date_to="2026-08-28T21:59:59Z"
        )

        assert [item["id"] for item in result] == ["r1"]

    async def test_status_is_still_sent_alongside_the_period(self):
        service, client = _service({"customerReturns": []})

        await service.get_customer_returns(
            status="DELIVERED", date_from="2026-08-01T00:00:00Z", date_to="2026-08-28T21:59:59Z"
        )

        assert client.get.call_args.kwargs["params"]["status"] == "DELIVERED"

    async def test_falls_back_to_client_side_filtering_on_400(self):
        """customer-returns is beta; if it rejects createdAt.* the query still
        has to be answered, just filtered here instead of there."""
        service, client = _service({})
        rejected = MagicMock(status_code=400, text="unsupported parameter")
        accepted = MagicMock(status_code=200)
        accepted.json.return_value = {
            "customerReturns": [
                {"id": "march", "createdAt": "2026-03-07T10:43:00Z"},
                {"id": "august", "createdAt": "2026-08-14T09:00:00Z"},
            ]
        }
        client.get = AsyncMock(side_effect=[rejected, accepted])

        result = await service.get_customer_returns(
            date_from="2026-08-01T00:00:00Z", date_to="2026-08-28T21:59:59Z"
        )

        assert [item["id"] for item in result] == ["august"]
        assert "createdAt.gte" not in client.get.call_args.kwargs["params"]

    async def test_non_400_errors_still_propagate(self):
        service, client = _service({})
        client.get = AsyncMock(return_value=MagicMock(status_code=500, text="boom"))

        with pytest.raises(AllegroAPIError):
            await service.get_customer_returns(
                date_from="2026-08-01T00:00:00Z", date_to="2026-08-28T21:59:59Z"
            )

    async def test_no_period_stays_a_single_page(self):
        """The returns monitor and the plain 'nowe zwroty' listing must keep
        their cheap one-page call — no pagination, no date params."""
        service, client = _service({"customerReturns": _returns(50, "2026-08")})

        result = await service.get_customer_returns(limit=50)

        assert len(result) == 50
        assert client.get.call_count == 1
        assert client.get.call_args.kwargs["params"] == {"limit": 50}


class TestIssuesPeriodFilter:
    async def test_period_filters_issues_client_side(self):
        """/sale/issues has no documented date filter, so the window is applied
        to the paged listing here."""
        service, client = _service({
            "issues": [
                {"id": "old", "openedDate": "2026-03-02T08:00:00Z"},
                {"id": "recent", "openedDate": "2026-08-20T08:00:00Z"},
            ]
        })

        result = await service.get_issues(
            date_from="2026-08-01T00:00:00Z", date_to="2026-08-28T21:59:59Z"
        )

        assert [item["id"] for item in result] == ["recent"]
        assert "createdAt.gte" not in client.get.call_args.kwargs["params"]

    async def test_no_period_stays_a_single_page(self):
        service, client = _service({"issues": [{"id": "i1"}]})

        assert await service.get_issues(limit=50) == [{"id": "i1"}]
        assert client.get.call_count == 1


def _agent() -> AllegroAgent:
    agent = AllegroAgent.__new__(AllegroAgent)
    agent._allegro = MagicMock()
    agent._returns_monitoring_status_block = AsyncMock(return_value="[monitoring]")
    return agent


class TestAgentPeriodHandling:
    async def test_count_only_passes_the_window_and_labels_it(self):
        agent = _agent()
        agent._allegro.get_customer_returns = AsyncMock(return_value=_returns(3, "2026-08"))

        text = await AllegroAgent._dispatch(
            agent,
            "get_new_returns",
            {"count_only": True, "date_from_local": "2026-08-01", "date_to_local": "2026-08-28"},
        )

        assert "Liczba zwrotów (2026-08-01 – 2026-08-28): 3." in text
        kwargs = agent._allegro.get_customer_returns.call_args.kwargs
        # Warsaw local midnight, not UTC midnight — August is UTC+2.
        assert kwargs["date_from"] == "2026-07-31T22:00:00Z"
        assert kwargs["date_to"] == "2026-08-28T21:59:59Z"

    async def test_returns_to_process_period_keeps_the_delivered_status(self):
        agent = _agent()
        agent._allegro.get_customer_returns = AsyncMock(return_value=[])

        text = await AllegroAgent._dispatch(
            agent,
            "get_returns_to_process",
            {"count_only": True, "date_from_local": "2026-08-01", "date_to_local": "2026-08-28"},
        )

        assert "Liczba zwrotów do obsłużenia (2026-08-01 – 2026-08-28): 0." in text
        assert agent._allegro.get_customer_returns.call_args.kwargs["status"] == "DELIVERED"

    async def test_complaints_period_is_forwarded(self):
        agent = _agent()
        agent._allegro.get_issues = AsyncMock(return_value=[{"id": "i1", "openedDate": "2026-08-02T08:00:00Z"}])

        text = await AllegroAgent._dispatch(
            agent,
            "get_new_complaints",
            {"count_only": True, "date_from_local": "2026-08-01", "date_to_local": "2026-08-28"},
        )

        assert "Liczba reklamacji (2026-08-01 – 2026-08-28): 1." in text
        assert agent._allegro.get_issues.call_args.kwargs["date_from"] == "2026-07-31T22:00:00Z"

    async def test_no_period_keeps_the_old_wording(self):
        agent = _agent()
        agent._allegro.get_customer_returns = AsyncMock(return_value=_returns(2, "2026-08"))

        text = await AllegroAgent._dispatch(agent, "get_new_returns", {"count_only": True})

        assert text.startswith("Liczba zwrotów: 2.")
        kwargs = agent._allegro.get_customer_returns.call_args.kwargs
        assert kwargs["date_from"] is None and kwargs["date_to"] is None

    async def test_malformed_period_degrades_to_no_period(self):
        agent = _agent()
        agent._allegro.get_customer_returns = AsyncMock(return_value=[])

        text = await AllegroAgent._dispatch(
            agent,
            "get_new_returns",
            {"count_only": True, "date_from_local": "w tym miesiącu", "date_to_local": "dzisiaj"},
        )

        assert text.startswith("Liczba zwrotów: 0.")
        assert agent._allegro.get_customer_returns.call_args.kwargs["date_from"] is None

    async def test_listing_is_newest_first_with_a_count_header(self):
        agent = _agent()
        agent._allegro.get_customer_returns = AsyncMock(return_value=[
            {"id": "older", "createdAt": "2026-08-02T10:00:00Z"},
            {"id": "newer", "createdAt": "2026-08-20T10:00:00Z"},
        ])

        text = await AllegroAgent._dispatch(
            agent, "get_new_returns", {"date_from_local": "2026-08-01", "date_to_local": "2026-08-28"}
        )

        assert "**Zwroty (2026-08-01 – 2026-08-28)** — 2" in text
        assert text.index("newer") < text.index("older")

    async def test_long_listing_is_capped_with_a_remainder_note(self):
        agent = _agent()
        agent._allegro.get_customer_returns = AsyncMock(return_value=_returns(62, "2026-08"))

        text = await AllegroAgent._dispatch(
            agent, "get_new_returns", {"date_from_local": "2026-08-01", "date_to_local": "2026-08-28"}
        )

        assert "— 62" in text
        assert "…i 12 więcej." in text


class TestPeriodQueriesReachTheLLM:
    """The deterministic layer has no clock, so a period query must fall
    through to the LLM instead of answering without the filter."""

    @pytest.mark.parametrize("query", [
        "a powiedz mi ile mialem zwrotów w tym miesiacu",
        "ile zwrotów miałem w marcu",
        "czy mam jakieś zwroty z dzisiaj",
        "ile zwrotów do obsłużenia w tym tygodniu",
        "ile reklamacji miałem w zeszłym miesiącu",
    ])
    def test_period_query_falls_through(self, query):
        assert resolve_deterministic(query, matched_labels(query)) is None

    @pytest.mark.parametrize("query,expected", [
        ("czy mam jakieś zwroty", ("get_new_returns", {"count_only": True})),
        ("zwroty do obsłużenia", ("get_returns_to_process", {})),
        ("ile mam reklamacji", ("get_new_complaints", {"count_only": True})),
    ])
    def test_dateless_queries_still_resolve_deterministically(self, query, expected):
        assert resolve_deterministic(query, matched_labels(query)) == expected


class TestToolSchemas:
    @pytest.mark.parametrize("tool_name", ["get_new_returns", "get_returns_to_process", "get_new_complaints"])
    def test_period_parameters_are_exposed(self, tool_name):
        tool = next(t for t in ALLEGRO_TOOLS if t["function"]["name"] == tool_name)
        props = tool["function"]["parameters"]["properties"]
        assert "date_from_local" in props and "date_to_local" in props
        assert tool["function"]["parameters"].get("required") in (None, [])
