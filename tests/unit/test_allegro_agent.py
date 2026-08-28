"""Unit tests for the get_new_orders interpret-bypass optimization in
agents/allegro/allegro_agent.py — see AllegroAgent.run() for the rationale:
get_new_orders' dispatch output already matches _TOOL_SPECIFIC_INSTRUCTIONS
verbatim for a Polish query, so the interpret LLM call is pure passthrough
and can be skipped entirely.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.allegro.allegro_agent import _is_confidently_polish


class TestIsConfidentlyPolish:
    def test_polish_diacritics_detected(self):
        assert _is_confidently_polish("jakie mam nowe zamówienia")
        assert _is_confidently_polish("pokaż moje oferty")

    def test_ascii_only_english_not_detected(self):
        assert not _is_confidently_polish("show me my new orders")

    def test_ascii_only_polish_not_detected(self):
        """Conservative by design: missing this optimization is fine, wrongly
        assuming Polish and skipping translation for a real English query is not."""
        assert not _is_confidently_polish("ile mam zamowien")


def _tool_call(name: str, arguments: dict) -> MagicMock:
    tc = MagicMock()
    tc.id = "c1"
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    tc.model_dump.return_value = {
        "id": "c1",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return tc


def _tool_select_response(tool_calls: list) -> MagicMock:
    msg = MagicMock()
    msg.tool_calls = tool_calls
    msg.content = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _no_more_tools_response() -> MagicMock:
    """The chained tool loop always offers a second, tool_choice='auto' round
    after the first — this is what the model returns to end it there."""
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


def _make_agent():
    from agents.allegro.allegro_agent import AllegroAgent

    with patch("agents.base_agent.AsyncOpenAI"), \
         patch("agents.allegro.allegro_agent.AllegroService") as MockService:
        mock_service = MagicMock()
        mock_service._tokens = MagicMock()
        mock_service._tokens.is_expired.return_value = False
        MockService.get_instance.return_value = mock_service
        agent = AllegroAgent()
    return agent


class TestGetNewOrdersInterpretBypass:
    @pytest.mark.asyncio
    async def test_polish_query_skips_interpret_call(self):
        # "jakie mam nowe zamówienia" is also a deterministic-dispatch match
        # (see agents/allegro/deterministic_dispatch.py) — the tool-select
        # LLM call is skipped too, on top of the interpret bypass this class
        # is about, so call_count is 0 rather than needing a mocked response.
        agent = _make_agent()
        agent._client.chat.completions.create = AsyncMock(
            side_effect=AssertionError("no LLM call expected — deterministic match")
        )

        raw_text = "**Zamówienie** `X`\n- Zamawiający: **jan_kowalski**"
        agent._dispatch = AsyncMock(return_value=raw_text)

        response = await agent.run("jakie mam nowe zamówienia")

        assert response.text == raw_text
        assert agent._client.chat.completions.create.call_count == 0
        assert "interpret_llm" not in response.metadata["perf_stages"]
        assert response.metadata["tools"] == ["get_new_orders"]

    @pytest.mark.asyncio
    async def test_english_query_still_uses_interpret_call(self):
        agent = _make_agent()
        tool_select_resp = _tool_select_response([_tool_call("get_new_orders", {})])

        interp_msg = MagicMock()
        interp_msg.tool_calls = None
        interp_msg.content = "**Order** `X`\n- Buyer: **jan_kowalski**"
        interp_choice = MagicMock()
        interp_choice.message = interp_msg
        interp_resp = MagicMock()
        interp_resp.choices = [interp_choice]

        agent._client.chat.completions.create = AsyncMock(
            side_effect=[tool_select_resp, _no_more_tools_response(), interp_resp]
        )
        agent._dispatch = AsyncMock(
            return_value="**Zamówienie** `X`\n- Zamawiający: **jan_kowalski**"
        )

        response = await agent.run("show me my new orders")

        assert response.text == "**Order** `X`\n- Buyer: **jan_kowalski**"
        assert agent._client.chat.completions.create.call_count == 3
        assert "interpret_llm" in response.metadata["perf_stages"]

    @pytest.mark.asyncio
    async def test_count_only_still_uses_interpret_call(self):
        """count_only's raw dispatch text ("Liczba nowych zamówień: N.") differs
        in wording from what the tool instruction asks for ("Masz N nowych
        zamówień.") — real transformation, not passthrough, so it must not
        bypass even for a Polish query."""
        agent = _make_agent()
        tool_select_resp = _tool_select_response(
            [_tool_call("get_new_orders", {"count_only": True})]
        )

        interp_msg = MagicMock()
        interp_msg.tool_calls = None
        interp_msg.content = "Masz 3 nowe zamówienia."
        interp_choice = MagicMock()
        interp_choice.message = interp_msg
        interp_resp = MagicMock()
        interp_resp.choices = [interp_choice]

        agent._client.chat.completions.create = AsyncMock(
            side_effect=[tool_select_resp, _no_more_tools_response(), interp_resp]
        )
        agent._dispatch = AsyncMock(return_value="Liczba nowych zamówień: 3.")

        response = await agent.run("ile mam nowych zamówień")

        assert response.text == "Masz 3 nowe zamówienia."
        assert agent._client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_multi_tool_turn_still_uses_interpret_call(self):
        """A turn calling get_new_orders alongside another tool still needs the
        LLM to weave the results together — passthrough only applies when it
        was the ONLY tool called."""
        agent = _make_agent()
        tool_select_resp = _tool_select_response([
            _tool_call("get_new_orders", {}),
            _tool_call("get_account_info", {}),
        ])

        interp_msg = MagicMock()
        interp_msg.tool_calls = None
        interp_msg.content = "combined answer"
        interp_choice = MagicMock()
        interp_choice.message = interp_msg
        interp_resp = MagicMock()
        interp_resp.choices = [interp_choice]

        agent._client.chat.completions.create = AsyncMock(
            side_effect=[tool_select_resp, _no_more_tools_response(), interp_resp]
        )
        agent._dispatch = AsyncMock(side_effect=["orders text", "account text"])

        response = await agent.run("nowe zamówienia i moje konto")

        assert response.text == "combined answer"
        assert agent._client.chat.completions.create.call_count == 3


class TestPassthroughGeneralizedToOtherTools:
    """The get_new_orders bypass generalizes to every tool in
    _PASSTHROUGH_TOOLS — spot-check a list tool with no dedicated
    _TOOL_SPECIFIC_INSTRUCTIONS entry at all (unlike get_new_orders, so
    count_only bypasses too) and an action-toggle tool, and confirm a tool
    NOT in the set (get_message_threads — its raw pipe-delimited dispatch
    output genuinely differs from its dedicated instruction) still uses the
    interpret call."""

    @pytest.mark.asyncio
    async def test_get_new_returns_count_only_also_bypasses(self):
        # Also a deterministic-dispatch match — see comment on
        # test_polish_query_skips_interpret_call in test_allegro_agent.py.
        agent = _make_agent()
        agent._client.chat.completions.create = AsyncMock(
            side_effect=AssertionError("no LLM call expected — deterministic match")
        )
        agent._dispatch = AsyncMock(return_value="Liczba zwrotów: 2.")

        response = await agent.run("czy mam jakieś zwroty")

        assert response.text == "Liczba zwrotów: 2."
        assert agent._client.chat.completions.create.call_count == 0

    @pytest.mark.asyncio
    async def test_monitoring_toggle_bypasses(self):
        # Also a deterministic-dispatch match — see comment above.
        agent = _make_agent()
        agent._client.chat.completions.create = AsyncMock(
            side_effect=AssertionError("no LLM call expected — deterministic match")
        )
        status_block = (
            "💡 Mogę automatycznie sprawdzać nowe zamówienia...\n\n"
            '<button onclick="OrderMonitor.enable()">🔔 Włącz</button>'
        )
        agent._dispatch = AsyncMock(return_value=status_block)

        response = await agent.run("włącz monitoring zamówień")

        assert response.text == status_block
        assert agent._client.chat.completions.create.call_count == 0

    @pytest.mark.asyncio
    async def test_tool_not_in_passthrough_set_still_uses_interpret_call(self):
        agent = _make_agent()
        tool_select_resp = _tool_select_response(
            [_tool_call("get_message_threads", {})]
        )

        interp_msg = MagicMock()
        interp_msg.tool_calls = None
        interp_msg.content = "- **jan_kowalski** — nieprzeczytana"
        interp_choice = MagicMock()
        interp_choice.message = interp_msg
        interp_resp = MagicMock()
        interp_resp.choices = [interp_choice]

        agent._client.chat.completions.create = AsyncMock(
            side_effect=[tool_select_resp, _no_more_tools_response(), interp_resp]
        )
        agent._dispatch = AsyncMock(
            return_value="Thread t1 | Buyer: jan_kowalski | Unread: True | Last message: 2026-08-26"
        )

        response = await agent.run("pokaż wiadomości")

        assert response.text == "- **jan_kowalski** — nieprzeczytana"
        assert agent._client.chat.completions.create.call_count == 3


class TestGetOrderDetailsDispatch:
    """_dispatch's get_order_details branch now builds the final, ready-to-
    display plain-text bullet list directly in Python instead of handing
    raw fields to the interpret LLM — _TOOL_SPECIFIC_INSTRUCTIONS["get_order_details"]
    already fully prescribed the shape, so there was no real judgment left
    for the LLM to apply. See _PASSTHROUGH_TOOLS."""

    def _make_order(self, **overrides):
        from models.allegro import AllegroOrder, AllegroOrderLine
        defaults = dict(
            order_id="abc-123",
            buyer_login="jan_kowalski",
            buyer_email="jan@example.com",
            status="BOUGHT",
            fulfillment_status="READY_FOR_PROCESSING",
            total_price=189.98,
            currency="PLN",
            created_at="2026-08-27T10:15:00Z",
            paid_at="2026-08-27T10:20:00Z",
            delivery={"method": {"name": "InPost Paczkomaty"}, "trackingCode": "PL123456789"},
            line_items=[
                AllegroOrderLine(offer_id="111", offer_name="Sweter wełniany M", quantity=1, price=129.99),
                AllegroOrderLine(offer_id="222", offer_name="Skarpety wełniane 3-pak", quantity=1, price=59.99),
            ],
            invoice_required=False,
        )
        defaults.update(overrides)
        return AllegroOrder(**defaults)

    def _make_agent_with_order(self, order, billing_entries=None, existing_invoices=None):
        agent = _make_agent()
        agent._allegro.get_order = AsyncMock(return_value=order)
        agent._allegro.get_billing_entries_for_order = AsyncMock(return_value=billing_entries or [])
        agent._allegro.get_order_invoices = AsyncMock(return_value=existing_invoices or [])
        return agent

    @pytest.mark.asyncio
    async def test_plain_text_bullet_list_no_headers_or_code_fences(self):
        order = self._make_order()
        agent = self._make_agent_with_order(order)

        result = await agent._dispatch("get_order_details", {"order_id": "abc-123"})

        assert "```" not in result
        assert "#" not in result
        assert "- Zamówienie: `abc-123`" in result
        assert "- Kupujący: jan_kowalski" in result
        assert "- Wartość: 189,98 PLN" in result
        assert "- Produkty:" in result
        assert "  - Sweter wełniany M (ID: 111): 1 × 129,99 PLN" in result
        assert "- Dostawa:" in result
        assert "  - Metoda: InPost Paczkomaty" in result
        assert "  - Tracking: PL123456789" in result

    @pytest.mark.asyncio
    async def test_no_billing_entries_omits_billing_section(self):
        order = self._make_order()
        agent = self._make_agent_with_order(order, billing_entries=[])

        result = await agent._dispatch("get_order_details", {"order_id": "abc-123"})

        assert "Rozliczenie" not in result

    @pytest.mark.asyncio
    async def test_every_billing_entry_shown_separately(self):
        order = self._make_order()
        billing_entries = [
            {"value": {"amount": "-12.99"}, "type": {"description": "Prowizja od sprzedaży"}, "offer": {"name": "Sweter"}, "occurredAt": "2026-08-27T00:00:00Z"},
            {"value": {"amount": "-6.00"}, "type": {"description": "Prowizja od sprzedaży"}, "offer": {"name": "Skarpety"}, "occurredAt": "2026-08-27T00:00:00Z"},
            {"value": {"amount": "-1.00"}, "type": {"description": "Opłata za wystawienie oferty"}, "occurredAt": "2026-08-27T00:00:00Z"},
        ]
        agent = self._make_agent_with_order(order, billing_entries=billing_entries)

        result = await agent._dispatch("get_order_details", {"order_id": "abc-123"})

        assert "- Rozliczenie:" in result
        assert result.count("Prowizja od sprzedaży") == 2
        assert "Opłata za wystawienie oferty" in result
        assert "Suma opłat: -19.99 PLN" in result
        assert "Zysk netto: 169.99 PLN" in result

    @pytest.mark.asyncio
    async def test_invoice_status_variants(self):
        cases = [
            (dict(invoice_required=False), [], "Kupujący nie poprosił o fakturę."),
            (dict(invoice_required=True), [{"id": "inv-1"}], "faktura już wystawiona."),
            (dict(invoice_required=True), [], "NIE WYSTAWIONO jeszcze faktury."),
        ]
        for order_overrides, existing_invoices, expected_substr in cases:
            order = self._make_order(**order_overrides)
            agent = self._make_agent_with_order(order, existing_invoices=existing_invoices)

            result = await agent._dispatch("get_order_details", {"order_id": "abc-123"})

            assert expected_substr in result, result
