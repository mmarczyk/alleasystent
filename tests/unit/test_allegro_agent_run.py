"""Unit tests for AllegroAgent.run()'s tool loop and its empty-reply guards.

Regression cover for the production symptom "Empty reply from source=allegro_orders
— substituting fallback": the query "podaj szczegóły ostatniego nowego zamówienia"
needs two chained tool calls (list the orders → look one up by its UUID), which the
loop used to make impossible by breaking after the first round.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret")
    # Two models, so a blank reply has exactly one other model to rotate to.
    monkeypatch.setenv("GEMINI_MODEL_FAST_POOL", "model-a,model-b")


def _tool_call(call_id: str, name: str, args: dict | None = None):
    payload = json.dumps(args or {})
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = payload
    tc.model_dump.return_value = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": payload},
    }
    return tc


def _resp(text: str | None = None, tool_calls: list | None = None, finish_reason: str = "stop"):
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    resp = MagicMock()
    resp.choices = [choice]
    resp.model = "model-a"
    resp.usage = None
    return resp


def _agent(tool_results: dict[str, str] | None = None):
    """AllegroAgent with authenticated Allegro service and stubbed tool execution."""
    from agents.allegro.allegro_agent import AllegroAgent

    service = MagicMock()
    service._tokens = MagicMock()
    service._tokens.is_expired.return_value = False
    with patch("agents.allegro.allegro_agent.AllegroService.get_instance", return_value=service):
        agent = AllegroAgent(user_id="u1")

    results = tool_results or {}
    agent._execute_tool = AsyncMock(side_effect=lambda name, _input: results.get(name, f"[{name} data]"))
    return agent


class TestToolChaining:
    """One tool call is not always enough: get_order_details takes a UUID that
    only get_new_orders/get_orders can supply."""

    @pytest.mark.asyncio
    async def test_second_round_can_look_up_the_order_it_just_listed(self):
        agent = _agent({
            "get_new_orders": "- Zamówienie: 11111111-2222-3333-4444-555555555555",
            "get_order_details": "Kupujący: jan_kowalski, Wartość: 120,00 zł",
        })
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(tool_calls=[_tool_call("c1", "get_new_orders", {})]),
            _resp(tool_calls=[_tool_call("c2", "get_order_details", {"order_id": "11111111-2222-3333-4444-555555555555"})]),
            _resp(),  # no more tools — the model has what it needs
            _resp("# Szczegóły zamówienia"),
        ])

        response = await agent.run("Podaj mi szczegóły ostatniego nowego zamówienia.")

        assert response.text == "# Szczegóły zamówienia"
        assert response.metadata["tools"] == ["get_new_orders", "get_order_details"]
        assert [c.args[0] for c in agent._execute_tool.await_args_list] == [
            "get_new_orders", "get_order_details",
        ]

    @pytest.mark.asyncio
    async def test_first_round_forces_a_tool_later_rounds_do_not(self):
        agent = _agent()
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(tool_calls=[_tool_call("c1", "get_new_orders", {})]),
            _resp(),
            _resp("gotowe"),
        ])

        # "z dzisiaj" bails deterministic dispatch (see deterministic_dispatch.
        # _ORDERS_BAIL_RE) so this still goes through the LLM tool-select loop
        # this test exercises.
        await agent.run("ile mam nowych zamówień z dzisiaj?")

        choices = [
            call.kwargs.get("tool_choice")
            for call in agent._client.chat.completions.create.call_args_list
        ]
        assert choices[0] == "required"
        assert choices[1] == "auto"

    @pytest.mark.asyncio
    async def test_stops_after_max_rounds(self):
        agent = _agent()
        # A model that would keep calling tools forever: the round cap stops it.
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(tool_calls=[_tool_call(f"c{i}", "get_new_orders", {"limit": i})]) for i in range(3)
        ] + [_resp("podsumowanie")])

        # "z tego miesiąca" bails deterministic dispatch (see
        # deterministic_dispatch._ORDERS_BAIL_RE) so this still exercises the
        # LLM tool-select loop's round cap.
        response = await agent.run("pokaż zamówienia z tego miesiąca")

        assert response.text == "podsumowanie"
        assert agent._execute_tool.await_count == 3

    @pytest.mark.asyncio
    async def test_big_tool_result_skips_the_follow_up_round(self):
        """A full listing is already the answer — don't pay for a round that
        re-sends it plus every tool schema just to hear 'nothing else'."""
        from agents.allegro.allegro_agent import _CHAIN_RESULT_BUDGET

        # An English query, so the interpret call still runs (every tool renders
        # its own view now, and a Polish single-tool turn skips interpret
        # entirely — see _PASSTHROUGH_TOOLS). Without the budget guard there
        # would be a third call: the "anything else?" follow-up round.
        agent = _agent({"get_billing_summary": "x" * (_CHAIN_RESULT_BUDGET + 1)})
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(tool_calls=[_tool_call("c1", "get_billing_summary", {})]),
            _resp("| Nazwa | Stan |"),
        ])

        response = await agent.run("show me all fees from this month")

        assert response.text == "| Nazwa | Stan |"
        assert agent._client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_repeated_identical_call_reuses_the_first_result(self):
        agent = _agent({"get_new_orders": "2 zamówienia"})
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(tool_calls=[_tool_call("c1", "get_new_orders", {"limit": 5})]),
            _resp(tool_calls=[_tool_call("c2", "get_new_orders", {"limit": 5})]),
            _resp(),
            _resp("2 zamówienia"),
        ])

        await agent.run("pokaż nowe zamówienia")

        assert agent._execute_tool.await_count == 1


class TestEmptyReplyGuards:
    """Nothing the agent returns may be blank — the orchestrator turns that into
    'Przepraszam, nie udało się wygenerować odpowiedzi' for the user."""

    @pytest.mark.asyncio
    async def test_blank_interpret_retries_without_the_format_instruction(self):
        # An English query: every tool's dispatch now renders the finished view,
        # so a Polish single-tool turn skips the interpret round this test is
        # about (see _PASSTHROUGH_TOOLS / TestLatestOrderChain).
        agent = _agent({"get_billing_summary": "Brak wpisów rozliczeniowych w tym okresie."})
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(tool_calls=[_tool_call("c1", "get_billing_summary", {})]),
            _resp(),                      # tool-select round 2: no more tools needed
            _resp(),                      # interpret: blank
            _resp(),                      # interpret: blank again on the other model
            _resp("Nie udało się ustalić rozliczeń dla tego okresu."),  # plain retry, no format instruction
        ])

        response = await agent.run("what fees did I pay this month")

        assert response.text == "Nie udało się ustalić rozliczeń dla tego okresu."
        assert response.metadata["output_format"] == "chat"
        # The retry drops the format instruction that the model balked at.
        last_messages = agent._client.chat.completions.create.call_args.kwargs["messages"]
        assert "[GOTOWA ODPOWIEDŹ — PRZEPISZ BEZ ZMIAN]" not in json.dumps(last_messages, ensure_ascii=False)

    @pytest.mark.asyncio
    async def test_no_tool_call_and_no_text_falls_through_to_interpret(self):
        agent = _agent()
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(),  # round 1 ignored tool_choice=required and said nothing
            _resp(),  # ... on the other model too
            _resp("Nie mam dostępu do tych danych."),  # interpret answers anyway
        ])

        response = await agent.run("co u ciebie?")

        assert response.text == "Nie mam dostępu do tych danych."
        assert agent._execute_tool.await_count == 0

    @pytest.mark.asyncio
    async def test_text_without_a_tool_call_is_returned_as_is(self):
        agent = _agent()
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp("Nie potrzebuję do tego danych ze sklepu."),
        ])

        response = await agent.run("dzięki!")

        assert response.text == "Nie potrzebuję do tego danych ze sklepu."
        assert agent._execute_tool.await_count == 0


class TestFormatInstruction:
    """Every tool's dispatch renders the finished view (table, document,
    dashboard, bullet list) in Python — see _RENDERED_VIEW_TOOLS. So when the
    interpret call does run (English query, or several tools in one turn) it
    gets one instruction: hand the view back unchanged."""

    @pytest.mark.asyncio
    async def test_chained_turn_is_told_to_hand_the_rendered_view_back(self):
        agent = _agent()
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(tool_calls=[_tool_call("c1", "get_new_orders", {})]),
            _resp(tool_calls=[_tool_call("c2", "get_order_details", {"order_id": "x"})]),
            _resp(),
            _resp("- Zamówienie: `x`"),
        ])

        response = await agent.run("szczegóły ostatniego nowego zamówienia")

        assert response.metadata["output_format"] == "chat"
        sent = json.dumps(agent._client.chat.completions.create.call_args.kwargs["messages"], ensure_ascii=False)
        assert "[GOTOWA ODPOWIEDŹ — PRZEPISZ BEZ ZMIAN]" in sent
        # Nothing tells the model to build a view out of the data any more.
        assert "Zbuduj tabelę" not in sent

    @pytest.mark.asyncio
    async def test_english_single_tool_turn_gets_the_same_instruction(self):
        agent = _agent()
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(tool_calls=[_tool_call("c1", "get_active_offers", {})]),
            _resp(),
            _resp("# Active offers"),
        ])

        response = await agent.run("show me my active offers")

        assert response.metadata["output_format"] == "table"
        sent = json.dumps(agent._client.chat.completions.create.call_args.kwargs["messages"], ensure_ascii=False)
        assert "[GOTOWA ODPOWIEDŹ — PRZEPISZ BEZ ZMIAN]" in sent

    @pytest.mark.asyncio
    async def test_interpret_call_gets_the_short_system_prompt(self):
        """The tool-select system prompt (~17.5K chars of routing rules —
        MANDATORY TOOL CALLS, chaining, billing/monitoring routing) has
        nothing to do with formatting an already-fetched result — the
        interpret call gets AllegroAgent._interpret_system_prompt instead,
        swapped into messages[0] right before this call."""
        # English, so the interpret call runs at all (a Polish single-tool turn
        # hands the rendered view straight back — see _PASSTHROUGH_TOOLS).
        agent = _agent()
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(tool_calls=[_tool_call("c1", "get_message_threads", {})]),
            _resp(),
            _resp("- Kupujący: jan"),
        ])

        await agent.run("show me the message threads")

        # First call (tool-select) still gets the full routing prompt.
        first_system = agent._client.chat.completions.create.call_args_list[0].kwargs["messages"][0]
        assert first_system["role"] == "system"
        assert "MANDATORY TOOL CALLS" in first_system["content"]

        # Last call (interpret) gets the short one instead.
        last_system = agent._client.chat.completions.create.call_args_list[-1].kwargs["messages"][0]
        assert last_system["role"] == "system"
        assert "MANDATORY TOOL CALLS" not in last_system["content"]
        assert "TWO-STEP LOOKUPS" not in last_system["content"]
        assert "BILLING ENTRIES — CRITICAL" in last_system["content"]
        assert len(last_system["content"]) < len(first_system["content"]) / 2


class TestToolContextFilter:
    """See agents/allegro/allegro_tools.py select_tools_for_context() — most
    turns are about one topic, so the tool-select call doesn't need to see
    all ~37 schemas."""

    @pytest.mark.asyncio
    async def test_domain_query_sends_a_trimmed_tool_list(self):
        agent = _agent({"get_new_orders": "- Zamówienie: 1"})
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(tool_calls=[_tool_call("c1", "get_new_orders", {})]),
            _resp(),
            _resp("Masz jedno zamówienie."),
        ])

        # "z tego miesiąca" bails deterministic dispatch (see
        # deterministic_dispatch._ORDERS_BAIL_RE) so the call actually reaches
        # the LLM, letting this test inspect what tool schemas it was sent.
        await agent.run("jakie mam zamówienia z tego miesiąca")

        sent_tools = agent._client.chat.completions.create.call_args_list[0].kwargs["tools"]
        names = {t["function"]["name"] for t in sent_tools}
        assert "get_new_orders" in names
        assert "issue_invoice_for_order" not in names
        assert len(names) < 37

    @pytest.mark.asyncio
    async def test_unrecognized_query_falls_back_to_full_tool_list(self):
        from agents.allegro.allegro_tools import ALLEGRO_TOOLS
        agent = _agent()
        agent._client.chat.completions.create = AsyncMock(side_effect=[_resp("Nie ma za co!")])

        await agent.run("dzięki wielkie!")

        sent_tools = agent._client.chat.completions.create.call_args_list[0].kwargs["tools"]
        assert len(sent_tools) == len(ALLEGRO_TOOLS)

    @pytest.mark.asyncio
    async def test_message_followup_with_no_stem_in_context_still_gets_the_tool(self):
        """The word that makes _wants_message_content fire can be tucked
        inside punctuation ("(wiadomości)") in a way the word-prefix label
        matcher alone would miss — the explicit force-include in run() is
        what keeps get_thread_messages reachable regardless."""
        agent = _agent({"get_thread_messages": "Treść: cześć!"})
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(tool_calls=[_tool_call("c1", "get_thread_messages", {})]),
            _resp(),
            _resp("Treść: cześć!"),
        ])
        history = [
            {"role": "user", "content": "sprawdź"},
            {"role": "assistant", "content": "Masz 2 (wiadomości) od jan_kowalski."},
        ]

        await agent.run("pokaż to jeszcze raz", conversation_history=history)

        sent_tools = agent._client.chat.completions.create.call_args_list[0].kwargs["tools"]
        names = {t["function"]["name"] for t in sent_tools}
        assert "get_thread_messages" in names

    @pytest.mark.asyncio
    async def test_deterministic_dispatch_ignores_unrelated_history_topic(self):
        """Deterministic-dispatch confidence is judged from the current query
        alone, not the (history-inclusive) Layer 1 label set — an earlier
        turn about an unrelated topic must not turn an unambiguous current
        query into a false 'multi-topic, skip this layer' read."""
        agent = _agent({"get_new_orders": "- Zamówienie: 1"})
        agent._client.chat.completions.create = AsyncMock(
            side_effect=AssertionError("no LLM call expected — deterministic match")
        )
        history = [
            {"role": "user", "content": "jakie mam opłaty w tym miesiącu"},
            {"role": "assistant", "content": "Suma opłat: 42,00 zł."},
        ]

        response = await agent.run("jakie mam nowe zamówienia", conversation_history=history)

        assert response.text == "- Zamówienie: 1"
        assert agent._client.chat.completions.create.call_count == 0


class TestLatestOrderChain:
    """See AllegroAgent._resolve_latest_order_chain — "szczegóły ostatniego
    zamówienia" mechanically needs get_new_orders(limit=1) to learn which
    order before get_order_details can run. That two-hop lookup used to cost
    2-3 tool-select LLM rounds (round 1 forced, round 2+ "need more?" checks)
    even though which two tools run, and in what order, was never actually
    in doubt — this resolves it with one direct Allegro API call instead."""

    @pytest.mark.asyncio
    async def test_resolves_directly_to_order_details_when_an_order_exists(self):
        agent = _agent({"get_order_details": "- Zamówienie: abc-123"})
        agent._allegro.get_orders = AsyncMock(return_value=[MagicMock(order_id="abc-123")])
        agent._client.chat.completions.create = AsyncMock(
            side_effect=AssertionError("no LLM call expected — deterministic chain + passthrough")
        )

        response = await agent.run("szczegóły ostatniego nowego zamówienia")

        agent._allegro.get_orders.assert_awaited_once_with(
            status="READY_FOR_PROCESSING", fulfillment_status="NEW", limit=1,
        )
        agent._execute_tool.assert_awaited_once_with("get_order_details", {"order_id": "abc-123"})
        assert response.text == "- Zamówienie: abc-123"
        assert response.metadata["output_format"] == "chat"
        # Zero LLM calls at all: the chain resolver skips tool-select, and
        # get_order_details is now in _PASSTHROUGH_TOOLS (its dispatch
        # already builds the final plain-text reply) so interpret is
        # skipped too.
        assert agent._client.chat.completions.create.call_count == 0

    @pytest.mark.asyncio
    async def test_falls_back_to_bare_new_orders_when_none_exist(self):
        agent = _agent({"get_new_orders": "Brak nowych zamówień.\n\n💡 ..."})
        agent._allegro.get_orders = AsyncMock(return_value=[])
        agent._client.chat.completions.create = AsyncMock(
            side_effect=AssertionError("no LLM call expected — deterministic match")
        )

        response = await agent.run("status ostatniego nowego zamówienia")

        agent._execute_tool.assert_awaited_once_with("get_new_orders", {"limit": 1})
        assert response.text == "Brak nowych zamówień.\n\n💡 ..."
        # get_new_orders is a passthrough tool for a single, confidently
        # Polish-language call — no interpret call either.
        assert agent._client.chat.completions.create.call_count == 0

    @pytest.mark.asyncio
    async def test_lookup_error_falls_back_to_the_llm_path(self):
        agent = _agent({"get_new_orders": "- Zamówienie: 1", "get_order_details": "# Szczegóły"})
        agent._allegro.get_orders = AsyncMock(side_effect=RuntimeError("Allegro API down"))
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(tool_calls=[_tool_call("c1", "get_new_orders", {})]),
            _resp(tool_calls=[_tool_call("c2", "get_order_details", {"order_id": "x"})]),
            _resp(),
            _resp("# Szczegóły zamówienia x"),
        ])

        response = await agent.run("szczegóły ostatniego nowego zamówienia")

        assert response.text == "# Szczegóły zamówienia x"
        assert agent._client.chat.completions.create.call_count == 4

    @pytest.mark.asyncio
    async def test_plain_listing_query_does_not_trigger_the_chain(self):
        """No detail-intent word — must not call the extra lookup at all."""
        agent = _agent({"get_new_orders": "- Zamówienie: 1"})
        agent._allegro.get_orders = AsyncMock(
            side_effect=AssertionError("chain lookup should not run for a plain listing")
        )
        agent._client.chat.completions.create = AsyncMock(
            side_effect=AssertionError("no LLM call expected — deterministic match")
        )

        response = await agent.run("jakie mam nowe zamówienia")

        assert response.text == "- Zamówienie: 1"
