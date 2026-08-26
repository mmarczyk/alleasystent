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

        await agent.run("ile mam nowych zamówień?")

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

        response = await agent.run("pokaż nowe zamówienia")

        assert response.text == "podsumowanie"
        assert agent._execute_tool.await_count == 3

    @pytest.mark.asyncio
    async def test_big_tool_result_skips_the_follow_up_round(self):
        """A full listing is already the answer — don't pay for a round that
        re-sends it plus every tool schema just to hear 'nothing else'."""
        from agents.allegro.allegro_agent import _CHAIN_RESULT_BUDGET

        agent = _agent({"get_active_offers": "x" * (_CHAIN_RESULT_BUDGET + 1)})
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(tool_calls=[_tool_call("c1", "get_active_offers", {})]),
            _resp("| Nazwa | Stan |"),
        ])

        response = await agent.run("pokaż wszystkie oferty")

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
        agent = _agent({"get_order_details": "Nie znaleziono zamówienia o podanym ID."})
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(tool_calls=[_tool_call("c1", "get_order_details", {"order_id": "x"})]),
            _resp(),                      # tool-select round 2: no more tools needed
            _resp(),                      # interpret: blank
            _resp(),                      # interpret: blank again on the other model
            _resp("Nie udało się znaleźć tego zamówienia."),  # plain retry, no format instruction
        ])

        response = await agent.run("szczegóły zamówienia x")

        assert response.text == "Nie udało się znaleźć tego zamówienia."
        assert response.metadata["output_format"] == "chat"
        # The retry drops the format instruction that the model balked at.
        last_messages = agent._client.chat.completions.create.call_args.kwargs["messages"]
        assert "[FORMAT ODPOWIEDZI: PODSUMOWANIE + DOKUMENT]" not in json.dumps(last_messages, ensure_ascii=False)

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
    @pytest.mark.asyncio
    async def test_chain_uses_the_instruction_of_the_tool_that_answered(self):
        """get_new_orders asks for a bullet list, get_order_details for a document.
        The chain resolves to 'document', so the document instruction must win —
        the listing tool only supplied the ID."""
        agent = _agent()
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(tool_calls=[_tool_call("c1", "get_new_orders", {})]),
            _resp(tool_calls=[_tool_call("c2", "get_order_details", {"order_id": "x"})]),
            _resp(),
            _resp("# Szczegóły zamówienia x"),
        ])

        response = await agent.run("szczegóły ostatniego nowego zamówienia")

        assert response.metadata["output_format"] == "document"
        sent = json.dumps(agent._client.chat.completions.create.call_args.kwargs["messages"], ensure_ascii=False)
        assert "[FORMAT ODPOWIEDZI: PODSUMOWANIE + DOKUMENT]" in sent
        assert "[FORMAT ODPOWIEDZI: LISTA PUNKTOWANA — NIGDY TABELA]" not in sent

    @pytest.mark.asyncio
    async def test_single_tool_keeps_its_own_instruction(self):
        agent = _agent()
        agent._client.chat.completions.create = AsyncMock(side_effect=[
            _resp(tool_calls=[_tool_call("c1", "get_new_orders", {})]),
            _resp(),
            _resp("- Zamawiający: jan"),
        ])

        response = await agent.run("pokaż nowe zamówienia")

        assert response.metadata["output_format"] == "chat"
        sent = json.dumps(agent._client.chat.completions.create.call_args.kwargs["messages"], ensure_ascii=False)
        assert "[FORMAT ODPOWIEDZI: LISTA PUNKTOWANA — NIGDY TABELA]" in sent
