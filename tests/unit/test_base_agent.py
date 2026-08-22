"""Unit tests for agents/base_agent.py."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


def _make_response(text: str):
    """Create a minimal mock OpenAI chat completion response."""
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestCallWithRetry:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        from agents.base_agent import _call_with_retry
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_make_response("ok"))
        result = await _call_with_retry(client, ["model-a"], "test", messages=[])
        assert result.choices[0].message.content == "ok"

    @pytest.mark.asyncio
    async def test_rotates_on_rate_limit(self):
        from agents.base_agent import _call_with_retry
        client = MagicMock()
        resp = _make_response("success")
        # First call raises RateLimitError, second succeeds
        rate_err = RateLimitError(
            "rate limited",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        client.chat.completions.create = AsyncMock(side_effect=[rate_err, resp])
        result = await _call_with_retry(client, ["model-a", "model-b"], "test",
                                        messages=[])
        assert result.choices[0].message.content == "success"

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self):
        from agents.base_agent import _call_with_retry, _BACKOFF_DELAYS
        client = MagicMock()
        rate_err = RateLimitError(
            "rate limited",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        client.chat.completions.create = AsyncMock(side_effect=rate_err)
        with patch("agents.base_agent.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RateLimitError):
                await _call_with_retry(client, ["model-a"], "test", messages=[])

    @pytest.mark.asyncio
    async def test_empty_pool_raises(self):
        from agents.base_agent import _call_with_retry
        client = MagicMock()
        # Empty pool — the function will exhaust all backoff rounds immediately
        with patch("agents.base_agent.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception):
                await _call_with_retry(client, [], "test", messages=[])


def _status_error(status: int):
    """Build a real openai APIStatusError with the given HTTP status."""
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(status, request=request, json={"error": {"message": "boom"}})
    cls = BadRequestError if status == 400 else APIStatusError
    return cls("boom", response=response, body=None)


@pytest.fixture(autouse=True)
def clear_quirk_memo():
    """_MODEL_QUIRKS is process-global — reset it so tests don't leak into each other."""
    from agents.base_agent import _MODEL_QUIRKS
    _MODEL_QUIRKS.clear()
    yield
    _MODEL_QUIRKS.clear()


class TestBadRequestFallbackLadder:
    """A 400 is not retryable — rotating models just reproduces it. Gemini often
    names no field ("Request contains an invalid argument."), so the offending
    part of the payload is found by re-sending without it."""

    @pytest.mark.asyncio
    async def test_drops_reasoning_effort_on_400(self):
        from agents.base_agent import _call_with_retry
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[_status_error(400), _make_response("ok")]
        )
        result = await _call_with_retry(
            client, ["model-a"], "test", messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="none",
        )
        assert result.choices[0].message.content == "ok"
        assert "reasoning_effort" not in client.chat.completions.create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_quirk_is_remembered_for_the_model(self):
        from agents.base_agent import _call_with_retry
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[_status_error(400), _make_response("ok"), _make_response("ok2")]
        )
        kwargs = {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "none"}
        await _call_with_retry(client, ["model-a"], "test", **dict(kwargs))
        await _call_with_retry(client, ["model-a"], "test", **dict(kwargs))
        # 2 calls for the first request (400 + retry), 1 for the second — the
        # rejected parameter is not sent again.
        assert client.chat.completions.create.call_count == 3
        assert "reasoning_effort" not in client.chat.completions.create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_strips_empty_tool_parameters_on_second_rung(self):
        from agents.base_agent import _call_with_retry
        tools = [
            {"type": "function", "function": {
                "name": "get_account_info",
                "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {
                "name": "get_orders",
                "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}},
        ]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[_status_error(400), _status_error(400), _make_response("ok")]
        )
        await _call_with_retry(
            client, ["model-a"], "test", messages=[{"role": "user", "content": "hi"}],
            tools=tools, reasoning_effort="none",
        )
        sent = client.chat.completions.create.call_args.kwargs["tools"]
        assert "parameters" not in sent[0]["function"]
        assert sent[1]["function"]["parameters"]["properties"] == {"limit": {"type": "integer"}}

    @pytest.mark.asyncio
    async def test_flattens_tool_turns_on_last_rung(self):
        from agents.base_agent import _call_with_retry
        messages = [
            {"role": "user", "content": "pokaż zamówienia"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "function": {"name": "get_new_orders", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "2 zamówienia"},
        ]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[_status_error(400), _status_error(400), _status_error(400), _make_response("ok")]
        )
        await _call_with_retry(
            client, ["model-a"], "test", messages=messages, reasoning_effort="none",
        )
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        assert not any(m.get("tool_calls") or m.get("role") == "tool" for m in sent)
        assert any("2 zamówienia" in (m.get("content") or "") for m in sent)

    @pytest.mark.asyncio
    async def test_drops_leading_assistant_turn_on_last_rung(self):
        """Possible since the invoice reminder writes itself into a conversation
        unprompted, so a history can now open with an assistant turn."""
        from agents.base_agent import _call_with_retry
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "🧾 Masz 2 niewystawione faktury. Wystawić?"},
            {"role": "user", "content": "wystaw"},
        ]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[_status_error(400)] * 4 + [_make_response("ok")]
        )
        await _call_with_retry(client, ["model-a"], "test", messages=messages)
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        assert [m["role"] for m in sent] == ["system", "user"]

    @pytest.mark.asyncio
    async def test_raises_when_no_variant_works(self):
        from agents.base_agent import _call_with_retry
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_status_error(400))
        with pytest.raises(APIStatusError):
            await _call_with_retry(
                client, ["model-a"], "test", messages=[{"role": "user", "content": "hi"}],
            )

    @pytest.mark.asyncio
    async def test_non_400_status_error_is_not_retried(self):
        """A 403 is a real error, not a payload quirk — no fallback attempts."""
        from agents.base_agent import _call_with_retry
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_status_error(403))
        with pytest.raises(APIStatusError):
            await _call_with_retry(
                client, ["model-a"], "test", messages=[{"role": "user", "content": "hi"}],
                reasoning_effort="none",
            )
        assert client.chat.completions.create.call_count == 1


class TestSanitizeMessages:
    """Gemini rejects any request carrying an empty text part with a
    non-retryable 400, so blank turns must never reach the API."""

    def test_drops_blank_assistant_turn(self):
        from agents.base_agent import sanitize_messages
        out = sanitize_messages([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "faktury?"},
        ])
        assert out == [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "faktury?"},
        ]

    def test_drops_whitespace_only_turn(self):
        from agents.base_agent import sanitize_messages
        out = sanitize_messages([{"role": "assistant", "content": "  \n\t "}])
        assert out == []

    def test_keeps_tool_call_turn_and_nulls_blank_content(self):
        from agents.base_agent import sanitize_messages
        tool_calls = [{"id": "c1", "function": {"name": "get_orders", "arguments": "{}"}}]
        out = sanitize_messages([{"role": "assistant", "content": "", "tool_calls": tool_calls}])
        assert out == [{"role": "assistant", "content": None, "tool_calls": tool_calls}]

    def test_keeps_tool_result_with_placeholder(self):
        """Dropping a tool result would orphan its tool_call — also a 400."""
        from agents.base_agent import sanitize_messages
        out = sanitize_messages([{"role": "tool", "tool_call_id": "c1", "content": ""}])
        assert out == [{"role": "tool", "tool_call_id": "c1", "content": "(no result)"}]

    def test_leaves_valid_messages_untouched(self):
        from agents.base_agent import sanitize_messages
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert sanitize_messages(msgs) == msgs

    @pytest.mark.asyncio
    async def test_call_with_retry_sanitizes_before_sending(self):
        from agents.base_agent import _call_with_retry
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_make_response("ok"))
        await _call_with_retry(
            client, ["model-a"], "test",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": ""},
            ],
        )
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        assert sent == [{"role": "user", "content": "hi"}]


class TestBuildSystemPrompt:
    def _make_agent(self):
        """Create a concrete subclass of BaseAgent for testing."""
        from agents.base_agent import BaseAgent

        class ConcreteAgent(BaseAgent):
            agent_name = "test"
            system_prompt = "You are a test assistant."

            def _get_tools(self):
                return []

            async def _execute_tool(self, tool_name, tool_input):
                return ""

        with patch("agents.base_agent.AsyncOpenAI"):
            return ConcreteAgent()

    def test_contains_system_prompt(self):
        agent = self._make_agent()
        result = agent._build_system_prompt(None)
        assert "You are a test assistant." in result

    def test_contains_datetime(self):
        agent = self._make_agent()
        result = agent._build_system_prompt(None)
        assert "Current date and time" in result

    def test_contains_language_rule(self):
        agent = self._make_agent()
        result = agent._build_system_prompt(None)
        assert "LANGUAGE RULE" in result

    def test_context_included_when_provided(self):
        agent = self._make_agent()
        result = agent._build_system_prompt("Some RAG context here")
        assert "Some RAG context here" in result
        assert "Relevant context" in result

    def test_no_context_section_when_none(self):
        agent = self._make_agent()
        result = agent._build_system_prompt(None)
        assert "Relevant context" not in result


class TestGetModelPool:
    def test_uses_settings_pool_by_default(self):
        from agents.base_agent import BaseAgent

        class ConcreteAgent(BaseAgent):
            agent_name = "test"
            system_prompt = "prompt"
            model_override = None

            def _get_tools(self):
                return []

            async def _execute_tool(self, t, i):
                return ""

        with patch("agents.base_agent.AsyncOpenAI"):
            agent = ConcreteAgent()
        pool = agent._get_model_pool()
        assert isinstance(pool, list)
        assert len(pool) >= 1

    def test_override_added_to_front(self):
        from agents.base_agent import BaseAgent

        class OverrideAgent(BaseAgent):
            agent_name = "override"
            system_prompt = "prompt"
            model_override = "my-special-model"

            def _get_tools(self):
                return []

            async def _execute_tool(self, t, i):
                return ""

        with patch("agents.base_agent.AsyncOpenAI"):
            agent = OverrideAgent()
        pool = agent._get_model_pool()
        assert pool[0] == "my-special-model"
