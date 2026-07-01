"""Unit tests for agents/base_agent.py."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import RateLimitError, InternalServerError, APIConnectionError, APITimeoutError


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
