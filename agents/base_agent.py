from __future__ import annotations

"""
Base agent class implementing the agentic tool-use loop via Gemini.

All specialized agents inherit from BaseAgent and register their tools
by overriding `_get_tools()` and `_execute_tool()`.

Tool definitions use OpenAI/Gemini format:
    {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from openai import AsyncOpenAI, RateLimitError

from config.settings import get_settings
from models.conversation import AgentResponse

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10
_RETRY_DELAYS = (2, 4, 8)  # seconds between retries on 429


async def _call_with_retry(coro_factory, label: str):
    """Call an async factory (returns a coroutine) with exponential backoff on 429."""
    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            return await coro_factory()
        except RateLimitError:
            if delay is None:
                logger.error("%s: rate-limited after %d attempts, giving up", label, attempt)
                raise
            logger.warning("%s: rate-limited (attempt %d), retrying in %ds…", label, attempt, delay)
            await asyncio.sleep(delay)


class BaseAgent(ABC):
    agent_name: str = "base"
    system_prompt: str = "You are a helpful AI assistant for an e-commerce store owner."
    model_override: str | None = None

    def __init__(self):
        self._settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=self._settings.google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    @abstractmethod
    def _get_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool definitions for this agent."""

    @abstractmethod
    async def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool call and return its string result."""

    async def run(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        context: str | None = None,
    ) -> AgentResponse:
        system = self._build_system_prompt(context)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages += list(conversation_history or [])
        messages.append({"role": "user", "content": query})

        tools = self._get_tools()
        model = self.model_override or self._settings.gemini_model

        for iteration in range(MAX_ITERATIONS):
            logger.debug("[%s] Iteration %d", self.agent_name, iteration + 1)
            logger.info("[%s] Calling %s (iteration %d)…", self.agent_name, model, iteration + 1)

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": self._settings.gemini_max_tokens,
            }
            if tools:
                kwargs["tools"] = tools

            response = await _call_with_retry(
                lambda kw=kwargs: self._client.chat.completions.create(**kw),
                f"{self.agent_name}/iter{iteration+1}",
            )
            choice = response.choices[0]
            msg = choice.message

            if not msg.tool_calls:
                return AgentResponse(
                    text=msg.content or "",
                    agent_type=self.agent_name,
                )

            # Append assistant turn with tool calls
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Execute each tool and append results
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}
                logger.info("[%s] Calling tool: %s(%s)", self.agent_name, tool_name, tool_input)
                try:
                    result_text = await self._execute_tool(tool_name, tool_input)
                except Exception as exc:
                    logger.exception("[%s] Tool %s failed: %s", self.agent_name, tool_name, exc)
                    result_text = "An internal error occurred. Please try again."
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })

        logger.warning("[%s] Reached max iterations (%d)", self.agent_name, MAX_ITERATIONS)
        return AgentResponse(
            text="I'm sorry, I couldn't complete the request within the allowed steps.",
            agent_type=self.agent_name,
        )

    def _build_system_prompt(self, context: str | None) -> str:
        parts = [self.system_prompt]
        if context:
            parts.append(f"## Relevant context\n{context}")
        parts.append(
            "LANGUAGE RULE: Detect the language of the user's message. "
            "If it is Polish, your ENTIRE response must be in Polish. "
            "If it is English, respond in English. "
            "Never mix languages. Tool results may be in English — ignore that and still reply in the user's language."
        )
        return "\n\n".join(parts)
