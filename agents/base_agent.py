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

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError

from agents.tool_gap_analyzer import analyze_for_tool_gap
from config.settings import get_settings
from models.conversation import AgentResponse

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10
# After exhausting all models in the pool, wait this many seconds before trying again.
_BACKOFF_DELAYS = (2, 4, 8)

# Exceptions that are worth rotating/retrying (transient infrastructure errors).
_RETRYABLE = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)


async def _call_with_retry(
    client: AsyncOpenAI,
    model_pool: list[str],
    label: str,
    **api_kwargs,
):
    """
    Call client.chat.completions.create with model rotation + exponential backoff.

    Strategy:
      1. Try each model in model_pool in order.
         On a retryable error (429, 500, connection/timeout) → rotate to next model.
      2. After all models exhausted → wait (exponential backoff) → restart from first model.
      3. After all backoff rounds exhausted → raise the last seen exception.
    """
    last_exc: Exception = RuntimeError("No models in pool")
    for backoff_round, delay in enumerate((*_BACKOFF_DELAYS, None)):
        for model in model_pool:
            try:
                logger.debug("%s: calling model %s", label, model)
                return await client.chat.completions.create(model=model, **api_kwargs)
            except _RETRYABLE as exc:
                last_exc = exc
                logger.warning(
                    "%s: %s on %s, rotating to next model",
                    label, type(exc).__name__, model,
                )

        # All models in pool returned retryable errors
        if delay is None:
            logger.error(
                "%s: all models failed after %d rounds, giving up",
                label, backoff_round + 1,
            )
            raise last_exc
        logger.warning(
            "%s: full pool failed (round %d), backing off %ds",
            label, backoff_round + 1, delay,
        )
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

    def _get_model_pool(self) -> list[str]:
        """Return rotation pool: starts with model_override/gemini_model, falls back to fast."""
        if self.model_override:
            pool = [self.model_override]
            if self._settings.gemini_model_fast not in pool:
                pool.append(self._settings.gemini_model_fast)
            return pool
        return self._settings.model_pool()

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
        model_pool = self._get_model_pool()

        for iteration in range(MAX_ITERATIONS):
            logger.info("[%s] Iteration %d, pool=%s", self.agent_name, iteration + 1, model_pool)

            api_kwargs: dict[str, Any] = {
                "messages": messages,
                "max_tokens": self._settings.gemini_max_tokens,
            }
            if tools:
                api_kwargs["tools"] = tools

            response = await _call_with_retry(
                self._client,
                model_pool,
                f"{self.agent_name}/iter{iteration+1}",
                **api_kwargs,
            )
            choice = response.choices[0]
            msg = choice.message

            if not msg.tool_calls:
                final_text = msg.content or ""
                if tools:
                    asyncio.create_task(
                        analyze_for_tool_gap(
                            client=self._client,
                            model=self._settings.gemini_model_fast,
                            query=query,
                            response=final_text,
                            existing_tool_names=[
                                t["function"]["name"] for t in tools
                            ],
                        )
                    )
                return AgentResponse(
                    text=final_text,
                    agent_type=self.agent_name,
                )

            # Append assistant turn — serialize via model_dump() so Gemini-specific
            # fields like thought_signature (required by thinking models) are preserved.
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump(exclude_none=True) for tc in msg.tool_calls],
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
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        parts = [self.system_prompt]
        parts.append(
            f"Current UTC datetime: {now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}. "
            "Polish timezone: UTC+2 in summer (CEST, late March–late Oct), UTC+1 in winter (CET). "
            "When the user gives a time like '12:00' or 'po 12', assume Polish local time and convert to UTC for API calls."
        )
        if context:
            parts.append(f"## Relevant context\n{context}")
        parts.append(
            "LANGUAGE RULE: Detect the language of the user's message. "
            "If it is Polish, your ENTIRE response must be in Polish. "
            "If it is English, respond in English. "
            "Never mix languages. Tool results may be in English — ignore that and still reply in the user's language."
        )
        return "\n\n".join(parts)
