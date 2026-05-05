from __future__ import annotations

"""
Base agent class implementing the Anthropic agentic tool-use loop.

All specialized agents inherit from BaseAgent and register their tools
by overriding `_get_tools()` and `_execute_tool()`.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import anthropic

from config.settings import get_settings
from models.conversation import AgentResponse

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10  # safety cap on the agentic loop


class BaseAgent(ABC):
    """
    Abstract base for all agents.

    Subclasses define their tools and tool-execution logic.
    The agentic loop (call → execute tools → loop) lives here.
    """

    agent_name: str = "base"
    system_prompt: str = "You are a helpful AI assistant for an e-commerce store owner."

    # Subclasses can set this to override the model (e.g. use fast model)
    model_override: str | None = None

    def __init__(self):
        self._settings = get_settings()
        self._client = anthropic.AsyncAnthropic(
            api_key=self._settings.anthropic_api_key,
            max_retries=3,
            timeout=60.0,
        )

    @abstractmethod
    def _get_tools(self) -> list[dict[str, Any]]:
        """Return Anthropic tool definitions for this agent."""

    @abstractmethod
    async def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool call and return its string result."""

    async def run(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        context: str | None = None,
    ) -> AgentResponse:
        """
        Execute the agentic loop for a given query.

        Args:
            query: The user's message or question.
            conversation_history: Prior turns in Anthropic format [{role, content}, ...].
            context: Extra context string prepended to the system prompt (e.g. RAG docs).
        """
        system = self._build_system_prompt(context)
        messages = self._build_initial_messages(query, conversation_history)
        tools = self._get_tools()

        sources: list[str] = []

        for iteration in range(MAX_ITERATIONS):
            logger.debug("[%s] Iteration %d", self.agent_name, iteration + 1)

            model = self.model_override or self._settings.claude_model
            logger.info("[%s] Calling %s (iteration %d)…", self.agent_name, model, iteration + 1)
            response = await self._client.messages.create(
                model=model,
                max_tokens=max(self._settings.claude_max_tokens, 1024),
                system=system,
                tools=tools,
                messages=messages,
            )

            # Collect any sources from tool metadata
            if hasattr(response, "_tool_sources"):
                sources.extend(response._tool_sources)

            if response.stop_reason == "end_turn":
                text = self._extract_text(response.content)
                return AgentResponse(
                    text=text,
                    agent_type=self.agent_name,
                    sources=sources,
                )

            if response.stop_reason != "tool_use":
                logger.warning(
                    "[%s] Unexpected stop_reason: %s", self.agent_name, response.stop_reason
                )
                text = self._extract_text(response.content)
                return AgentResponse(text=text, agent_type=self.agent_name, sources=sources)

            # Append assistant turn, execute tools, append results
            messages.append({"role": "assistant", "content": response.content})
            tool_results = await self._process_tool_calls(response.content)
            messages.append({"role": "user", "content": tool_results})

        logger.warning("[%s] Reached max iterations (%d)", self.agent_name, MAX_ITERATIONS)
        return AgentResponse(
            text="I'm sorry, I couldn't complete the request within the allowed steps.",
            agent_type=self.agent_name,
            sources=sources,
        )

    async def _process_tool_calls(
        self, content: list[Any]
    ) -> list[dict[str, Any]]:
        """Execute all tool_use blocks in the content and return tool_result list."""
        results = []
        for block in content:
            if block.type != "tool_use":
                continue
            logger.info("[%s] Calling tool: %s(%s)", self.agent_name, block.name, block.input)
            try:
                result_text = await self._execute_tool(block.name, block.input)
            except Exception as exc:
                logger.exception("[%s] Tool %s failed: %s", self.agent_name, block.name, exc)
                result_text = "An internal error occurred. Please try again."
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                    "is_error": True,
                })
                continue
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
            })
        return results

    def _build_system_prompt(self, context: str | None) -> str:
        if context:
            return f"{self.system_prompt}\n\n## Relevant context\n{context}"
        return self.system_prompt

    def _build_initial_messages(
        self,
        query: str,
        history: list[dict[str, str]] | None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = list(history or [])
        messages.append({"role": "user", "content": query})
        return messages

    @staticmethod
    def _extract_text(content: list[Any]) -> str:
        parts = [block.text for block in content if hasattr(block, "text") and block.type == "text"]
        return "\n".join(parts) if parts else ""
