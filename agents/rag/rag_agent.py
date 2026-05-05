from __future__ import annotations

"""
RAG Agent — retrieves relevant knowledge and augments Claude's context.

Used by the orchestrator before routing to a specialized agent, so that
every agent response is grounded in store-specific knowledge.
"""

import logging
from typing import Any

from agents.base_agent import BaseAgent
from agents.rag.retriever import Document, build_retriever
from config.settings import get_settings
from models.conversation import AgentResponse

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """
    Retrieves relevant documents from the vector store and generates answers.

    Also exposes a `retrieve()` helper used by the orchestrator to inject
    context into other agents.
    """

    agent_name = "rag"
    system_prompt = (
        "You are a knowledgeable assistant for an e-commerce store. "
        "Answer questions based on the provided context documents. "
        "If the context does not contain enough information, say so clearly. "
        "Always respond in the same language as the customer's question. "
        "Be concise and helpful."
    )

    def __init__(self):
        super().__init__()
        self._retriever = build_retriever()

    def _get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "search_knowledge_base",
                "description": (
                    "Search the store's knowledge base for product information, "
                    "policies, FAQs, and shipping details."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query in any language.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of results to return (1-10).",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            }
        ]

    async def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        if tool_name == "search_knowledge_base":
            query = tool_input["query"]
            top_k = min(int(tool_input.get("top_k", 5)), 10)
            docs = await self._retriever.query(query, top_k=top_k)
            if not docs:
                return "No relevant information found in the knowledge base."
            return self._format_docs(docs)
        return f"Unknown tool: {tool_name}"

    async def retrieve(self, query: str) -> tuple[str, list[str]]:
        """
        Retrieve relevant context for a query.

        Returns:
            (context_text, list_of_source_ids)
        """
        settings = get_settings()
        docs = await self._retriever.query(query, top_k=settings.rag_top_k)
        if not docs:
            return "", []
        context = self._format_docs(docs)
        sources = [d.source for d in docs]
        return context, sources

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.source or doc.doc_id
            parts.append(f"[{i}] (source: {source})\n{doc.content}")
        return "\n\n---\n\n".join(parts)
