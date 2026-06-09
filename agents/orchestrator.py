from __future__ import annotations

"""
Orchestrator Agent — the central brain of the system.

Responsibilities:
  1. Receive normalized IncomingMessage from any communication channel.
  2. Load/save conversation history from Firestore.
  3. Run the RAG retriever to get relevant context.
  4. Classify the query intent and route to the appropriate specialized agent.
  5. Return the AgentResponse.

Routing logic uses a fast Claude call to classify the query intent before
dispatching to the heavier specialized agents.
"""

import logging
from typing import Any

from openai import AsyncOpenAI

from agents.allegro.allegro_agent import AllegroAgent
from agents.base_agent import BaseAgent
from agents.rag.rag_agent import RAGAgent
from config.settings import get_settings
from models.conversation import AgentResponse, IncomingMessage, MessageRole
from services.gcp_service import FirestoreService

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """
You are an intent classifier for an e-commerce store AI assistant.
Classify the user's message into exactly one of these intents:

- allegro_orders: questions about orders, shipping, delivery, tracking, returns, invoices
- allegro_offers: questions about products, listings, prices, stock, offer management
- allegro_messaging: requests to send messages to buyers or check buyer messages
- allegro_account: questions about the seller account, billing, fees, statistics
- general_knowledge: general product questions, FAQs, policies, store information
- chitchat: greetings, small talk, off-topic messages

Respond with ONLY the intent name, nothing else.
""".strip()


class Orchestrator:
    """
    Routes incoming messages to the correct specialized agent.

    Agent pool:
      - RAGAgent: knowledge base Q&A (always used for context enrichment)
      - AllegroAgent: marketplace operations
      (More agents can be registered via register_agent())
    """

    def __init__(self):
        self._settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=self._settings.google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self._firestore = FirestoreService()
        self._rag_agent = RAGAgent()
        self._allegro_agent = AllegroAgent()
        self._extra_agents: dict[str, BaseAgent] = {}

    def register_agent(self, intent_prefix: str, agent: BaseAgent) -> None:
        """Register an additional specialized agent for a custom intent prefix."""
        self._extra_agents[intent_prefix] = agent

    async def handle(self, message: IncomingMessage) -> AgentResponse:
        """
        Main entry point — process an incoming message end-to-end.
        """
        # 1. Load conversation history
        session = await self._firestore.get_or_create_session(
            session_id=message.session_id,
            channel=message.channel,
            sender_id=message.sender_id,
        )

        # 2. Retrieve RAG context
        rag_context, sources = await self._rag_agent.retrieve(message.text)

        # 3. Classify intent
        intent = await self._classify_intent(message.text, session.to_anthropic_messages())
        logger.info("Classified intent: %s for message: %.60s...", intent, message.text)

        # 4. Route to specialized agent
        response = await self._route(intent, message, session.to_anthropic_messages(), rag_context)
        response.sources = sources

        # 5. Persist conversation
        session.add_message(MessageRole.USER, message.text)
        session.add_message(MessageRole.ASSISTANT, response.text)
        await self._firestore.save_session(session)

        return response

    async def _classify_intent(
        self,
        query: str,
        history: list[dict[str, str]],
    ) -> str:
        """Use a fast Claude call to classify the query intent."""
        # Use a recent snippet of history for context (last 4 turns)
        history_snippet = history[-4:] if len(history) > 4 else history
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}" for m in history_snippet
        )

        prompt = query
        if history_text:
            prompt = f"Recent conversation:\n{history_text}\n\nNew message: {query}"

        try:
            resp = await self._client.chat.completions.create(
                model=self._settings.gemini_model_fast,
                max_tokens=20,
                messages=[
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            intent = resp.choices[0].message.content.strip().lower()
            # Validate against known intents
            known_intents = {
                "allegro_orders", "allegro_offers", "allegro_messaging",
                "allegro_account", "general_knowledge", "chitchat",
            }
            known_intents.update(self._extra_agents.keys())
            if intent not in known_intents:
                logger.warning("Unknown intent '%s', falling back to general_knowledge", intent)
                return "general_knowledge"
            return intent
        except Exception as exc:
            logger.error("Intent classification failed: %s", exc)
            return "general_knowledge"

    async def _route(
        self,
        intent: str,
        message: IncomingMessage,
        history: list[dict[str, str]],
        rag_context: str,
    ) -> AgentResponse:
        """Dispatch to the appropriate agent based on intent."""
        # Extra registered agents
        for prefix, agent in self._extra_agents.items():
            if intent.startswith(prefix):
                return await agent.run(message.text, history, rag_context or None)

        # Allegro intents → AllegroAgent (with RAG context for product knowledge)
        if intent.startswith("allegro_"):
            return await self._allegro_agent.run(
                message.text,
                history,
                rag_context or None,
            )

        # General knowledge → RAGAgent
        if intent == "general_knowledge":
            return await self._rag_agent.run(message.text, history)

        # Chitchat → lightweight response
        if intent == "chitchat":
            return await self._handle_chitchat(message.text, history)

        # Default fallback
        return await self._rag_agent.run(message.text, history)

    async def _handle_chitchat(
        self,
        query: str,
        history: list[dict[str, str]],
    ) -> AgentResponse:
        """Handle greetings and small talk without hitting specialized agents."""
        resp = await self._client.chat.completions.create(
            model=self._settings.gemini_model_fast,
            max_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly assistant for an online store. "
                        "Keep responses brief and warm. "
                        "Respond in the same language as the customer. "
                        "After greeting, gently ask how you can help with their order or products."
                    ),
                },
                *list(history),
                {"role": "user", "content": query},
            ],
        )
        text = resp.choices[0].message.content or "Hello! How can I help you?"
        return AgentResponse(text=text, agent_type="chitchat")
