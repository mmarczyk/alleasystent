from __future__ import annotations

"""
Orchestrator Agent — the central brain of the system.

Responsibilities:
  1. Receive normalized IncomingMessage from any communication channel.
  2. Load/save conversation history from Firestore.
  3. Classify the query intent (keyword rules first, LLM fallback).
  4. Route to the appropriate specialized agent.
  5. Return the AgentResponse.
"""

import logging
from typing import Any

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from agents.base_agent import _call_with_retry

from agents.allegro.allegro_agent import AllegroAgent
from agents.base_agent import BaseAgent
from agents.rag.rag_agent import RAGAgent
from config.settings import get_settings
from models.conversation import AgentResponse, IncomingMessage, MessageRole
from services.gcp_service import FirestoreService

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """
Classify the user's message. Reply with EXACTLY ONE of these labels and nothing else:

allegro_orders      — requests for order data, shipping, delivery, tracking, returns, invoices, "zamówienia", "paczka";
                       ALSO follow-up data questions about orders in conversation context
                       (e.g. "Ile ich jest?", "Co kupiło się u mnie ostatnio?", "Ile to razem?")
allegro_offers      — listings, prices, stock, offer management, "oferty", "cena", "stan magazynowy"
allegro_messaging   — buyer messages, send message, "wiadomości", "napisz do kupującego"
allegro_account     — seller account, fees, billing, statistics, "konto", "opłaty", "prowizja"
general_knowledge   — product FAQs, store policies, shipping schedule
                       (e.g. "polityka zwrotów", "kiedy wysyłacie zamówienia?", "ile dni na zwrot?")
chitchat            — greetings, small talk, questions about assistant capabilities/features,
                       AND meta-questions where the user asks to recall what THEY THEMSELVES said
                       or requested earlier in this conversation (NOT follow-up data requests)
                       (e.g. "What was I asking about?", "Jakich zamówień szukam?", "Co chciałem?")

Output the label only. No punctuation, no explanation.
""".strip()


class Orchestrator:
    """
    Routes incoming messages to the correct specialized agent.

    Agent pool:
      - AllegroAgent: all Allegro marketplace operations (orders, offers, messages, account)
      - RAGAgent: store knowledge base Q&A — only for general_knowledge intent, lazy-loaded
      (More agents can be registered via register_agent())
    """

    def __init__(self):
        self._settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=self._settings.google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self._firestore = FirestoreService()
        self._allegro_agents: dict[str, AllegroAgent] = {}
        self._rag_agent: RAGAgent | None = None  # lazy — only loaded for general_knowledge
        self._extra_agents: dict[str, BaseAgent] = {}

    def _get_rag_agent(self) -> RAGAgent:
        """Return (and lazily create) the RAGAgent."""
        if self._rag_agent is None:
            self._rag_agent = RAGAgent()
        return self._rag_agent

    def _get_allegro_agent(self, user_id: str | None = None) -> AllegroAgent:
        key = user_id or "default"
        if key not in self._allegro_agents:
            self._allegro_agents[key] = AllegroAgent(user_id=user_id)
        return self._allegro_agents[key]

    def register_agent(self, intent_prefix: str, agent: BaseAgent) -> None:
        """Register an additional specialized agent for a custom intent prefix."""
        self._extra_agents[intent_prefix] = agent

    async def handle(self, message: IncomingMessage, user_id: str | None = None) -> AgentResponse:
        """Main entry point — process an incoming message end-to-end."""
        # 1. Load conversation history
        session = await self._firestore.get_or_create_session(
            session_id=message.session_id,
            channel=message.channel,
            sender_id=message.sender_id,
        )

        # 2. Classify intent
        intent = await self._classify_intent(message.text, session.to_anthropic_messages())
        logger.info("Intent: %s | message: %.60s…", intent, message.text)

        # 3. Route to specialized agent
        try:
            response = await self._route(intent, message, session.to_anthropic_messages(), user_id=user_id)
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError) as exc:
            logger.error("LLM API error during routing (intent=%s): %s", intent, exc)
            response = AgentResponse(
                text="Przepraszam, usługa AI jest chwilowo przeciążona. Spróbuj ponownie za chwilę.",
                agent_type="base",
            )

        # 4. Persist conversation
        session.add_message(MessageRole.USER, message.text)
        session.add_message(MessageRole.ASSISTANT, response.text)
        await self._firestore.save_session(session)

        return response

    # ── Keyword pre-routing (no LLM call needed for obvious patterns) ──────────
    _KEYWORD_MAP: list[tuple[list[str], str]] = [
        # Capability / meta questions always → chitchat
        # Also: "szukam?" catches "Jakich zamówień szukam?" and similar recall meta-questions
        (["funkcj", "możliwości", "co potrafisz", "co umiesz", "co możesz", "jakie masz",
          "capabilities", "what can you", "what do you", "features", "help me", "pomoc",
          "szukam?", "co chciałem", "o co mi chodzi", "czego szukam"],
         "chitchat"),
        # Greetings
        (["cześć", "hej", "witaj", "dzień dobry", "dobry wieczór", "siema",
          "hello", "hi ", "hey "],
         "chitchat"),
        # Store policies / FAQs — checked BEFORE orders so "polityka zwrotów" → general_knowledge
        (["polityk", "faq", "regulamin", "kiedy wysyłacie", "kiedy wysyłają"],
         "general_knowledge"),
        # Orders — use "zamówien" (matches zamówień/zamówienia etc.) but NOT "zamówi"
        # which would also match "zamówień" in meta-questions like "Jakich zamówień szukam?"
        (["zamówien", "zamowien", "order", "paczk", "dostaw", "śledzeni", "sledzeni",
          "zwrot", "reklamacj", "faktur", "invoice", "tracking", "shipment",
          "niespakow", "wysłan", "niewysłan", "nieopakow", "wartość zam"],
         "allegro_orders"),
        # Offers
        (["ofert", "offer", "listing", "produkt", "cen", "price", "stock",
          "stan magaz", "aktywn", "wystawion", "dodaj ofert"],
         "allegro_offers"),
        # Messaging
        (["wiadomoś", "wiadomo", "message", "napisz do", "wyślij do", "kupując",
          "buyer", "odpowiedz"],
         "allegro_messaging"),
        # Account
        (["konto", "opłat", "prowizj", "statystyk", "rozliczen", "account",
          "fees", "billing", "limit sprzedaży"],
         "allegro_account"),
    ]

    def _keyword_classify(self, query: str) -> str | None:
        q = query.lower()
        for keywords, intent in self._KEYWORD_MAP:
            if any(kw in q for kw in keywords):
                logger.info("Keyword pre-route: %r -> %s", query[:60], intent)
                return intent
        return None

    async def _classify_intent(
        self,
        query: str,
        history: list[dict[str, str]],
    ) -> str:
        # Fast keyword check — no LLM call needed for obvious patterns
        keyword_intent = self._keyword_classify(query)
        if keyword_intent:
            return keyword_intent

        # Use a recent snippet of history for context (last 4 turns)
        history_snippet = history[-4:] if len(history) > 4 else history
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}" for m in history_snippet
        )

        prompt = query
        if history_text:
            prompt = f"Recent conversation:\n{history_text}\n\nNew message: {query}"

        known_intents = [
            "allegro_orders", "allegro_offers", "allegro_messaging",
            "allegro_account", "general_knowledge", "chitchat",
            *self._extra_agents.keys(),
        ]
        try:
            msgs = [
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            resp = await _call_with_retry(
                self._client,
                self._settings.model_fast_pool(),
                "orchestrator/intent",
                max_tokens=30,
                messages=msgs,
            )
            raw = resp.choices[0].message.content.strip().lower()
            logger.info("Intent classifier raw output: %r", raw)

            # Exact match first
            if raw in known_intents:
                return raw

            # Substring match — handles verbose output like "The intent is allegro_orders"
            for ki in known_intents:
                if ki in raw:
                    logger.info("Intent matched via substring: %r -> %s", raw, ki)
                    return ki

            logger.warning("Unknown intent %r, falling back to chitchat", raw)
            return "chitchat"
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError):
            raise  # propagate — handle() will catch and return a friendly error
        except Exception as exc:
            logger.error("Intent classification failed: %s", exc)
            return "chitchat"

    async def _route(
        self,
        intent: str,
        message: IncomingMessage,
        history: list[dict[str, str]],
        user_id: str | None = None,
    ) -> AgentResponse:
        """Dispatch to the appropriate agent based on intent."""
        # Extra registered agents
        for prefix, agent in self._extra_agents.items():
            if intent.startswith(prefix):
                return await agent.run(message.text, history)

        # Allegro intents → live API via AllegroAgent (no static context needed)
        if intent.startswith("allegro_"):
            return await self._get_allegro_agent(user_id).run(message.text, history)

        # Store knowledge (FAQs, policies) → RAGAgent (lazy-loaded)
        if intent == "general_knowledge":
            try:
                return await self._get_rag_agent().run(message.text, history)
            except Exception as exc:
                logger.error("RAGAgent failed, falling back to chitchat: %s", exc)
                return await self._handle_chitchat(message.text, history)

        # Chitchat / capabilities
        if intent == "chitchat":
            return await self._handle_chitchat(message.text, history)

        # Default fallback → chitchat (safe, no auth side-effects)
        return await self._handle_chitchat(message.text, history)

    async def _handle_chitchat(
        self,
        query: str,
        history: list[dict[str, str]],
    ) -> AgentResponse:
        """Handle greetings and small talk without hitting specialized agents."""
        # Deterministic guard: name queries always return a canned "don't know" response.
        # Users authenticate via Allegro OAuth, not by self-introduction in chat.
        # This prevents hallucinated or contaminated history from leaking names.
        q_lower = query.lower()
        name_query = any(kw in q_lower for kw in [
            "na imię", "jak się nazywam", "jakie mam imię", "my name", "what is my name",
        ])
        if name_query:
            text = (
                "Nie powiedziałeś mi swojego imienia w tej rozmowie — nie wiem jak masz na imię. "
                "W czym mogę Ci pomóc?" if "imię" in q_lower or "nazywam" in q_lower
                else "You haven't told me your name in this conversation, so I don't know it. How can I help you?"
            )
            return AgentResponse(text=text, agent_type="chitchat")

        msgs = [
            {
                "role": "system",
                "content": (
                    "You are AllEasystent — a friendly AI assistant for Allegro store owners. "
                    "Keep responses brief and warm. "
                    "When asked about your capabilities, list what you can actually do:\n"
                    "- Sprawdzanie nowych i historycznych zamówień (statusy, dane kupujących, adresy)\n"
                    "- Przeglądanie i aktualizacja ofert (tytuł, cena, stan magazynowy)\n"
                    "- Czytanie i wysyłanie wiadomości do kupujących\n"
                    "- Informacje o koncie sprzedawcy (opłaty, statystyki, limity)\n"
                    "- Odpowiedzi na pytania z bazy wiedzy sklepu (polityki, FAQ, wysyłka)\n"
                    "After greeting, gently ask how you can help.\n\n"
                    "ABSOLUTE RULE — PERSONAL DETAILS: You have zero knowledge of the user's "
                    "real name, company, or identity unless they explicitly stated it in THIS "
                    "conversation. NEVER guess, invent, or assume a name (not 'Jan', 'Anna', "
                    "or any other). If asked and no name was given, say you don't know.\n\n"
                    "LANGUAGE RULE: If the user writes in Polish, respond entirely in Polish. "
                    "If in English, respond in English. Never mix languages."
                ),
            },
            *list(history),
            {"role": "user", "content": query},
        ]
        resp = await _call_with_retry(
            self._client,
            self._settings.model_fast_pool(),
            "orchestrator/chitchat",
            max_tokens=512,
            messages=msgs,
        )
        text = resp.choices[0].message.content or "Hello! How can I help you?"
        return AgentResponse(text=text, agent_type="chitchat")
