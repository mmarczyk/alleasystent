from __future__ import annotations

"""
Orchestrator Agent — the central brain of the system.

Responsibilities:
  1. Receive normalized IncomingMessage from any communication channel.
  2. Load/save conversation history from Firestore.
  3. Classify the query on ONE dimension: which data source is needed.
  4. Route to the appropriate specialized agent.
  5. Return the AgentResponse.

Routing model:
  DATA SOURCE — what data is needed to answer?
    allegro_orders | allegro_offers | allegro_messaging | allegro_account | rag | none

  OUTPUT FORMAT — chat | table | document | dashboard | action — is NOT classified
  here anymore. It used to be guessed from the user's wording before any tool ran,
  which was unreliable (e.g. a yes/no question naming a plural entity — "czy mam
  nowe wiadomości?" — could get misclassified as "table", producing an empty table
  for a one-sentence answer). Each agent now derives the format itself from WHICH
  TOOL it actually called (see agents/allegro/allegro_tools.py TOOL_OUTPUT_FORMAT)
  and reports it back via AgentResponse.metadata["output_format"] — the orchestrator
  just forwards that into the "<source>:<format>" tag the frontend reads.
"""

import logging

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)
from agents.base_agent import _call_with_retry

from agents.allegro.allegro_agent import AllegroAgent
from agents.base_agent import BaseAgent
from agents.rag.rag_agent import RAGAgent
from config.settings import get_settings
from models.conversation import AgentResponse, IncomingMessage, MessageRole
from services.gcp_service import FirestoreService

logger = logging.getLogger(__name__)

# ── Context-aware data-source classifier prompt ────────────────────────────────
# Single LLM call with full conversation history → returns the data source label.
# This is the primary classifier; keywords are used only as a cheap fast-path
# for long, self-contained queries where context is irrelevant.

_CLASSIFY_SYSTEM = """
Jesteś klasyfikatorem routingu dla asystenta AI właścicieli sklepów Allegro.

Sklasyfikuj wiadomość użytkownika NA PODSTAWIE PEŁNEJ HISTORII ROZMOWY.
Odpowiedz TYLKO jedną etykietą — nic więcej.

ŹRÓDŁO DANYCH (co pobrać żeby odpowiedzieć):
  allegro_orders    — dane zamówień: statusy, wysyłka, śledzenie, zwroty, faktury
  allegro_offers    — oferty: ceny, stany magazynowe, produkty, dostawcy
  allegro_messaging — wiadomości od kupujących: czytanie, odpowiadanie
  allegro_account   — konto sprzedawcy: opłaty, prowizje, statystyki, limity
  rag               — statyczna baza wiedzy sklepu: polityki, FAQ (nie żywe dane)
  none              — nie trzeba danych: pozdrowienia, rozmowa, pytania o asystenta

KLUCZOWE ZASADY:
- Uwzględnij CAŁĄ historię rozmowy, szczególnie gdy bieżąca wiadomość jest krótka
- Krótkie follow-upy ("A teraz", "Spróbuj ponownie", "Ok", "Zrób to", "I co?", "Dobra")
  dziedziczą kontekst z poprzednich wiadomości — nie traktuj ich jako nowych tematów
- Jeśli user kontynuuje temat z historii, użyj tego samego źródła
- Gdy user poprawia lub doprecyzowuje — to nadal ten sam kontekst

Odpowiedź: jedna etykieta źródła danych.
Przykłady: allegro_offers   allegro_orders   none
""".strip()

# ── Keyword map ─────────────────────────────────────────────────────────────────
# Ordered list — first match wins. Each entry: (keyword_list, label)

_SOURCE_KEYWORDS: list[tuple[list[str], str]] = [
    # Store policies / FAQs — check BEFORE orders so "polityka zwrotów" → rag not orders
    (["polityk", "faq", "regulamin", "kiedy wysyłacie", "kiedy wysyłają"],
     "rag"),
    # Offers / products — checked BEFORE orders so "dostawcy" (supplier) → offers not orders.
    # "dostawc" covers: dostawca, dostawcy, dostawcę, dostawców (all mean supplier)
    (["ofert", "offer", "listing", "produkt", "cen", "price", "stock",
      "stan magaz", "aktywn", "wystawion", "dodaj ofert", "dostawc",
      "włóczk", "tkanin", "materiał", "przędz", "lista produktów", "lista towarów"],
     "allegro_offers"),
    # Orders — "dostaw" covers dostawy/dostawę (delivery) but comes after offers
    # so "dostawcy" (supplier) is already caught above
    # "zamówi"/"zamowi" (stem, no case ending) covers zamówienie/zamówienia/zamówień/
    # zamówić/zamówię — the old "zamówien" keyword missed genitive plural "zamówień"
    # because that form ends in "ń", a different character than the "n" it looked for.
    (["zamówi", "zamowi", "order", "paczk", "dostaw", "śledzeni", "sledzeni",
      "zwrot", "reklamacj", "faktur", "invoice", "tracking", "shipment",
      "niespakow", "wysłan", "niewysłan", "nieopakow", "wartość zam"],
     "allegro_orders"),
    # Messaging
    (["wiadomoś", "wiadomo", "message", "napisz do kupując", "wyślij do kupując",
      "kupując", "buyer", "odpowiedz na wiadomość"],
     "allegro_messaging"),
    # Account / billing
    (["konto", "opłat", "prowizj", "statystyk", "rozliczen", "account",
      "fees", "billing", "limit sprzedaży"],
     "allegro_account"),
    # Chitchat / meta — check last so Allegro keywords take priority
    (["cześć", "hej", "witaj", "dzień dobry", "dobry wieczór", "siema",
      "hello", "hi ", "hey ", "funkcj", "możliwości", "co potrafisz", "co umiesz",
      "capabilities", "what can you", "pomoc", "co chciałem", "czego szukam"],
     "none"),
]


class Orchestrator:
    """
    Routes incoming messages to the correct specialized agent.

    Classification is 1D: data_source — which Allegro sub-system (or rag/none)
    to query. The reply's presentation (chat/table/document/dashboard/action) is
    decided downstream by the agent, from which tool it actually called.

    Agent pool:
      - AllegroAgent: all Allegro marketplace operations
      - RAGAgent: store knowledge base Q&A (lazy-loaded)
    """

    def __init__(self):
        self._settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=self._settings.google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self._firestore = FirestoreService()
        self._allegro_agents: dict[str, AllegroAgent] = {}
        self._rag_agent: RAGAgent | None = None
        self._extra_agents: dict[str, BaseAgent] = {}

    def _get_rag_agent(self) -> RAGAgent:
        if self._rag_agent is None:
            self._rag_agent = RAGAgent()
        return self._rag_agent

    def _get_allegro_agent(self, user_id: str | None = None) -> AllegroAgent:
        key = user_id or "default"
        if key not in self._allegro_agents:
            self._allegro_agents[key] = AllegroAgent(user_id=user_id)
        return self._allegro_agents[key]

    def register_agent(self, intent_prefix: str, agent: BaseAgent) -> None:
        self._extra_agents[intent_prefix] = agent

    async def handle(self, message: IncomingMessage, user_id: str | None = None) -> AgentResponse:
        """Main entry point — classify, route, persist, return."""
        session = await self._firestore.get_or_create_session(
            session_id=message.session_id,
            channel=message.channel,
            sender_id=message.sender_id,
        )

        # Classify the data source
        try:
            data_source = await self._classify(
                message.text,
                session.to_anthropic_messages(),
                last_source=session.metadata.get("last_data_source"),
            )
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError, NotFoundError) as exc:
            logger.error("LLM API error during classification: %s", exc)
            response = AgentResponse(
                text="Przepraszam, usługa AI jest chwilowo przeciążona. Spróbuj ponownie za chwilę.",
                agent_type="error",
            )
            session.add_message(MessageRole.USER, message.text)
            session.add_message(MessageRole.ASSISTANT, response.text)
            await self._firestore.save_session(session)
            return response

        logger.info("Routing: source=%s | %.60s…", data_source, message.text)

        # Route to the right agent
        try:
            response = await self._route(data_source, message, session.to_anthropic_messages(), user_id)
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError, NotFoundError) as exc:
            logger.error("LLM API error during routing (source=%s): %s", data_source, exc)
            response = AgentResponse(
                text="Przepraszam, usługa AI jest chwilowo przeciążona. Spróbuj ponownie za chwilę.",
                agent_type=data_source,
            )

        # Remember what this turn was about so a keyword-less follow-up
        # ("sprawdź jeszcze raz") can anchor to it instead of defaulting to none.
        session.metadata["last_data_source"] = data_source

        # Persist conversation
        session.add_message(MessageRole.USER, message.text)
        session.add_message(MessageRole.ASSISTANT, response.text)
        await self._firestore.save_session(session)

        return response

    # ── Classification ─────────────────────────────────────────────────────────

    _KNOWN_SOURCES = frozenset([
        "allegro_orders", "allegro_offers", "allegro_messaging",
        "allegro_account", "rag", "none",
    ])

    def _keyword_source(self, query: str) -> str | None:
        q = query.lower()
        for keywords, source in _SOURCE_KEYWORDS:
            if any(kw in q for kw in keywords):
                return source
        return None

    def _is_self_contained(self, query: str) -> bool:
        """True when the query is long enough to be understood without conversation history."""
        return len(query.split()) >= 6

    async def _classify_with_llm(
        self,
        query: str,
        history: list[dict[str, str]],
        known_sources: list[str],
    ) -> str:
        """Single LLM call — full history context → returns the data source label."""
        # Build a conversation-style prompt so the LLM sees the full flow
        history_snippet = history[-8:] if len(history) > 8 else history
        messages = [{"role": "system", "content": _CLASSIFY_SYSTEM}]
        # Add recent history so the model understands context
        for m in history_snippet:
            role = m.get("role", "user")
            content = m.get("content", "")[:300]
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": f"[KLASYFIKUJ] {query}"})

        all_sources = list(known_sources) + list(self._extra_agents.keys())
        try:
            resp = await _call_with_retry(
                self._client,
                self._settings.model_fast_pool(),
                "orchestrator/classify",
                # Thinking-capable models (gemini-3.5-flash, 2.5-flash) spend part of
                # max_tokens on invisible reasoning before the visible answer — 30
                # tokens left no room for it, so the visible output was sometimes
                # empty/truncated. reasoning_effort="none" skips that for this
                # simple, deterministic task; max_tokens raised as a safety margin.
                max_tokens=100,
                reasoning_effort="none",
                messages=messages,
            )
            raw = (resp.choices[0].message.content or "").strip().lower()
            logger.info("LLM classifier raw: %r (query=%.50r)", raw, query)

            matched = next((s for s in all_sources if s in raw), None)
            if matched:
                return matched

            logger.warning("LLM classifier fallback to 'none' from %r", raw)
            return "none"

        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError, NotFoundError):
            raise
        except Exception as exc:
            logger.error("Classification LLM failed: %s", exc)
            return "none"

    async def _classify(
        self,
        query: str,
        history: list[dict[str, str]],
        last_source: str | None = None,
    ) -> str:
        """Classify query into a data_source using full conversation context.

        Strategy:
          1. Always compute a keyword-based source guess. Domain nouns like
             "faktur" or "zamówien" are unambiguous regardless of sentence length.
          2. If the query is long/self-contained AND matches keywords, skip the
             LLM entirely (cheap fast-path).
          3. Otherwise call the LLM — but never let it silently override an
             unambiguous keyword match. The classifier occasionally defaults an
             obvious topic to "none" (flaky output, timeout, parse edge case),
             which used to leak straight to the user as "I don't have access to
             your data" even though the query clearly named e.g. invoices.
          4. Keyword-less follow-ups ("sprawdź jeszcze raz", "spróbuj ponownie")
             that the LLM defaults to "none" almost always mean "continue the
             previous topic", not a genuine switch to chitchat — anchor to the
             previous turn's data source instead of trusting that default.
        """
        known_sources = list(self._KNOWN_SOURCES)
        kw_source = self._keyword_source(query)

        if self._is_self_contained(query) and kw_source is not None:
            logger.info("Keyword fast-path: src=%s | %.60s", kw_source, query)
            return kw_source

        source = await self._classify_with_llm(query, history, known_sources)

        if kw_source is not None and source != kw_source:
            logger.warning(
                "LLM source (%s) overridden by keyword match (%s) | %.60s",
                source, kw_source, query,
            )
            source = kw_source

        if (
            source == "none"
            and kw_source is None
            and last_source
            and last_source != "none"
            and not self._is_self_contained(query)
        ):
            logger.warning(
                "Keyword-less follow-up classified as none — inheriting last source %s | %.60s",
                last_source, query,
            )
            source = last_source

        logger.info("LLM routing: src=%s | %.60s", source, query)
        return source

    # ── Routing ────────────────────────────────────────────────────────────────

    async def _route(
        self,
        data_source: str,
        message: IncomingMessage,
        history: list[dict[str, str]],
        user_id: str | None = None,
    ) -> AgentResponse:
        """Dispatch to the right agent based on data source. Each agent decides its
        own output format (see AgentResponse.metadata["output_format"]) from
        whichever tool it actually called — the orchestrator just forwards it."""

        # Extra registered agents
        for prefix, agent in self._extra_agents.items():
            if data_source.startswith(prefix):
                response = await agent.run(message.text, history)
                response.agent_type = f"{data_source}:{response.metadata.get('output_format', 'chat')}"
                return response

        # Allegro sub-systems → AllegroAgent
        if data_source.startswith("allegro_"):
            response = await self._get_allegro_agent(user_id).run(message.text, history)
            response.agent_type = f"{data_source}:{response.metadata.get('output_format', 'chat')}"
            return response

        # Knowledge base
        if data_source == "rag":
            try:
                response = await self._get_rag_agent().run(message.text, history)
                response.agent_type = "rag:chat"
                return response
            except Exception as exc:
                logger.error("RAGAgent failed, falling back to chitchat: %s", exc)
                response = await self._handle_chitchat(message.text, history)
                response.agent_type = "rag:fallback"
                return response

        # No data needed → conversational handler
        response = await self._handle_chitchat(message.text, history)
        response.agent_type = "none:chat"
        return response

    # ── Chitchat handler ───────────────────────────────────────────────────────

    async def _handle_chitchat(
        self,
        query: str,
        history: list[dict[str, str]],
    ) -> AgentResponse:
        """Handle greetings, small talk, and document generation without store data."""
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
            return AgentResponse(text=text, agent_type="none:chat")

        msgs = [
            {
                "role": "system",
                "content": (
                    "LANGUAGE RULE (HIGHEST PRIORITY): Detect the language of the user's message "
                    "and respond ONLY in that language. English message → English reply. "
                    "Polish message → Polish reply. NEVER mix languages.\n\n"
                    "You are AllEasystent — a friendly AI assistant for Allegro store owners. "
                    "Keep responses brief and warm. "
                    "When asked about your capabilities, list what you can actually do:\n"
                    "- Sprawdzanie nowych i historycznych zamówień (statusy, dane kupujących, adresy)\n"
                    "- Przeglądanie i aktualizacja ofert (tytuł, cena, stan magazynowy)\n"
                    "- Czytanie i wysyłanie wiadomości do kupujących\n"
                    "- Informacje o koncie sprzedawcy (opłaty, statystyki, limity)\n"
                    "- Odpowiedzi na pytania z bazy wiedzy sklepu (polityki, FAQ, wysyłka)\n"
                    "- Generowanie dokumentów i maili na podstawie danych sklepu\n"
                    "- Zestawianie danych w tabele i dashboardy\n"
                    "After greeting, gently ask how you can help.\n\n"
                    "ABSOLUTE RULE — NO STORE DATA: You have NO access to the user's real orders, "
                    "offers, prices, stock levels, messages, billing, or any live Allegro data. "
                    "You MUST NOT invent, estimate, or guess any business figures. "
                    "If the user asks about their orders, offers, statistics, invoices, or any "
                    "real store data — tell them to ask specifically (e.g. 'ile mam nowych zamówień?' "
                    "or 'pokaż moje oferty') so the system can fetch it from Allegro.\n\n"
                    "ABSOLUTE RULE — PERSONAL DETAILS: You have zero knowledge of the user's "
                    "real name, company, or identity unless they explicitly stated it in THIS "
                    "conversation. NEVER guess, invent, or assume a name. If asked and no name "
                    "was given, say you don't know."
                ),
            },
            *list(history),
            {"role": "user", "content": query},
        ]
        resp = await _call_with_retry(
            self._client,
            self._settings.model_fast_pool(),
            "orchestrator/chitchat",
            # See comment in _classify_with_llm — thinking models can eat the token
            # budget on invisible reasoning and cut the visible reply off mid-sentence.
            # Plain chitchat needs no reasoning, so disable it and raise the ceiling.
            max_tokens=2048,
            reasoning_effort="none",
            messages=msgs,
        )
        text = resp.choices[0].message.content or "Cześć! W czym mogę pomóc?"
        return AgentResponse(text=text, agent_type="none:chat")
