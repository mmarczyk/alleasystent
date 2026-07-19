from __future__ import annotations

"""
Orchestrator Agent — the central brain of the system.

Responsibilities:
  1. Receive normalized IncomingMessage from any communication channel.
  2. Load/save conversation history from Firestore.
  3. Classify the query on two dimensions: data source + output format.
  4. Route to the appropriate specialized agent with the right output mode.
  5. Return the AgentResponse.

Routing model (2D):
  Dimension 1 — DATA SOURCE: what data is needed to answer?
    allegro_orders | allegro_offers | allegro_messaging | allegro_account | rag | none

  Dimension 2 — OUTPUT FORMAT: how should the answer look?
    chat | table | document | dashboard
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

# ── Context-aware 2D classifier prompt ────────────────────────────────────────
# Single LLM call with full conversation history → returns "source|format".
# This is the primary classifier; keywords are used only as a cheap fast-path
# for long, self-contained queries where context is irrelevant.

_CLASSIFY_SYSTEM = """
Jesteś klasyfikatorem routingu dla asystenta AI właścicieli sklepów Allegro.

Sklasyfikuj wiadomość użytkownika NA PODSTAWIE PEŁNEJ HISTORII ROZMOWY na dwóch wymiarach.
Odpowiedz TYLKO dwoma etykietami oddzielonymi "|" — nic więcej.

WYMIAR 1 — ŹRÓDŁO DANYCH (co pobrać żeby odpowiedzieć):
  allegro_orders    — dane zamówień: statusy, wysyłka, śledzenie, zwroty, faktury
  allegro_offers    — oferty: ceny, stany magazynowe, produkty, dostawcy
  allegro_messaging — wiadomości od kupujących: czytanie, odpowiadanie
  allegro_account   — konto sprzedawcy: opłaty, prowizje, statystyki, limity
  rag               — statyczna baza wiedzy sklepu: polityki, FAQ (nie żywe dane)
  none              — nie trzeba danych: pozdrowienia, rozmowa, pytania o asystenta

WYMIAR 2 — FORMAT ODPOWIEDZI (jak ma wyglądać wynik):
  chat      — krótka konwersacyjna odpowiedź
  table     — tabela markdown z wierszami i kolumnami
  document  — pełny sformatowany dokument (mail, raport, list, szablon)
  dashboard — wielosekcyjne podsumowanie z kilkoma metrykami

KLUCZOWE ZASADY:
- Uwzględnij CAŁĄ historię rozmowy, szczególnie gdy bieżąca wiadomość jest krótka
- Krótkie follow-upy ("A teraz", "Spróbuj ponownie", "Ok", "Zrób to", "I co?", "Dobra")
  dziedziczą kontekst z poprzednich wiadomości — nie traktuj ich jako nowych tematów
- Jeśli user kontynuuje temat z historii, użyj tego samego źródła i formatu
- Gdy user poprawia lub doprecyzowuje — to nadal ten sam kontekst

Format odpowiedzi: source|format
Przykłady: allegro_offers|document   allegro_orders|table   none|chat
""".strip()

# ── Output-format prefixes injected before the user's query ───────────────────

_FORMAT_PREFIXES: dict[str, str] = {
    "document": (
        "[TRYB DOKUMENTU — ZASADY BEZWZGLĘDNE]\n"
        "1. NIE pisz żadnych wstępów ani potwierdzeń — ŻADNEGO 'Jasne', 'Rozumiem', 'Oczywiście', "
        "'Przygotuję', 'Oto' ani żadnej innej preambuły. Zacznij NATYCHMIAST od dokumentu.\n"
        "2. Pierwsza linia odpowiedzi MUSI być nagłówkiem markdown: # Tytuł dokumentu\n"
        "3. Użyj narzędzi, aby pobrać aktualne dane ze sklepu PRZED napisaniem treści. "
        "Dane muszą być prawdziwe — nie wymyślaj produktów, cen ani stanów.\n"
        "4. Dokument musi być kompletny i profesjonalny: data, pełna treść, "
        "tabele z danymi jeśli potrzebne, podpis/stopka.\n"
        "5. Minimum 300 słów — dokument musi nadawać się do natychmiastowego użycia.\n\n"
        "Polecenie użytkownika: "
    ),
    "table": (
        "[TRYB TABELI — ZASADY BEZWZGLĘDNE]\n"
        "KROK 1 — DANE PIERWSZE: wywołaj odpowiednie narzędzie(a) i poczekaj na wynik.\n"
        "         BEZ WYWOŁANIA NARZĘDZIA = BEZ TABELI. Nigdy nie generuj danych z pamięci.\n"
        "KROK 2 — TABELA: zbuduj tabelę markdown WYŁĄCZNIE z danych zwróconych przez narzędzie.\n"
        "         Pierwsza linia: nagłówek | kolumna1 | kolumna2 | ...\n"
        "         NIE pisz żadnego wstępu, potwierdzenia ani preambuły przed tabelą.\n"
        "KROK 3 — maksymalnie 1-2 zdania podsumowania po tabeli.\n"
        "UWAGA: ID ofert Allegro to 11-cyfrowe liczby (np. 12345678901), NIE UUID.\n"
        "Jeśli w odpowiedzi pojawiają się UUID — to halucynacja. Wywołaj narzędzie.\n\n"
        "Zapytanie: "
    ),
    "dashboard": (
        "[TRYB DASHBOARD — ZASADY]\n"
        "1. Zacznij od ## nagłówka sekcji — NIE od wstępu.\n"
        "2. Przygotuj wielosekcyjny raport zarządczy: każda sekcja z nagłówkiem ##, "
        "kluczowe liczby pogrubione (**x**), porównania i trendy gdzie możliwe.\n"
        "3. Pobierz WSZYSTKIE potrzebne dane narzędziami.\n\n"
        "Zapytanie: "
    ),
}

# ── Keyword maps ───────────────────────────────────────────────────────────────

# Ordered list — first match wins.
# Each entry: (keyword_list, label)

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
    (["zamówien", "zamowien", "order", "paczk", "dostaw", "śledzeni", "sledzeni",
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

_FORMAT_KEYWORDS: list[tuple[list[str], str]] = [
    # Document — email, report, letter, template
    (["wygeneruj", "generuj ", "napisz mail", "napisz email", "email do ",
      "mail do ", "stwórz raport", "utwórz raport", "przygotuj raport",
      "przygotuj dokument", "stwórz dokument", "napisz list", "szablon maila",
      "szablon email", "napisz pismo", "przygotuj pismo",
      "przygotuj maila", "przygotuj mail", "przygotuj email",
      "przygotuj wiadomość", "napisz wiadomość", "utwórz maila",
      "utwórz mail", "utwórz email", "utwórz wiadomość",
      "przygotuj ofertę", "napisz ofertę", "przygotuj list",
      "wyślij maila", "wyślij mail", "wyślij email"],
     "document"),
    # Dashboard — multi-metric summary
    (["dashboard", "panel sterowania", "podsumowanie całościowe", "przegląd całościowy",
      "raport zarządczy", "zestawienie zbiorcze"],
     "dashboard"),
    # Table — structured data
    (["w tabeli", "jako tabela", "tabelę", "tabelarycznie", "zestawienie w tabeli",
      "pokaż tabelę", "csv", "w formie tabeli"],
     "table"),
]


class Orchestrator:
    """
    Routes incoming messages to the correct specialized agent.

    Classification is 2D:
      - data_source: which Allegro sub-system (or rag/none) to query
      - output_format: chat | table | document | dashboard

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

        # Classify on both dimensions
        try:
            data_source, output_format = await self._classify(
                message.text, session.to_anthropic_messages()
            )
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError) as exc:
            logger.error("LLM API error during classification: %s", exc)
            response = AgentResponse(
                text="Przepraszam, usługa AI jest chwilowo przeciążona. Spróbuj ponownie za chwilę.",
                agent_type="error",
            )
            session.add_message(MessageRole.USER, message.text)
            session.add_message(MessageRole.ASSISTANT, response.text)
            await self._firestore.save_session(session)
            return response

        logger.info(
            "Routing: source=%s format=%s | %.60s…",
            data_source, output_format, message.text,
        )

        # Route to the right agent + format mode
        try:
            response = await self._route(
                data_source, output_format, message, session.to_anthropic_messages(), user_id
            )
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError) as exc:
            logger.error("LLM API error during routing (source=%s): %s", data_source, exc)
            response = AgentResponse(
                text="Przepraszam, usługa AI jest chwilowo przeciążona. Spróbuj ponownie za chwilę.",
                agent_type=data_source,
            )

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
    _KNOWN_FORMATS = frozenset(["chat", "table", "document", "dashboard"])

    def _keyword_source(self, query: str) -> str | None:
        q = query.lower()
        for keywords, source in _SOURCE_KEYWORDS:
            if any(kw in q for kw in keywords):
                return source
        return None

    def _keyword_format(self, query: str) -> str | None:
        q = query.lower()
        for keywords, fmt in _FORMAT_KEYWORDS:
            if any(kw in q for kw in keywords):
                return fmt
        return None

    def _is_self_contained(self, query: str) -> bool:
        """True when the query is long enough to be understood without conversation history."""
        return len(query.split()) >= 6

    async def _classify_with_llm(
        self,
        query: str,
        history: list[dict[str, str]],
        known_sources: list[str],
    ) -> tuple[str, str]:
        """Single LLM call — full history context → returns (source, format)."""
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
                max_tokens=30,
                messages=messages,
            )
            raw = (resp.choices[0].message.content or "").strip().lower()
            logger.info("LLM classifier raw: %r (query=%.50r)", raw, query)

            # Parse "source|format"
            parts = raw.replace(" ", "").split("|")
            if len(parts) == 2:
                src, fmt = parts[0].strip(), parts[1].strip()
                # Validate — fuzzy match for robustness
                matched_src = next((s for s in all_sources if s in src or src in s), None)
                matched_fmt = next((f for f in self._KNOWN_FORMATS if f in fmt or fmt in f), None)
                if matched_src and matched_fmt:
                    return matched_src, matched_fmt

            # Fallback: try substring match on the whole output
            src = next((s for s in all_sources if s in raw), "none")
            fmt = next((f for f in self._KNOWN_FORMATS if f in raw), "chat")
            logger.warning("LLM classifier fallback parse: src=%s fmt=%s from %r", src, fmt, raw)
            return src, fmt

        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError):
            raise
        except Exception as exc:
            logger.error("Classification LLM failed: %s", exc)
            return "none", "chat"

    async def _classify(
        self,
        query: str,
        history: list[dict[str, str]],
    ) -> tuple[str, str]:
        """Classify query into (data_source, output_format) using full conversation context.

        Strategy:
          1. If query is long and self-contained: try keywords for both dimensions.
             Only skip LLM when BOTH are found — guarantees no ambiguity.
          2. Everything else (short queries, follow-ups, ambiguous): single LLM call
             with last 8 turns of history. The LLM understands context like a human.
        """
        known_sources = list(self._KNOWN_SOURCES)

        if self._is_self_contained(query):
            kw_source = self._keyword_source(query)
            kw_format = self._keyword_format(query)
            if kw_source is not None and kw_format is not None:
                logger.info("Keyword fast-path: src=%s fmt=%s | %.60s", kw_source, kw_format, query)
                return kw_source, kw_format
            # One dimension known — LLM still sees context for the other
            # (fall through to LLM which is smarter and costs only ~30 tokens)

        source, fmt = await self._classify_with_llm(query, history, known_sources)
        logger.info("LLM routing: src=%s fmt=%s | %.60s", source, fmt, query)
        return source, fmt

    # ── Routing ────────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_format_prefix(query: str, output_format: str) -> str:
        """Prepend format instructions to the user query when not in chat mode."""
        prefix = _FORMAT_PREFIXES.get(output_format, "")
        return prefix + query if prefix else query

    async def _route(
        self,
        data_source: str,
        output_format: str,
        message: IncomingMessage,
        history: list[dict[str, str]],
        user_id: str | None = None,
    ) -> AgentResponse:
        """Dispatch to the right agent based on data source + output format."""

        # Extra registered agents
        for prefix, agent in self._extra_agents.items():
            if data_source.startswith(prefix):
                query = self._apply_format_prefix(message.text, output_format)
                response = await agent.run(query, history)
                response.agent_type = f"{data_source}:{output_format}"
                return response

        query = self._apply_format_prefix(message.text, output_format)

        # Allegro sub-systems → AllegroAgent
        if data_source.startswith("allegro_"):
            response = await self._get_allegro_agent(user_id).run(query, history)
            response.agent_type = f"{data_source}:{output_format}"
            return response

        # Knowledge base
        if data_source == "rag":
            try:
                response = await self._get_rag_agent().run(query, history)
                response.agent_type = f"rag:{output_format}"
                return response
            except Exception as exc:
                logger.error("RAGAgent failed, falling back to chitchat: %s", exc)
                response = await self._handle_chitchat(message.text, history)
                response.agent_type = "rag:fallback"
                return response

        # No data needed → conversational handler
        response = await self._handle_chitchat(query, history)
        response.agent_type = f"none:{output_format}"
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
            max_tokens=512,
            messages=msgs,
        )
        text = resp.choices[0].message.content or "Cześć! W czym mogę pomóc?"
        return AgentResponse(text=text, agent_type="none:chat")
