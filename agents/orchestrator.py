from __future__ import annotations

"""
Orchestrator Agent — the central brain of the system.

Responsibilities:
  1. Receive normalized IncomingMessage from any communication channel.
  2. Load/save conversation history from the session store (Redis/in-memory).
  3. Classify the query on ONE dimension: which data source is needed.
  4. Route to the appropriate specialized agent.
  5. Return the AgentResponse.

Routing model:
  DATA SOURCE — which agent can answer?
    allegro | rag | none

  OUTPUT FORMAT — chat | table | document | dashboard | action — is NOT classified
  here anymore. It used to be guessed from the user's wording before any tool ran,
  which was unreliable (e.g. a yes/no question naming a plural entity — "czy mam
  nowe wiadomości?" — could get misclassified as "table", producing an empty table
  for a one-sentence answer). Each agent now derives the format itself from WHICH
  TOOL it actually called (see agents/allegro/allegro_tools.py TOOL_OUTPUT_FORMAT)
  and reports it back via AgentResponse.metadata["output_format"] — the orchestrator
  just forwards that into the "<source>:<format>" tag the frontend reads.

  The DATA SOURCE label used to carry the same kind of pre-tool guess: it split
  Allegro four ways (allegro_orders | allegro_offers | allegro_messaging |
  allegro_account). Nothing routed on that split — all four reached the same
  AllegroAgent below, which picks its own tool from the full query anyway — so
  the only thing the four-way guess produced was an analytics label, computed
  from the user's wording before any tool ran and wrong whenever the wording and
  the tool disagreed ("ile zapłaciłem prowizji od tego zamówienia" reads as
  orders, answers from the billing tools). Analytics now labels a turn by the
  tool that actually ran — see services/analytics_service.py._intent_label,
  which is the same correction the output format got above — and this classifier
  only decides what it actually routes on: which agent gets the turn.

  Collapsing the four labels also removed the reason the keyword map below had
  to order its Allegro entries so carefully: "dostawca" (supplier → offers) vs
  "dostawa" (delivery → orders) only ever needed telling apart to pick between
  two labels that now both read "allegro".
"""

import asyncio
import logging
import time
from typing import NamedTuple

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)
from agents.base_agent import _call_for_reply, _call_with_retry

from agents.allegro.allegro_agent import AllegroAgent
from agents.base_agent import BaseAgent
from agents.perf import StageTimer
from agents.rag.rag_agent import RAGAgent
from config.settings import get_settings
from models.conversation import AgentResponse, IncomingMessage, MessageRole
from services import analytics_service
from services.gcp_service import SessionStore

logger = logging.getLogger(__name__)

# ── Cold-start visibility ────────────────────────────────────────────────────
# Cloud Run runs this service with --min-instances=0 (cloudbuild.yaml,
# deploy-backend.yml), so a request after an idle period can hit a container
# that just booted — image pull, Python startup, module imports — all of
# which happens before this module's StageTimer ever starts timing. That
# delay was invisible in the analytics dashboard's phase chart, prompting the
# question of whether it's real. This doesn't fix it — it just tags the first
# request this process handles so it's visible in the logged perf data
# instead of silently inflating (or explaining) an otherwise-unaccounted-for
# total_ms on whatever query happened to land first.
_PROCESS_STARTED_AT = time.time()
_requests_handled = 0


def _mark_request() -> tuple[bool, float]:
    """Returns (is_first_request_this_process, seconds_since_process_start)."""
    global _requests_handled
    _requests_handled += 1
    return _requests_handled == 1, time.time() - _PROCESS_STARTED_AT


# ── Context-aware data-source classifier prompt ────────────────────────────────
# Single LLM call with full conversation history → returns the data source label.
# This is the primary classifier; keywords are used only as a cheap fast-path
# for long, self-contained queries where context is irrelevant.

_CLASSIFY_SYSTEM = """
Jesteś klasyfikatorem routingu dla asystenta AI właścicieli sklepów Allegro.

Sklasyfikuj wiadomość użytkownika NA PODSTAWIE PEŁNEJ HISTORII ROZMOWY.
Odpowiedz TYLKO jedną etykietą — nic więcej.

ŹRÓDŁO DANYCH (co pobrać żeby odpowiedzieć):
  allegro — żywe dane ze sklepu Allegro: zamówienia (statusy, wysyłka, śledzenie,
            zwroty, faktury), oferty (ceny, stany magazynowe, produkty, dostawcy),
            wiadomości od kupujących, konto sprzedawcy (opłaty, prowizje,
            statystyki, limity), kupujący i ich dane kontaktowe
  rag     — statyczna baza wiedzy sklepu: polityki, FAQ (nie żywe dane)
  none    — nie trzeba danych: pozdrowienia, rozmowa, pytania o asystenta

KLUCZOWE ZASADY:
- Pytanie o OGÓLNĄ MOŻLIWOŚĆ ("czy jesteś w stanie...", "czy potrafisz...", "czy umiesz...")
  to zawsze "none" — nawet jeśli wymienia dane (zamówienia, wiadomości, oferty). Użytkownik
  pyta czy dana funkcja istnieje, nie prosi o wykonanie jej teraz. Dopiero konkretne polecenie
  ("sprawdź nowe wiadomości", "pokaż zamówienia") jest właściwym źródłem danych.
- Uwzględnij CAŁĄ historię rozmowy, szczególnie gdy bieżąca wiadomość jest krótka
- Krótkie follow-upy ("A teraz", "Spróbuj ponownie", "Ok", "Zrób to", "I co?", "Dobra")
  dziedziczą kontekst z poprzednich wiadomości — nie traktuj ich jako nowych tematów
- Jeśli user kontynuuje temat z historii, użyj tego samego źródła
- Gdy user poprawia lub doprecyzowuje — to nadal ten sam kontekst

Odpowiedź: jedna etykieta źródła danych.
Przykłady: allegro   rag   none
""".strip()

# ── Keyword map ─────────────────────────────────────────────────────────────────
# Ordered list — first match wins. Each entry: (keyword_list, label)

_SOURCE_KEYWORDS: list[tuple[list[str], str]] = [
    # Capability questions ("czy jesteś w stanie sprawdzać wiadomości?", "czy potrafisz...")
    # — checked FIRST, before any domain keyword. These ask what the assistant CAN do in
    # general, not for live data right now, even though they often name a domain noun
    # ("wiadomości", "zamówienia"). Without this, "czy jesteś w stanie sprawdzać nowe
    # wiadomości" would match the messaging bucket below and force an immediate (and
    # irrelevant) tool call instead of a plain "yes, I can do that" answer.
    (["czy jesteś w stanie", "jesteś w stanie", "czy potrafisz", "czy umiesz",
      "are you able to", "do you know how to"],
     "none"),
    # Store policies / FAQs — check BEFORE allegro so "polityka zwrotów" → rag
    # not live order data
    (["polityk", "faq", "regulamin", "kiedy wysyłacie", "kiedy wysyłają"],
     "rag"),
    # Live Allegro data — one bucket for every sub-system, since they all reach
    # the same AllegroAgent (see _route), which picks its own tool from the full
    # query. This used to be four ordered buckets, and the ordering was load-
    # bearing only for telling apart words that pointed at two different LABELS:
    # "dostawca" (supplier → offers) had to be caught before "dostawa" (delivery
    # → orders), and a contact lookup ("czy mam klienta z takim nr telefonu")
    # had to sit after messaging so "co pisał klient" stayed messaging. With one
    # label the ordering carries no meaning and the near-duplicates collapse:
    # "dostaw" already covers "dostawca", "wiadomo" covers "odpowiedz na
    # wiadomość", "produkt" covers "lista produktów".
    #
    # "zamówi"/"zamowi" (stem, no case ending) covers zamówienie/zamówienia/
    # zamówień/zamówić/zamówię — the old "zamówien" keyword missed genitive
    # plural "zamówień" because that form ends in "ń", a different character
    # than the "n" it looked for.
    ([
        # zamówienia + wysyłka
        "zamówi", "zamowi", "order", "paczk", "dostaw", "śledzeni", "sledzeni",
        "zwrot", "reklamacj", "faktur", "invoice", "tracking", "shipment",
        "niespakow", "wysłan", "nieopakow",
        # oferty + produkty
        "ofert", "offer", "listing", "produkt", "towar", "cen", "price", "stock",
        "stan magaz", "aktywn", "wystawion",
        "włóczk", "tkanin", "materiał", "przędz",
        # wiadomości + kupujący
        "wiadomo", "message", "kupując", "buyer",
        "klient", "nr tel", "numer tel", "telefon", "kontrahent", "nabywc",
        # konto + rozliczenia
        "konto", "opłat", "prowizj", "statystyk", "rozliczen", "account",
        "fees", "billing", "limit sprzedaży",
     ],
     "allegro"),
    # Chitchat / meta — check last so Allegro keywords take priority
    (["cześć", "hej", "witaj", "dzień dobry", "dobry wieczór", "siema",
      "hello", "hi ", "hey ", "funkcj", "możliwości", "co potrafisz", "co umiesz",
      "capabilities", "what can you", "pomoc", "co chciałem", "czego szukam"],
     "none"),
]


# How many of the most recent conversation turns are replayed into each request,
# and how far back those turns may be. Sessions live for 30 days (services/
# gcp_service.py), so without a cap an old thread grows unboundedly and every
# message re-sends the whole thing — including past order/offer listings, which
# are the largest turns by far. The age cap matters independently of the count
# cap: a seller returning after a multi-hour gap doesn't need that stale
# context re-sent into the tool-selection/interpret prompts on their next
# query — it was still costing full _HISTORY_TURNS-worth of tokens even
# though it had nothing to do with a fresh question.
_HISTORY_TURNS = 10
_HISTORY_MAX_AGE_HOURS = 12

# Shown instead of an empty reply. A blank assistant turn must never reach the
# user OR the session store: replayed as history it makes Gemini reject every
# later message in the thread with a non-retryable 400 (see
# agents/base_agent.py.sanitize_messages), which killed whole conversations.
_EMPTY_REPLY_FALLBACK = (
    "Przepraszam, nie udało się wygenerować odpowiedzi na to pytanie. "
    "Spróbuj ponownie lub sformułuj je inaczej."
)


# ── Which branch of _classify decided ───────────────────────────────────────
# Recorded per turn (AgentResponse.metadata["classify_path"] → the analytics
# record) because the keyword/LLM split is the number that decides whether
# replacing the LLM fallback with a local classifier is worth anything: only
# the LLM branch costs a round-trip, and only its queries are worth learning
# from. A corpus mixing in the keyword branch would teach a model to reproduce
# the regex matcher that still runs in front of it — no gain, and it skews the
# training distribution towards keyword-shaped queries, which are exactly the
# ones the model would never be asked about.
PATH_KEYWORD = "keyword"      # a domain noun matched — no LLM call was made
PATH_LLM = "llm"              # no keyword; the LLM's own answer stands
PATH_INHERITED = "inherited"  # LLM said "none"; overridden by the last turn's source


class Classification(NamedTuple):
    """What _classify decided, and which branch decided it."""
    source: str
    path: str


def _normalize_source(source: str | None) -> str | None:
    """Map a stored pre-collapse label onto the current three.

    Sessions live for 30 days (services/gcp_service.py), so for a month after
    this deploys, session.metadata["last_data_source"] can still hold one of
    the four old Allegro labels written by the previous version — and that
    value is fed straight back into routing by the follow-up inheritance in
    _classify(). Without this it would reach _route() as an unknown source and
    silently answer a "sprawdź jeszcze raz" follow-up with chitchat.
    """
    if source and source.startswith("allegro"):
        return "allegro"
    return source


def _last_assistant_text(session) -> str | None:
    """The last thing the assistant said in this thread, or None if it hasn't
    spoken yet. Used to tell which open question a short reply ("tak") is
    answering — see the reminder check in Orchestrator.handle().

    Deliberately not age-limited like the history sent to the agents: a stale
    question is exactly the case where a bare "tak" must NOT be read as
    consent to a write action.
    """
    for msg in reversed(session.messages):
        if msg.role == MessageRole.ASSISTANT and msg.content.strip():
            return msg.content
    return None


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
            # See agents/base_agent.py BaseAgent.__init__ for why this matters:
            # without it a degraded-but-not-erroring model can sit inside one
            # call for up to the SDK's 600s default with no rotation.
            timeout=30.0,
        )
        self._session_store = SessionStore()
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
        is_cold_start, since_process_start_s = _mark_request()
        if is_cold_start:
            logger.info(
                "First request handled %.1fs after process start — likely a Cloud Run cold start",
                since_process_start_s,
            )
        perf = StageTimer("orchestrator.handle")

        with perf.stage("session_load"):
            session = await self._session_store.get_or_create_session(
                session_id=message.session_id,
                channel=message.channel,
                sender_id=message.sender_id,
            )

        # An open REMINDER — unissued invoices (services/invoice_reminder.py)
        # or unread buyer messages (services/message_reminder.py) — gets first
        # refusal on this message, because it may be the answer to a question
        # the assistant asked proactively. Both keep their STATE in Redis
        # rather than in this session, since the seller may reply from a
        # different chat thread than the reminder was written into (see those
        # modules' docstrings).
        #
        # The last assistant turn of THIS thread goes with it, because Redis
        # state alone cannot tell "tak" answering a reminder from "tak"
        # answering whatever the assistant asked a second ago ("Masz 1 nową
        # wiadomość. Pokazać szczegóły?"). Without it that "tak" was taken as
        # consent and issued real VAT invoices.
        #
        # services/reminder_router.py picks between them and, when a reply
        # could belong to either, asks the seller which one instead of
        # guessing. A message unrelated to any open reminder falls through
        # unchanged.
        if user_id:
            with perf.stage("reminder_check"):
                try:
                    from services.reminder_router import handle_reply as _handle_reminder_reply
                    reminder_text = await _handle_reminder_reply(
                        user_id, message.text, _last_assistant_text(session),
                    )
                except Exception as exc:
                    logger.warning("Reminder reply handling failed: %s", exc)
                    reminder_text = None
            if reminder_text is not None:
                response = AgentResponse(text=reminder_text, agent_type="reminder")
                session.add_message(MessageRole.USER, message.text)
                session.add_message(
                    MessageRole.ASSISTANT, response.text, {"agent": response.agent_type},
                )
                await self._session_store.save_session(session)
                perf.log(source="reminder", channel=message.channel)
                return response

        # Classify the data source
        try:
            with perf.stage("classify"):
                data_source, classify_path = await self._classify(
                    message.text,
                    session.to_anthropic_messages(limit=_HISTORY_TURNS, max_age_hours=_HISTORY_MAX_AGE_HOURS),
                    last_source=_normalize_source(session.metadata.get("last_data_source")),
                )
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError, NotFoundError) as exc:
            logger.error("LLM API error during classification: %s", exc)
            response = AgentResponse(
                text="Przepraszam, usługa AI jest chwilowo przeciążona. Spróbuj ponownie za chwilę.",
                agent_type="error",
            )
            session.add_message(MessageRole.USER, message.text)
            session.add_message(
                MessageRole.ASSISTANT, response.text, {"agent": response.agent_type},
            )
            await self._session_store.save_session(session)
            perf.log(source="error", channel=message.channel)
            return response

        logger.info("Routing: source=%s | %.60s…", data_source, message.text)

        # Route to the right agent
        try:
            with perf.stage("route"):
                response = await self._route(
                    data_source, message, session.to_anthropic_messages(limit=_HISTORY_TURNS, max_age_hours=_HISTORY_MAX_AGE_HOURS), user_id,
                )
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError, NotFoundError) as exc:
            logger.error("LLM API error during routing (source=%s): %s", data_source, exc)
            response = AgentResponse(
                text="Przepraszam, usługa AI jest chwilowo przeciążona. Spróbuj ponownie za chwilę.",
                agent_type=data_source,
            )
        except APIStatusError as exc:
            # Non-retryable API errors (e.g. 400 INVALID_ARGUMENT) used to crash all
            # the way out to a raw 500 — _classify() already had this safety net,
            # _route() didn't. Log the query so a recurring bad payload is diagnosable.
            logger.error(
                "LLM API error during routing (source=%s): %s | query=%.200r",
                data_source, exc, message.text,
            )
            response = AgentResponse(
                text="Przepraszam, nie udało się przetworzyć tej wiadomości. Spróbuj sformułować pytanie inaczej.",
                agent_type=data_source,
            )

        # Which classifier branch decided this turn, carried out to the caller
        # so the analytics record can store it (see PATH_* above and
        # services/analytics_service.py.log_query). Set on every routing
        # outcome including the two error responses above — a turn whose route
        # failed was still classified, and dropping it would quietly bias the
        # split towards whichever branch fails less often.
        response.metadata["classify_path"] = classify_path

        # An agent can return an empty string (reply truncated by max_tokens, a
        # safety filter, or a tool round that produced no text). Never show that
        # blank bubble, and above all never store it — see _EMPTY_REPLY_FALLBACK.
        if not (response.text or "").strip():
            logger.warning(
                "Empty reply from source=%s (agent=%s, tools=%s) — substituting fallback | query=%.200r",
                data_source,
                response.agent_type,
                ",".join(response.metadata.get("tools") or []) or "none",
                message.text,
            )
            response.text = _EMPTY_REPLY_FALLBACK

        # Remember what this turn was about so a keyword-less follow-up
        # ("sprawdź jeszcze raz") can anchor to it instead of defaulting to none.
        session.metadata["last_data_source"] = data_source

        # Persist conversation
        with perf.stage("session_save"):
            session.add_message(MessageRole.USER, message.text)
            # agent_type is "<data_source>:<output_format>" — storing it is what
            # lets a table/document reply come back as a table/document when the
            # thread is reopened on another device, instead of as flat text.
            session.add_message(
                MessageRole.ASSISTANT, response.text, {"agent": response.agent_type},
            )
            await self._session_store.save_session(session)

        perf.log(source=data_source, channel=message.channel)

        # Fire-and-forget: feed the phase-timing breakdown to the analytics
        # dashboard's query-performance chart. The "route" stage above is this
        # level's one opaque bucket for whatever the routed agent did — when
        # that agent reports its own breakdown (AllegroAgent/BaseAgent do, via
        # metadata["perf_stages"]: auth check, tool-select LLM call, the
        # Allegro API call(s), interpret LLM call), swap it in for the real
        # picture instead of double-counting both.
        own_stages = perf.snapshot()
        own_stages.pop("route", None)
        combined_phases = {**own_stages, **response.metadata.get("perf_stages", {})}
        perf_label = analytics_service.label_for_perf(data_source, response.metadata.get("tools"))
        asyncio.create_task(
            analytics_service.log_perf(perf_label, combined_phases, perf.elapsed_ms(), cold=is_cold_start)
        )

        return response

    # ── Classification ─────────────────────────────────────────────────────────

    _KNOWN_SOURCES = frozenset(["allegro", "rag", "none"])

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
    ) -> Classification:
        """Classify query into a data_source using full conversation context.

        Returns the label AND which branch produced it (see Classification) —
        the caller records the branch so the keyword/LLM split is measurable.

        Strategy:
          1. Always compute a keyword-based source guess. Domain nouns like
             "faktur" or "zamówien" are unambiguous regardless of sentence length.
          2. Any keyword match skips the LLM entirely (cheap fast-path). This used
             to be gated on the query being long/self-contained, but a keyword
             match was ALWAYS trusted over the LLM's answer when they disagreed —
             so for a short-but-unambiguous query ("nowe zamówienia", "moje
             oferty": 2 words, always kw-matched — the single most common
             phrasing for a direct command) the LLM call ran, its result got
             thrown away the moment it disagreed with the keyword, and even
             agreeing was a wasted round-trip that changed nothing. There's no
             query shape where consulting the LLM changes the returned value once
             a keyword has matched, so the call is skipped outright now.
          3. No keyword match → call the LLM. A keyword-less query is exactly
             the case where the LLM's own judgment (and conversation history)
             is what decides, so nothing here is skippable.
          4. Keyword-less follow-ups ("sprawdź jeszcze raz", "spróbuj ponownie")
             that the LLM defaults to "none" almost always mean "continue the
             previous topic", not a genuine switch to chitchat — anchor to the
             previous turn's data source instead of trusting that default.
        """
        known_sources = list(self._KNOWN_SOURCES)
        kw_source = self._keyword_source(query)

        if kw_source is not None:
            logger.info("Keyword fast-path: src=%s | %.60s", kw_source, query)
            return Classification(kw_source, PATH_KEYWORD)

        source = await self._classify_with_llm(query, history, known_sources)
        path = PATH_LLM

        # kw_source is always None here (the fast-path above already returned
        # otherwise), so the LLM's own answer stands unless overridden below.
        if (
            source == "none"
            and last_source
            and last_source != "none"
            and not self._is_self_contained(query)
        ):
            logger.warning(
                "Keyword-less follow-up classified as none — inheriting last source %s | %.60s",
                last_source, query,
            )
            source = last_source
            path = PATH_INHERITED

        logger.info("LLM routing: src=%s | %.60s", source, query)
        return Classification(source, path)

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

        # Live store data → AllegroAgent, which picks its own tool
        if data_source == "allegro":
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
                    "You are AlleAsystent — a friendly AI assistant for Allegro store owners. "
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
        resp = await _call_for_reply(
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
