"""Unit tests for agents/orchestrator.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


def _make_orchestrator():
    with patch("agents.orchestrator.AsyncOpenAI"), \
         patch("agents.orchestrator.SessionStore"):
        from agents.orchestrator import Orchestrator
        return Orchestrator()


class TestClientTimeout:
    def test_client_gets_explicit_timeout(self):
        """See agents/base_agent.py TestClientTimeout for why: without this,
        a merely-slow (not erroring) model call can sit for up to the openai
        SDK's 600s default with no exception for _call_with_retry to rotate
        on."""
        with patch("agents.orchestrator.AsyncOpenAI") as MockOpenAI, \
             patch("agents.orchestrator.SessionStore"):
            from agents.orchestrator import Orchestrator
            Orchestrator()
        assert MockOpenAI.call_args.kwargs["timeout"] == 30.0


class TestKeywordClassify:
    def test_chitchat_greeting(self):
        orc = _make_orchestrator()
        assert orc._keyword_classify("cześć, jak się masz?") == "chitchat"

    def test_chitchat_hello_en(self):
        orc = _make_orchestrator()
        assert orc._keyword_classify("hi there") == "chitchat"

    def test_chitchat_capabilities(self):
        orc = _make_orchestrator()
        assert orc._keyword_classify("co potrafisz zrobić?") == "chitchat"

    def test_allegro_orders(self):
        orc = _make_orchestrator()
        assert orc._keyword_classify("pokaż moje zamówień") == "allegro_orders"

    def test_allegro_orders_en(self):
        orc = _make_orchestrator()
        assert orc._keyword_classify("show me my order status") == "allegro_orders"

    def test_allegro_offers(self):
        orc = _make_orchestrator()
        assert orc._keyword_classify("lista moich ofert") == "allegro_offers"

    def test_allegro_messaging(self):
        orc = _make_orchestrator()
        assert orc._keyword_classify("wiadomości od kupujących") == "allegro_messaging"

    def test_allegro_account(self):
        orc = _make_orchestrator()
        assert orc._keyword_classify("moje konto allegro") == "allegro_account"

    def test_general_knowledge_policy(self):
        orc = _make_orchestrator()
        assert orc._keyword_classify("jaka jest polityka zwrotów?") == "general_knowledge"

    def test_returns_none_for_unknown(self):
        orc = _make_orchestrator()
        assert orc._keyword_classify("xyzzy frobble quux") is None

    def test_paczka_maps_to_orders(self):
        orc = _make_orchestrator()
        assert orc._keyword_classify("gdzie jest moja paczka?") == "allegro_orders"

    def test_case_insensitive(self):
        orc = _make_orchestrator()
        assert orc._keyword_classify("CZEŚĆ") == "chitchat"


class TestHandleChitchatNameGuard:
    @pytest.mark.asyncio
    async def test_name_query_polish(self):
        orc = _make_orchestrator()
        response = await orc._handle_chitchat("jak się nazywam?", [])
        assert "imię" in response.text.lower() or "nie" in response.text.lower()
        assert response.agent_type == "chitchat"

    @pytest.mark.asyncio
    async def test_name_query_english(self):
        orc = _make_orchestrator()
        response = await orc._handle_chitchat("what is my name?", [])
        assert "name" in response.text.lower()
        assert response.agent_type == "chitchat"

    @pytest.mark.asyncio
    async def test_jakie_mam_imie(self):
        orc = _make_orchestrator()
        response = await orc._handle_chitchat("jakie mam imię?", [])
        assert response.agent_type == "chitchat"
        # Should return canned response, no LLM call needed

    @pytest.mark.asyncio
    async def test_chitchat_calls_llm_for_non_name_query(self):
        orc = _make_orchestrator()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Hello! How can I help?"
        orc._client.chat.completions.create = AsyncMock(return_value=mock_resp)
        with patch("agents.orchestrator._call_with_retry",
                   new_callable=AsyncMock, return_value=mock_resp):
            response = await orc._handle_chitchat("cześć!", [])
        assert response.agent_type == "chitchat"


class TestRegisterAgent:
    def test_register_and_retrieve(self):
        orc = _make_orchestrator()
        mock_agent = MagicMock()
        orc.register_agent("custom_intent", mock_agent)
        assert orc._extra_agents["custom_intent"] is mock_agent

    def test_register_multiple_agents(self):
        orc = _make_orchestrator()
        a1 = MagicMock()
        a2 = MagicMock()
        orc.register_agent("intent_a", a1)
        orc.register_agent("intent_b", a2)
        assert len(orc._extra_agents) == 2


class TestClassifySkipsLLMOnKeywordMatch:
    """A keyword match always wins over the LLM's own answer when they
    disagree (see _classify's docstring), so for a keyword-matched query the
    LLM call changes nothing — it used to still run for anything under 6
    words, e.g. "nowe zamówienia" (2 words), pure wasted latency."""

    @pytest.mark.asyncio
    async def test_short_keyword_query_skips_llm(self):
        orc = _make_orchestrator()
        orc._classify_with_llm = AsyncMock(side_effect=AssertionError("LLM must not be called"))

        source = await orc._classify("nowe zamówienia", [])

        assert source == "allegro_orders"
        orc._classify_with_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_keyword_query_still_skips_llm(self):
        orc = _make_orchestrator()
        orc._classify_with_llm = AsyncMock(side_effect=AssertionError("LLM must not be called"))

        source = await orc._classify("pokaż mi proszę wszystkie moje nowe zamówienia z dzisiaj", [])

        assert source == "allegro_orders"
        orc._classify_with_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_keyword_less_query_still_calls_llm(self):
        orc = _make_orchestrator()
        orc._classify_with_llm = AsyncMock(return_value="none")

        source = await orc._classify("a teraz?", [])

        assert source == "none"
        orc._classify_with_llm.assert_called_once()


class TestEmptyReplyNeverPersisted:
    """A blank agent reply used to be shown as an empty bubble AND stored in the
    session. Replayed as history, Gemini rejected every later message in that
    thread with a non-retryable 400 ("empty text parameter"), so the user got
    "nie udało się przetworzyć tej wiadomości" forever — the whole conversation
    was dead until they started a new one."""

    def _orchestrator_with_session(self):
        from models.conversation import ChannelType, ConversationSession

        orc = _make_orchestrator()
        session = ConversationSession(session_id="s1", channel=ChannelType.API, sender_id="u1")
        orc._session_store.get_or_create_session = AsyncMock(return_value=session)
        orc._session_store.save_session = AsyncMock()
        return orc, session

    def _message(self, text: str):
        from models.conversation import ChannelType, IncomingMessage
        return IncomingMessage(text=text, session_id="s1", channel=ChannelType.API, sender_id="u1")

    @pytest.mark.asyncio
    async def test_blank_reply_replaced_with_fallback(self):
        from models.conversation import AgentResponse, MessageRole

        orc, session = self._orchestrator_with_session()
        orc._classify = AsyncMock(return_value="allegro_orders")
        orc._route = AsyncMock(return_value=AgentResponse(text="   ", agent_type="allegro_orders:chat"))

        response = await orc.handle(self._message("Czy są jakieś faktury do wystawienia"))

        assert response.text.strip()
        stored = [m.content for m in session.messages if m.role == MessageRole.ASSISTANT]
        assert stored and all(c.strip() for c in stored)

    @pytest.mark.asyncio
    async def test_stored_blank_turn_is_not_replayed(self):
        """Belt and braces for sessions already poisoned before the fix."""
        from models.conversation import AgentResponse, MessageRole

        orc, session = self._orchestrator_with_session()
        session.add_message(MessageRole.USER, "pokaż zamówienia")
        session.add_message(MessageRole.ASSISTANT, "")
        orc._classify = AsyncMock(return_value="allegro_orders")
        orc._route = AsyncMock(return_value=AgentResponse(text="ok", agent_type="allegro_orders:chat"))

        await orc.handle(self._message("a faktury?"))

        history = orc._route.call_args[0][2]
        assert all(m["content"].strip() for m in history)

    @pytest.mark.asyncio
    async def test_history_is_capped(self):
        from models.conversation import AgentResponse, MessageRole
        from agents.orchestrator import _HISTORY_TURNS

        orc, session = self._orchestrator_with_session()
        for i in range(_HISTORY_TURNS * 2):
            session.add_message(MessageRole.USER, f"q{i}")
            session.add_message(MessageRole.ASSISTANT, f"a{i}")
        orc._classify = AsyncMock(return_value="none")
        orc._route = AsyncMock(return_value=AgentResponse(text="ok", agent_type="none:chat"))

        await orc.handle(self._message("i co dalej?"))

        assert len(orc._route.call_args[0][2]) == _HISTORY_TURNS


class TestReplyFormatStoredWithTheTurn:
    """A table/document reply has to come back as a table/document when the
    thread is reopened somewhere else, so the stored turn carries the agent
    type the output format is read from."""

    def _orchestrator_with_session(self):
        from models.conversation import ChannelType, ConversationSession

        orc = _make_orchestrator()
        session = ConversationSession(session_id="u1:c1", channel=ChannelType.API, sender_id="u1")
        orc._session_store.get_or_create_session = AsyncMock(return_value=session)
        orc._session_store.save_session = AsyncMock()
        return orc, session

    def _message(self, text: str):
        from models.conversation import ChannelType, IncomingMessage
        return IncomingMessage(text=text, session_id="u1:c1", channel=ChannelType.API, sender_id="u1")

    @pytest.mark.asyncio
    async def test_assistant_turn_records_the_agent_type(self):
        from models.conversation import AgentResponse, MessageRole

        orc, session = self._orchestrator_with_session()
        orc._classify = AsyncMock(return_value="allegro_orders")
        orc._route = AsyncMock(
            return_value=AgentResponse(text="| nr |", agent_type="allegro_orders:table"),
        )

        await orc.handle(self._message("pokaż zamówienia"))

        assistant = [m for m in session.messages if m.role == MessageRole.ASSISTANT]
        assert assistant[-1].metadata["agent"] == "allegro_orders:table"


class TestMarkRequest:
    """See agents/orchestrator.py._mark_request — flags the first request
    this process handles as a likely Cloud Run cold start (--min-instances=0)
    for services.analytics_service.log_perf(), since that container-boot
    time happens before any StageTimer starts."""

    def test_first_call_is_cold(self, monkeypatch):
        import agents.orchestrator as orch
        monkeypatch.setattr(orch, "_requests_handled", 0)

        is_cold, _ = orch._mark_request()

        assert is_cold is True

    def test_second_call_is_warm(self, monkeypatch):
        import agents.orchestrator as orch
        monkeypatch.setattr(orch, "_requests_handled", 0)

        orch._mark_request()
        is_cold, _ = orch._mark_request()

        assert is_cold is False

    def test_returns_seconds_since_process_start(self, monkeypatch):
        import agents.orchestrator as orch
        monkeypatch.setattr(orch, "_requests_handled", 0)
        monkeypatch.setattr(orch, "_PROCESS_STARTED_AT", orch.time.time() - 5)

        _, since_start_s = orch._mark_request()

        assert since_start_s >= 5


class TestInvoiceReminderGetsTheOpenQuestion:
    """The invoice reminder gets first look at every message, and its state
    lives in Redis rather than in the session — so on its own it cannot tell
    a "Tak" meant for it from a "Tak" answering the question the assistant
    asked a second ago. It once took the seller's "Tak" to "Masz 1 nową
    wiadomość (od: Modelinarnia). Pokazać szczegóły?" and issued 3 real VAT
    invoices. handle() must hand it the thread's last assistant turn."""

    def _orchestrator_with_session(self):
        from models.conversation import ChannelType, ConversationSession

        orc = _make_orchestrator()
        session = ConversationSession(session_id="s1", channel=ChannelType.API, sender_id="u1")
        orc._session_store.get_or_create_session = AsyncMock(return_value=session)
        orc._session_store.save_session = AsyncMock()
        return orc, session

    def _message(self, text: str):
        from models.conversation import ChannelType, IncomingMessage
        return IncomingMessage(text=text, session_id="s1", channel=ChannelType.API, sender_id="u1")

    @pytest.mark.asyncio
    async def test_last_assistant_turn_is_passed_to_the_reminder(self):
        from models.conversation import AgentResponse, MessageRole

        orc, session = self._orchestrator_with_session()
        session.add_message(MessageRole.USER, "Czy mam jakieś nowe wiadomości?")
        session.add_message(
            MessageRole.ASSISTANT,
            "Masz **1** nową wiadomość (od: Modelinarnia). Pokazać szczegóły?",
        )
        orc._classify = AsyncMock(return_value="allegro_messages")
        orc._route = AsyncMock(return_value=AgentResponse(text="ok", agent_type="allegro_messages:chat"))
        handle_reply = AsyncMock(return_value=None)

        with patch("services.invoice_reminder.handle_reply", handle_reply):
            await orc.handle(self._message("Tak"), user_id="u1")

        handle_reply.assert_awaited_once_with(
            "u1", "Tak", "Masz **1** nową wiadomość (od: Modelinarnia). Pokazać szczegóły?",
        )

    @pytest.mark.asyncio
    async def test_falling_through_still_routes_normally(self):
        from models.conversation import AgentResponse, MessageRole

        orc, session = self._orchestrator_with_session()
        session.add_message(MessageRole.ASSISTANT, "Masz **1** nową wiadomość. Pokazać szczegóły?")
        orc._classify = AsyncMock(return_value="allegro_messages")
        orc._route = AsyncMock(
            return_value=AgentResponse(text="treść wiadomości", agent_type="allegro_messages:chat"),
        )

        with patch("services.invoice_reminder.handle_reply", AsyncMock(return_value=None)):
            response = await orc.handle(self._message("Tak"), user_id="u1")

        assert response.text == "treść wiadomości"
        orc._route.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_thread_passes_none(self):
        from models.conversation import AgentResponse

        orc, _ = self._orchestrator_with_session()
        orc._classify = AsyncMock(return_value="none")
        orc._route = AsyncMock(return_value=AgentResponse(text="ok", agent_type="none:chat"))
        handle_reply = AsyncMock(return_value=None)

        with patch("services.invoice_reminder.handle_reply", handle_reply):
            await orc.handle(self._message("tak"), user_id="u1")

        assert handle_reply.await_args[0][2] is None

    @pytest.mark.asyncio
    async def test_handled_reply_short_circuits_routing(self):
        orc, _ = self._orchestrator_with_session()
        orc._classify = AsyncMock()
        orc._route = AsyncMock()

        with patch("services.invoice_reminder.handle_reply", AsyncMock(return_value="Wystawiam 1 fakturę:")):
            response = await orc.handle(self._message("tak, wystaw faktury"), user_id="u1")

        assert response.agent_type == "invoice_reminder"
        assert response.text == "Wystawiam 1 fakturę:"
        orc._route.assert_not_awaited()


class TestLastAssistantText:
    def _session(self):
        from models.conversation import ChannelType, ConversationSession
        return ConversationSession(session_id="s1", channel=ChannelType.API, sender_id="u1")

    def test_none_when_the_assistant_has_not_spoken(self):
        from agents.orchestrator import _last_assistant_text
        from models.conversation import MessageRole

        session = self._session()
        session.add_message(MessageRole.USER, "cześć")
        assert _last_assistant_text(session) is None

    def test_returns_the_most_recent_assistant_turn(self):
        from agents.orchestrator import _last_assistant_text
        from models.conversation import MessageRole

        session = self._session()
        session.add_message(MessageRole.ASSISTANT, "pierwsza")
        session.add_message(MessageRole.USER, "ok")
        session.add_message(MessageRole.ASSISTANT, "druga")
        session.add_message(MessageRole.USER, "tak")
        assert _last_assistant_text(session) == "druga"

    def test_blank_assistant_turns_are_skipped(self):
        """A stored empty reply must not mask the real open question."""
        from agents.orchestrator import _last_assistant_text
        from models.conversation import MessageRole

        session = self._session()
        session.add_message(MessageRole.ASSISTANT, "Pokazać szczegóły?")
        session.add_message(MessageRole.USER, "tak")
        session.add_message(MessageRole.ASSISTANT, "   ")
        assert _last_assistant_text(session) == "Pokazać szczegóły?"
