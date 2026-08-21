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
