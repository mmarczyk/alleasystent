"""Unit tests for agents/orchestrator.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


def _make_orchestrator():
    with patch("agents.orchestrator.AsyncOpenAI"), \
         patch("agents.orchestrator.FirestoreService"):
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
