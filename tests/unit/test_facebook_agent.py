"""Unit tests for agents/communication/facebook_agent.py."""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


def _make_agent():
    from agents.communication.facebook_agent import FacebookCommunicationAgent
    with patch("agents.communication.facebook_agent.FacebookService"):
        agent = FacebookCommunicationAgent()
    return agent


class TestSplitMessage:
    def _split(self, text):
        from agents.communication.facebook_agent import FacebookCommunicationAgent
        return FacebookCommunicationAgent._split_message(text)

    def test_short_message_not_split(self):
        text = "Hello world"
        assert self._split(text) == [text]

    def test_exactly_at_limit_not_split(self):
        from agents.communication.facebook_agent import MAX_MESSAGE_LENGTH
        text = "x" * MAX_MESSAGE_LENGTH
        chunks = self._split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_at_newline(self):
        from agents.communication.facebook_agent import MAX_MESSAGE_LENGTH
        part1 = "A" * 1800
        part2 = "B" * 200
        text = part1 + "\n" + part2
        chunks = self._split(text)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= MAX_MESSAGE_LENGTH

    def test_very_long_no_boundary_splits_hard(self):
        from agents.communication.facebook_agent import MAX_MESSAGE_LENGTH
        text = "A" * (MAX_MESSAGE_LENGTH * 3)
        chunks = self._split(text)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk) <= MAX_MESSAGE_LENGTH

    def test_empty_string(self):
        assert self._split("") == [""]


class TestParseEvent:
    def _agent(self):
        return _make_agent()

    def test_text_message(self):
        agent = self._agent()
        event = {
            "sender": {"id": "user-1"},
            "message": {"text": "Hello", "mid": "m1"},
        }
        msg = agent._parse_event(event, page_id="page-1")
        assert msg is not None
        assert msg.sender_id == "user-1"
        assert msg.text == "Hello"
        assert msg.session_id == "fb_user-1"

    def test_echo_ignored(self):
        agent = self._agent()
        event = {
            "sender": {"id": "user-1"},
            "message": {"text": "echo", "is_echo": True},
        }
        assert agent._parse_event(event, page_id="page-1") is None

    def test_own_page_echo_ignored(self):
        agent = self._agent()
        event = {
            "sender": {"id": "page-1"},
            "message": {"text": "hi"},
        }
        assert agent._parse_event(event, page_id="page-1") is None

    def test_empty_sender_ignored(self):
        agent = self._agent()
        event = {"sender": {"id": ""}, "message": {"text": "hi"}}
        assert agent._parse_event(event, page_id="page-1") is None

    def test_attachment_message(self):
        agent = self._agent()
        event = {
            "sender": {"id": "user-1"},
            "message": {
                "attachments": [{"type": "image", "payload": {"url": "http://img"}}]
            },
        }
        msg = agent._parse_event(event, page_id="page-1")
        assert msg is not None
        assert "image" in msg.text

    def test_no_text_no_attachments_ignored(self):
        agent = self._agent()
        event = {
            "sender": {"id": "user-1"},
            "message": {"text": ""},
        }
        assert agent._parse_event(event, page_id="page-1") is None

    def test_postback(self):
        agent = self._agent()
        event = {
            "sender": {"id": "user-1"},
            "postback": {"payload": "GET_STARTED", "title": "Get Started"},
        }
        msg = agent._parse_event(event, page_id="page-1")
        assert msg is not None
        assert msg.text == "GET_STARTED"

    def test_unknown_event_returns_none(self):
        agent = self._agent()
        event = {"sender": {"id": "user-1"}, "read": {"watermark": 123}}
        assert agent._parse_event(event, page_id="page-1") is None


class TestParseIncoming:
    @pytest.mark.asyncio
    async def test_parse_single_message(self):
        agent = _make_agent()
        payload = {
            "entry": [{
                "id": "page-1",
                "messaging": [{
                    "sender": {"id": "user-1"},
                    "message": {"text": "hi"},
                }]
            }]
        }
        msgs = await agent.parse_incoming(payload)
        assert len(msgs) == 1
        assert msgs[0].text == "hi"

    @pytest.mark.asyncio
    async def test_parse_empty_payload(self):
        agent = _make_agent()
        msgs = await agent.parse_incoming({"entry": []})
        assert msgs == []

    @pytest.mark.asyncio
    async def test_parse_multiple_messages(self):
        agent = _make_agent()
        payload = {
            "entry": [{
                "id": "page-1",
                "messaging": [
                    {"sender": {"id": "u1"}, "message": {"text": "first"}},
                    {"sender": {"id": "u2"}, "message": {"text": "second"}},
                ]
            }]
        }
        msgs = await agent.parse_incoming(payload)
        assert len(msgs) == 2
