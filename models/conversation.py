from __future__ import annotations

"""Conversation and message data models."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChannelType(str, Enum):
    FACEBOOK = "facebook"
    WHATSAPP = "whatsapp"  # prepared for future use
    API = "api"


class IncomingMessage(BaseModel):
    """Normalized message received from any communication channel."""

    channel: ChannelType
    sender_id: str
    sender_name: str | None = None
    session_id: str  # unique per conversation thread
    text: str
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    """Single turn in a conversation (stored via SessionStore)."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationSession(BaseModel):
    """Full conversation session persisted via SessionStore (Redis/in-memory)."""

    session_id: str
    channel: ChannelType
    sender_id: str
    messages: list[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_message(self, role: MessageRole, content: str, metadata: dict | None = None) -> None:
        self.messages.append(
            ConversationMessage(role=role, content=content, metadata=metadata or {})
        )
        self.updated_at = datetime.utcnow()

    def to_anthropic_messages(
        self, limit: int | None = None, max_age_hours: float | None = None,
    ) -> list[dict[str, str]]:
        """Convert to Anthropic API messages format (user/assistant alternation).

        Blank turns are skipped: a stored empty assistant reply makes Gemini
        reject the whole request with a non-retryable 400 ("empty text
        parameter"), which would break every later message in the thread —
        see agents/base_agent.py.sanitize_messages.

        `limit` keeps only the most recent N turns. Sessions live for 30 days,
        so an old thread otherwise replays hundreds of turns (including full
        order/offer listings) into every single request.

        `max_age_hours`, when given, additionally drops any turn older than
        that — a seller returning after a long gap doesn't need days-old
        order listings replayed into the tool-selection/interpret prompts,
        and a stale thread was otherwise still paying full `limit` turns of
        token cost for context that's no longer relevant to a fresh query.
        Applied before `limit`, so the final result is at most `limit` turns
        that are also within the age window.
        """
        messages = self.messages
        if max_age_hours is not None:
            cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
            messages = [m for m in messages if m.timestamp >= cutoff]
        result = []
        for msg in messages:
            if msg.role in (MessageRole.USER, MessageRole.ASSISTANT) and msg.content.strip():
                result.append({"role": msg.role.value, "content": msg.content})
        return result[-limit:] if limit else result


class AgentResponse(BaseModel):
    """Structured response produced by an agent."""

    text: str
    agent_type: str
    confidence: float = 1.0
    sources: list[str] = Field(default_factory=list)  # RAG document sources
    metadata: dict[str, Any] = Field(default_factory=dict)
