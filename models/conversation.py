from __future__ import annotations

"""Conversation and message data models."""

from datetime import datetime
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
    """Single turn in a conversation (stored in Firestore)."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationSession(BaseModel):
    """Full conversation session persisted in Firestore."""

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

    def to_anthropic_messages(self) -> list[dict[str, str]]:
        """Convert to Anthropic API messages format (user/assistant alternation)."""
        result = []
        for msg in self.messages:
            if msg.role in (MessageRole.USER, MessageRole.ASSISTANT):
                result.append({"role": msg.role.value, "content": msg.content})
        return result


class AgentResponse(BaseModel):
    """Structured response produced by an agent."""

    text: str
    agent_type: str
    confidence: float = 1.0
    sources: list[str] = Field(default_factory=list)  # RAG document sources
    metadata: dict[str, Any] = Field(default_factory=dict)
