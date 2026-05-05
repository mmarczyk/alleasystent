from __future__ import annotations

"""
Abstract base for communication channel agents.

Each channel (Facebook, WhatsApp, etc.) implements this interface.
The orchestrator works exclusively with IncomingMessage / AgentResponse,
so adding a new channel never requires changes to the core logic.
"""

from abc import ABC, abstractmethod

from models.conversation import AgentResponse, IncomingMessage


class BaseCommunicationAgent(ABC):
    """
    Handles the send/receive contract for a specific messaging channel.

    Responsibilities:
      1. Parse raw webhook payloads into normalized IncomingMessage objects.
      2. Deliver AgentResponse text back to the user via the channel's API.
    """

    channel_name: str = "base"

    @abstractmethod
    async def parse_incoming(self, raw_payload: dict) -> list[IncomingMessage]:
        """
        Parse a raw webhook payload into one or more IncomingMessage objects.
        Returns an empty list if the payload contains no actionable messages.
        """

    @abstractmethod
    async def send_response(self, message: IncomingMessage, response: AgentResponse) -> None:
        """
        Deliver the agent's response back to the sender via this channel.
        Handles channel-specific formatting (message splitting, typing indicators, etc.).
        """
