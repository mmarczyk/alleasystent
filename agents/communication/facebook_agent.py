from __future__ import annotations

"""
Facebook Messenger communication agent.

Parses Messenger webhook payloads and delivers responses via the
Facebook Graph API Send API.  Handles message splitting for long responses,
typing indicators, and postback events.
"""

import logging

from agents.communication.base_communication import BaseCommunicationAgent
from models.conversation import AgentResponse, ChannelType, IncomingMessage
from services.facebook_service import FacebookService

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 1900  # keep below FB's 2000-char limit with safety margin


class FacebookCommunicationAgent(BaseCommunicationAgent):
    """
    Handles Facebook Messenger webhook events and message delivery.

    Webhook events handled:
      - messages (text, attachments)
      - messaging_postbacks (quick reply / button postbacks)

    Ignored:
      - message_reads, message_reactions, message_echoes
    """

    channel_name = "facebook"

    def __init__(self):
        self._fb = FacebookService()

    async def parse_incoming(self, raw_payload: dict) -> list[IncomingMessage]:
        """
        Parse a Facebook Messenger webhook POST body.

        Facebook sends batched events; this method normalizes them all.
        """
        messages: list[IncomingMessage] = []

        entries = raw_payload.get("entry", [])
        for entry in entries:
            page_id = str(entry.get("id", ""))
            for event in entry.get("messaging", []):
                msg = self._parse_event(event, page_id)
                if msg:
                    messages.append(msg)

        return messages

    def _parse_event(self, event: dict, page_id: str) -> IncomingMessage | None:
        sender_id = str(event.get("sender", {}).get("id", ""))
        if not sender_id or sender_id == page_id:
            return None  # ignore page's own echoes

        # Text message
        if message := event.get("message"):
            if message.get("is_echo"):
                return None
            text = message.get("text", "")
            if not text:
                # Handle attachments (images, files)
                attachments = message.get("attachments", [])
                if attachments:
                    text = f"[Attachment: {attachments[0].get('type', 'unknown')}]"
                else:
                    return None

            return IncomingMessage(
                channel=ChannelType.FACEBOOK,
                sender_id=sender_id,
                session_id=f"fb_{sender_id}",
                text=text,
                attachments=message.get("attachments", []),
                raw_payload=event,
            )

        # Postback (button / quick reply)
        if postback := event.get("postback"):
            payload = postback.get("payload", "")
            title = postback.get("title", payload)
            return IncomingMessage(
                channel=ChannelType.FACEBOOK,
                sender_id=sender_id,
                session_id=f"fb_{sender_id}",
                text=f"[Button: {title}]" if not payload else payload,
                raw_payload=event,
            )

        return None

    async def send_response(self, message: IncomingMessage, response: AgentResponse) -> None:
        """
        Deliver response to the Facebook user.

        Shows typing indicator, then sends the response split into
        chunks if it exceeds Facebook's message length limit.
        """
        await self._fb.send_typing_indicator(message.sender_id, on=True)

        chunks = self._split_message(response.text)
        for chunk in chunks:
            await self._fb.send_text_message(message.sender_id, chunk)

        await self._fb.send_typing_indicator(message.sender_id, on=False)
        logger.info(
            "Sent %d chunk(s) to FB user %s", len(chunks), message.sender_id
        )

    @staticmethod
    def _split_message(text: str) -> list[str]:
        """Split long messages at paragraph/sentence boundaries."""
        if len(text) <= MAX_MESSAGE_LENGTH:
            return [text]

        chunks: list[str] = []
        while len(text) > MAX_MESSAGE_LENGTH:
            split_at = text.rfind("\n", 0, MAX_MESSAGE_LENGTH)
            if split_at == -1:
                split_at = text.rfind(". ", 0, MAX_MESSAGE_LENGTH)
            if split_at == -1:
                split_at = MAX_MESSAGE_LENGTH
            else:
                split_at += 1
            chunks.append(text[:split_at].strip())
            text = text[split_at:].strip()
        if text:
            chunks.append(text)
        return chunks
