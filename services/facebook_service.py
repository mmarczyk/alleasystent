from __future__ import annotations

"""Facebook Messenger Graph API client."""

import hashlib
import hmac
import logging
from typing import Any

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)

GRAPH_API_URL = "https://graph.facebook.com/v19.0"


class FacebookService:
    """Sends messages and manages Facebook Messenger interactions."""

    def __init__(self):
        self._settings = get_settings()
        self._client = httpx.AsyncClient(base_url=GRAPH_API_URL, timeout=15.0)

    def verify_webhook_signature(self, payload: bytes, signature_header: str) -> bool:
        """Validate X-Hub-Signature-256 header to ensure payload is from Facebook."""
        if not signature_header.startswith("sha256="):
            return False
        expected = hmac.new(
            self._settings.facebook_app_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(f"sha256={expected}", signature_header)

    async def send_text_message(self, recipient_id: str, text: str) -> dict[str, Any]:
        """Send a plain text message to a Messenger user."""
        body = {
            "recipient": {"id": recipient_id},
            "message": {"text": text[:2000]},  # FB limit
            "messaging_type": "RESPONSE",
        }
        return await self._call_send_api(body)

    async def send_typing_indicator(self, recipient_id: str, on: bool = True) -> None:
        """Show/hide the typing bubble."""
        body = {
            "recipient": {"id": recipient_id},
            "sender_action": "typing_on" if on else "typing_off",
        }
        try:
            await self._call_send_api(body)
        except Exception as exc:
            logger.warning("Typing indicator failed: %s", exc)

    async def send_quick_replies(
        self,
        recipient_id: str,
        text: str,
        replies: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Send a message with quick reply buttons."""
        quick_replies = [
            {"content_type": "text", "title": r["title"], "payload": r.get("payload", r["title"])}
            for r in replies[:11]  # FB limit
        ]
        body = {
            "recipient": {"id": recipient_id},
            "message": {"text": text[:640], "quick_replies": quick_replies},
            "messaging_type": "RESPONSE",
        }
        return await self._call_send_api(body)

    async def send_generic_template(
        self,
        recipient_id: str,
        elements: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Send a horizontal scroll card template."""
        body = {
            "recipient": {"id": recipient_id},
            "message": {
                "attachment": {
                    "type": "template",
                    "payload": {
                        "template_type": "generic",
                        "elements": elements[:10],
                    },
                }
            },
            "messaging_type": "RESPONSE",
        }
        return await self._call_send_api(body)

    async def get_user_profile(self, user_id: str) -> dict[str, Any]:
        """Fetch basic profile info (name, locale) for a sender."""
        resp = await self._client.get(
            f"/{user_id}",
            params={
                "fields": "first_name,last_name,locale",
                "access_token": self._settings.facebook_page_access_token,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def _call_send_api(self, body: dict[str, Any]) -> dict[str, Any]:
        resp = await self._client.post(
            "/me/messages",
            params={"access_token": self._settings.facebook_page_access_token},
            json=body,
        )
        if resp.status_code >= 400:
            logger.error("Facebook Send API error %d: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._client.aclose()
