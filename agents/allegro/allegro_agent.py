from __future__ import annotations

"""
Allegro Agent — handles all Allegro marketplace operations.

Receives queries from the orchestrator, uses Allegro API tools
to fetch/update data, and returns structured responses.
"""

import json
import logging
from typing import Any

from agents.allegro.allegro_tools import ALLEGRO_TOOLS
from agents.base_agent import BaseAgent
from models.conversation import AgentResponse
from services.allegro_service import AllegroAPIError, AllegroAuthError, AllegroService

logger = logging.getLogger(__name__)


class AllegroAgent(BaseAgent):
    """
    Specialized agent for all Allegro marketplace operations.
    Uses claude_model_fast by default — tool-heavy API calls don't need the largest model.
      - Order management & tracking
      - Offer listing, price, and stock updates
      - Buyer messaging
      - Account and billing information
    """

    agent_name = "allegro"
    model_override = None  # resolved to claude_model_fast in __init__
    system_prompt = (
        "You are an AI assistant specialized in managing an Allegro (Polish e-commerce) store. "
        "You have access to the store's orders, offers, and messaging system. "
        "When answering questions about orders or offers, always fetch fresh data via tools. "
        "Present information clearly and concisely. "
        "When showing prices, always include the PLN currency. "
        "Respond in the same language as the user's question (Polish or English). "
        "Do not make up or guess any order IDs or prices — always retrieve them via tools."
    )

    def __init__(self):
        super().__init__()
        self.model_override = self._settings.claude_model_fast
        self._allegro = AllegroService()

    async def run(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        context: str | None = None,
    ) -> AgentResponse:
        # No tokens at all
        if self._allegro._tokens is None:
            # A device flow was already started — try to complete it
            if self._allegro._pending_device_code:
                return await self._try_complete_auth(query, conversation_history, context)
            return await self._request_auth()

        # Tokens exist but expired — try refresh, fall back to fresh device flow
        if self._allegro._tokens.is_expired():
            try:
                await self._allegro._refresh_tokens()
            except AllegroAuthError:
                return await self._request_auth()

        return await super().run(query, conversation_history, context)

    async def _request_auth(self) -> AgentResponse:
        import asyncio
        try:
            flow = await self._allegro.start_device_flow()
            url = flow.get("verification_uri_complete") or flow.get("verification_uri", "")
            user_code = flow.get("user_code", "")
            device_code = flow.get("device_code", "")
            interval = int(flow.get("interval", 5))
            # Background polling — saves tokens to disk automatically when approved
            asyncio.create_task(self._allegro.poll_device_flow(device_code, interval))
            text = (
                "To access your Allegro store I need authorization first.\n\n"
                f"Please open this link and approve the request:\n{url}"
            )
            if user_code:
                text += f"\n\nIf prompted for a code, enter: **{user_code}**"
            text += "\n\nOnce you've approved, send me your question again."
        except Exception as exc:
            logger.error("Failed to start Allegro device flow: %s", exc)
            text = "Allegro authorization is required but could not be started due to an internal error. Please try again later."
        return AgentResponse(text=text, agent_type=self.agent_name)

    async def _try_complete_auth(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None,
        context: str | None,
    ) -> AgentResponse:
        try:
            success = await self._allegro.try_complete_device_flow()
        except AllegroAuthError as exc:
            # Hard failure (expired device code, denied, etc.) — start fresh
            logger.warning("Device flow completion failed: %s — starting fresh", exc)
            return await self._request_auth()

        if success:
            # Tokens obtained — answer the original question immediately
            return await super().run(query, conversation_history, context)

        return AgentResponse(
            text=(
                "I wasn't able to confirm your Allegro authorization yet.\n\n"
                "Did you open the link and approve the request on Allegro? "
                "If yes, please send your question again and I'll try once more. "
                "If not, please approve it first."
            ),
            agent_type=self.agent_name,
        )

    def _get_tools(self) -> list[dict[str, Any]]:
        return ALLEGRO_TOOLS

    async def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        try:
            return await self._dispatch(tool_name, tool_input)
        except AllegroAuthError as exc:
            # Tokens expired mid-session; surface clearly (run() handles pre-check)
            logger.error("Allegro auth error during tool call: %s", exc)
            return "Allegro authentication expired mid-session. Please re-authorize and retry."
        except AllegroAPIError as exc:
            logger.error("Allegro API error: %s", exc)
            return "An internal error occurred while contacting Allegro. Please try again."
        except Exception as exc:
            logger.exception("Unexpected error in Allegro tool %s: %s", tool_name, exc)
            return "An internal error occurred. Please try again."

    async def _dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        if tool_name == "get_orders":
            orders = await self._allegro.get_orders(
                status=tool_input.get("status"),
                buyer_login=tool_input.get("buyer_login"),
                fulfillment_status=tool_input.get("fulfillment_status"),
                line_items_sent=tool_input.get("line_items_sent"),
                limit=min(int(tool_input.get("limit", 10)), 50),
            )
            if not orders:
                return "No orders found matching the given criteria."
            lines = []
            for o in orders:
                items_str = ", ".join(f"{li.offer_name} x{li.quantity}" for li in o.line_items[:3])
                lines.append(
                    f"Order {o.order_id} | Buyer: {o.buyer_login} | "
                    f"Status: {o.status} | Total: {o.total_price} {o.currency} | "
                    f"Items: {items_str}"
                )
            return "\n".join(lines)

        if tool_name == "get_order_details":
            order = await self._allegro.get_order(tool_input["order_id"])
            items_str = "\n".join(
                f"  - {li.offer_name} (ID: {li.offer_id}): {li.quantity} × {li.price} {li.currency}"
                for li in order.line_items
            )
            delivery = order.delivery
            return (
                f"Order ID: {order.order_id}\n"
                f"Buyer: {order.buyer_login} ({order.buyer_email})\n"
                f"Status: {order.status}\n"
                f"Total: {order.total_price} {order.currency}\n"
                f"Created: {order.created_at}\n"
                f"Delivery method: {delivery.get('method', {}).get('name', 'N/A')}\n"
                f"Delivery status: {delivery.get('smart', {}).get('trackingCode', 'N/A')}\n"
                f"Items:\n{items_str}"
            )

        if tool_name == "get_active_offers":
            offers = await self._allegro.get_offers(
                name=tool_input.get("name"),
                limit=min(int(tool_input.get("limit", 10)), 50),
            )
            if not offers:
                return "No active offers found."
            lines = []
            for o in offers:
                price = o.get("sellingMode", {}).get("price", {})
                stock = o.get("stock", {})
                lines.append(
                    f"Offer {o.get('id')} | {o.get('name', 'N/A')} | "
                    f"Price: {price.get('amount')} {price.get('currency', 'PLN')} | "
                    f"Stock: {stock.get('available', 'N/A')}"
                )
            return "\n".join(lines)

        if tool_name == "get_offer_details":
            offer = await self._allegro.get_offer(tool_input["offer_id"])
            return json.dumps(offer, ensure_ascii=False, indent=2)[:3000]

        if tool_name == "update_offer_price":
            price = float(tool_input["price"])
            if price <= 0:
                return "Error: price must be greater than 0."
            result = await self._allegro.update_offer_price(tool_input["offer_id"], price)
            return f"Price updated successfully. New price: {price} PLN. Response: {json.dumps(result)[:200]}"

        if tool_name == "update_offer_stock":
            available = int(tool_input["available"])
            if available < 0:
                return "Error: stock quantity cannot be negative."
            result = await self._allegro.update_offer_stock(tool_input["offer_id"], available)
            return f"Stock updated to {available}. Response: {json.dumps(result)[:200]}"

        if tool_name == "send_message_to_buyer":
            result = await self._allegro.send_message(tool_input["thread_id"], tool_input["text"])
            return f"Message sent successfully. Message ID: {result.get('id', 'N/A')}"

        if tool_name == "get_message_threads":
            threads = await self._allegro.get_message_threads(
                limit=min(int(tool_input.get("limit", 10)), 50)
            )
            if not threads:
                return "No message threads found."
            lines = []
            for t in threads:
                lines.append(
                    f"Thread {t.get('id')} | Subject: {t.get('subject', {}).get('name', 'N/A')} | "
                    f"Unread: {t.get('hasUnreadMessages', False)} | "
                    f"Last message: {t.get('lastMessageCreatedAt', 'N/A')}"
                )
            return "\n".join(lines)

        if tool_name == "get_account_info":
            info = await self._allegro.get_user_info()
            return (
                f"Login: {info.get('login', 'N/A')}\n"
                f"Email: {info.get('email', 'N/A')}\n"
                f"Company: {info.get('company', {}).get('name', 'Individual seller')}\n"
                f"Registered: {info.get('registeredAt', 'N/A')}"
            )

        if tool_name == "get_billing_summary":
            entries = await self._allegro.get_billing_entries(
                limit=min(int(tool_input.get("limit", 10)), 50)
            )
            if not entries:
                return "No billing entries found."
            lines = []
            for e in entries:
                amount = e.get("value", {})
                lines.append(
                    f"{e.get('occurredAt', 'N/A')} | {e.get('type', {}).get('description', 'N/A')} | "
                    f"{amount.get('amount', 'N/A')} {amount.get('currency', 'PLN')}"
                )
            return "\n".join(lines)

        return f"Unknown tool: {tool_name}"
