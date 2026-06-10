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
    model_override = None  # resolved to gemini_model_fast in __init__
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
        self.model_override = self._settings.gemini_model_fast
        self._allegro = AllegroService()

    async def run(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        context: str | None = None,
    ) -> AgentResponse:
        # Try to restore tokens from Redis when file storage was wiped (e.g. after redeploy)
        if self._allegro._tokens is None:
            await self._allegro._load_tokens_from_redis()

        # Still no tokens — trigger auth flow
        if self._allegro._tokens is None:
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
                "Aby uzyskać dostęp do Twojego sklepu Allegro, potrzebuję autoryzacji.\n\n"
                f"Otwórz poniższy link i zatwierdź dostęp:\n{url}"
            )
            if user_code:
                text += f"\n\nJeśli zostaniesz poproszony o kod, wpisz: **{user_code}**"
            text += "\n\nPo zatwierdzeniu wyślij swoje pytanie jeszcze raz."
        except Exception as exc:
            logger.error("Failed to start Allegro device flow: %s", exc)
            text = "Wymagana autoryzacja Allegro, ale nie udało się jej uruchomić. Spróbuj ponownie za chwilę."
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
                "Nie udało mi się jeszcze potwierdzić autoryzacji Allegro.\n\n"
                "Czy otworzyłeś link i zatwierdziłeś dostęp na Allegro? "
                "Jeśli tak, wyślij swoje pytanie jeszcze raz. "
                "Jeśli nie, najpierw zatwierdź dostęp."
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
            return "Sesja Allegro wygasła. Wyślij dowolne pytanie, aby rozpocząć ponowną autoryzację."
        except AllegroAPIError as exc:
            logger.error("Allegro API error: %s", exc)
            return "An internal error occurred while contacting Allegro. Please try again."
        except Exception as exc:
            logger.exception("Unexpected error in Allegro tool %s: %s", tool_name, exc)
            return "An internal error occurred. Please try again."

    # ── Formatting helpers ────────────────────────────────────────────────────

    @staticmethod
    def _format_price(amount: float, currency: str = "PLN") -> str:
        return f"{amount:.2f}".replace(".", ",") + f" {currency}"

    @classmethod
    def _order_block(cls, o: Any, extra_lines: list[str] | None = None) -> str:
        """Render a single order as a markdown bullet-point block."""
        price = cls._format_price(o.total_price, o.currency)
        delivery = o.delivery.get("method", {}).get("name", "—")
        total_qty = sum(li.quantity for li in o.line_items)
        link = f"https://allegro.pl/sprzedaz/zamowienia/{o.order_id}"
        lines = [
            f"**Zamówienie** `{o.order_id}`",
            f"- Kupujący: **{o.buyer_login}**",
            f"- Wartość: **{price}**",
            f"- Dostawa: {delivery}",
            f"- Produkty: {total_qty} szt.",
        ]
        if extra_lines:
            lines.extend(f"- {l}" for l in extra_lines)
        lines.append(f"- Link: {link}")
        return "\n".join(lines)

    # ── Tool dispatch ─────────────────────────────────────────────────────────

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
                return "Brak zamówień spełniających podane kryteria."
            return "\n\n".join(self._order_block(o) for o in orders)

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

        if tool_name == "get_orders_delivery":
            orders = await self._allegro.get_orders(
                status=tool_input.get("status", "READY_FOR_PROCESSING"),
                fulfillment_status=tool_input.get("fulfillment_status"),
                limit=min(int(tool_input.get("limit", 20)), 50),
            )
            if not orders:
                return "Brak zamówień spełniających podane kryteria."
            blocks = []
            for o in orders:
                d = o.delivery
                tracking = (
                    d.get("smart", {}).get("trackingCode")
                    or d.get("trackingCode")
                    or "—"
                )
                pickup = d.get("pickupPoint", {})
                extra = [f"Numer śledzenia: {tracking}"]
                if pickup:
                    extra.append(f"Punkt odbioru: {pickup.get('name', '—')}")
                blocks.append(self._order_block(o, extra_lines=extra))
            return "\n\n".join(blocks)

        if tool_name == "get_orders_pending_invoice":
            orders = await self._allegro.get_orders_needing_invoice(
                limit=min(int(tool_input.get("limit", 50)), 50)
            )
            if not orders:
                return "Brak zamówień wymagających wystawienia faktury."
            header = f"**Zamówień bez faktury: {len(orders)}**\n"
            blocks = []
            for o in orders:
                items_str = ", ".join(f"{li.offer_name} ×{li.quantity}" for li in o.line_items[:3])
                extra = [
                    f"E-mail: {o.buyer_email}",
                    f"Data: {o.created_at[:10] if o.created_at else '—'}",
                    f"Produkty: {items_str}",
                    "**Faktura: niewystawiona**",
                ]
                blocks.append(self._order_block(o, extra_lines=extra))
            return header + "\n\n".join(blocks)

        return f"Unknown tool: {tool_name}"
