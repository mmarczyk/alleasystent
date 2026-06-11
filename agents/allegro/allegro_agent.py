from __future__ import annotations

"""
Allegro Agent — handles all Allegro marketplace operations.

Receives queries from the orchestrator, uses Allegro API tools
to fetch/update data, and returns structured responses.
"""

import json
import logging
from collections import Counter, defaultdict
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

    def __init__(self, user_id: str | None = None):
        super().__init__()
        self.model_override = self._settings.gemini_model_fast
        self._allegro = AllegroService(user_id=user_id)

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

    @staticmethod
    def _offer_fields(offer: dict) -> tuple[str, str, float, str, int]:
        """Extract (id, name, price, currency, stock) from raw offer dict."""
        oid = offer.get("id", "?")
        name = offer.get("name", "N/A")
        price_data = (offer.get("sellingMode") or {}).get("price") or {}
        price = float(price_data.get("amount") or 0)
        currency = price_data.get("currency", "PLN")
        stock = int((offer.get("stock") or {}).get("available") or 0)
        return oid, name, price, currency, stock

    @classmethod
    def _aggregate_offers_by_name(cls, offers: list[dict]) -> list[dict]:
        """Group offers by name (case-insensitive), summing stock. Returns list of aggregated dicts."""
        groups: dict[str, dict] = defaultdict(lambda: {"ids": [], "name": "", "price": 0.0, "currency": "PLN", "total_stock": 0})
        for offer in offers:
            oid, name, price, currency, stock = cls._offer_fields(offer)
            key = name.strip().lower()
            g = groups[key]
            g["ids"].append(oid)
            g["name"] = name
            g["price"] = price
            g["currency"] = currency
            g["total_stock"] += stock
        return sorted(groups.values(), key=lambda g: g["total_stock"])

    @classmethod
    def _order_block(cls, o: Any, extra_lines: list[str] | None = None) -> str:
        """Render a single order as a markdown bullet-point block."""
        price = cls._format_price(o.total_price, o.currency)
        delivery = (o.delivery.get("method") or {}).get("name", "—")
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
            delivery = order.delivery if isinstance(order.delivery, dict) else {}
            return (
                f"Order ID: {order.order_id}\n"
                f"Buyer: {order.buyer_login} ({order.buyer_email})\n"
                f"Status: {order.status}\n"
                f"Total: {order.total_price} {order.currency}\n"
                f"Created: {order.created_at}\n"
                f"Delivery method: {(delivery.get('method') or {}).get('name', 'N/A')}\n"
                f"Delivery status: {(delivery.get('smart') or {}).get('trackingCode', 'N/A')}\n"
                f"Items:\n{items_str}"
            )

        if tool_name == "get_active_offers":
            name_filter = tool_input.get("name")
            if name_filter:
                offers = await self._allegro.get_offers(name=name_filter, limit=50)
            else:
                offers = await self._allegro.get_all_offers()
            if not offers:
                return "Brak aktywnych ofert."
            lines = [f"Łącznie **{len(offers)}** aktywnych ofert:\n"]
            for o in offers:
                oid, name, price, currency, stock = self._offer_fields(o)
                lines.append(
                    f"- **{name}** (ID: `{oid}`) — "
                    f"{self._format_price(price, currency)} — "
                    f"stan: **{stock} szt.**"
                )
            return "\n".join(lines)

        if tool_name == "get_offers_summary":
            offers = await self._allegro.get_all_offers()
            if not offers:
                return "Brak aktywnych ofert."
            total = len(offers)
            total_stock = 0
            stock_buckets = {"0 szt.": 0, "1–9 szt.": 0, "10–49 szt.": 0, "50–199 szt.": 0, "200+ szt.": 0}
            price_buckets = {"do 50 zł": 0, "50–200 zł": 0, "200–500 zł": 0, "500+ zł": 0}
            for o in offers:
                _, _, price, _, stock = self._offer_fields(o)
                total_stock += stock
                if stock == 0:
                    stock_buckets["0 szt."] += 1
                elif stock < 10:
                    stock_buckets["1–9 szt."] += 1
                elif stock < 50:
                    stock_buckets["10–49 szt."] += 1
                elif stock < 200:
                    stock_buckets["50–199 szt."] += 1
                else:
                    stock_buckets["200+ szt."] += 1
                if price < 50:
                    price_buckets["do 50 zł"] += 1
                elif price < 200:
                    price_buckets["50–200 zł"] += 1
                elif price < 500:
                    price_buckets["200–500 zł"] += 1
                else:
                    price_buckets["500+ zł"] += 1
            stock_lines = "\n".join(f"  - {k}: **{v}** ofert" for k, v in stock_buckets.items() if v)
            price_lines = "\n".join(f"  - {k}: **{v}** ofert" for k, v in price_buckets.items() if v)
            return (
                f"**Podsumowanie ofert**\n\n"
                f"Łącznie aktywnych ofert: **{total}**\n"
                f"Łączny stan magazynowy: **{total_stock:,} szt.**\n\n"
                f"**Stany magazynowe:**\n{stock_lines}\n\n"
                f"**Ceny:**\n{price_lines}"
            )

        if tool_name == "query_offers_by_stock":
            max_stock = tool_input.get("max_stock")
            min_stock = tool_input.get("min_stock")
            offers = await self._allegro.get_all_offers()
            aggregated = self._aggregate_offers_by_name(offers)
            results = []
            for g in aggregated:
                s = g["total_stock"]
                if max_stock is not None and s > max_stock:
                    continue
                if min_stock is not None and s < min_stock:
                    continue
                results.append(g)
            if not results:
                return "Brak produktów spełniających podane kryteria stanów magazynowych."
            label = []
            if max_stock is not None:
                label.append(f"≤ {max_stock} szt.")
            if min_stock is not None:
                label.append(f"≥ {min_stock} szt.")
            header = f"Znaleziono **{len(results)}** produktów ({', '.join(label) or 'wszystkie'}):\n"
            lines = [header]
            for g in results:
                ids_str = ", ".join(f"`{i}`" for i in g["ids"])
                ofert_str = f"({len(g['ids'])} {'oferta' if len(g['ids']) == 1 else 'ofert'})" if len(g["ids"]) > 1 else ""
                lines.append(
                    f"- **{g['name']}** {ofert_str}— "
                    f"stan łącznie: **{g['total_stock']} szt.** — "
                    f"{self._format_price(g['price'], g['currency'])} — "
                    f"ID: {ids_str}"
                )
            return "\n".join(lines)

        if tool_name == "query_offers_by_price":
            max_price = tool_input.get("max_price")
            min_price = tool_input.get("min_price")
            offers = await self._allegro.get_all_offers()
            results = []
            for o in offers:
                oid, name, price, currency, stock = self._offer_fields(o)
                if max_price is not None and price > max_price:
                    continue
                if min_price is not None and price < min_price:
                    continue
                results.append((oid, name, price, currency, stock))
            if not results:
                return "Brak ofert spełniających podane kryteria cenowe."
            results.sort(key=lambda x: x[2])
            label = []
            if min_price is not None:
                label.append(f"≥ {self._format_price(min_price)}")
            if max_price is not None:
                label.append(f"≤ {self._format_price(max_price)}")
            header = f"Znaleziono **{len(results)}** ofert ({', '.join(label) or 'wszystkie'}):\n"
            lines = [header]
            for oid, name, price, currency, stock in results:
                lines.append(
                    f"- **{name}** — **{self._format_price(price, currency)}** — "
                    f"stan: {stock} szt. — ID: `{oid}`"
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
                    f"Thread {t.get('id')} | Subject: {(t.get('subject') or {}).get('name', 'N/A')} | "
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
            fulfillment_status = tool_input.get("fulfillment_status")
            orders = await self._allegro.get_orders(
                status=tool_input.get("status", "READY_FOR_PROCESSING"),
                fulfillment_status=fulfillment_status,
                limit=min(int(tool_input.get("limit", 50)), 50),
            )
            # When no fulfillment_status given, exclude already-sent orders
            if not fulfillment_status:
                orders = [o for o in orders if o.fulfillment_status not in ("SENT", "PICKED_UP", "CANCELLED")]
            if not orders:
                return "Brak zamówień do wysłania."
            # Group by delivery method for a quick summary
            courier_counts: Counter = Counter()
            blocks = []
            for o in orders:
                d = o.delivery
                method_name = (d.get("method") or {}).get("name") or "—"
                courier_counts[method_name] += 1
                tracking = (
                    (d.get("smart") or {}).get("trackingCode")
                    or d.get("trackingCode")
                    or "—"
                )
                pickup = d.get("pickupPoint") or {}
                extra = [f"Kurier/dostawa: **{method_name}**", f"Numer śledzenia: {tracking}"]
                if pickup:
                    extra.append(f"Punkt odbioru: {pickup.get('name', '—')}")
                blocks.append(self._order_block(o, extra_lines=extra))
            summary = "**Podsumowanie kurierów:**\n" + "\n".join(
                f"- {method}: {count} zamówień" for method, count in courier_counts.most_common()
            )
            return summary + "\n\n---\n\n" + "\n\n".join(blocks)

        if tool_name == "get_orders_pending_invoice":
            orders = await self._allegro.get_orders_needing_invoice(
                month=tool_input.get("month"),
                year=tool_input.get("year"),
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
