from __future__ import annotations

"""
Allegro Agent — handles all Allegro marketplace operations.

Receives queries from the orchestrator, uses Allegro API tools
to fetch/update data, and returns structured responses.
"""

import asyncio
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from agents.allegro.allegro_tools import ALLEGRO_TOOLS, resolve_output_format
from agents.base_agent import BaseAgent
from models.conversation import AgentResponse
from services.allegro_service import AllegroAPIError, AllegroAuthError, AllegroService

logger = logging.getLogger(__name__)

# Injected as a user-turn message right before the final "interpret" call, once
# the output format is known from which tool(s) were just called (see
# agents/allegro/allegro_tools.py TOOL_OUTPUT_FORMAT). "chat" and "action" need
# no extra steering — the model already answers naturally / includes the tool's
# button HTML verbatim per the system prompt.
_OUTPUT_FORMAT_INSTRUCTIONS: dict[str, str] = {
    "table": (
        "[FORMAT ODPOWIEDZI: TABELA]\n"
        "Zbuduj tabelę markdown WYŁĄCZNIE z danych zwróconych przez narzędzie powyżej. "
        "Pierwsza linia — nagłówek: | kolumna1 | kolumna2 | ... . Bez wstępu ani potwierdzenia "
        "przed tabelą. Maksymalnie 1-2 zdania podsumowania PO tabeli.\n"
        "Jeśli narzędzie nie zwróciło żadnych wyników — NIE twórz tabeli (pustej ani "
        "przykładowej), odpowiedz jednym krótkim zdaniem."
    ),
    "dashboard": (
        "[FORMAT ODPOWIEDZI: DASHBOARD]\n"
        "Zbuduj wielosekcyjny raport z danych zwróconych przez narzędzie powyżej: każda sekcja "
        "z nagłówkiem ##, kluczowe liczby pogrubione (**x**), porównania/trendy gdzie to możliwe. "
        "Zacznij od razu od pierwszego nagłówka ## — bez wstępu.\n"
        "Jeśli dane zawierają liczby nadające się do porównania (np. przychód wg produktu, "
        "rozkład stanów magazynowych, koszty vs zysk, top produkty) — dołącz po odpowiedniej "
        "sekcji jeden blok kodu oznaczony jako 'chart' z wykresem w formacie JSON:\n"
        "```chart\n"
        '{"type":"bar","title":"Tytuł wykresu","labels":["Etykieta 1","Etykieta 2"],'
        '"series":[{"name":"Nazwa serii","data":[100,200]}]}\n'
        "```\n"
        "   - type: \"bar\" | \"line\" | \"pie\" | \"doughnut\"\n"
        "   - labels: kategorie/dni/produkty; series: jedna lub więcej serii liczbowych tej "
        "samej długości co labels\n"
        "   - Użyj WYŁĄCZNIE prawdziwych liczb z danych powyżej — nigdy nie zmyślaj; pomiń "
        "blok chart, jeśli danych jest za mało na sensowny wykres."
    ),
    "document": (
        "[FORMAT ODPOWIEDZI: DOKUMENT]\n"
        "Sformatuj odpowiedź jako pełny, gotowy do użycia dokument. Pierwsza linia — nagłówek: "
        "# Tytuł dokumentu. Bez wstępu ani potwierdzenia. Kompletna treść, z podpisem/stopką "
        "jeśli pasuje."
    ),
}

# Per-tool overrides, checked BEFORE the generic per-format instruction above —
# for cases where "chat" isn't specific enough (get_new_orders is "chat" format
# but still needs a fixed shape, not whatever prose/table the model feels like).
_TOOL_SPECIFIC_INSTRUCTIONS: dict[str, str] = {
    "get_new_orders": (
        "[FORMAT ODPOWIEDZI: LISTA PUNKTOWANA — NIGDY TABELA]\n"
        "Przedstaw zamówienia z danych powyżej jako listę punktowaną — jeden blok na "
        "zamówienie, dokładnie jak w danych narzędzia. NIGDY nie buduj tabeli markdown "
        "dla zamówień.\n"
        "Dla każdego zamówienia pokaż TYLKO te pola: Zamawiający, Rodzaj dostawy, "
        "Ilość szt., Wartość, Opłacone (data i godzina). Nie dodawaj innych pól "
        "(bez statusu realizacji, bez linku, bez daty złożenia), chyba że user o nie "
        "poprosi osobno. Bez wstępu przed listą."
    ),
    "get_new_orders:count_only": (
        "[FORMAT ODPOWIEDZI: TYLKO LICZBA]\n"
        "Użytkownik zapytał ILE jest nowych zamówień — odpowiedz JEDNYM krótkim zdaniem podającym "
        "TYLKO liczbę z danych narzędzia powyżej (np. 'Masz 5 nowych zamówień.'). NIE wypisuj "
        "poszczególnych zamówień ani ich szczegółów, chyba że user o to poprosi osobno."
    ),
    "get_order_details": (
        "[FORMAT ODPOWIEDZI: PODSUMOWANIE + DOKUMENT]\n"
        "Odpowiedź MUSI składać się z dwóch części, w tej kolejności:\n"
        "1. Na samym początku, jako PIERWSZA rzecz w odpowiedzi (bez żadnego wstępu) — "
        "blok kodu oznaczony jako 'summary' z krótkim podsumowaniem w formie listy "
        "punktowanej, dokładnie te pola:\n"
        "```summary\n"
        "- Zamówienie: `ID zamówienia`\n"
        "- Kupujący: ...\n"
        "- Status: ...\n"
        "- Wartość: ...\n"
        "- Faktura: ...\n"
        "```\n"
        "   Pole 'Faktura' musi jasno mówić, czy kupujący poprosił o fakturę, a jeśli "
        "tak — czy została już wystawiona (dane z narzędzia to mówią wprost).\n"
        "2. Zaraz po tym bloku — pełny dokument szczegółów: pierwsza linia # Szczegóły "
        "zamówienia {ID}, a w nim sekcje ## Produkty (KAŻDY produkt osobno, z ilością i "
        "ceną), ## Dostawa (metoda, tracking), ## Rozliczenie (wszystkie wpisy "
        "billingowe osobno, jeśli są w danych, plus zysk netto).\n"
        "Użyj WYŁĄCZNIE danych zwróconych przez narzędzie powyżej — nigdy nie zmyślaj "
        "produktów, cen ani statusu faktury."
    ),
}


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
        "Do not make up or guess any order IDs or prices — always retrieve them via tools. "
        "MANDATORY TOOL CALLS — these question types MUST trigger a tool, never be answered from memory:\n"
        "• Order counts / 'ile zamówień' / 'ile jest wszystkich nowych' / 'ile mam nowych' "
        "(user wants a NUMBER, not the list) → get_new_orders with count_only=true. Do NOT return "
        "the whole order list when the user only asked HOW MANY.\n"
        "• Last/newest new order, SINGULAR ('ostatnie zamówienie', 'ostatnie nowe zamówienie', "
        "'najnowsze zamówienie', 'ostatniego nowego zamówienia') → get_new_orders with limit=1. "
        "Do NOT return the whole list when the user asks about just ONE order — check singular vs "
        "plural phrasing carefully before calling.\n"
        "• Unsent / 'niewysłane' / 'do wysłania' → get_orders_delivery (fulfillment_status=READY_FOR_SHIPMENT)\n"
        "• Not packed / 'niespakowane' / 'do spakowania' → get_new_orders (fulfillment_status=NEW)\n"
        "• Delivery providers / 'dostawcy' / 'kurierzy' → get_orders_delivery\n"
        "• 'Jakich mam dostawców/kurierów w zamówieniach do wysłania?' / which couriers are used for "
        "orders needing shipment / group orders by courier → get_orders_delivery. There is NO separate "
        "'get_orders_by_courier' tool — get_orders_delivery already groups and counts orders by courier; "
        "always use it for any question combining orders with courier/delivery-provider grouping.\n"
        "• Order value / 'wartość zamówień' / 'łączna wartość' → get_new_orders, then sum total_price\n"
        "• Offer list / 'lista ofert' / 'zestawienie ofert' / 'pokaż oferty' / 'moje oferty' → get_active_offers\n"
        "• Offer stock / 'stany magazynowe' / 'ile mam' / 'które kończą się' → query_offers_by_stock\n"
        "• Offer prices / 'ceny ofert' / 'drogie oferty' / 'najtańsze' → query_offers_by_price\n"
        "• Reorder/restock email to a SUPPLIER — 'wygeneruj mail z zamówieniem do dostawcy', "
        "'przygotuj zamówienie uzupełniające', 'lista produktów do zamówienia' — "
        "→ get_products_to_reorder (NOT get_orders_delivery — that tool is for couriers/shipping "
        "on existing customer orders, this one is for restocking FROM a supplier)\n"
        "Even if the conversation history contains order or offer data, you MUST re-fetch via tool for any of the above. "
        "ANTI-HALLUCINATION — Allegro offer IDs are always 11-digit numbers (e.g. '12345678901'). "
        "If you find yourself writing UUID-format IDs (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx), "
        "STOP — you are hallucinating. Call get_active_offers instead. "
        "ANTI-HALLUCINATION — TOOL NAMES: only call a tool whose exact name appears in your tool list. "
        "Never invent a plausible-sounding tool name (e.g. 'get_orders_by_courier') just because it "
        "matches the phrasing of the question — find the closest EXISTING tool from the MANDATORY TOOL "
        "CALLS list above instead. If truly no listed tool fits, say so in your reply instead of calling "
        "a name that doesn't exist. "
        "BILLING ENTRIES — CRITICAL RULE: When a tool returns billing entries, you MUST list "
        "EVERY entry as a separate line EXACTLY as received. NEVER group, aggregate, merge, or "
        "summarize entries by type. Each 'Prowizja od sprzedaży' for a different product is a "
        "SEPARATE entry and must be shown as a separate row. Showing 2 rows when there are 5 "
        "entries is WRONG. If the tool says '5 wpisów', show 5 rows, not 2. "
        "• Invoice address / 'dane do faktury' / 'NIP' / 'adres nabywcy' for a specific order (no issuance verb) → get_order_invoice_data\n"
        "• Which orders need an invoice / 'jakie mam faktury do wystawienia' / 'brakujące faktury' "
        "(read-only list, nothing created) → get_orders_pending_invoice (includes address automatically)\n"
        "• ISSUE/CREATE invoice(s) — ONLY with an explicit issuance verb ('wystaw fakturę/faktury', "
        "'wystaw brakujące faktury', 'utwórz fakturę dla zamówienia X'):\n"
        "  - ONE specific order named (a concrete order_id from context or given directly by the "
        "user) → issue_invoice_for_order. This is REAL — it actually creates the invoice in inFakt. "
        "Never invent or guess an order_id; if you don't have one, ask the user or look it up first.\n"
        "  - Batch / whole month, no single order named ('wystaw wszystkie faktury', 'wystaw brakujące "
        "faktury za ten miesiąc') → preview_pending_invoices. This does NOT send anything — bulk "
        "issuance is preview-only by design. Tell the user clearly that nothing was actually issued and "
        "that they can ask you to issue a specific order individually instead.\n"
        "A question about WHETHER invoices are pending ('czy mam faktury do wystawienia?', "
        "'czy są jakieś faktury?') is NOT an issuance command — use get_orders_pending_invoice for that, "
        "never issue_invoice_for_order or preview_pending_invoices for a yes/no question.\n"
        "AFTER ISSUING AN INVOICE (issue_invoice_for_order succeeded) — delivering it further:\n"
        "  - attach_invoice_to_allegro_order → downloads the PDF from inFakt and attaches it to the "
        "Allegro order, so the buyer sees it on their order page. Needs order_id + invoice_uuid "
        "(invoice_uuid comes from the issue_invoice_for_order result earlier in this conversation — "
        "never guess it, ask if it's not in context).\n"
        "  - send_invoice_to_ksef → submits the invoice to KSeF (Poland's e-invoicing system). Needs "
        "invoice_uuid, same rule — never guess it.\n"
        "  - If the user's ORIGINAL request already named the channel(s) ('wystaw i wyślij do KSeF i "
        "Allegro', 'wystaw i dodaj do Allegro') — just call the matching tool(s) directly, no need to ask.\n"
        "  - Otherwise: once the user confirms the issued invoice looks fine ('ok', 'faktura jest ok', "
        "'wygląda dobrze') and hasn't named a channel yet, call get_order_invoice_data for that order to "
        "check whether the buyer is a company or private person, then ASK in your reply: "
        "for a company buyer — 'Wysłać fakturę do KSeF i dołączyć ją do zamówienia w Allegro?'; "
        "for a private person — 'Dołączyć fakturę do zamówienia w Allegro?' (don't default to KSeF for "
        "a private person — only call send_invoice_to_ksef for one if the user explicitly asks). "
        "Only call the delivery tool(s) after the user answers that question, unless they already "
        "specified the channel(s) upfront as above.\n"
        "BILLING ROUTING: "
        "1) Specific order costs → ALWAYS get_order_details (uses order.id filter, exact results). "
        "2) Period earnings/profit → get_sales_summary. "
        "3) Period billing only → get_billing_summary. "
        "NEVER use get_billing_summary for a specific order — it covers ALL orders in date range. "
        "MONITORING — CRITICAL: You CANNOT monitor orders or invoices yourself. You have NO ability "
        "to run background tasks, check anything automatically, or send proactive messages. "
        "get_new_orders ALREADY appends the current automatic-checking status and an enable/disable "
        "button to its own result — do NOT also call suggest_order_monitoring or disable_order_monitoring "
        "right after get_new_orders, that would show a duplicate button. Only call suggest_order_monitoring "
        "/ disable_order_monitoring when the user brings up monitoring/notifications OUTSIDE of an "
        "order query (e.g. general 'wyłącz monitoring', 'chcę powiadomienia o zamówieniach' with no "
        "get_new_orders call in this turn), or for INVOICE monitoring (suggest_invoice_monitoring / "
        "disable_invoice_monitoring), which has no equivalent auto-status block. "
        "NEVER say 'I will monitor', 'I am monitoring', 'będę sprawdzać', 'będę Cię powiadamiał' "
        "as a standalone promise — you cannot do this. Always call a tool and let it render the button. "
        "HTML — CRITICAL: When a tool result contains HTML tags (e.g. <button ...>), you MUST include them VERBATIM "
        "in your response, character-for-character, without translating, paraphrasing, or modifying them in any way. "
        "JSON PREVIEWS — CRITICAL: When a tool result contains ```json code blocks (e.g. from preview_pending_invoices), "
        "include them VERBATIM, character-for-character — this is exact data the user will verify, never summarize, "
        "reformat, or reorder the JSON fields. "
        "INVOICE UUID — CRITICAL: When a tool result (e.g. issue_invoice_for_order) contains a line like "
        "'ID faktury w inFakt: `...`', you MUST keep that exact line, with the exact ID in backticks, "
        "VERBATIM in your reply — never drop it as 'technical detail' and never paraphrase or shorten the "
        "ID. It is the ONLY way a later attach_invoice_to_allegro_order or send_invoice_to_ksef call in "
        "this conversation can find the right invoice — losing it forces you to guess an ID later, which "
        "fails against the real inFakt API (404)."
    )

    def __init__(self, user_id: str | None = None):
        super().__init__()
        self.model_override = self._settings.gemini_model_fast
        self._allegro = AllegroService.get_instance(user_id)

    async def run(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        context: str | None = None,
    ) -> AgentResponse:
        from agents.base_agent import _call_with_retry

        # ── Auth guard ────────────────────────────────────────────────────────
        if self._allegro._tokens is None:
            await self._allegro._load_tokens_from_redis()
        if self._allegro._tokens is None:
            return self._request_auth()
        if self._allegro._tokens.is_expired():
            try:
                await self._allegro._refresh_tokens()
            except AllegroAuthError:
                return self._request_auth()

        tools = self._get_tools()
        model_pool = self._get_model_pool()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt(context)},
            *list(conversation_history or []),
            {"role": "user", "content": query},
        ]

        # ── Step 1: force tool call(s) ────────────────────────────────────────
        # The LLM MUST pick a real tool — it cannot answer from memory.
        # We allow up to MAX_TOOL_ROUNDS sequential tool calls (rare: e.g. look up
        # a thread ID then send a message), but stop as soon as the model signals
        # it has enough data (no new tool calls after a round).
        MAX_TOOL_ROUNDS = 3
        tool_rounds = 0
        called_tools: list[str] = []
        new_orders_count_only = False

        while tool_rounds < MAX_TOOL_ROUNDS:
            resp = await _call_with_retry(
                self._client,
                model_pool,
                f"allegro/tool-select-{tool_rounds + 1}",
                messages=messages,
                tools=tools,
                tool_choice="required",
                max_tokens=400,
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                # model ignored tool_choice — interpret whatever it said
                logger.warning("AllegroAgent: model skipped tool_choice=required (round %d)", tool_rounds + 1)
                return AgentResponse(text=msg.content or "", agent_type=self.agent_name, metadata={"output_format": "chat"})

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump(exclude_none=True) for tc in msg.tool_calls],
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                called_tools.append(tool_name)
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}
                if tool_name == "get_new_orders" and tool_input.get("count_only"):
                    new_orders_count_only = True
                logger.info("[allegro] tool call: %s(%s)", tool_name, tool_input)
                try:
                    result = await self._execute_tool(tool_name, tool_input)
                except Exception as exc:
                    logger.exception("[allegro] tool %s failed: %s", tool_name, exc)
                    result = "An internal error occurred. Please try again."
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            tool_rounds += 1

            # After one round, most queries are satisfied.
            # Allow a second round only if the tool result itself signals that
            # more data is needed (e.g. it returned a partial list + "fetch more").
            # For safety we just break after round 1 unless MAX_TOOL_ROUNDS > 1
            # and we detect the model wants to chain (handled by the loop limit).
            break

        # ── Step 1b: resolve output format from the tool(s) just called ───────
        # Deterministic — decided by WHICH TOOL ran, not guessed from the
        # user's wording (see TOOL_OUTPUT_FORMAT in allegro_tools.py for why).
        output_format = resolve_output_format(called_tools)
        instruction_keys = [
            "get_new_orders:count_only" if (t == "get_new_orders" and new_orders_count_only) else t
            for t in called_tools
        ]
        tool_instruction = next(
            (_TOOL_SPECIFIC_INSTRUCTIONS[k] for k in instruction_keys if k in _TOOL_SPECIFIC_INSTRUCTIONS),
            None,
        )
        format_instruction = tool_instruction or _OUTPUT_FORMAT_INSTRUCTIONS.get(output_format)
        if format_instruction:
            messages.append({"role": "user", "content": format_instruction})

        # ── Step 2: interpret — NO tools ─────────────────────────────────────
        # The LLM now sees the real API results and formats the answer.
        # Without tools it literally cannot hallucinate store data.
        interp_resp = await _call_with_retry(
            self._client,
            model_pool,
            "allegro/interpret",
            messages=messages,
            max_tokens=self._settings.gemini_max_tokens,
            # no `tools` parameter — model can only read and format
        )
        final_text = interp_resp.choices[0].message.content or ""
        return AgentResponse(text=final_text, agent_type=self.agent_name, metadata={"output_format": output_format})

    def _request_auth(self) -> AgentResponse:
        text = (
            "Aby uzyskać dostęp do Twojego sklepu Allegro, potrzebuję autoryzacji.\n\n"
            "[➡ Zaloguj się przez Allegro](/allegro/login)\n\n"
            "Po zalogowaniu wróć tutaj i zadaj swoje pytanie ponownie."
        )
        return AgentResponse(text=text, agent_type=self.agent_name, metadata={"output_format": "chat"})

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
    def _dig(obj: Any, *keys: str, default: Any = None) -> Any:
        """Safely navigate nested dicts. Any non-dict level returns default."""
        cur = obj
        for key in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(key)
            if cur is None:
                return default
        return cur if cur is not None else default

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

    _FULFILLMENT_PL: dict[str, str] = {
        "NEW":                "Nowe",
        "PROCESSING":         "W realizacji",
        "READY_FOR_SHIPMENT": "Gotowe do wysyłki",
        "SENT":               "Wysłane",
        "IN_TRANSIT":         "W transporcie",
        "READY_FOR_PICKUP":   "Gotowe do odbioru",
        "PICKED_UP":          "Odebrane",
        "CANCELLED":          "Anulowane",
        "SUSPENDED":          "Wstrzymane",
    }

    _STATUS_PL: dict[str, str] = {
        "BOUGHT":               "Opłacone",
        "FILLED_IN":            "Wypełnione",
        "READY_FOR_PROCESSING": "Gotowe do realizacji",
        "CANCELLED":            "Anulowane",
    }

    @classmethod
    def _fulfillment_pl(cls, status: str | None) -> str:
        return cls._FULFILLMENT_PL.get(status or "", status or "—")

    @classmethod
    def _status_pl(cls, status: str | None) -> str:
        return cls._STATUS_PL.get(status or "", status or "—")
    _TRACKING_URLS: dict[str, str] = {
        "INPOST":          "https://inpost.pl/sledzenie-przesylek?number={code}",
        "DHL":             "https://www.dhl.com/pl-pl/home/tracking.html?tracking-id={code}&submit=1",
        "DPD":             "https://www.dpd.com.pl/tracking?q={code}",
        "GLS":             "https://gls-group.eu/PL/pl/sledzenie-paczek?match={code}",
        "POCZTA_POLSKA":   "https://emonitoring.poczta-polska.pl/?numer={code}",
        "ORLEN":           "https://orlenpaczka.pl/sledz-paczke/?nr={code}",
        "UPS":             "https://www.ups.com/track?tracknum={code}",
        "FEDEX":           "https://www.fedex.com/apps/fedextrack/?tracknumbers={code}",
    }

    _WARSAW = ZoneInfo("Europe/Warsaw")
    _UTC = ZoneInfo("UTC")

    @classmethod
    def _local_to_utc(cls, time_str: str) -> str:
        """Convert Polish local time string to UTC ISO 8601 for Allegro API.

        Accepts 'HH:MM' (today) or 'YYYY-MM-DD HH:MM' (specific date).
        """
        if len(time_str) <= 5:  # "HH:MM"
            today = datetime.now(cls._WARSAW).date()
            local_dt = datetime.strptime(f"{today.isoformat()} {time_str}", "%Y-%m-%d %H:%M")
        else:  # "YYYY-MM-DD HH:MM"
            local_dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        local_dt = local_dt.replace(tzinfo=cls._WARSAW)
        return local_dt.astimezone(cls._UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    @classmethod
    def _tracking_url(cls, carrier_id: str, code: str) -> str | None:
        """Return tracking URL for a carrier+code pair, or None if unknown carrier."""
        if not code or code == "—":
            return None
        carrier_upper = (carrier_id or "").upper()
        for prefix, template in cls._TRACKING_URLS.items():
            if carrier_upper.startswith(prefix):
                return template.format(code=code)
        return None

    async def _monitoring_status_block(self) -> str:
        """Deterministic (non-LLM) status + action button for automatic order checking.

        Always reflects the real Redis flag, not the model's guess — the invoice-issuance
        bug (misfiring on a yes/no question) showed prompt-only judgement isn't reliable
        for this kind of state, so it's computed here instead of left to the LLM.
        """
        from services.order_monitor import is_monitor_enabled

        if await is_monitor_enabled(self._allegro._user_id):
            return (
                "🔔 Automatyczne sprawdzanie nowych zamówień jest włączone — dam Ci znać, "
                "gdy pojawi się coś nowego.\n\n"
                '<button class="btn-monitoring" style="background:#6b7280" '
                'onclick="OrderMonitor.disable();this.outerHTML=\'<span>✓ Monitoring zamówień wyłączony</span>\'">'
                '🔕 Wyłącz automatyczne sprawdzanie</button>'
            )
        return (
            "💡 Mogę automatycznie sprawdzać nowe zamówienia i wysyłać Ci powiadomienia, "
            "nawet gdy ta karta jest w tle.\n\n"
            '<button class="btn-monitoring" onclick="OrderMonitor.enable()">'
            '🔔 Włącz automatyczne sprawdzanie</button>'
        )

    @classmethod
    def _format_dt_pl(cls, iso_str: str) -> str:
        """ISO 8601 (UTC) → 'DD.MM.RRRR, HH:MM' in Warsaw local time."""
        if not iso_str:
            return "—"
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return dt.astimezone(cls._WARSAW).strftime("%d.%m.%Y, %H:%M")
        except ValueError:
            return iso_str

    @classmethod
    def _new_order_bullet(cls, o: Any) -> str:
        """Render one new order as a bullet-list block with exactly the fields
        the store owner asked for: buyer, delivery type, item count, value,
        payment date/time. No status, no link, no order-creation date — those
        are for get_orders / get_order_details, not this compact new-orders view."""
        d = o.delivery if isinstance(o.delivery, dict) else {}
        delivery_name = cls._dig(d, "method", "name", default="—")
        total_qty = sum(li.quantity for li in o.line_items)
        price = cls._format_price(o.total_price, o.currency)
        return (
            f"**Zamówienie** `{o.order_id}`\n"
            f"- Zamawiający: **{o.buyer_login}**\n"
            f"- Rodzaj dostawy: {delivery_name}\n"
            f"- Ilość: {total_qty} szt.\n"
            f"- Wartość: **{price}**\n"
            f"- Opłacone: {cls._format_dt_pl(o.paid_at)}"
        )

    @classmethod
    def _order_block(cls, o: Any, extra_lines: list[str] | None = None) -> str:
        """Render a single order as a markdown bullet-point block."""
        price = cls._format_price(o.total_price, o.currency)
        d = o.delivery if isinstance(o.delivery, dict) else {}
        delivery_name = cls._dig(d, "method", "name", default="—")
        total_qty = sum(li.quantity for li in o.line_items)
        link = f"https://allegro.pl/sprzedaz/zamowienia/{o.order_id}"
        fulfillment = cls._fulfillment_pl(o.fulfillment_status)
        lines = [
            f"**Zamówienie** `{o.order_id}`",
            f"- Kupujący: **{o.buyer_login}**",
            f"- Wartość: **{price}**",
            f"- Status realizacji: **{fulfillment}**",
            f"- Dostawa: {delivery_name}",
            f"- Produkty: {total_qty} szt.",
        ]
        if o.created_at:
            lines.insert(1, f"- Data złożenia: {o.created_at} (UTC)")
        if o.paid_at:
            lines.insert(2 if o.created_at else 1, f"- Opłacone: {o.paid_at} (UTC)")
        if extra_lines:
            lines.extend(f"- {l}" for l in extra_lines)
        lines.append(f"- Link: {link}")
        return "\n".join(lines)

    async def _preview_pending_invoices(self, month: int | None, year: int | None) -> str:
        """Build the exact inFakt invoice payload for each pending order WITHOUT sending it.

        Real submission (InfaktService.create_invoice/get_share_link) is
        intentionally not called here — automatic issuance is disabled until
        someone has reviewed a batch of these payloads and turned it back on.
        No inFakt API call at all means no INFAKT_API_KEY is even needed for
        this preview to work.
        """
        from services.infakt_service import build_invoice_payload

        orders = await self._allegro.get_orders_needing_invoice(month=month, year=year)
        if not orders:
            return "Brak zamówień wymagających wystawienia faktury."

        is_production = self._settings.is_production
        blocks = []
        for o in orders:
            try:
                address = await self._allegro.get_order_invoice_data(o.order_id)
                payload = build_invoice_payload(o, address, is_production)
                payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
                blocks.append(
                    f"**Zamówienie `{o.order_id}` ({o.buyer_login}):**\n```json\n{payload_json}\n```"
                )
            except Exception:
                logger.exception("preview_pending_invoices: failed to build payload for %s", o.order_id)
                blocks.append(f"**Zamówienie `{o.order_id}` ({o.buyer_login}):** ❌ nie udało się zbudować danych faktury")

        header = (
            f"**Podgląd danych faktur dla {len(orders)} zamówień — NIC nie zostało wysłane do inFakt:**\n\n"
        )
        footer = (
            "\n\nSprawdź dane, a wystawienie w inFakt zrób ręcznie, albo poproś o wystawienie "
            "konkretnego zamówienia pojedynczo (np. „wystaw fakturę dla zamówienia <id>”)."
        )
        return header + "\n\n".join(blocks) + footer

    async def _issue_invoice_for_order(self, order_id: str) -> str:
        """Actually create ONE real VAT invoice in inFakt for a single named order.

        Deliberately single-order only — bulk issuance stays preview-only
        (see _preview_pending_invoices) after an earlier misfire on an
        ambiguous yes/no question created a real (failed) issuance attempt.
        Requiring a concrete order_id per call keeps the blast radius of any
        future misfire to at most one invoice.
        """
        from services.infakt_service import (
            InfaktAPIError,
            InfaktService,
            InfaktTaskError,
            build_invoice_payload,
        )

        order = await self._allegro.get_order(order_id)
        if not order.invoice_required:
            return f"Zamówienie `{order_id}`: kupujący nie poprosił o fakturę VAT — nic nie wystawiono."

        existing = await self._allegro.get_order_invoices(order_id)
        if existing:
            return f"Zamówienie `{order_id}`: faktura już istnieje w Allegro — nie wystawiono kolejnej."

        try:
            address = await self._allegro.get_order_invoice_data(order_id)
            payload = build_invoice_payload(order, address, self._settings.is_production)
            infakt = InfaktService.get_instance()
            status = await infakt.create_invoice(payload)
            invoice_uuid = status["invoice_uuid"]
            link = await infakt.get_share_link(invoice_uuid)
            buyer_kind = "firma" if address.get("company_name") else "osoba prywatna"
            return (
                f"✅ Faktura dla zamówienia `{order_id}` ({order.buyer_login}) wystawiona w inFakt: {link}\n"
                f"ID faktury w inFakt: `{invoice_uuid}`\n"
                f"Nabywca: {buyer_kind}.\n"
                "Zweryfikuj fakturę pod linkiem, zanim pójdzie dalej (do Allegro / KSeF)."
            )
        except (InfaktAPIError, InfaktTaskError, TimeoutError) as exc:
            logger.error("issue_invoice_for_order: order %s failed: %s", order_id, exc)
            return f"❌ Nie udało się wystawić faktury dla zamówienia `{order_id}`: {exc}"

    async def _attach_invoice_to_allegro_order(self, order_id: str, invoice_uuid: str) -> str:
        """Fetch the invoice PDF from inFakt and attach it to the Allegro order.

        Allegro allows exactly one PDF invoice per order (2 MB max) via a
        two-step API: POST registers the invoice metadata, PUT uploads the
        actual file bytes against the id from that response.
        """
        from services.infakt_service import InfaktAPIError, InfaktService

        infakt = InfaktService.get_instance()
        try:
            invoice = await infakt.get_invoice(invoice_uuid)
            pdf_bytes = await infakt.get_invoice_pdf(invoice_uuid)
        except InfaktAPIError as exc:
            logger.error("attach_invoice_to_allegro_order: fetch from inFakt failed for %s: %s", invoice_uuid, exc)
            if exc.status_code == 404:
                return (
                    f"❌ inFakt nie zna faktury `{invoice_uuid}` (404) — to ID jest nieprawidłowe albo "
                    "zgubione. Sprawdź w panelu inFakt prawidłowe ID albo wystaw fakturę dla tego "
                    "zamówienia ponownie przez issue_invoice_for_order."
                )
            return f"❌ Nie udało się pobrać faktury `{invoice_uuid}` z inFakt: {exc}"

        number = invoice.get("number", "")
        filename = f"faktura-{number or invoice_uuid}.pdf"
        try:
            allegro_invoice_id = await self._allegro.create_order_invoice_record(order_id, number, filename)
            await self._allegro.upload_order_invoice_file(order_id, allegro_invoice_id, pdf_bytes)
        except AllegroAPIError as exc:
            logger.error("attach_invoice_to_allegro_order: order %s failed: %s", order_id, exc)
            return f"❌ Nie udało się dołączyć faktury do zamówienia `{order_id}` w Allegro: {exc}"

        return f"✅ Faktura {number or invoice_uuid} dołączona do zamówienia `{order_id}` w Allegro — kupujący zobaczy ją na stronie zamówienia."

    async def _send_invoice_to_ksef(self, invoice_uuid: str) -> str:
        """Submit an already-issued inFakt invoice to KSeF."""
        from services.infakt_service import InfaktAPIError, InfaktService

        infakt = InfaktService.get_instance()
        try:
            result = await infakt.send_to_ksef(invoice_uuid)
        except InfaktAPIError as exc:
            logger.error("send_invoice_to_ksef: invoice %s failed: %s", invoice_uuid, exc)
            if exc.status_code == 404:
                return (
                    f"❌ inFakt nie zna faktury `{invoice_uuid}` (404) — to ID jest nieprawidłowe albo "
                    "zgubione. Sprawdź w panelu inFakt prawidłowe ID albo wystaw fakturę dla tego "
                    "zamówienia ponownie przez issue_invoice_for_order."
                )
            return f"❌ Nie udało się wysłać faktury `{invoice_uuid}` do KSeF: {exc}"

        status = result.get("status", "?")
        return (
            f"📤 Faktura `{invoice_uuid}` wysłana do KSeF (status zgłoszenia: {status}). "
            "Wysyłka do KSeF jest asynchroniczna — ostateczny status sprawdź w panelu inFakt."
        )

    # ── Tool dispatch ─────────────────────────────────────────────────────────

    async def _dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        logger.info("DEBUG _dispatch: tool=%s input=%s", tool_name, tool_input)
        if tool_name == "get_new_orders":
            orders = await self._allegro.get_orders(
                status="READY_FOR_PROCESSING",
                fulfillment_status="NEW",
                buyer_login=tool_input.get("buyer_login"),
                limit=min(int(tool_input.get("limit", 100)), 100),
            )
            if tool_input.get("count_only"):
                body = f"Liczba nowych zamówień: {len(orders)}."
            else:
                body = "Brak nowych zamówień." if not orders else "\n\n".join(self._new_order_bullet(o) for o in orders)
            return body + "\n\n" + await self._monitoring_status_block()

        if tool_name == "get_orders":
            bought_after = tool_input.get("bought_after_local")
            bought_before = tool_input.get("bought_before_local")
            paid_after = tool_input.get("paid_after_local")
            paid_before = tool_input.get("paid_before_local")
            orders = await self._allegro.get_orders(
                status=tool_input.get("status"),
                buyer_login=tool_input.get("buyer_login"),
                fulfillment_status=tool_input.get("fulfillment_status"),
                line_items_sent=tool_input.get("line_items_sent"),
                bought_at_gte=self._local_to_utc(bought_after) if bought_after else None,
                bought_at_lte=self._local_to_utc(bought_before) if bought_before else None,
                paid_at_gte=self._local_to_utc(paid_after) if paid_after else None,
                paid_at_lte=self._local_to_utc(paid_before) if paid_before else None,
                limit=min(int(tool_input.get("limit", 50)), 100),
            )
            if not orders:
                return "Brak zamówień spełniających podane kryteria."
            return "\n\n".join(self._order_block(o) for o in orders)

        if tool_name == "get_order_details":
            logger.info("DEBUG get_order_details called for order_id=%s", tool_input.get("order_id"))
            order, billing_entries, existing_invoices = await asyncio.gather(
                self._allegro.get_order(tool_input["order_id"]),
                self._allegro.get_billing_entries_for_order(tool_input["order_id"]),
                self._allegro.get_order_invoices(tool_input["order_id"]),
                return_exceptions=True,
            )
            if isinstance(order, BaseException):
                raise order
            billing_entries = billing_entries if not isinstance(billing_entries, BaseException) else []
            existing_invoices = existing_invoices if not isinstance(existing_invoices, BaseException) else []
            if not order.invoice_required:
                invoice_str = "Kupujący nie poprosił o fakturę."
            elif existing_invoices:
                invoice_str = "Kupujący poprosił o fakturę — faktura już wystawiona."
            else:
                invoice_str = "Kupujący poprosił o fakturę — NIE WYSTAWIONO jeszcze faktury."
            items_str = "\n".join(
                f"  - {li.offer_name} (ID: {li.offer_id}): {li.quantity} × {li.price} {li.currency}"
                for li in order.line_items
            )
            d = order.delivery if isinstance(order.delivery, dict) else {}
            method_name = self._dig(d, "method", "name", default="N/A")
            tracking = (
                self._dig(d, "smart", "trackingCode", default=None)
                or self._dig(d, "trackingCode", default="N/A")
            )
            billing_str = ""
            if billing_entries:
                total_fees = 0.0
                total_credits = 0.0
                fee_lines = []
                for i, e in enumerate(billing_entries, 1):
                    amount = float((e.get("value") or {}).get("amount", 0) or 0)
                    desc = (e.get("type") or {}).get("description", "Inne")
                    offer_name = (e.get("offer") or {}).get("name", "")
                    occurred = e.get("occurredAt", "")[:10]
                    offer_part = f" — {offer_name}" if offer_name else ""
                    sign = "+" if amount > 0 else "-"
                    fee_lines.append(
                        f"  Wpis {i}/{len(billing_entries)}: {occurred} | {desc}{offer_part} | {sign}{abs(amount):.2f} PLN"
                    )
                    if amount < 0:
                        total_fees += abs(amount)
                    else:
                        total_credits += amount
                net = order.total_price - total_fees + total_credits
                billing_str = (
                    f"\n[BILLING: {len(billing_entries)} osobnych wpisów — wyświetl każdy wiersz oddzielnie]\n"
                    + "\n".join(fee_lines)
                    + f"\n  SUMA OPŁAT: -{total_fees:.2f} PLN"
                    + (f" | ZWROTY: +{total_credits:.2f} PLN" if total_credits else "")
                    + f" | ZYSK NETTO: {net:.2f} PLN"
                )
            return (
                f"Order ID: {order.order_id}\n"
                f"Buyer: {order.buyer_login} ({order.buyer_email})\n"
                f"Status: {order.status}\n"
                f"Fulfillment status: {self._fulfillment_pl(order.fulfillment_status)}\n"
                f"Total: {order.total_price} {order.currency}\n"
                f"Created: {order.created_at}\n"
                f"Paid: {self._format_dt_pl(order.paid_at)}\n"
                f"Invoice: {invoice_str}\n"
                f"Delivery method: {method_name}\n"
                f"Tracking: {tracking}\n"
                f"Items:\n{items_str}"
                f"{billing_str}"
            )

        if tool_name == "get_order_invoice_data":
            inv = await self._allegro.get_order_invoice_data(tool_input["order_id"])
            if inv.get("dontWant"):
                return f"Zamówienie {tool_input['order_id']}: kupujący zrezygnował z faktury."
            if not inv.get("required"):
                return f"Zamówienie {tool_input['order_id']}: kupujący nie zaznaczył prośby o fakturę."
            lines = [f"Dane do faktury dla zamówienia {tool_input['order_id']}:"]
            if inv.get("company_name"):
                lines.append(f"Firma: {inv['company_name']}")
            if inv.get("vat_id"):
                lines.append(f"NIP: {inv['vat_id']}")
            if inv.get("first_name") or inv.get("last_name"):
                lines.append(f"Nabywca: {inv.get('first_name', '')} {inv.get('last_name', '')}".strip())
            if inv.get("street"):
                lines.append(f"Ulica: {inv['street']}")
            if inv.get("zip_code") or inv.get("city"):
                lines.append(f"Kod pocztowy / miasto: {inv.get('zip_code', '')} {inv.get('city', '')}".strip())
            if inv.get("country_code"):
                lines.append(f"Kraj: {inv['country_code']}")
            return "\n".join(lines)

        if tool_name == "get_active_offers":
            name_filter = tool_input.get("name")
            if name_filter:
                raw, _ = await self._allegro.get_offers(name=name_filter, limit=50)
            else:
                raw = await self._allegro.get_all_offers()
            logger.info("get_active_offers: %d raw offers fetched", len(raw))
            if not raw:
                return "Brak aktywnych ofert."
            aggregated = self._aggregate_offers_by_name(raw)
            logger.info("get_active_offers: %d unique products after name aggregation", len(aggregated))
            lines = [f"Łącznie **{len(raw)}** ofert / **{len(aggregated)}** unikalnych produktów:\n"]
            for g in sorted(aggregated, key=lambda x: x["name"].lower()):
                ids_str = ", ".join(f"`{i}`" for i in g["ids"])
                price_str = self._format_price(g["price"], g["currency"])
                lines.append(
                    f"- **{g['name']}** — {price_str} — "
                    f"stan: **{g['total_stock']} szt.** — ID: {ids_str}"
                )
            return "\n".join(lines)

        if tool_name == "get_offers_summary":
            offers = await self._allegro.get_all_offers()
            logger.info("get_offers_summary: %d raw offers fetched", len(offers))
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
            logger.info("query_offers_by_stock: %d raw offers, max_stock=%s min_stock=%s", len(offers), max_stock, min_stock)
            aggregated = self._aggregate_offers_by_name(offers)
            logger.info("query_offers_by_stock: %d unique products after aggregation", len(aggregated))
            results = []
            for g in aggregated:
                s = g["total_stock"]
                if max_stock is not None and s > max_stock:
                    continue
                if min_stock is not None and s < min_stock:
                    continue
                results.append(g)
            logger.info("query_offers_by_stock: %d products match the stock filter", len(results))
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

        if tool_name == "get_products_to_reorder":
            assortment = tool_input.get("assortment")
            max_stock = tool_input.get("max_stock", 5)
            if assortment:
                offers, _ = await self._allegro.get_offers(name=assortment, limit=100)
            else:
                offers = await self._allegro.get_all_offers()
            logger.info(
                "get_products_to_reorder: %d raw offers (assortment=%r, max_stock=%s)",
                len(offers), assortment, max_stock,
            )
            aggregated = self._aggregate_offers_by_name(offers)
            results = [g for g in aggregated if g["total_stock"] <= max_stock]
            results.sort(key=lambda g: g["total_stock"])
            logger.info("get_products_to_reorder: %d products at or below threshold", len(results))
            if not results:
                scope = f" w asortymencie „{assortment}”" if assortment else ""
                return f"Brak produktów{scope} wymagających uzupełnienia (próg: {max_stock} szt.)."
            scope = f" — asortyment: „{assortment}”" if assortment else ""
            header = f"Produkty do zamówienia (stan ≤ {max_stock} szt.{scope}) — {len(results)} pozycji:\n"
            lines = [header]
            for g in results:
                ids_str = ", ".join(f"`{i}`" for i in g["ids"])
                lines.append(
                    f"- **{g['name']}** — stan obecny: **{g['total_stock']} szt.** — "
                    f"cena sprzedaży: {self._format_price(g['price'], g['currency'])} — ID: {ids_str}"
                )
            return "\n".join(lines)

        if tool_name == "get_sales_summary":
            date_from = tool_input["date_from"]
            date_to = tool_input["date_to"]
            logger.info("get_sales_summary: fetching orders and billing %s → %s", date_from, date_to)
            # Every fee/rebate type except account subscriptions is tied to an order.id, and
            # matching happens by that ID — not by the entry's own date — so the date window
            # on the billing fetch only exists to bound how much we pull from the API. Widen it
            # a month either side of the order period: any fee/rebate for an order paid in this
            # period should have posted (or been pre-calculated) within that margin.
            from datetime import datetime, timedelta, timezone
            dt_from = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
            dt_to = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
            billing_date_from = (dt_from - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
            billing_date_to = (dt_to + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
            results = await asyncio.gather(
                self._allegro.get_all_paid_orders_in_period(date_from, date_to),
                self._allegro.get_billing_entries_in_period(billing_date_from, billing_date_to),
                return_exceptions=True,
            )
            orders = results[0] if not isinstance(results[0], BaseException) else []
            all_billing = results[1] if not isinstance(results[1], BaseException) else []
            billing_error = results[1] if isinstance(results[1], BaseException) else None
            if isinstance(results[0], BaseException):
                raise results[0]
            if billing_error:
                logger.warning("get_sales_summary: billing fetch failed (%s), continuing without cost data", billing_error)
            # Entries tied to one of our orders (matched by order.id) count regardless of their
            # own date — the widened fetch above is what guarantees they were found. Entries
            # with no order ref at all (subscriptions, listing fees) have nothing to match by,
            # so those are counted only if they actually occurred within the strict period —
            # an entry for some OTHER order outside this period is correctly excluded either way,
            # since its revenue isn't part of this report.
            order_ids = {o.order_id for o in orders}
            billing_entries_with_order = [
                e for e in all_billing
                if (e.get("order") or {}).get("id") in order_ids
            ]
            billing_entries_no_order = [
                e for e in all_billing
                if not (e.get("order") or {}).get("id")
                and date_from <= e.get("occurredAt", "") <= date_to
            ]
            billing_entries = billing_entries_with_order + billing_entries_no_order
            logger.info(
                "get_sales_summary: %d paid orders, %d billing entries (%d matched by order.id, %d no-order in period)",
                len(orders), len(billing_entries), len(billing_entries_with_order), len(billing_entries_no_order),
            )
            if not orders:
                return f"Brak opłaconych zamówień w okresie {date_from[:10]} – {date_to[:10]}."
            total_revenue = sum(o.total_price for o in orders)
            order_count = len(orders)
            avg_value = total_revenue / order_count if order_count else 0
            # Top products by revenue
            product_revenue: dict[str, float] = {}
            for o in orders:
                for li in o.line_items:
                    product_revenue[li.offer_name] = product_revenue.get(li.offer_name, 0) + li.price * li.quantity
            top = sorted(product_revenue.items(), key=lambda x: x[1], reverse=True)[:10]
            top_lines = "\n".join(
                f"  {i+1}. **{name}** — {self._format_price(rev)}"
                for i, (name, rev) in enumerate(top)
            )
            billing_section = ""
            if billing_entries:
                # Group billing entries by order ID
                fees_per_order: dict[str, float] = {}   # order_id → total fees (costs, positive value)
                refunds_per_order: dict[str, float] = {}  # order_id → total refunds/credits
                fee_by_type: dict[str, float] = {}
                refund_by_type: dict[str, float] = {}
                total_fees = 0.0
                total_refunds = 0.0
                for e in billing_entries:
                    amount = float((e.get("value") or {}).get("amount", 0) or 0)
                    type_desc = (e.get("type") or {}).get("description", "Inne")
                    order_id = (e.get("order") or {}).get("id", "")
                    if amount < 0:
                        total_fees += abs(amount)
                        fee_by_type[type_desc] = fee_by_type.get(type_desc, 0) + abs(amount)
                        if order_id:
                            fees_per_order[order_id] = fees_per_order.get(order_id, 0) + abs(amount)
                    elif amount > 0:
                        total_refunds += amount
                        refund_by_type[type_desc] = refund_by_type.get(type_desc, 0) + amount
                        if order_id:
                            refunds_per_order[order_id] = refunds_per_order.get(order_id, 0) + amount
                net_profit = total_revenue - total_fees + total_refunds
                # Fees/refunds not tied to any single order (account subscriptions, listing
                # fees) are still real costs and belong in the total above, but they can't
                # appear in any order's row below — track them so the per-order table's sum
                # doesn't silently disagree with the total.
                unattributed_fees = total_fees - sum(fees_per_order.values())
                unattributed_refunds = total_refunds - sum(refunds_per_order.values())
                billing_lines = "\n".join(
                    f"  - {desc}: {self._format_price(amt)}"
                    for desc, amt in sorted(fee_by_type.items(), key=lambda x: x[1], reverse=True)
                )
                refund_lines = "\n".join(
                    f"  - {desc}: +{self._format_price(amt)}"
                    for desc, amt in sorted(refund_by_type.items(), key=lambda x: x[1], reverse=True)
                )
                # Per-order table (sorted by date)
                order_rows = []
                for o in sorted(orders, key=lambda x: x.paid_at or x.created_at):
                    oid = o.order_id
                    rev = o.total_price
                    fees = fees_per_order.get(oid, 0.0)
                    refunds = refunds_per_order.get(oid, 0.0)
                    net = rev - fees + refunds
                    date_str = (o.paid_at or o.created_at)[:10]
                    buyer = o.buyer_login[:15] if o.buyer_login else "—"
                    items_short = ", ".join(
                        f"{li.offer_name[:25]}×{li.quantity}" for li in o.line_items[:2]
                    ) + ("…" if len(o.line_items) > 2 else "")
                    order_rows.append(
                        f"  {date_str} | {buyer:<15} | przychód: {self._format_price(rev)} | "
                        f"opłaty: {self._format_price(fees)} | zysk: {self._format_price(net)}"
                        + (f"\n    {items_short}" if items_short else "")
                    )
                per_order_section = ""
                if order_rows:
                    per_order_section = (
                        "\n\n**Zestawienie per zamówienie** (opłaty przypisane do konkretnego zamówienia; "
                        "nie obejmuje opłat kontowych, np. abonamentu):\n" + "\n".join(order_rows)
                    )
                    if abs(unattributed_fees) > 0.01 or abs(unattributed_refunds) > 0.01:
                        per_order_section += (
                            f"\n\n  Uwaga: suma opłat/zwrotów z powyższej tabeli nie zsumuje się do "
                            f"\"Zysk netto\" powyżej — różnicę ({self._format_price(unattributed_refunds - unattributed_fees)}) "
                            f"stanowią opłaty/zwroty nieprzypisane do żadnego zamówienia (patrz niżej)."
                        )
                billing_section = (
                    f"\n\n**Koszty Allegro** ({date_from[:10]} – {date_to[:10]})\n"
                    f"- Łączne opłaty: **{self._format_price(total_fees)}**\n"
                    + (f"- Zwroty/rabaty: **+{self._format_price(total_refunds)}**\n" if total_refunds > 0 else "")
                    + (f"{billing_lines}\n" if billing_lines else "")
                    + (f"{refund_lines}\n" if refund_lines else "")
                    + (
                        f"- w tym opłaty/zwroty nieprzypisane do zamówienia (abonament, inne): "
                        f"**{self._format_price(unattributed_fees - unattributed_refunds)}**\n"
                        if abs(unattributed_fees) > 0.01 or abs(unattributed_refunds) > 0.01 else ""
                    )
                    + f"\n**Zysk netto (przychód − opłaty): {self._format_price(net_profit)}**"
                    + per_order_section
                )
            elif billing_error:
                billing_section = (
                    "\n\n⚠️ Dane o kosztach Allegro niedostępne (brak uprawnień do rozliczeń). "
                    "Zaloguj się ponownie przez /allegro/login, aby uzyskać dostęp do billing."
                )
            return (
                f"**Podsumowanie sprzedaży** ({date_from[:10]} – {date_to[:10]})\n\n"
                f"- Liczba zamówień: **{order_count}**\n"
                f"- Łączny przychód: **{self._format_price(total_revenue)}**\n"
                f"- Średnia wartość zamówienia: **{self._format_price(avg_value)}**\n\n"
                f"**Top produkty wg przychodu:**\n{top_lines}"
                f"{billing_section}"
            )

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
            date_from = tool_input.get("date_from")
            date_to = tool_input.get("date_to")
            try:
                if date_from and date_to:
                    from datetime import datetime, timedelta
                    dt_to = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
                    billing_date_to = (dt_to + timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    entries = await self._allegro.get_billing_entries_in_period(date_from, billing_date_to)
                    # For entries with order.id, keep all; for no-order entries keep only in strict period
                    entries = [
                        e for e in entries
                        if (e.get("order") or {}).get("id")
                        or date_from <= e.get("occurredAt", "") <= date_to
                    ]
                    period_label = f"{date_from[:10]} – {date_to[:10]}"
                else:
                    entries = await self._allegro.get_billing_entries(
                        limit=min(int(tool_input.get("limit", 50)), 100)
                    )
                    period_label = "ostatnie operacje"
            except AllegroAPIError as exc:
                if "403" in str(exc):
                    return (
                        "Brak dostępu do danych rozliczeniowych (błąd 403). "
                        "Token OAuth nie zawiera uprawnienia `allegro:api:billing:read`. "
                        "Zaloguj się ponownie przez /allegro/login, aby odświeżyć token z pełnymi uprawnieniami."
                    )
                raise
            if not entries:
                return f"Brak wpisów rozliczeniowych ({period_label})."
            fee_by_type: dict[str, float] = {}
            total_fees = 0.0
            total_refunds = 0.0
            detail_lines = []
            for e in entries:
                amount_val = float((e.get("value") or {}).get("amount", 0) or 0)
                currency = (e.get("value") or {}).get("currency", "PLN")
                type_desc = (e.get("type") or {}).get("description", "Inne")
                occurred = e.get("occurredAt", "")[:10]
                order_id = (e.get("order") or {}).get("id", "")
                order_ref = f" | zamówienie {order_id}" if order_id else ""
                sign = "+" if amount_val > 0 else ""
                detail_lines.append(
                    f"{occurred} | {type_desc}{order_ref} | {sign}{amount_val:.2f} {currency}"
                )
                if amount_val < 0:
                    total_fees += abs(amount_val)
                    fee_by_type[type_desc] = fee_by_type.get(type_desc, 0) + abs(amount_val)
                elif amount_val > 0:
                    total_refunds += amount_val
            net_cost = total_fees - total_refunds
            breakdown = "\n".join(
                f"  - {desc}: {self._format_price(amt)}"
                for desc, amt in sorted(fee_by_type.items(), key=lambda x: x[1], reverse=True)
            )
            summary = (
                f"**Koszty Allegro** ({period_label}) — {len(entries)} operacji\n\n"
                f"- Łączne opłaty: **{self._format_price(total_fees)}**\n"
                + (f"- Zwroty/rabaty: **+{self._format_price(total_refunds)}**\n" if total_refunds > 0 else "")
                + f"- Koszt netto: **{self._format_price(net_cost)}**\n\n"
                + (f"**Podział wg rodzaju opłaty:**\n{breakdown}\n\n" if breakdown else "")
                + f"**Szczegóły:**\n" + "\n".join(detail_lines[:50])
                + (f"\n… i {len(detail_lines) - 50} więcej" if len(detail_lines) > 50 else "")
            )
            return summary

        if tool_name == "get_orders_delivery":
            fulfillment_status = tool_input.get("fulfillment_status")
            orders, carriers_raw = await asyncio.gather(
                self._allegro.get_orders(
                    status=tool_input.get("status", "READY_FOR_PROCESSING"),
                    fulfillment_status=fulfillment_status,
                    limit=min(int(tool_input.get("limit", 50)), 50),
                ),
                self._allegro.get_carriers(),
                return_exceptions=True,
            )
            if isinstance(orders, BaseException):
                raise orders
            carriers_raw = carriers_raw if not isinstance(carriers_raw, BaseException) else []
            # Build id→name map from carriers endpoint
            carrier_map: dict[str, str] = {c["id"]: c.get("name", c["id"]) for c in carriers_raw}

            if not fulfillment_status:
                orders = [o for o in orders if o.fulfillment_status == "READY_FOR_SHIPMENT"]
            if not orders:
                return "Brak zamówień gotowych do wysłania."
            courier_counts: Counter = Counter()
            blocks = []
            for o in orders:
                d = o.delivery if isinstance(o.delivery, dict) else {}
                carrier_id = self._dig(d, "method", "id", default="")
                # Prefer name from carriers endpoint, fall back to order's method.name
                method_name = carrier_map.get(carrier_id) or self._dig(d, "method", "name", default="—")
                courier_counts[method_name] += 1
                tracking = (
                    self._dig(d, "smart", "trackingCode", default=None)
                    or self._dig(d, "trackingCode", default="—")
                )
                pickup_name = self._dig(d, "pickupPoint", "name", default=None)
                tracking_url = self._tracking_url(carrier_id, tracking)
                tracking_str = (
                    f"[{tracking}]({tracking_url})" if tracking_url and tracking != "—"
                    else tracking
                )
                extra = [
                    f"Kurier/dostawa: **{method_name}**",
                    f"Status: **{self._fulfillment_pl(o.fulfillment_status)}**",
                    f"Numer śledzenia: {tracking_str}",
                ]
                if pickup_name:
                    extra.append(f"Punkt odbioru: {pickup_name}")
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
            # Fetch invoice address data for all orders in parallel
            inv_results = await asyncio.gather(
                *[self._allegro.get_order_invoice_data(o.order_id) for o in orders],
                return_exceptions=True,
            )
            header = f"**Zamówień bez faktury: {len(orders)}**\n"
            blocks = []
            for o, inv in zip(orders, inv_results):
                items_str = ", ".join(f"{li.offer_name} ×{li.quantity}" for li in o.line_items[:3])
                extra = [
                    f"E-mail: {o.buyer_email}",
                    f"Status zamówienia: **{self._status_pl(o.status)}**",
                    f"Data: {o.created_at[:10] if o.created_at else '—'}",
                    f"Produkty: {items_str}",
                    "**Faktura: niewystawiona**",
                ]
                if isinstance(inv, dict) and inv.get("required"):
                    if inv.get("company_name"):
                        extra.append(f"Firma: {inv['company_name']}")
                    if inv.get("vat_id"):
                        extra.append(f"NIP: {inv['vat_id']}")
                    if inv.get("first_name") or inv.get("last_name"):
                        extra.append(f"Nabywca: {inv.get('first_name', '')} {inv.get('last_name', '')}".strip())
                    if inv.get("street"):
                        extra.append(f"Adres: {inv['street']}, {inv.get('zip_code', '')} {inv.get('city', '')}".strip(", "))
                blocks.append(self._order_block(o, extra_lines=extra))
            return header + "\n\n".join(blocks)

        if tool_name == "preview_pending_invoices":
            return await self._preview_pending_invoices(
                month=tool_input.get("month"),
                year=tool_input.get("year"),
            )

        if tool_name == "issue_invoice_for_order":
            return await self._issue_invoice_for_order(tool_input["order_id"])

        if tool_name == "attach_invoice_to_allegro_order":
            return await self._attach_invoice_to_allegro_order(
                tool_input["order_id"], tool_input["invoice_uuid"]
            )

        if tool_name == "send_invoice_to_ksef":
            return await self._send_invoice_to_ksef(tool_input["invoice_uuid"])

        if tool_name == "suggest_order_monitoring":
            return (
                "Mogę automatycznie sprawdzać nowe zamówienia co 5 minut i wysyłać Ci powiadomienia "
                "w przeglądarce — nawet gdy ta zakładka jest w tle.\n\n"
                '<button class="btn-monitoring" onclick="OrderMonitor.enable()">🔔 Włącz monitoring zamówień</button>'
            )

        if tool_name == "suggest_invoice_monitoring":
            return (
                "Mogę co 15 minut sprawdzać, czy pojawiły się nowe zamówienia wymagające faktury VAT, "
                "i natychmiast Cię powiadamiać — nawet gdy zakładka jest w tle.\n\n"
                '<button class="btn-invoice-monitoring" onclick="InvoiceMonitor.enable()">🧾 Włącz monitoring faktur</button>'
            )

        if tool_name == "disable_order_monitoring":
            return (
                "Kliknij poniższy przycisk, aby wyłączyć monitoring zamówień w tej przeglądarce.\n\n"
                '<button class="btn-monitoring" style="background:#6b7280" '
                'onclick="OrderMonitor.disable();this.outerHTML=\'<span>✓ Monitoring zamówień wyłączony</span>\'">🔕 Wyłącz monitoring zamówień</button>'
            )

        if tool_name == "disable_invoice_monitoring":
            return (
                "Kliknij poniższy przycisk, aby wyłączyć monitoring faktur w tej przeglądarce.\n\n"
                '<button class="btn-invoice-monitoring" style="filter:grayscale(1)" '
                'onclick="InvoiceMonitor.disable();this.outerHTML=\'<span>✓ Monitoring faktur wyłączony</span>\'">🔕 Wyłącz monitoring faktur</button>'
            )

        return f"Unknown tool: {tool_name}"
