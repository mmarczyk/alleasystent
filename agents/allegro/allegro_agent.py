from __future__ import annotations

"""
Allegro Agent — handles all Allegro marketplace operations.

Receives queries from the orchestrator, uses Allegro API tools
to fetch/update data, and returns structured responses.
"""

import asyncio
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from agents.allegro.allegro_tools import (
    ALLEGRO_TOOLS,
    TOOL_OUTPUT_FORMAT,
    matched_labels,
    resolve_output_format,
    tools_for_labels,
)
from agents.allegro.deterministic_dispatch import resolve_deterministic, wants_latest_order_details
from agents.base_agent import BaseAgent
from agents.perf import StageTimer
from models.conversation import AgentResponse
from services.allegro_service import (
    AllegroAPIError, AllegroAuthError, AllegroService,
    is_thread_unread, thread_last_message_at,
)

logger = logging.getLogger(__name__)

# Deterministic override for get_message_threads' count_only flag — prompt-only
# judgement has repeatedly proven unreliable for this exact yes/no-vs-list
# distinction (see the get_new_orders count_only fix and the order-monitoring
# status block), so the user's own wording decides, not the model's tool-call
# guess. A "show me [the list]" word always wins (user explicitly wants the
# thread list); otherwise a question word next to "wiadomo..." means they just
# want yes/no or a number. Content-intent words ("treść", "przeczytaj", ...)
# live in _MESSAGE_CONTENT_INTENT_RE below, not here — those mean the user
# wants a specific message's TEXT, which get_message_threads never has.
_MESSAGE_LIST_OVERRIDE_RE = re.compile(
    r"poka[żz]|wyświetl|wypisz|\blista\b|jakie\s+(są|mam)|zobacz",
    re.IGNORECASE,
)
_MESSAGE_QUESTION_WORD_RE = re.compile(r"\b(czy|ile)\b", re.IGNORECASE)
_MESSAGE_TOPIC_WORD_RE = re.compile(r"wiadomo", re.IGNORECASE)


def _wants_message_count_only(query: str) -> bool:
    if _MESSAGE_LIST_OVERRIDE_RE.search(query):
        return False
    return bool(_MESSAGE_QUESTION_WORD_RE.search(query) and _MESSAGE_TOPIC_WORD_RE.search(query))


# Deterministic nudge toward get_thread_messages (actual message TEXT) instead
# of get_message_threads (metadata only) — same rationale as above. Two ways
# to signal this:
#   1. The current message itself asks to read/see the content of ONE message
#      ('treść wiadomości', 'przeczytaj', 'co napisał', 'tę/tą wiadomość',
#      'wiadomość z dzisiaj/od X').
#   2. A short follow-up ('jeszcze raz', 'ponownie', 'spróbuj', 'pokaż to')
#      right after a turn that was itself about messages — conversation
#      history only carries the rendered text (see
#      ConversationSession.to_anthropic_messages), never the tool args, so
#      this is the only way to tell "jeszcze raz" meant "show that message's
#      content again" rather than some unrelated repeat request.
_MESSAGE_CONTENT_INTENT_RE = re.compile(
    r"treść|co\s+napisał|co\s+pisze|przeczytaj|t[eę]\s+wiadomo|ta\s+wiadomo|"
    r"wiadomo\w*\s+(z\s+dzisiaj|z\s+wczoraj|od\s+\w+)",
    re.IGNORECASE,
)
_MESSAGE_FOLLOWUP_REPEAT_RE = re.compile(
    r"jeszcze\s+raz|ponownie|spróbuj|pokaż\s+to|raz\s+jeszcze",
    re.IGNORECASE,
)


def _wants_message_content(query: str, conversation_history: list[dict[str, str]] | None) -> bool:
    if _MESSAGE_CONTENT_INTENT_RE.search(query):
        return True
    if _MESSAGE_FOLLOWUP_REPEAT_RE.search(query) and conversation_history:
        for turn in reversed(conversation_history):
            if turn.get("role") == "assistant":
                return bool(_MESSAGE_TOPIC_WORD_RE.search(turn.get("content", "")))
    return False


# Used to skip the interpret LLM call entirely: every tool's dispatch already
# returns the finished view (see _RENDERED_VIEW_TOOLS below), so the interpret
# call's only remaining job is translating the Polish labels for a non-Polish
# query — which makes it safe to skip exactly when the query is confidently
# Polish. A diacritic-free Polish query ("ile mam zamowien") just falls through
# to the LLM — missing the optimization there is fine, guessing wrong and
# mangling the language isn't.
_POLISH_DIACRITICS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")


def _is_confidently_polish(query: str) -> bool:
    return any(ch in _POLISH_DIACRITICS for ch in query)


# Above this many characters of tool results, AllegroAgent.run stops offering the
# model another tool round — see the cost guard in run() for why.
_CHAIN_RESULT_BUDGET = 8000

# Every data tool renders its own finished view in _dispatch — the table, the
# document, the dashboard, the bullet list the user actually sees. Turning rows
# of data into a view is mechanical work with no judgment in it, so it belongs
# in Python: a model doing it is slower, costs tokens on every turn, and can
# drop, reorder or truncate rows, or quietly invent a number. See
# AllegroAgent._render_offers_table / _orders_listing / _render_dashboard
# and the "Skip the interpret call entirely" block in AllegroAgent.run().
#
# So the interpret call is skipped for these tools whenever the query is
# confidently Polish (see _is_confidently_polish). It still runs for a
# non-Polish query and for a turn that called several tools, and then this is
# its whole brief: hand the finished view back, never rebuild it.
_RENDERED_VIEW_INSTRUCTION = (
    "[GOTOWA ODPOWIEDŹ — PRZEPISZ BEZ ZMIAN]\n"
    "Dane powyżej to już gotowa, sformatowana odpowiedź dla użytkownika: nagłówki, "
    "tabele, listy, bloki ```chart i zdanie podsumowania — dokładnie w tej kolejności, "
    "w jakiej mają się wyświetlić. Przepisz je bez zmian: nie buduj tabeli od nowa, nie "
    "sortuj, nie grupuj, nie skracaj, nie pomijaj wierszy, nie dopisuj wstępu ani "
    "komentarza i nie przenoś podsumowania sprzed/za tabelę (tabela trafia do dokumentu, "
    "a zdanie po niej do dymka czatu). Jeśli użytkownik pytał po angielsku — przetłumacz "
    "WYŁĄCZNIE etykiety (nagłówki, nazwy kolumn, zdania podsumowania); wartości w "
    "wierszach, kwoty, daty i identyfikatory zostaw dokładnie takie, jakie są."
)

# Tools whose dispatch output is the finished answer. That is every tool that
# touches store data plus the monitoring toggles (their button block is built
# from the stored flag, never guessed) — i.e. everything the agent can call.
# (ask_clarifying_question is in here for free and never uses it: run() answers
# that one before any dispatch happens.)
_RENDERED_VIEW_TOOLS = frozenset(TOOL_OUTPUT_FORMAT)

# Single-tool Polish turns skip the interpret call altogether — nothing is left
# for it to do.
_PASSTHROUGH_TOOLS = _RENDERED_VIEW_TOOLS



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
        "ORDER FIELDS — ALWAYS PRESENT: every answer that lists or describes orders MUST show, "
        "for each order, its current status ('Status' / 'Status realizacji') and the dispatch "
        "deadline ('Wysyłka do' — the moment by which the parcel has to be handed over to the "
        "carrier). In a table these are two of the columns; in a bullet list, two of the lines. "
        "Copy both verbatim from the tool data (including the '⚠️ po terminie' marker and the "
        "'—' used when Allegro didn't return a deadline) — never compute, guess or omit them.\n"
        "ORDER LISTINGS — get_new_orders, get_orders and get_orders_delivery are the SAME "
        "listing with different preset filters, and all three answer as a plain-text bullet "
        "list, never a table. Pick the one whose preset matches the intent (new/unprocessed, "
        "arbitrary filters, ready-to-ship + couriers) and pass the filters the question "
        "actually names: status, fulfillment_status, buyer_login, bought_after/before_local "
        "(when the order was placed), paid_after/before_local (when it was paid), "
        "dispatch_after/before_local (the 'Wysyłka do' deadline — use it for 'co muszę wysłać "
        "dzisiaj/do jutra' and for overdue orders), count_only for 'ile', limit for a singular "
        "'ostatnie zamówienie'. Never answer a filtered question with an unfiltered list.\n"
        "When answering questions about orders or offers, always fetch fresh data via tools. "
        "Present information clearly and concisely. "
        "When showing prices, always include the PLN currency. "
        "Respond in the same language as the user's question (Polish or English). "
        "Do not make up or guess any order IDs or prices — always retrieve them via tools. "
        "CLARIFYING QUESTIONS — CRITICAL: You MUST call a tool every turn, but that tool does not "
        "have to be a data tool. If the message plus conversation history does not give you enough "
        "to pick the right tool OR to fill one of its REQUIRED parameters (order_id, invoice_uuid, "
        "buyer_login, a date/period, a product/assortment name, a price, ...) without inventing or "
        "guessing it, call ask_clarifying_question instead — never fabricate a value just to satisfy "
        "a tool call, and never silently pick between two plausible tools when the wording is truly "
        "ambiguous. Do not overuse it: if the value is already in this conversation (given earlier "
        "by the user or by you), or the MANDATORY TOOL CALLS rules below already resolve which tool "
        "and arguments to use, call that tool directly — do not ask for information you already have.\n"
        "MANDATORY TOOL CALLS — these question types MUST trigger a tool, never be answered from memory:\n"
        "• Order LIST for a period, no cost/profit/earnings wording — 'lista zamówień', 'pokaż "
        "wszystkie zamówienia z tego miesiąca', 'zamówienia z ostatniego tygodnia' → get_orders "
        "(bought_after/before_local or paid_after/before_local for the period). Do NOT use "
        "get_sales_summary for this — that tool is ONLY for earnings/profit/fee questions "
        "(see BILLING ROUTING below), never for 'just show me the orders'.\n"
        "• Order counts / 'ile zamówień' / 'ile jest wszystkich nowych' / 'ile mam nowych' "
        "(user wants a NUMBER, not the list) → get_new_orders with count_only=true. Do NOT return "
        "the whole order list when the user only asked HOW MANY.\n"
        "• Last/newest new order, SINGULAR ('ostatnie zamówienie', 'ostatnie nowe zamówienie', "
        "'najnowsze zamówienie', 'ostatniego nowego zamówienia') → get_new_orders with limit=1. "
        "Do NOT return the whole list when the user asks about just ONE order — check singular vs "
        "plural phrasing carefully before calling.\n"
        "• TWO-STEP LOOKUPS — you may call tools in SEQUENCE, and for details of an order you "
        "cannot name yet you MUST: 'szczegóły ostatniego nowego zamówienia' / 'co zawiera "
        "najnowsze zamówienie' / 'pełne dane ostatniego zamówienia' → FIRST get_new_orders "
        "(limit=1), then, once you can see that order's ID in the tool result, get_order_details "
        "with THAT order_id. Same for a thread you must find before reading it (get_message_threads "
        "→ get_thread_messages). NEVER invent or guess a UUID to satisfy get_order_details — a made-up "
        "ID just fails against the Allegro API and the user gets no answer at all. When a step "
        "returns nothing to continue with, call ask_clarifying_question instead of guessing.\n"
        "• COMPOUND REQUESTS — when one message asks for two or more INDEPENDENT things that each "
        "resolve to their own tool with everything they need already given (not a chain — neither "
        "needs the other's result, and neither is missing a required parameter), call ALL of them "
        "in this same round, as separate tool calls, instead of answering only one and silently "
        "dropping the rest. Example: 'podsumuj mi sprzedaż z dzisiaj i wystaw te zaległe faktury' → "
        "get_sales_summary (today's period) AND preview_pending_invoices (see BILLING ROUTING and "
        "the invoice-issuance rules below for why each resolves the way it does), both called now, "
        "not just one of them. This is different from TWO-STEP LOOKUPS above: there the second call "
        "NEEDS an ID the first call produces, so it waits for round 2; here both calls can run side "
        "by side because neither depends on the other. If a part of the message is itself missing "
        "required information (not just \"which preset/period\" — a genuinely missing value), resolve "
        "that part the normal way (ask_clarifying_question, alone, for the whole turn) rather than "
        "guessing a value just to bundle a call alongside the resolvable part.\n"
        "• ORDER STAGE — pick the tool from the STAGE the user names, however they name it "
        "(the same stage gets asked formally, colloquially and as a count — for a count add "
        "count_only=true, the stage choice does not change):\n"
        "   – NOWE: 'nowe', 'świeże', 'do obsłużenia', 'złożone', 'zarejestrowane', "
        "'oczekujące na potwierdzenie', 'co nowego wpadło', 'co mam zacząć', 'co czeka na start', "
        "'nietknięte', 'ile w kolejce' — and the still-to-pack wording 'do spakowania' / "
        "'co mam spakować' / 'niespakowane' → get_new_orders (fulfillment_status=NEW)\n"
        "   – W REALIZACJI: 'w trakcie', 'w realizacji', 'przetwarzane', 'w toku', 'kompletowane', "
        "'co teraz kompletuję', 'co mam w robocie', 'nad czym siedzę', 'nieskończone', "
        "'do dokończenia' → get_orders with fulfillment_status=PROCESSING\n"
        "   – DO WYSŁANIA: 'gotowe do wysyłki', 'oczekujące/czekają na wysyłkę', 'do wysłania', "
        "'niewysłane', 'przygotowane do nadania', 'do nadania', 'zapakowane' (już spakowane), "
        "'co czeka na kuriera', 'gotowe do wywózki', 'ile paczek do nadania' → get_orders_delivery "
        "(its default filter is already fulfillment_status=READY_FOR_SHIPMENT)\n"
        "   – WYSŁANE: 'wysłane', 'nadane', 'w transporcie', 'przekazane przewoźnikowi', "
        "'co już poszło', 'co odebrał kurier', 'ile dziś wysłałem', 'ile już wyjechało' → "
        "get_orders_delivery with fulfillment_status=SENT (the courier and tracking details are "
        "exactly what a question about a parcel already on its way is after)\n"
        "   – ODEBRANE: 'odebrane', 'dostarczone', 'zrealizowane', 'zakończone', 'co już dotarło', "
        "'co klient odebrał', 'ile dostarczonych', 'ile zamkniętych' → get_orders with "
        "fulfillment_status=PICKED_UP\n"
        "   – NO stage named at all ('pokaż zamówienia', 'lista zamówień', a period or a buyer) → "
        "get_orders with no fulfillment_status — it is the fallback for every order question the "
        "stages above do not cover, never the first choice when a stage IS named.\n"
        "• DEADLINE, not stage — 'co muszę wysłać dzisiaj', 'co mam dziś do wysyłki', 'ile paczek "
        "muszę dziś nadać', 'jakie mam dzisiaj terminy' → get_orders_due_today (everything still to "
        "be handed over whose 'Wysyłka do' falls by the end of today, overdue included, soonest "
        "first). Do NOT answer this with get_orders_delivery: that lists what is PACKED regardless "
        "of deadline, so it both hides orders nobody packed yet and shows parcels not due for days. "
        "Other horizons on the same tool: 'do jutra'/'do piątku' → dispatch_before_local with that "
        "date at 23:59; 'co jest po terminie' → dispatch_before_local with the CURRENT time.\n"
        "• Delivery providers / 'dostawcy' / 'kurierzy' → get_orders_delivery\n"
        "• 'Jakich mam dostawców/kurierów w zamówieniach do wysłania?' / which couriers are used for "
        "orders needing shipment / group orders by courier → get_orders_delivery. There is NO separate "
        "'get_orders_by_courier' tool — get_orders_delivery already groups and counts orders by courier; "
        "always use it for any question combining orders with courier/delivery-provider grouping.\n"
        "• Order value / 'wartość zamówień' / 'łączna wartość' → get_new_orders, then sum total_price\n"
        "• BUYERS/CUSTOMERS rather than orders — 'lista kupujących', 'lista klientów', 'kto u mnie "
        "kupował', 'ilu miałem klientów', 'moi najlepsi klienci', 'stali klienci', 'jakie firmy u "
        "mnie kupowały', 'kupujący, dla których wystawiłem faktury VAT' → get_buyers (one row per "
        "buyer with their order count, total spend and invoice count). Pass buyer_type='company' "
        "when the question names firms/NIP/B2B and 'person' for private buyers, invoice_status="
        "'issued' for 'dla których wystawiłem fakturę', 'missing' for buyers still owed one, "
        "count_only=true for 'ilu/ile'. Resolve the period yourself ('w tym roku' → 1 January of "
        "the current year through today) and omit both dates only when no period is named — the "
        "tool then defaults to the current year. NEVER answer a buyer question with get_orders or "
        "get_sales_summary: neither groups anything by buyer, so the seller would be left counting "
        "rows themselves.\n"
        "• Status / any detail of ONE SPECIFIC, already-identified order — 'jaki jest status tego "
        "zamówienia', 'co się dzieje z zamówieniem <id>', 'sprawdź zamówienie <id>', a bare order_id "
        "(UUID) pasted by the user, or a follow-up like 'a teraz?'/'sprawdź jeszcze raz' referring "
        "to an order named earlier in this conversation → get_order_details with that order_id. The "
        "order_id may come directly from the user's message or from earlier in the conversation "
        "history (e.g. you or the user already named/showed this exact order) — reuse it, never ask "
        "the user to repeat an ID that's already in context. NEVER call get_orders for this — it has "
        "NO order_id parameter, so given a specific order_id it silently ignores it and returns an "
        "unrelated list of unrelated orders (a real bug seen in production: a status question about "
        "one named order returned a 50-row table of unrelated orders). Only fall back to get_orders "
        "if you genuinely have no order_id at all (e.g. you'd have to search by buyer name/company, "
        "which get_orders can't do either — buyer_login is the Allegro login, not a company or "
        "person's name — in that case call ask_clarifying_question asking for the order_id or the "
        "buyer's Allegro login).\n"
        "• Whether/how many orders were PLACED in a specific TIME WINDOW ('czy dzisiaj po 8 rano były "
        "jakieś zamówienia', 'ile zamówień wpłynęło wczoraj', 'zamówienia złożone po godzinie X', "
        "'zamówienia z ostatniej godziny') → get_orders with bought_after_local/bought_before_local. "
        "NEVER use get_new_orders for this — it only shows currently-unprocessed (NEW) orders "
        "regardless of when they were placed, and has NO time-window parameters at all, so it would "
        "silently ignore 'po 8 rano' and answer a different question (is anything still pending) "
        "instead of the one actually asked (was anything placed in that window — it may already be "
        "packed/shipped and still count as a 'yes').\n"
        "• Offer list / 'lista ofert' / 'zestawienie ofert' / 'pokaż oferty' / 'moje oferty' → get_active_offers\n"
        "• Offer stock / 'stany magazynowe' / 'ile mam' / 'które kończą się' → query_offers_by_stock\n"
        "• Offer prices / 'ceny ofert' / 'drogie oferty' / 'najtańsze' → query_offers_by_price\n"
        "• Whenever the user's offer/stock question names a specific product, brand, model or "
        "collection ('kolekcja X', 'produkty Y', 'włóczka Z') — you MUST pass that term through "
        "the tool's name/assortment argument (get_active_offers: name, query_offers_by_stock: "
        "name, get_products_to_reorder: assortment). NEVER omit it and dump the full unfiltered "
        "list — that answers a different, broader question than the one asked. If two or more "
        "collections are named ('kolekcja X i Y'), call the tool once per name and merge the "
        "results into a single reply.\n"
        "• New/unread buyer messages — YES/NO or COUNT question ('czy mam nowe wiadomości', "
        "'czy są jakieś nowe wiadomości', 'ile mam nowych wiadomości') → get_message_threads with "
        "count_only=true. Do NOT return the whole thread list when the user only asked WHETHER or "
        "HOW MANY. Only drop count_only when the user also wants to see the threads/messages "
        "themselves ('pokaż wiadomości', 'jakie mam wiadomości', or a follow-up like 'pokaż "
        "szczegóły'/'tak' right after you already told them the count).\n"
        "• Reading the actual TEXT/content of a message ('pokaż mi wiadomość od X', 'co napisał "
        "kupujący', 'wiadomość z dzisiaj', 'treść wiadomości', 'przeczytaj wiadomość') → "
        "get_thread_messages — NEVER get_message_threads for this, that tool only returns metadata "
        "(buyer, read status, last-message date), never the message text. Pass buyer_login and/or "
        "date ('dzisiaj'/'today' or 'YYYY-MM-DD') if you don't already have a thread_id from earlier "
        "in this conversation — the tool finds the matching thread for you, no need to call "
        "get_message_threads first.\n"
        "• New/recent customer returns, ANY status — 'nowe zwroty', 'jakie mam zwroty', 'czy są "
        "jakieś zwroty', 'ile zwrotów' → get_new_returns (count_only=true for a plain number "
        "question). NEVER confuse this with complaints/disputes even if the user's wording is loose.\n"
        "• RETURNS/COMPLAINTS FOR A PERIOD — 'ile miałem zwrotów w tym miesiącu', 'zwroty z marca', "
        "'reklamacje z zeszłego tygodnia', 'zwroty do obsłużenia z tego tygodnia' → the SAME tool as "
        "above (get_new_returns / get_returns_to_process / get_new_complaints), but you MUST pass "
        "date_from_local and date_to_local as 'YYYY-MM-DD' Warsaw-local dates computed from the "
        "current date given in your context (e.g. 'w tym miesiącu' = the 1st of the current month "
        "through today). Omitting them answers a DIFFERENT question — all recent returns regardless "
        "of date — and the resulting count is simply wrong for what was asked.\n"
        "• Returns that need SELLER ACTION right now (parcel already physically back, status "
        "DELIVERED, awaiting accept/refund or reject) — 'zwroty do obsłużenia', 'zwroty do "
        "rozpatrzenia', 'czy mam jakieś zwroty do obsłużenia', 'zwroty czekające na decyzję', "
        "'ile zwrotów do obsłużenia' → get_returns_to_process, NEVER get_new_returns. A return can "
        "exist (buyer requested it) long before the parcel arrives, so 'nowe zwroty' and 'zwroty do "
        "obsłużenia' are DIFFERENT questions with different answers — pick the tool matching what "
        "the user actually asked: existence/count of ANY recent return vs. ones needing action now.\n"
        "• New/recent complaints or disputes — 'nowe reklamacje', 'reklamacje kupujących', 'czy są "
        "jakieś spory', 'ile reklamacji' → get_new_complaints (count_only=true for a plain number "
        "question). A 'zwrot' (return, no dispute) and a 'reklamacja'/'spór' (complaint/dispute) are "
        "different Allegro processes — pick the tool matching the word actually used.\n"
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
        "Never invent or guess an order_id; if you don't have one, call ask_clarifying_question or "
        "look it up first.\n"
        "  - Batch / whole month, no single order named ('wystaw wszystkie faktury', 'wystaw brakujące "
        "faktury za ten miesiąc') → preview_pending_invoices. This does NOT send anything — bulk "
        "issuance is preview-only by design. Tell the user clearly that nothing was actually issued and "
        "that they can ask you to issue a specific order individually instead.\n"
        "A question about WHETHER invoices are pending ('czy mam faktury do wystawienia?', "
        "'czy są jakieś faktury?') is NOT an issuance command — use get_orders_pending_invoice for that, "
        "never issue_invoice_for_order or preview_pending_invoices for a yes/no question.\n"
        "REFUSED AS ALREADY ISSUED: when issue_invoice_for_order answers that the invoice was already "
        "issued or already ordered for that order ('faktura została już wystawiona', 'nie wystawiam "
        "drugiej', 'nie zlecam tego drugi raz'), that is a duplicate guard, not a failure — do NOT call "
        "issue_invoice_for_order again for that order, whatever the user says. Report what the tool said. "
        "ONLY if the user then states the invoice is not actually in inFakt ('nie ma tam takiej faktury', "
        "'odblokuj to zamówienie') call unblock_invoice_for_order — it re-checks inFakt itself and clears "
        "the block only if inFakt confirms nothing was created. Pass seller_confirmed=true ONLY when the "
        "user explicitly said they looked in the inFakt panel and the invoice is not there; otherwise "
        "leave it out. After the block is cleared, issuing still takes a separate "
        "issue_invoice_for_order call.\n"
        "AFTER ISSUING AN INVOICE (issue_invoice_for_order succeeded) — delivering it further:\n"
        "  - attach_invoice_to_allegro_order → downloads the PDF from inFakt and attaches it to the "
        "Allegro order, so the buyer sees it on their order page. Needs order_id + invoice_uuid "
        "(invoice_uuid comes from the issue_invoice_for_order result earlier in this conversation — "
        "never guess it, call ask_clarifying_question if it's not in context).\n"
        "  - send_invoice_to_ksef → submits the invoice to KSeF (Poland's e-invoicing system). Needs "
        "invoice_uuid, same rule — never guess it, call ask_clarifying_question instead.\n"
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
        "2) Period earnings/profit ('ile zarobiłem', 'zysk', 'przychód po opłatach', 'podsumuj sprzedaż') "
        "→ get_sales_summary, ONE call covering the WHOLE period asked about. Resolve the period from the "
        "current date in your context ('z tego roku'/'w tym roku' → 1 January of the current year through "
        "today — never January alone; 'w tym miesiącu' → the 1st of the current month through today; "
        "'w zeszłym roku' → 1 January to 31 December of the previous year). NEVER call it once per month to "
        "build a 'podział na miesiące' — the report already breaks any period longer than one month down "
        "month by month, and a per-month call answers a narrower question than the one asked. "
        "3) Period billing only → get_billing_summary. "
        "4) Plain order LIST for a period, no cost/profit wording → get_orders, NOT get_sales_summary "
        "(see the order-list rule above). "
        "NEVER use get_billing_summary for a specific order — it covers ALL orders in date range. "
        "MONITORING — CRITICAL: You CANNOT monitor orders, invoices, messages, returns, or complaints "
        "yourself. You have NO ability to run background tasks, check anything automatically, or send "
        "proactive messages. "
        "get_new_orders ALREADY appends the current automatic-checking status and an enable/disable "
        "button to its own result — do NOT also call suggest_order_monitoring or disable_order_monitoring "
        "right after get_new_orders, that would show a duplicate button. Only call suggest_order_monitoring "
        "/ disable_order_monitoring when the user brings up monitoring/notifications OUTSIDE of an "
        "order query (e.g. general 'wyłącz monitoring', 'chcę powiadomienia o zamówieniach' with no "
        "get_new_orders call in this turn), or for the INVOICE REMINDER "
        "(suggest_invoice_reminder / disable_invoice_reminder — a scheduled 7:00-20:00 check that "
        "actively ASKS in chat whether to issue pending invoices for shipped orders and adapts to "
        "the reply; this is the ONLY invoice automation there is, so use this pair for ANY request "
        "about invoice notifications/monitoring/reminders), MESSAGE monitoring (suggest_message_monitoring / "
        "disable_message_monitoring), or RETURNS/COMPLAINTS monitoring (suggest_returns_monitoring / "
        "disable_returns_monitoring — a single shared toggle covering BOTH zwroty and reklamacje, "
        "there is no separate tool for each) — call suggest_message_monitoring after get_message_threads "
        "when the user asks about being notified of new buyer messages, and likewise "
        "suggest_returns_monitoring after get_new_returns/get_returns_to_process/get_new_complaints, and "
        "suggest_invoice_reminder after get_orders_pending_invoice when the user wants to be notified or "
        "actively asked about pending invoices "
        "(though get_new_orders/get_message_threads/get_new_returns/get_returns_to_process/"
        "get_new_complaints/get_orders_pending_invoice ALREADY append their own status block — "
        "don't double-call). "
        "The invoice REMINDER also handles its own conversation once it has asked — if the user's "
        "current message looks like a reply to that chat question ('tak wystaw', 'później', 'za 3 "
        "godziny', 'przestań pytać'), it is intercepted before reaching you at all "
        "(agents.orchestrator.Orchestrator.handle), so you will never see it; don't try to replicate "
        "that logic yourself. "
        "Whichever of the pair you call (suggest "
        "or disable), the reply always reflects the REAL stored status, not your guess — never assert "
        "in your own words whether monitoring is on/off ('nie jest włączone', 'jest aktywne'), always let "
        "the tool result's text and button say it, verbatim. "
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

    # `system_prompt` above (~17.5K chars) is almost entirely tool-ROUTING
    # rules — which tool to call for which phrasing, chaining, billing/
    # monitoring routing, anti-hallucination tool-name guardrails. None of
    # that is relevant once a tool has already run and the only remaining
    # job is turning its result into a reply — sending the full prompt to
    # every interpret call was ~4300 tokens of pure dead weight on top of
    # whatever the tool actually returned. This is the interpret step's own,
    # much shorter prompt: just enough to format/preserve data faithfully.
    _interpret_system_prompt = (
        "You are passing an Allegro (Polish e-commerce) store reply on to the store owner. "
        "The tool result above is ALREADY the finished, formatted view — headings, tables, "
        "charts, summary sentence, buttons — built for the app that renders it. Hand it back "
        "as it is; the only thing you may change is the LANGUAGE of the labels, and only when "
        "the user asked in another language. Never rebuild a table, re-sort or regroup rows, "
        "drop rows, or add a preamble. Use ONLY the data in the tool result(s) above — never "
        "invent order IDs, prices, offer IDs, or any other data not present there. When "
        "showing prices, always include the PLN currency.\n"
        "ORDER FIELDS — ALWAYS PRESENT: every reply that lists or describes orders MUST show, "
        "for each order, its current status and dispatch deadline ('Wysyłka do'), copied "
        "VERBATIM from the tool data (including any '⚠️ po terminie' marker and the '—' used "
        "when Allegro didn't return a deadline) — never compute, guess, or omit them.\n"
        "BILLING ENTRIES — CRITICAL: list EVERY entry as its own separate line, exactly as "
        "received. NEVER group, aggregate, merge, or summarize entries by type — if the tool "
        "returned 5 entries, show 5 rows, not 2.\n"
        "HTML — CRITICAL: HTML tags in a tool result (e.g. <button ...>) go into your reply "
        "VERBATIM, character-for-character — never translate, paraphrase, or modify them.\n"
        "JSON PREVIEWS — CRITICAL: ```json code blocks in a tool result go into your reply "
        "VERBATIM — never summarize, reformat, or reorder the fields.\n"
        "INVOICE UUID — CRITICAL: a line like 'ID faktury w inFakt: `...`' in a tool result "
        "must stay VERBATIM in your reply, exact ID included — it's the only way a later "
        "step in this conversation can find the right invoice."
    )

    def _build_interpret_system_prompt(self, context: str | None) -> str:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now_local = datetime.now(ZoneInfo("Europe/Warsaw"))
        parts = [
            self._interpret_system_prompt,
            f"Current date and time: {now_local.strftime('%Y-%m-%d %H:%M %Z')}",
        ]
        if context:
            parts.append(f"## Relevant context\n{context}")
        parts.append(
            "LANGUAGE RULE: Detect the language of the user's message. "
            "If it is Polish, your ENTIRE response must be in Polish. "
            "If it is English, respond in English. Never mix languages. "
            "Tool results may be in English — ignore that and still reply in the user's language."
        )
        return "\n\n".join(parts)

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
        from agents.base_agent import _call_for_reply, _call_with_retry

        perf = StageTimer("allegro_agent.run")

        # ── Auth guard ────────────────────────────────────────────────────────
        with perf.stage("auth_check"):
            if self._allegro._tokens is None:
                await self._allegro._load_tokens_from_redis()
            if self._allegro._tokens is None:
                perf.log(result="auth_required")
                return self._request_auth()
            if self._allegro._tokens.is_expired():
                try:
                    await self._allegro._refresh_tokens()
                except AllegroAuthError:
                    perf.log(result="auth_expired")
                    return self._request_auth()

        # ── Tool-select context filter ──────────────────────────────────────────
        # Sending all ~37 tool schemas to every tool-select call is dead weight
        # for the overwhelming majority of turns, which are about exactly one
        # topic — see select_tools_for_context()'s docstring in allegro_tools.py
        # for the labeling/stem-matching design. No label found at all falls
        # back to the full, unfiltered list rather than asking the user to
        # clarify — a truly topic-less turn (chit-chat riding on an
        # already-Allegro-classified conversation, or phrasing the stems
        # don't cover) still needs the model to see everything it might need,
        # exactly as before this filter existed; only the common, clearly-
        # labeled majority gets the trimmed prompt.
        wants_msg_content = _wants_message_content(query, conversation_history)
        history_text = "\n".join(m.get("content", "") for m in (conversation_history or []))
        # Forces the "wiadomosci" label in even when the trigger was a bare
        # follow-up like "przeczytaj jeszcze raz" that names no topic word
        # itself — see _wants_message_content's docstring for why detecting
        # that doesn't always hinge on the literal word being present.
        label_text = f"{query}\n{history_text}" + ("\nwiadomość" if wants_msg_content else "")
        labels = matched_labels(label_text)
        tools = tools_for_labels(labels) if labels else self._get_tools()

        model_pool = self._get_model_pool()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt(context)},
            *list(conversation_history or []),
            {"role": "user", "content": query},
        ]

        if wants_msg_content:
            messages.append({
                "role": "user",
                "content": (
                    "[WYMAGANE NARZĘDZIE: get_thread_messages]\n"
                    "Użytkownik prosi o TREŚĆ konkretnej wiadomości (albo o pokazanie jej ponownie). "
                    "Użyj get_thread_messages — NIGDY get_message_threads, ta druga zwraca tylko "
                    "metadane (kupujący/status/data), nigdy samej treści. Jeśli w tej rozmowie był już "
                    "wcześniej wskazany kupujący/data/wątek, użyj tych samych wartości ponownie "
                    "(buyer_login/date/thread_id)."
                ),
            })

        # ── Step 1: resolve the tool call — deterministically if possible ───────
        # Layer 2/3 of the tool-select pipeline (Layer 1 = the label filter
        # above): see agents/allegro/deterministic_dispatch.py for the
        # matchers and the conservative-by-design rationale — a confident
        # match skips the tool-select LLM call entirely. Never attempted for
        # a message-content request: get_thread_messages' correct arguments
        # depend on conversational context this layer doesn't parse.
        #
        # Judged from the CURRENT query alone, not `labels` above (which also
        # scans history for Layer 1's broader recall) — older turns on a
        # different topic would otherwise make a perfectly unambiguous
        # current query look multi-topic and needlessly skip this layer.
        query_labels = matched_labels(query)
        called_tools: list[str] = []
        single_tool_raw_result: str | None = None
        # tool call signature → result, so a model that re-asks for data it was
        # already given costs no second Allegro API round-trip.
        tool_results: dict[str, str] = {}

        det_match = None
        if not wants_msg_content:
            # "Szczegóły ostatniego zamówienia" always needs get_new_orders
            # to learn WHICH order before get_order_details can run — a
            # mechanical two-hop lookup, but one _match_get_new_orders
            # deliberately bails on (its detail-intent words are in
            # _ORDERS_BAIL_RE) since a single-tool matcher can't chain.
            # Resolve it here instead, before falling back to the
            # single-tool matchers.
            if wants_latest_order_details(query) and query_labels == {"zamowienia"}:
                det_match = await self._resolve_latest_order_chain()
            if det_match is None:
                det_match = resolve_deterministic(query, query_labels)
        if det_match is not None:
            det_tool, det_input = det_match
            called_tools.append(det_tool)
            logger.info("[allegro] deterministic tool match: %s(%s)", det_tool, det_input)
            try:
                with perf.stage(f"tool:{det_tool}"):
                    result = await self._execute_tool(det_tool, det_input)
            except Exception as exc:
                logger.exception("[allegro] tool %s failed: %s", det_tool, exc)
                result = "An internal error occurred. Please try again."
            single_tool_raw_result = result
            tool_results[f"{det_tool}:{json.dumps(det_input, sort_keys=True, default=str)}"] = result
            # Same assistant/tool message shape a real LLM tool call would
            # produce (see the loop below) — the interpret call, if it runs,
            # needs this history to look identical either way.
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "det-1",
                    "type": "function",
                    "function": {"name": det_tool, "arguments": json.dumps(det_input)},
                }],
            })
            messages.append({"role": "tool", "tool_call_id": "det-1", "content": result})
        else:
            # ── Force tool call(s) via the LLM ───────────────────────────────
            # The LLM MUST pick a real tool — it cannot answer from memory.
            # We allow up to MAX_TOOL_ROUNDS sequential tool calls (rare: e.g. look up
            # a thread ID then send a message), but stop as soon as the model signals
            # it has enough data (no new tool calls after a round).
            MAX_TOOL_ROUNDS = 3
            tool_rounds = 0

            while tool_rounds < MAX_TOOL_ROUNDS:
                # A reply with neither text nor a tool call means something went
                # wrong in round 1 (it was asked for a tool and produced nothing) but
                # is the normal "I have everything I need" signal from round 2 on —
                # so only round 1 gets the blank-reply model rotation.
                select_call = _call_for_reply if tool_rounds == 0 else _call_with_retry
                with perf.stage("tool_select_llm"):
                    resp = await select_call(
                        self._client,
                        model_pool,
                        f"allegro/tool-select-{tool_rounds + 1}",
                        messages=messages,
                        tools=tools,
                        # Round 1 must produce data — the model may not answer store
                        # questions from memory. From round 2 on it is "auto": the
                        # model has results in hand and signals it is done by
                        # returning no further tool calls (with "required" it could
                        # never say that, and would be forced to call something).
                        tool_choice="required" if tool_rounds == 0 else "auto",
                        max_tokens=400,
                        # This call only ever needs to pick a tool + arguments per the
                        # MANDATORY TOOL CALLS rules above — no open-ended reasoning is
                        # needed. Without this, thinking-capable models (gemini-3.5-flash,
                        # 2.5-flash) burn seconds of invisible reasoning tokens before
                        # emitting the tool call (same issue as orchestrator._classify_with_llm
                        # — see the comment there), which was the single biggest contributor
                        # to the 30s-60s response times reported for e.g. "new orders" queries.
                        reasoning_effort="none",
                    )
                msg = resp.choices[0].message

                if not msg.tool_calls:
                    if tool_rounds > 0:
                        # The model has its data and wants no more tools — that is the
                        # normal way out of a multi-round chain. Go format the answer.
                        break
                    # Round 1 with no tool call: the model ignored tool_choice=required.
                    # Interpret whatever it said — but never hand back an empty reply
                    # (the user would just see the orchestrator's generic "nie udało
                    # się wygenerować odpowiedzi"), so fall through to Step 2 instead
                    # and let the interpret call answer from the conversation.
                    logger.warning("AllegroAgent: model skipped tool_choice=required (round %d)", tool_rounds + 1)
                    if (msg.content or "").strip():
                        perf.log(result="no_tool_call")
                        return AgentResponse(text=msg.content, agent_type=self.agent_name, metadata={"output_format": "chat"})
                    logger.warning("AllegroAgent: no tool call and no text — falling through to interpret")
                    break

                # The model couldn't determine the right tool or a required parameter
                # from the query — it asked instead of guessing/hallucinating an ID.
                # Short-circuit here: no real Allegro call, no interpret step, and any
                # other tool call the model bundled into the same round is dropped
                # rather than executed against guessed arguments.
                clarify_call = next(
                    (tc for tc in msg.tool_calls if tc.function.name == "ask_clarifying_question"), None
                )
                if clarify_call:
                    try:
                        clarify_args = json.loads(clarify_call.function.arguments)
                    except json.JSONDecodeError:
                        clarify_args = {}
                    question = clarify_args.get("question") or "Czy możesz doprecyzować, o co dokładnie chodzi?"
                    logger.info("[allegro] ask_clarifying_question: %r | query=%.80r", question, query)
                    perf.log(result="ask_clarifying_question")
                    return AgentResponse(text=question, agent_type=self.agent_name, metadata={"output_format": "chat"})

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
                    if tool_name == "get_message_threads":
                        # The user's wording overrides whatever the model decided for
                        # count_only (see _wants_message_count_only above).
                        if _MESSAGE_LIST_OVERRIDE_RE.search(query):
                            tool_input["count_only"] = False
                        elif _wants_message_count_only(query):
                            tool_input["count_only"] = True
                    logger.info("[allegro] tool call: %s(%s)", tool_name, tool_input)
                    signature = f"{tool_name}:{json.dumps(tool_input, sort_keys=True, default=str)}"
                    if signature in tool_results:
                        logger.info("[allegro] repeat call to %s — reusing the previous result", tool_name)
                        result = tool_results[signature]
                    else:
                        try:
                            with perf.stage(f"tool:{tool_name}"):
                                result = await self._execute_tool(tool_name, tool_input)
                        except Exception as exc:
                            logger.exception("[allegro] tool %s failed: %s", tool_name, exc)
                            result = "An internal error occurred. Please try again."
                        tool_results[signature] = result
                    single_tool_raw_result = result if len(msg.tool_calls) == 1 else None
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                tool_rounds += 1

                # Cost guard on the follow-up round: it re-sends every tool result
                # collected so far alongside all ~30 tool schemas, so for a query
                # that already pulled a big listing (all offers, a 50-row order
                # table) asking "need anything else?" is both expensive and
                # pointless — that dump IS the answer. Chaining is for lookups,
                # whose results are small.
                collected = sum(len(r) for r in tool_results.values())
                if collected > _CHAIN_RESULT_BUDGET:
                    logger.info(
                        "[allegro] %d chars of tool results after round %d — formatting instead of chaining",
                        collected, tool_rounds,
                    )
                    break
                # No break here: most queries are satisfied by round 1 and the model
                # ends the loop itself by asking for no further tools (above). Some
                # genuinely need a chain, and cutting the loop off after round 1 left
                # those unanswerable — "szczegóły ostatniego nowego zamówienia" needs
                # get_new_orders to learn the order's UUID before get_order_details
                # can be called with it (that tool has no "latest" mode), so the model
                # was forced to either guess an ID or call the listing tool alone, and
                # the interpret step then had nothing that matched what was asked and
                # sometimes replied with nothing at all. MAX_TOOL_ROUNDS still caps it.

        # ── Skip the interpret call entirely when it would be pure passthrough ──
        # See _PASSTHROUGH_TOOLS above: the dispatch output IS the answer, down
        # to the count_only wording ("Masz 5 nowych zamówień."). Bypassed only
        # when this was the ONLY tool called in the turn — a multi-tool turn
        # still needs the LLM to weave the results together — and the query is
        # confidently Polish (see _is_confidently_polish); an English query
        # still needs the LLM to translate the Polish labels.
        if (
            len(called_tools) == 1
            and called_tools[0] in _PASSTHROUGH_TOOLS
            and single_tool_raw_result is not None
            and _is_confidently_polish(query)
        ):
            # Not hardcoded "chat": a table/document/dashboard tool renders as a
            # distinct artifact view in the frontend (web/js/app.js _isDoc) —
            # collapsing it into plain chat text here would silently drop that
            # presentation.
            bypass_format = resolve_output_format(called_tools)
            perf.log(
                source=self.agent_name, output_format=bypass_format,
                tools=called_tools[0], bypassed_interpret=True,
            )
            return AgentResponse(
                text=single_tool_raw_result,
                agent_type=self.agent_name,
                metadata={
                    "output_format": bypass_format,
                    "tools": called_tools,
                    "perf_stages": perf.snapshot(),
                    "perf_total_ms": perf.elapsed_ms(),
                },
            )

        # ── Step 1b: resolve output format from the tool(s) just called ───────
        # Deterministic — decided by WHICH TOOL ran, not guessed from the
        # user's wording (see TOOL_OUTPUT_FORMAT in allegro_tools.py for why).
        output_format = resolve_output_format(called_tools)
        # Any tool in the turn rendered its own view (that is every tool there
        # is — see _RENDERED_VIEW_TOOLS), so the interpret call is only reached
        # by a non-Polish or multi-tool turn and its brief is "hand it back".
        # A turn that called no tool at all has no view to protect and gets no
        # formatting instruction — there is nothing to format.
        format_instruction = (
            _RENDERED_VIEW_INSTRUCTION
            if any(t in _RENDERED_VIEW_TOOLS for t in called_tools)
            else None
        )
        if format_instruction:
            messages.append({"role": "user", "content": format_instruction})

        # The tool-routing system prompt (messages[0], used above to pick a
        # tool) no longer applies — swap in the much shorter interpret-only
        # prompt before this and any retry below reads `messages`.
        messages[0] = {"role": "system", "content": self._build_interpret_system_prompt(context)}

        # ── Step 2: interpret — NO tools ─────────────────────────────────────
        # The LLM now sees the real API results and formats the answer.
        # Without tools it literally cannot hallucinate store data.
        with perf.stage("interpret_llm"):
            interp_resp = await _call_for_reply(
                self._client,
                model_pool,
                "allegro/interpret",
                messages=messages,
                max_tokens=self._settings.gemini_max_tokens,
                # Pure formatting of data that's already in `messages` (see the
                # tool-select call above for why thinking-capable models eating
                # invisible reasoning tokens here — against a 16000 max_tokens
                # budget, easily tens of seconds — is the main driver of slow
                # replies) — no extended reasoning is needed to lay it out.
                reasoning_effort="none",
                # no `tools` parameter — model can only read and format
            )
        final_text = interp_resp.choices[0].message.content or ""

        # Last resort before the user gets the orchestrator's generic apology:
        # the tool data IS in `messages`, so a blank reply here means the model
        # balked at the format instruction — typically one demanding a full
        # document (get_products_to_reorder) from a tool result that turned out
        # to be an error or an empty list. Ask once more for a plain answer instead of
        # returning nothing.
        if not final_text.strip() and format_instruction:
            logger.warning(
                "[allegro] interpret returned nothing (tools=%s, format=%s) — "
                "retrying without the format instruction",
                ",".join(called_tools) or "none", output_format,
            )
            # messages[-1] is the format instruction appended just above.
            plain_messages = messages[:-1] + [{
                "role": "user",
                "content": (
                    "Odpowiedz na pytanie użytkownika krótko i po polsku, wyłącznie na "
                    "podstawie danych zwróconych przez narzędzia powyżej. Jeśli danych "
                    "brakuje albo narzędzie zwróciło błąd — napisz wprost, czego nie udało "
                    "się ustalić. Nie zmyślaj żadnych danych sklepu."
                ),
            }]
            with perf.stage("interpret_llm_retry"):
                retry_resp = await _call_for_reply(
                    self._client,
                    model_pool,
                    "allegro/interpret-plain",
                    messages=plain_messages,
                    max_tokens=self._settings.gemini_max_tokens,
                    reasoning_effort="none",
                )
            final_text = retry_resp.choices[0].message.content or ""
            if final_text.strip():
                output_format = "chat"

        if not final_text.strip():
            logger.error(
                "[allegro] empty reply for query=%.200r | tools=%s | format=%s",
                query, ",".join(called_tools) or "none", output_format,
            )
        perf.log(
            source=self.agent_name,
            output_format=output_format,
            tools=",".join(called_tools) or "none",
        )
        return AgentResponse(
            text=final_text,
            agent_type=self.agent_name,
            metadata={
                "output_format": output_format,
                "tools": called_tools,
                "perf_stages": perf.snapshot(),
                "perf_total_ms": perf.elapsed_ms(),
            },
        )

    def _request_auth(self) -> AgentResponse:
        # [ALLEGRO_LOGIN_BTN] is replaced client-side with a button that fetches
        # a signed auth URL from the backend and redirects there. A plain
        # "/allegro/login" markdown link would resolve against the frontend's
        # own origin on the split GitHub Pages / Cloud Run deployment (no such
        # route there) and 404 instead of reaching Allegro's login page.
        text = (
            "Aby uzyskać dostęp do Twojego sklepu Allegro, potrzebuję autoryzacji.\n\n"
            "[ALLEGRO_LOGIN_BTN]\n\n"
            "Po zalogowaniu wróć tutaj i zadaj swoje pytanie ponownie."
        )
        return AgentResponse(text=text, agent_type=self.agent_name, metadata={"output_format": "chat"})

    def _get_tools(self) -> list[dict[str, Any]]:
        return ALLEGRO_TOOLS

    async def _resolve_latest_order_chain(self) -> tuple[str, dict[str, Any]] | None:
        """Deterministically resolve "szczegóły ostatniego (nowego)
        zamówienia" without the tool-select LLM: look up which order is the
        latest directly (one Allegro API call — cheap relative to an LLM
        round-trip, see the "Zapytanie do Allegro" phase's share of total
        latency in the analytics dashboard), then hand back a normal
        single-tool det_match, reusing the existing deterministic-dispatch
        execution path either way — get_order_details when a new order
        exists, or get_new_orders bare (whose own dispatch already answers
        "Brak nowych zamówień.") when there isn't one to chain from.

        Returns None only on an unexpected error, falling through to the
        normal LLM tool-select path.
        """
        try:
            orders = await self._allegro.get_orders(
                status="READY_FOR_PROCESSING", fulfillment_status="NEW", limit=1,
            )
        except Exception as exc:
            logger.warning("[allegro] latest-order chain lookup failed, falling back to LLM: %s", exc)
            return None
        if not orders:
            return "get_new_orders", {"limit": 1}
        return "get_order_details", {"order_id": orders[0].order_id}

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
    def _is_balance_transfer_entry(entry: dict) -> bool:
        """PAD ("Pobranie opłat z wpływów") entries record Allegro sweeping money from
        the seller's Allegro Finanse proceeds to settle their account balance — an
        internal transfer, not a new charge or credit. The fee it settles is already
        its own billing entry, so counting PAD too double-counts that same money.
        It also carries no order.id (account-level, not order-level)."""
        type_desc = ((entry.get("type") or {}).get("description") or "").lower()
        type_id = (entry.get("type") or {}).get("id") or ""
        return type_id == "PAD" or "pobranie opłat z wpływów" in type_desc or "pobranie opłaty z wpływów" in type_desc

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
        """Group offers by name (case-insensitive), summing stock, sorted ascending by stock
        (lowest/most urgent first). Returns list of aggregated dicts. Tracks whether every
        listing behind a name is an ended/sold-out one (see _get_stock_relevant_offers) so
        callers can flag it."""
        groups: dict[str, dict] = defaultdict(
            lambda: {"ids": [], "name": "", "price": 0.0, "currency": "PLN", "total_stock": 0, "ended_only": True}
        )
        for offer in offers:
            oid, name, price, currency, stock = cls._offer_fields(offer)
            key = name.strip().lower()
            g = groups[key]
            g["ids"].append(oid)
            g["name"] = name
            g["price"] = price
            g["currency"] = currency
            g["total_stock"] += stock
            if not offer.get("_ended"):
                g["ended_only"] = False
        return sorted(groups.values(), key=lambda g: g["total_stock"])

    @staticmethod
    def _plural_pl(n: int, one: str, few: str, many: str) -> str:
        """Polish count form: 1 oferta / 2-4 oferty / 5+ ofert, with the usual
        11-14 exception (11 ofert, not 11 oferty)."""
        if n == 1:
            return one
        if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
            return few
        return many

    @classmethod
    def _count_sentence(
        cls, count: int, lead: str, forms: tuple[str, str, str], none: str, scope: str = ""
    ) -> str:
        """A count_only answer, finished: "Masz **3** nowe zamówienia.", never a
        "Liczba nowych zamówień: 3." label waiting for someone to reword it —
        this text reaches the user as-is (see _RENDERED_VIEW_TOOLS).

        `scope` is the optional period/filter note that goes inside the sentence
        ("Masz **3** zwroty (2026-08-01 – 2026-08-28).").
        """
        if not count:
            return none
        return f"{lead} **{count}** {cls._plural_pl(count, *forms)}{scope}."

    @staticmethod
    def _cell(value: Any) -> str:
        """One markdown table cell. A '|' inside the value would split the cell in
        two and shift every following column of that row; a newline would end the
        row outright — so neutralize both."""
        return str(value).replace("|", "\\|").replace("\n", " ").strip() or "—"

    @classmethod
    def _md_table(cls, headers: list[str], rows: list[list[Any]], align: str = "") -> list[str]:
        """Markdown table as a list of lines. `align` is one character per column:
        'r' right-aligns (numbers, prices), anything else left."""
        seps = []
        for i in range(len(headers)):
            seps.append("---:" if i < len(align) and align[i] == "r" else "---")
        return [
            "| " + " | ".join(cls._cell(h) for h in headers) + " |",
            "| " + " | ".join(seps) + " |",
            *["| " + " | ".join(cls._cell(c) for c in row) + " |" for row in rows],
        ]

    @classmethod
    def _offer_row(cls, g: dict) -> list[str]:
        """One aggregated-product row, shared by every offers table (active
        offers, stock query, price query, reorder list) so the seller sees the
        same columns everywhere."""
        ended_marker = " *(zakończona — wyprzedana)*" if g["ended_only"] else ""
        return [
            f"{g['name']}{ended_marker}",
            g["total_stock"],
            cls._format_price(g["price"], g["currency"]),
            ", ".join(f"`{i}`" for i in g["ids"]),
        ]

    _OFFER_COLUMNS = ["Nazwa", "Stan (szt.)", "Cena", "ID ofert"]

    @classmethod
    def _render_dashboard(
        cls, title: str, lead: str, sections: list[tuple[str, dict[str, int | float], str]]
    ) -> str:
        """A dashboard view: '## ' sections the PWA turns into cards, each with a
        bucket breakdown and a ```chart block the frontend swaps for a live
        Chart.js canvas (web/js/app.js _renderCharts).

        The chart JSON is built here, from the same numbers as the bullets, for
        the reason the whole view is: it is a mechanical transform of data, and
        having the interpret LLM type it out risked invented or mismatched
        figures in a chart the seller reads as fact.
        """
        out = [f"## {title}", "", lead]
        for name, buckets, chart_title in sections:
            filled = {k: v for k, v in buckets.items() if v}
            if not filled:
                continue
            out += ["", f"## {name}", ""]
            out += [
                f"- {k}: **{v}** {cls._plural_pl(int(v), 'oferta', 'oferty', 'ofert')}"
                for k, v in filled.items()
            ]
            out += [
                "",
                "```chart",
                json.dumps(
                    {
                        "type": "bar",
                        "title": chart_title,
                        "labels": list(filled.keys()),
                        "series": [{"name": chart_title, "data": list(filled.values())}],
                    },
                    ensure_ascii=False,
                ),
                "```",
            ]
        return "\n".join(out)

    # ── Sprzedaż wg miesięcy ──────────────────────────────────────────────────
    _MONTH_NAMES_PL = (
        "styczeń", "luty", "marzec", "kwiecień", "maj", "czerwiec",
        "lipiec", "sierpień", "wrzesień", "październik", "listopad", "grudzień",
    )
    # A month-by-month table only stays readable for so long; past this the
    # period is almost certainly a mistake (or an all-time question), and the
    # breakdown is dropped rather than printing a hundred rows.
    _MONTH_ROWS_CAP = 36

    @classmethod
    def _local_month(cls, iso_str: str) -> str:
        """ISO 8601 (UTC) → 'YYYY-MM' in Warsaw local time, '' when unparseable.

        Warsaw-local, not UTC: the period bounds the seller asked for are local
        calendar dates (see _local_day_bounds_to_utc), so an order paid at 00:30
        on 1 March local time belongs to March here too — bucketing it by its
        UTC month would file it under February and make the monthly rows
        disagree with the period total above them.
        """
        date_str = cls._to_warsaw_date(iso_str)
        return date_str[:7] if date_str else ""

    @classmethod
    def _month_label_pl(cls, month_key: str) -> str:
        """'2026-03' → 'marzec 2026'."""
        year, _, month = month_key.partition("-")
        try:
            return f"{cls._MONTH_NAMES_PL[int(month) - 1]} {year}"
        except (ValueError, IndexError):
            return month_key

    @classmethod
    def _months_in_range(cls, first: str, last: str) -> list[str]:
        """Every 'YYYY-MM' from `first` to `last` inclusive — gaps included, so a
        month with no sales shows up as a zero row instead of silently vanishing
        from a "podział na miesiące" report."""
        months: list[str] = []
        year, month = int(first[:4]), int(first[5:7])
        end_year, end_month = int(last[:4]), int(last[5:7])
        while (year, month) <= (end_year, end_month) and len(months) <= cls._MONTH_ROWS_CAP:
            months.append(f"{year:04d}-{month:02d}")
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return months

    @classmethod
    def _monthly_sales_section(cls, orders: list[Any], period_from: str, period_to: str) -> list[str]:
        """Per-month order count and revenue — the "podsumuj sprzedaż z podziałem
        na miesiące" view, built here rather than left to the interpret LLM for
        the same reason as every other table in this file: it is arithmetic over
        the order list, and a model retyping it can silently drop or invent a month.

        Returns [] for a period inside a single calendar month (the summary above
        already IS that month) or one too long to tabulate (see _MONTH_ROWS_CAP).
        """
        by_month: dict[str, dict[str, float]] = defaultdict(lambda: {"orders": 0.0, "revenue": 0.0})
        for o in orders:
            key = cls._local_month(o.paid_at or o.created_at)
            if not key:
                continue
            by_month[key]["orders"] += 1
            by_month[key]["revenue"] += o.total_price
        if not by_month:
            return []
        # Span the period the seller asked for, not just the months that happen to
        # have orders — a zero month is an answer, and a missing one reads as an
        # oversight. Fall back to the months with data when the period bounds
        # can't be read.
        first = period_from[:7] if len(period_from) >= 7 else min(by_month)
        last = period_to[:7] if len(period_to) >= 7 else max(by_month)
        months = cls._months_in_range(min(first, min(by_month)), max(last, max(by_month)))
        if len(months) < 2 or len(months) > cls._MONTH_ROWS_CAP:
            return []

        rows = []
        for m in months:
            count = int(by_month[m]["orders"]) if m in by_month else 0
            revenue = by_month[m]["revenue"] if m in by_month else 0.0
            avg = revenue / count if count else 0.0
            rows.append([
                cls._month_label_pl(m),
                count,
                cls._format_price(revenue),
                cls._format_price(avg),
            ])
        best = max(months, key=lambda m: by_month[m]["revenue"] if m in by_month else 0.0)
        return [
            "",
            "## Sprzedaż wg miesięcy",
            "",
            *cls._md_table(
                ["Miesiąc", "Zamówienia", "Przychód", "Śr. wartość"],
                rows,
                align="lrrr",
            ),
            "",
            f"Najlepszy miesiąc: **{cls._month_label_pl(best)}** "
            f"({cls._format_price(by_month[best]['revenue'])}).",
            "",
            "```chart",
            json.dumps(
                {
                    "type": "bar",
                    "title": "Przychód wg miesiąca",
                    # Short labels on purpose — a bar chart with twelve
                    # "październik 2026" ticks turns into an unreadable fan.
                    "labels": [f"{m[5:7]}.{m[:4]}" for m in months],
                    "series": [{
                        "name": "Przychód (PLN)",
                        "data": [round(by_month[m]["revenue"], 2) if m in by_month else 0 for m in months],
                    }],
                },
                ensure_ascii=False,
            ),
            "```",
        ]

    @classmethod
    def _render_offers_table(
        cls, aggregated: list[dict], raw: list[dict], name_filter: str | None = None
    ) -> str:
        """The final, ready-to-display answer for get_active_offers: a heading,
        the markdown table, and — last — a one-sentence summary.

        Rendering happens here, in Python, not in the interpret LLM: turning rows
        of data into the output view is mechanical work with no judgment in it,
        and a model doing it is slower, costs tokens and can drop, reorder or
        truncate rows. See _PASSTHROUGH_TOOLS — the model never sees this on the
        way out for a Polish query.

        The layout is what the PWA needs to split the reply the way the seller
        expects (web/js/app.js): the leading '# ' heading names the document tab
        (without it the tab title is a garbled '| Nazwa | Stan…' row), and the
        summary must come AFTER the table because _tablePreview shows the text
        following the table as the chat-bubble preview — so the chat says "masz
        obecnie XX aktywnych ofert" and the table stays in the document.
        """
        rows = cls._md_table(cls._OFFER_COLUMNS, [cls._offer_row(g) for g in aggregated], align="lrrl")

        ended_offers = sum(1 for o in raw if o.get("_ended"))
        active_offers = len(raw) - ended_offers
        products = len(aggregated)
        filter_note = f" (filtr nazwy: „{name_filter}”)" if name_filter else ""
        # Kept short on purpose: this sentence IS the chat bubble, and the
        # preview cuts off at 220 characters.
        summary = (
            f"Masz obecnie **{active_offers}** "
            f"{cls._plural_pl(active_offers, 'aktywną ofertę', 'aktywne oferty', 'aktywnych ofert')}"
            f"{filter_note} — w tabeli **{products}** "
            f"{cls._plural_pl(products, 'produkt', 'produkty', 'produktów')} "
            f"po zsumowaniu ofert o tej samej nazwie."
        )
        if ended_offers:
            summary += (
                f" W tym **{ended_offers}** "
                f"{cls._plural_pl(ended_offers, 'zakończona', 'zakończone', 'zakończonych')}"
                " przez wyprzedanie — do wznowienia."
            )
        return "\n".join(["# Aktywne oferty", "", *rows, "", summary])

    async def _get_stock_relevant_offers(self, name: str | None = None) -> list[dict]:
        """Active offers plus ended offers that sold out to zero stock.

        Allegro auto-ends an offer once its stock hits zero, so a sold-out product has no
        ACTIVE listing left even though it's exactly what needs reordering. An offer ended
        for any other reason (still has stock left) was stopped deliberately, so it's left
        out. Ended offers are tagged with _ended=True (on a copy, so the shared offers cache
        is never mutated) for _aggregate_offers_by_name to flag.

        The ENDED fetch is best-effort: if it fails, active offers are still returned rather
        than the whole query blowing up over a list that's a nice-to-have addition.
        """
        if name:
            active_result, ended_result = await asyncio.gather(
                self._allegro.get_offers(publication_status="ACTIVE", name=name, limit=100),
                self._allegro.get_offers(publication_status="ENDED", name=name, limit=100),
                return_exceptions=True,
            )
        else:
            active_result, ended_result = await asyncio.gather(
                self._allegro.get_all_offers("ACTIVE"),
                self._allegro.get_all_offers("ENDED"),
                return_exceptions=True,
            )
        if isinstance(active_result, BaseException):
            raise active_result
        active = active_result[0] if name else active_result
        if isinstance(ended_result, BaseException):
            logger.warning("_get_stock_relevant_offers: ENDED fetch failed, continuing with active only: %s", ended_result)
            ended = []
        else:
            ended = ended_result[0] if name else ended_result
        ended_sold_out = [
            dict(o, _ended=True) for o in ended
            if int((o.get("stock") or {}).get("available") or 0) == 0
        ]
        return active + ended_sold_out

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

    # Fulfillment states where the parcel has already left (or never will) —
    # the dispatch deadline is then history, not something to warn about.
    _DISPATCH_DONE_STATUSES: frozenset[str] = frozenset(
        {"SENT", "IN_TRANSIT", "READY_FOR_PICKUP", "PICKED_UP", "CANCELLED", "SUSPENDED"}
    )

    @classmethod
    def _fulfillment_pl(cls, status: str | None) -> str:
        return cls._FULFILLMENT_PL.get(status or "", status or "—")

    _PUBLICATION_PL: dict[str, str] = {
        "ACTIVE":    "Aktywna",
        "ACTIVATING": "Aktywowana",
        "INACTIVE":  "Nieaktywna",
        "ENDED":     "Zakończona",
    }

    @classmethod
    def _publication_pl(cls, status: str | None) -> str:
        return cls._PUBLICATION_PL.get(status or "", status or "—")

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
    def _local_day_bounds_to_utc(cls, date_from_local: str, date_to_local: str) -> tuple[str, str]:
        """Convert a Warsaw-local calendar date range ('YYYY-MM-DD', inclusive) to UTC ISO 8601 bounds.

        Allegro filters by UTC instant, but the seller thinks in Warsaw calendar days — an order
        placed at 00:30 local time is "today" to them even though it's still "yesterday" in UTC
        for part of the year. Anchoring the boundaries to Warsaw midnight (not UTC midnight) keeps
        those orders in the day the seller actually means.
        """
        start_local = datetime.strptime(date_from_local, "%Y-%m-%d").replace(tzinfo=cls._WARSAW)
        end_local = datetime.strptime(date_to_local, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=cls._WARSAW
        )
        return (
            start_local.astimezone(cls._UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            end_local.astimezone(cls._UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

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

    async def _invoice_reminder_status_block(self) -> str:
        """Deterministic (non-LLM) status + action button for the automatic
        invoice REMINDER — a scheduled 7:00-20:00 check (every 2h by default,
        adjustable by the seller) for unissued VAT invoices on already-shipped
        orders, which asks in chat whether to issue them and adapts to the
        reply (see services/invoice_reminder.py). This is the only invoice
        automation left — the client-side 15-minute "new order needs an
        invoice" notifier it used to sit next to was removed (kept on the
        archive/invoice-monitoring branch).
        """
        from services.invoice_reminder import is_monitor_enabled

        if await is_monitor_enabled(self._allegro._user_id):
            return (
                "⏰ Automatyczne przypomnienia o niewystawionych fakturach są włączone — co 2 "
                "godziny (7:00-20:00) sprawdzę, czy są niewystawione faktury dla wysłanych "
                "zamówień, i zapytam Cię o to na czacie.\n\n"
                '<button class="btn-invoice-reminder" style="filter:grayscale(1)" '
                'onclick="InvoiceReminder.disable();this.outerHTML=\'<span>✓ Przypomnienia o fakturach wyłączone</span>\'">'
                '🔕 Wyłącz przypomnienia o fakturach</button>'
            )
        return (
            "💡 Mogę co 2 godziny (7:00-20:00) sprawdzać, czy są niewystawione faktury dla już "
            "wysłanych zamówień, i pytać Cię o to na czacie — a jeśli poprosisz o odłożenie, "
            "zapamiętam na jak długo.\n\n"
            '<button class="btn-invoice-reminder" onclick="InvoiceReminder.enable()">'
            '⏰ Włącz przypomnienia o fakturach</button>'
        )

    async def _message_monitoring_status_block(self) -> str:
        """Deterministic (non-LLM) status + action button for automatic message
        checking — same rationale as _monitoring_status_block above."""
        from services.monitor_state import is_monitor_enabled

        if await is_monitor_enabled("message", self._allegro._user_id):
            return (
                "💬 Automatyczne sprawdzanie nowych wiadomości jest włączone — dam Ci znać, "
                "gdy kupujący napisze coś nowego, także gdy aplikacja jest zamknięta.\n\n"
                '<button class="btn-message-monitoring" style="filter:grayscale(1)" '
                'onclick="MessageMonitor.disable();this.outerHTML=\'<span>✓ Monitoring wiadomości wyłączony</span>\'">'
                '🔕 Wyłącz monitoring wiadomości</button>'
            )
        return (
            "💡 Mogę na bieżąco sprawdzać, czy pojawiły się nowe wiadomości od kupujących, "
            "i natychmiast Cię powiadamiać — także gdy aplikacja jest zamknięta.\n\n"
            '<button class="btn-message-monitoring" onclick="MessageMonitor.enable()">'
            '💬 Włącz monitoring wiadomości</button>'
        )

    async def _returns_monitoring_status_block(self) -> str:
        """Deterministic (non-LLM) status + action button for automatic returns/
        complaints checking — same rationale as _monitoring_status_block above."""
        from services.return_complaint_monitor import is_monitor_enabled

        if await is_monitor_enabled(self._allegro._user_id):
            return (
                "↩️ Automatyczne sprawdzanie nowych zwrotów i reklamacji jest włączone — dam Ci "
                "znać, gdy pojawi się coś nowego.\n\n"
                '<button class="btn-returns-monitoring" style="filter:grayscale(1)" '
                'onclick="ReturnsMonitor.disable();this.outerHTML=\'<span>✓ Monitoring zwrotów i reklamacji wyłączony</span>\'">'
                '🔕 Wyłącz monitoring zwrotów i reklamacji</button>'
            )
        return (
            "💡 Mogę automatycznie sprawdzać nowe zwroty i reklamacje i wysyłać Ci powiadomienia, "
            "nawet gdy ta karta jest w tle.\n\n"
            '<button class="btn-returns-monitoring" onclick="ReturnsMonitor.enable()">'
            '↩️ Włącz monitoring zwrotów i reklamacji</button>'
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
    def _dispatch_deadline_pl(cls, o: Any) -> str:
        """'Do kiedy paczka ma zostać wysłana' — delivery.time.dispatch.to, in
        Warsaw local time, with a '⚠️ po terminie' marker once the deadline has
        passed and the order still isn't sent.

        Allegro doesn't return the dispatch window on every order (older ones,
        some delivery methods), hence the '—' fallback: an unknown deadline must
        never be rendered as an urgent one.
        """
        deadline = getattr(o, "dispatch_to", "") or ""
        if not deadline:
            return "—"
        formatted = cls._format_dt_pl(deadline)
        # Already handed over (or no longer to be handed over) — the deadline is
        # history, not a warning.
        if getattr(o, "fulfillment_status", "") in cls._DISPATCH_DONE_STATUSES:
            return formatted
        try:
            dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
        except ValueError:
            return formatted
        if dt < datetime.now(cls._UTC):
            return f"{formatted} ⚠️ po terminie"
        return formatted

    @classmethod
    def _to_warsaw_date(cls, iso_str: str) -> str:
        """ISO 8601 (UTC) → 'YYYY-MM-DD' in Warsaw local time, for date matching."""
        if not iso_str:
            return ""
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return dt.astimezone(cls._WARSAW).date().isoformat()
        except ValueError:
            return ""

    @classmethod
    def _today_warsaw(cls) -> str:
        """Today's date ('YYYY-MM-DD') in Warsaw local time.

        Kept as its own classmethod rather than called inline from _dispatch —
        a local `from datetime import datetime, ...` elsewhere in that method
        makes `datetime` a local name for the whole function in Python's
        scoping, so a bare `datetime.now(...)` there raises UnboundLocalError
        on any branch that runs before that import line.
        """
        return datetime.now(cls._WARSAW).date().isoformat()

    @classmethod
    def _order_bullet(
        cls,
        o: Any,
        *,
        carrier_map: dict[str, str] | None = None,
        include_delivery: bool = False,
        extra_lines: list[str] | None = None,
    ) -> str:
        """Render ONE order as the bullet block that every order listing uses.

        Deliberately the only order renderer left: get_new_orders,
        get_orders, get_orders_delivery and get_orders_pending_invoice each
        had their own block before, differing in field order and even in
        timezone (one printed raw UTC timestamps, the other Warsaw local
        time), so the same order read differently depending on which tool
        happened to fetch it. Fields, in the order the store owner asked for:
        buyer, status, dispatch deadline, delivery type, quantity, value,
        payment time, order time — then courier details when the caller asked
        for them, any caller-specific extras, and the panel link.
        """
        d = o.delivery if isinstance(o.delivery, dict) else {}
        carrier_id = cls._dig(d, "method", "id", default="")
        delivery_name = (carrier_map or {}).get(carrier_id) or cls._dig(d, "method", "name", default="—")
        total_qty = sum(li.quantity for li in o.line_items)
        price = cls._format_price(o.total_price, o.currency)
        lines = [
            f"**Zamówienie** `{o.order_id}`",
            f"- Zamawiający: **{o.buyer_login}**",
            f"- Status: **{cls._fulfillment_pl(o.fulfillment_status)}**",
            f"- Wysyłka do: **{cls._dispatch_deadline_pl(o)}**",
            f"- Rodzaj dostawy: {delivery_name}",
            f"- Ilość: {total_qty} szt.",
            f"- Wartość: **{price}**",
            f"- Opłacone: {cls._format_dt_pl(o.paid_at)}",
        ]
        if o.created_at:
            lines.append(f"- Złożone: {cls._format_dt_pl(o.created_at)}")
        if include_delivery:
            tracking = (
                cls._dig(d, "smart", "trackingCode", default=None)
                or cls._dig(d, "trackingCode", default="—")
            )
            tracking_url = cls._tracking_url(carrier_id, tracking)
            tracking_str = f"[{tracking}]({tracking_url})" if tracking_url else tracking
            lines.append(f"- Numer śledzenia: {tracking_str}")
            pickup_name = cls._dig(d, "pickupPoint", "name", default=None)
            if pickup_name:
                lines.append(f"- Punkt odbioru: {pickup_name}")
        if extra_lines:
            lines.extend(f"- {line}" for line in extra_lines)
        lines.append(f"- Link: https://allegro.pl/sprzedaz/zamowienia/{o.order_id}")
        return "\n".join(lines)

    # A period query lists everything in the window, which for a busy month can
    # be hundreds of returns — far more than a chat reply should dump. 50 is the
    # single-page size the dateless listing has always shown, so nothing that
    # used to fit gets truncated now.
    _RETURNS_LISTING_CAP = 50

    @classmethod
    def _optional_period(cls, tool_input: dict) -> tuple[str | None, str | None, str | None]:
        """Resolve an optional date_from_local/date_to_local pair into UTC bounds.

        Returns (date_from, date_to, label) — all None when the tool was called
        without a period, which means "most recent, any date". A malformed date
        degrades to no period rather than raising: answering about all recent
        items is recoverable, erroring out on the user's question isn't.
        """
        date_from_local = tool_input.get("date_from_local")
        date_to_local = tool_input.get("date_to_local")
        if not (date_from_local and date_to_local):
            return None, None, None
        try:
            date_from, date_to = cls._local_day_bounds_to_utc(date_from_local, date_to_local)
        except (ValueError, TypeError):
            logger.warning(
                "_optional_period: unparseable period %r – %r, ignoring", date_from_local, date_to_local
            )
            return None, None, None
        return date_from, date_to, f"{date_from_local} – {date_to_local}"

    # ── Kupujący ──────────────────────────────────────────────────────────────
    # One row per buyer, not per order (get_buyers). Two caps bound the work:
    # how many rows the table may hold, and how many per-order invoice lookups
    # the report may spend (Allegro has no bulk endpoint — it is one GET per
    # order, see AllegroService.invoices_issued_map).
    _BUYERS_TABLE_CAP = 200
    _INVOICE_LOOKUP_CAP = 400

    _BUYER_FILTER_LABELS: dict[tuple[str, str], str] = {
        ("buyer_type", "company"):        "firmy",
        ("buyer_type", "person"):         "osoby prywatne",
        ("invoice_status", "issued"):     "z wystawioną fakturą VAT",
        ("invoice_status", "missing"):    "z niewystawioną fakturą VAT",
        ("invoice_status", "requested"):  "z prośbą o fakturę VAT",
    }

    @classmethod
    def _period_or_current_year(cls, tool_input: dict) -> tuple[str, str, str]:
        """UTC bounds + label for a period pair that defaults to the CURRENT
        CALENDAR YEAR.

        Unlike _optional_period (where "no period" legitimately means "the most
        recent ones, any date"), a buyer report with no period is a question
        about this year's customers — an all-time list would be a different,
        much slower answer nobody asked for. A malformed date degrades to the
        same default instead of raising: the period label is printed in the
        reply, so the seller always sees which period they actually got.
        """
        today = datetime.now(cls._WARSAW).date()
        default_from, default_to = f"{today.year}-01-01", today.isoformat()
        date_from_local = tool_input.get("date_from_local") or default_from
        date_to_local = tool_input.get("date_to_local") or default_to
        try:
            date_from, date_to = cls._local_day_bounds_to_utc(date_from_local, date_to_local)
        except (ValueError, TypeError):
            logger.warning(
                "_period_or_current_year: unparseable period %r – %r, using the current year",
                date_from_local, date_to_local,
            )
            date_from_local, date_to_local = default_from, default_to
            date_from, date_to = cls._local_day_bounds_to_utc(date_from_local, date_to_local)
        return date_from, date_to, f"{date_from_local} – {date_to_local}"

    @staticmethod
    def _buyer_key(order: Any) -> tuple[str, str]:
        """What makes two orders the same customer: the NIP where there is one
        (a company can buy from several Allegro accounts, and the NIP is the
        identity the seller invoices), otherwise the company name, otherwise the
        Allegro login. A company that ordered once with invoice data and once
        without therefore splits into two rows — nothing in the second order
        links it to the first, and inventing that link would merge unrelated
        buyers."""
        buyer = order.invoice_buyer
        if buyer.vat_id:
            return "nip", re.sub(r"[\s-]", "", buyer.vat_id)
        if buyer.company_name:
            return "firma", buyer.company_name.strip().lower()
        return "login", order.buyer_login

    @staticmethod
    def _delivery_recipient(order: Any) -> str:
        """Who the parcel is addressed to, from delivery.address: the company
        name when the recipient gave one, otherwise their first and last name.

        This is the second-best name Allegro carries after the invoice address —
        every order has it, including the ones where nobody asked for an invoice.
        Empty only when Allegro returned no delivery address at all.
        """
        address = order.delivery.get("address") if isinstance(order.delivery, dict) else None
        if not isinstance(address, dict):
            return ""
        company = (address.get("companyName") or "").strip()
        if company:
            return company
        return f"{address.get('firstName') or ''} {address.get('lastName') or ''}".strip()

    @classmethod
    def _aggregate_buyers(
        cls, orders: list[Any], invoice_flags: dict[str, bool | None]
    ) -> list[dict[str, Any]]:
        """Group orders into one entry per buyer, newest order first so the name
        and NIP shown are the most recent ones that buyer gave."""
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for order in sorted(orders, key=lambda o: (o.paid_at or o.created_at or ""), reverse=True):
            key = cls._buyer_key(order)
            group = groups.get(key)
            if group is None:
                buyer = order.invoice_buyer
                group = groups[key] = {
                    "name": "",
                    "is_company": buyer.is_company,
                    "vat_id": buyer.vat_id,
                    "logins": [],
                    "orders": 0,
                    "value": 0.0,
                    "currency": order.currency,
                    "invoices": 0,
                    "last_bought": "",
                }
            if not group["name"]:
                # Newest-first iteration, so this is the most recent order that
                # actually names the buyer: the invoice address if there is one,
                # otherwise whoever the parcel was addressed to. The Allegro
                # login is a machine handle ("kasia.w"), never a name — it only
                # steps in below when the order carries neither, and it has its
                # own column anyway.
                group["name"] = order.invoice_buyer.display_name or cls._delivery_recipient(order)
            if order.buyer_login and order.buyer_login not in group["logins"]:
                group["logins"].append(order.buyer_login)
            group["orders"] += 1
            group["value"] += order.total_price
            if invoice_flags.get(order.order_id) is True:
                group["invoices"] += 1
            # Kept as the raw UTC ISO string: it is both what sorts correctly
            # (sort_by='recent', and max() here) and what _format_dt_pl needs to
            # render the date in Warsaw local time at the end.
            group["last_bought"] = max(group["last_bought"], order.paid_at or order.created_at or "")
        for group in groups.values():
            group["name"] = group["name"] or (group["logins"] or ["—"])[0]
        return list(groups.values())

    async def _invoice_flags(self, orders: list[Any]) -> tuple[dict[str, bool | None], int]:
        """(order_id → invoice attached?, how many orders the cap left unchecked).

        Orders whose buyer asked for an invoice go first, so when the cap bites
        it falls on the orders least likely to carry one.
        """
        candidates = sorted(orders, key=lambda o: not o.invoice_required)
        checked = candidates[: self._INVOICE_LOOKUP_CAP]
        flags = await self._allegro.invoices_issued_map([o.order_id for o in checked])
        return flags, len(candidates) - len(checked)

    async def _buyers_report(self, tool_input: dict[str, Any]) -> str:
        """The finished get_buyers answer: heading, table, trailing summary.

        Company-vs-person and the invoice state are per-ORDER facts (they live in
        the order's VAT-invoice address block), so both filters run over orders
        first and only what survives is grouped — a buyer counts as a company
        here exactly when at least one of their orders in the period carried
        company invoice data.
        """
        date_from, date_to, period_label = self._period_or_current_year(tool_input)
        buyer_type = tool_input.get("buyer_type") or "any"
        invoice_status = tool_input.get("invoice_status") or "any"
        sort_by = tool_input.get("sort_by") or "value"
        limit = max(1, min(int(tool_input.get("limit") or 100), self._BUYERS_TABLE_CAP))

        orders = await self._allegro.get_all_paid_orders_in_period(date_from, date_to)
        if buyer_type == "company":
            orders = [o for o in orders if o.invoice_buyer.is_company]
        elif buyer_type == "person":
            orders = [o for o in orders if not o.invoice_buyer.is_company]
        if invoice_status in ("requested", "missing"):
            orders = [o for o in orders if o.invoice_required]

        # No invoice filter ⇒ no lookups at all: they cost one API call per order,
        # and a question that never mentioned invoices shouldn't pay for hundreds.
        flags: dict[str, bool | None] = {}
        unchecked = 0
        if invoice_status != "any" and orders:
            flags, unchecked = await self._invoice_flags(orders)
        if invoice_status == "issued":
            orders = [o for o in orders if flags.get(o.order_id) is True]
        elif invoice_status == "missing":
            orders = [o for o in orders if flags.get(o.order_id) is False]

        buyers = self._aggregate_buyers(orders, flags)
        if sort_by == "recent":
            buyers.sort(key=lambda g: (g["last_bought"], g["value"]), reverse=True)
        elif sort_by == "orders":
            buyers.sort(key=lambda g: (g["orders"], g["value"]), reverse=True)
        else:
            buyers.sort(key=lambda g: (g["value"], g["orders"]), reverse=True)

        filters = [
            label for (arg, value), label in self._BUYER_FILTER_LABELS.items()
            if {"buyer_type": buyer_type, "invoice_status": invoice_status}[arg] == value
        ]
        filter_note = f" ({', '.join(filters)})" if filters else ""
        logger.info(
            "get_buyers: %d orders → %d buyers (%s, typ=%s, faktury=%s)",
            len(orders), len(buyers), period_label, buyer_type, invoice_status,
        )

        if tool_input.get("count_only"):
            return self._count_sentence(
                len(buyers), "Miałeś", ("kupującego", "kupujących", "kupujących"),
                none=f"Brak kupujących{filter_note} w okresie {period_label}.",
                scope=f"{filter_note} w okresie {period_label}",
            )
        if not buyers:
            return f"Brak kupujących{filter_note} w okresie {period_label}."

        shown = buyers[:limit]
        # The invoice column earns its place only where it can vary: with
        # invoice_status='missing' every row is 0 by construction, and with 'any'
        # no lookup ran at all.
        with_invoices = invoice_status in ("issued", "requested")
        headers = ["Kupujący", "Typ", "NIP", "Login Allegro", "Zamówienia", "Wartość"]
        align = "llllrr"
        if with_invoices:
            headers.append("Faktury VAT")
            align += "r"
        headers.append("Ostatni zakup")
        align += "l"

        rows = []
        for group in shown:
            row: list[Any] = [
                group["name"],
                "Firma" if group["is_company"] else "Osoba prywatna",
                group["vat_id"],
                ", ".join(f"`{login}`" for login in group["logins"]),
                group["orders"],
                self._format_price(group["value"], group["currency"]),
            ]
            if with_invoices:
                row.append(group["invoices"])
            row.append(self._format_dt_pl(group["last_bought"])[:10])
            rows.append(row)

        total_orders = sum(g["orders"] for g in buyers)
        total_value = sum(g["value"] for g in buyers)
        # Kept short on purpose: this sentence IS the chat bubble (the table goes
        # to the document viewer), and the preview cuts off at 220 characters.
        # Phrased as a noun phrase, not "kupowało u Ciebie N…": the Polish verb
        # would have to agree with the count ("kupował" for 1, "kupowało" for 2+),
        # and a summary line that reads as broken grammar half the time is worse
        # than one with no verb at all.
        summary = (
            f"**{len(buyers)}** "
            f"{self._plural_pl(len(buyers), 'kupujący', 'kupujących', 'kupujących')}"
            f"{filter_note} w okresie {period_label} — łącznie **{total_orders}** "
            f"{self._plural_pl(total_orders, 'zamówienie', 'zamówienia', 'zamówień')} "
            f"na **{self._format_price(total_value)}**."
        )
        if len(shown) < len(buyers):
            summary += f" W tabeli pokazano pierwszych {len(shown)}."
        unknown = sum(1 for value in flags.values() if value is None)
        if unchecked or unknown:
            summary += (
                f" Statusu faktury nie udało się sprawdzić dla {unchecked + unknown} "
                f"{self._plural_pl(unchecked + unknown, 'zamówienia', 'zamówień', 'zamówień')} "
                "— mogły nie trafić do zestawienia."
            )
        return "\n".join(["# Kupujący", "", *self._md_table(headers, rows, align=align), "", summary])

    @classmethod
    def _sorted_by_date_desc(cls, items: list[dict], date_of) -> list[dict]:
        """Newest first — /order/customer-returns and /sale/issues don't promise
        any order, so 'the latest ones' has to be established here."""
        return sorted(items, key=lambda item: date_of(item) or "", reverse=True)

    @classmethod
    def _dated_listing(cls, items: list[dict], header: str, date_of, bullet) -> str:
        ordered = cls._sorted_by_date_desc(items, date_of)
        shown = ordered[: cls._RETURNS_LISTING_CAP]
        lines = [f"**{header}** — {len(ordered)}", ""]
        lines.append("\n\n".join(bullet(item) for item in shown))
        if len(ordered) > len(shown):
            lines.append(f"\n…i {len(ordered) - len(shown)} więcej.")
        return "\n".join(lines)

    @classmethod
    def _returns_listing(cls, returns: list[dict], header: str) -> str:
        from services.allegro_service import return_created_at

        return cls._dated_listing(returns, header, return_created_at, cls._return_bullet)

    @classmethod
    def _complaints_listing(cls, issues: list[dict], header: str) -> str:
        from services.allegro_service import issue_opened_at

        return cls._dated_listing(issues, header, issue_opened_at, cls._complaint_bullet)

    @classmethod
    def _return_bullet(cls, item: dict) -> str:
        """Render one customer return (zwrot) from /order/customer-returns as a
        bullet block. Field names are read defensively (several candidate keys
        per field) since the exact response shape isn't publicly documented in
        full — Allegro's own convention elsewhere pairs id/status/createdAt."""
        return_id = item.get("id", "—")
        order_id = (
            cls._dig(item, "order", "id")
            or cls._dig(item, "checkoutForm", "id")
            or item.get("orderId")
            or "—"
        )
        buyer = cls._dig(item, "buyer", "login") or cls._dig(item, "buyer", "email") or "—"
        status = item.get("status") or cls._dig(item, "reception", "status") or "—"
        created = (
            item.get("createdAt")
            or item.get("receivedAt")
            or cls._dig(item, "reception", "createdAt")
            or ""
        )
        return (
            f"**Zwrot** `{return_id}`\n"
            f"- Zamówienie: `{order_id}`\n"
            f"- Kupujący: **{buyer}**\n"
            f"- Status: {status}\n"
            f"- Data: {cls._format_dt_pl(created) if created else '—'}"
        )

    @classmethod
    def _complaint_bullet(cls, item: dict) -> str:
        """Render one dispute/claim (reklamacja) from /sale/issues as a bullet
        block — id, type, subject, checkoutForm.id, openedDate, decisionDueDate
        are the fields Allegro's own developer docs show for this resource."""
        issue_id = item.get("id", "—")
        order_id = cls._dig(item, "checkoutForm", "id") or "—"
        issue_type = item.get("type") or "—"
        subject = item.get("subject") or "—"
        opened = item.get("openedDate") or ""
        due = item.get("decisionDueDate") or ""
        lines = [
            f"**Reklamacja/spór** `{issue_id}`",
            f"- Zamówienie: `{order_id}`",
            f"- Typ: {issue_type}",
            f"- Temat: {subject}",
            f"- Otwarto: {cls._format_dt_pl(opened) if opened else '—'}",
        ]
        if due:
            lines.append(f"- Termin decyzji: {cls._format_dt_pl(due)}")
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
        # `<id>` has to stay inside backticks: the chat renders replies as markdown,
        # and a bare <id> is parsed as an HTML tag and dropped, leaving the sentence
        # ending in "…dla zamówienia ”".
        footer = (
            "\n\nSprawdź dane, a wystawienie w inFakt zrób ręcznie, albo poproś o wystawienie "
            "konkretnego zamówienia pojedynczo (np. „wystaw fakturę dla zamówienia `<id>`”)."
        )
        return header + "\n\n".join(blocks) + footer

    async def _issue_invoice_for_order(self, order_id: str) -> str:
        """Actually create ONE real VAT invoice in inFakt for a single named order.

        Deliberately single-order only — bulk issuance stays preview-only
        (see _preview_pending_invoices) after an earlier misfire on an
        ambiguous yes/no question created a real (failed) issuance attempt.
        Requiring a concrete order_id per call keeps the blast radius of any
        future misfire to at most one invoice.

        Delegates to services.infakt_service.issue_invoice_for_order, which is
        also called (once per order, same one-at-a-time path) by the invoice
        reminder's "issue now" action — see services/invoice_reminder.py.
        """
        from services.infakt_service import issue_invoice_for_order

        return await issue_invoice_for_order(self._allegro, order_id, self._settings.is_production)

    async def _unblock_invoice_for_order(self, order_id: str, seller_confirmed: bool) -> str:
        """Re-check inFakt and lift the duplicate block on one order, if inFakt agrees.

        The counterpart to the guard in issue_invoice_for_order: that guard
        deliberately errs towards "the invoice probably exists", which would
        otherwise strand an order whose issuance really did fail unrecorded.
        This does NOT issue anything — see
        services.infakt_service.unblock_invoice_for_order for which branch is
        allowed to clear the block and why the seller's own word only counts
        when there is nothing left to ask inFakt about.
        """
        from services.infakt_service import unblock_invoice_for_order

        return await unblock_invoice_for_order(self._allegro, order_id, seller_confirmed)

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

    # ── Order listing: one implementation behind three tools ─────────────────
    # get_new_orders / get_orders / get_orders_delivery exist as separate tool
    # schemas only so the model can route intents against named presets (see
    # the comment above _ORDER_PARAMS in allegro_tools.py). Functionally they
    # are this table plus _orders_listing() below — same call, same rendering,
    # same "chat" output format — so "nowe zamówienia" and "zamówienia do
    # wysłania" can no longer come back in two different shapes.
    _ORDERS_PRESETS: dict[str, dict[str, Any]] = {
        "get_new_orders": {
            "status": "READY_FOR_PROCESSING",
            "fulfillment_status": "NEW",
            "limit": 100,
            "empty": "Brak nowych zamówień.",
            # count_only answers with the finished sentence, like every other
            # view here — nothing downstream rewords it (see _RENDERED_VIEW_TOOLS).
            "count_lead": "Masz",
            "count_forms": ("nowe zamówienie", "nowe zamówienia", "nowych zamówień"),
            "count_none": "Nie masz nowych zamówień.",
            # Only the new-orders view carries the automatic-checking status
            # block + button; a filtered/period listing is a one-off question,
            # not the "watch my inbox" view the toggle belongs to.
            "monitoring_block": True,
        },
        "get_orders": {
            "limit": 50,
            "empty": "Brak zamówień spełniających podane kryteria.",
            "count_lead": "Masz łącznie",
            "count_forms": ("zamówienie", "zamówienia", "zamówień"),
            "count_none": "Brak zamówień spełniających podane kryteria.",
        },
        "get_orders_delivery": {
            "status": "READY_FOR_PROCESSING",
            "fulfillment_status": "READY_FOR_SHIPMENT",
            "limit": 50,
            "include_delivery": True,
            "empty": "Brak zamówień gotowych do wysłania.",
            "count_lead": "Do wysłania:",
            "count_forms": ("zamówienie", "zamówienia", "zamówień"),
            "count_none": "Brak zamówień gotowych do wysłania.",
        },
        # The deadline view. Unlike the presets above it is defined by WHEN the
        # parcel has to leave, not by how far along it is, so it filters by
        # exclusion: everything still to be handed over, whether nobody has
        # packed it yet or it is already sitting by the door.
        "get_orders_due_today": {
            "status": "READY_FOR_PROCESSING",
            "exclude_fulfillment": _DISPATCH_DONE_STATUSES,
            # End of today. No lower bound on purpose: an order whose deadline
            # passed yesterday is the most urgent thing on the list, not
            # something to hide from "co muszę wysłać dzisiaj".
            "dispatch_before_local": "23:59",
            "sort_by_dispatch": True,
            "limit": 50,
            "include_delivery": True,
            "empty": "Nic nie czeka na wysyłkę z dzisiejszym terminem.",
            "count_lead": "Na dziś do wysłania:",
            "count_forms": ("zamówienie", "zamówienia", "zamówień"),
            "count_none": "Nic nie czeka na wysyłkę z dzisiejszym terminem.",
        },
    }

    @classmethod
    def _optional_local_to_utc(cls, time_str: str | None) -> str | None:
        """_local_to_utc for an optional argument: None (no filter) when absent
        or unparseable — answering about a wider window is recoverable, blowing
        up on the user's question isn't."""
        if not time_str:
            return None
        try:
            return cls._local_to_utc(time_str)
        except (ValueError, TypeError):
            logger.warning("_optional_local_to_utc: unparseable time %r, ignoring", time_str)
            return None

    @classmethod
    def _dispatch_within(cls, o: Any, after: str | None, before: str | None) -> bool:
        """Does this order's dispatch deadline fall inside [after, before]?

        Allegro has no query parameter for delivery.time.dispatch.to, so this
        filter is applied client-side. An order with no deadline at all is
        excluded: "co muszę wysłać do jutra" must not answer with orders whose
        deadline is simply unknown.
        """
        deadline = getattr(o, "dispatch_to", "") or ""
        if not deadline:
            return False
        if after and deadline < after:
            return False
        if before and deadline > before:
            return False
        return True

    async def _orders_listing(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """The one order-listing implementation, shared by all three order
        tools. `tool_name` only picks the preset defaults in _ORDERS_PRESETS;
        every explicit argument the model passed wins over them."""
        preset = self._ORDERS_PRESETS.get(tool_name, self._ORDERS_PRESETS["get_orders"])
        status = tool_input.get("status") or preset.get("status")
        fulfillment_status = tool_input.get("fulfillment_status") or preset.get("fulfillment_status")
        include_delivery = bool(tool_input.get("include_delivery", preset.get("include_delivery", False)))
        try:
            limit = int(tool_input.get("limit") or preset["limit"])
        except (TypeError, ValueError):
            limit = int(preset["limit"])
        limit = max(1, min(limit, 100))

        # "23:59" in a preset is resolved against the CURRENT day on every call
        # (see _local_to_utc), so the deadline view means today whenever it runs
        # — no date is ever baked into the preset itself.
        dispatch_after = self._optional_local_to_utc(
            tool_input.get("dispatch_after_local") or preset.get("dispatch_after_local")
        )
        dispatch_before = self._optional_local_to_utc(
            tool_input.get("dispatch_before_local") or preset.get("dispatch_before_local")
        )
        exclude_fulfillment = preset.get("exclude_fulfillment") or frozenset()
        # Both the deadline filter and the status exclusion run client-side (the
        # Allegro API has no parameter for either — see _dispatch_within), so
        # fetch a full page and narrow afterwards; filtering a limit=1 fetch
        # would usually leave nothing at all.
        fetch_limit = 100 if (dispatch_after or dispatch_before or exclude_fulfillment) else limit

        orders = await self._allegro.get_orders(
            status=status,
            buyer_login=tool_input.get("buyer_login"),
            fulfillment_status=fulfillment_status,
            line_items_sent=tool_input.get("line_items_sent"),
            bought_at_gte=self._optional_local_to_utc(tool_input.get("bought_after_local")),
            bought_at_lte=self._optional_local_to_utc(tool_input.get("bought_before_local")),
            paid_at_gte=self._optional_local_to_utc(tool_input.get("paid_after_local")),
            paid_at_lte=self._optional_local_to_utc(tool_input.get("paid_before_local")),
            limit=fetch_limit,
        )
        if exclude_fulfillment:
            orders = [o for o in orders if (o.fulfillment_status or "") not in exclude_fulfillment]
        if dispatch_after or dispatch_before:
            orders = [o for o in orders if self._dispatch_within(o, dispatch_after, dispatch_before)]
        if preset.get("sort_by_dispatch"):
            # Soonest deadline first — the order the parcels have to be dealt
            # with, not the order they were bought in.
            orders.sort(key=lambda o: getattr(o, "dispatch_to", "") or "")
        orders = orders[:limit]

        suffix = "\n\n" + await self._monitoring_status_block() if preset.get("monitoring_block") else ""
        # An explicit fulfillment_status can override the preset's own stage —
        # the stage matchers in deterministic_dispatch do exactly that to reach
        # WYSŁANE/ODEBRANE — and then the preset's wording names the wrong one
        # ("Brak zamówień gotowych do wysłania" for a question about parcels
        # already sent). Fall back to the stage's own name in that case.
        if fulfillment_status and fulfillment_status != preset.get("fulfillment_status"):
            stage_pl = self._fulfillment_pl(fulfillment_status)
            empty_msg = count_none = f"Brak zamówień w statusie: {stage_pl}."
            count_lead = f"W statusie {stage_pl.lower()} masz"
            count_forms = ("zamówienie", "zamówienia", "zamówień")
        else:
            empty_msg, count_none = preset["empty"], preset["count_none"]
            count_lead, count_forms = preset["count_lead"], preset["count_forms"]
        if tool_input.get("count_only"):
            return self._count_sentence(len(orders), count_lead, count_forms, count_none) + suffix
        if not orders:
            return empty_msg + suffix

        carrier_map: dict[str, str] = {}
        if include_delivery:
            # id→name from the carriers endpoint, with the order's own
            # method.name as fallback (see _order_bullet). A failure here only
            # costs the nicer names, never the listing itself.
            try:
                carrier_map = {c["id"]: c.get("name", c["id"]) for c in await self._allegro.get_carriers()}
            except Exception as exc:
                logger.warning("[allegro] carrier lookup failed, using order delivery names: %s", exc)

        blocks = [
            self._order_bullet(o, carrier_map=carrier_map, include_delivery=include_delivery)
            for o in orders
        ]
        body = "\n\n".join(blocks)
        if include_delivery:
            courier_counts: Counter = Counter(
                carrier_map.get(self._dig(o.delivery if isinstance(o.delivery, dict) else {}, "method", "id", default=""))
                or self._dig(o.delivery if isinstance(o.delivery, dict) else {}, "method", "name", default="—")
                for o in orders
            )
            summary = "**Podsumowanie kurierów:**\n" + "\n".join(
                f"- {method}: {count} zamówień" for method, count in courier_counts.most_common()
            )
            body = summary + "\n\n---\n\n" + body
        return body + suffix

    # ── Tool dispatch ─────────────────────────────────────────────────────────

    async def _dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        logger.info("DEBUG _dispatch: tool=%s input=%s", tool_name, tool_input)
        if tool_name in self._ORDERS_PRESETS:
            return await self._orders_listing(tool_name, tool_input)

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
            d = order.delivery if isinstance(order.delivery, dict) else {}
            method_name = self._dig(d, "method", "name", default="N/A")
            tracking = (
                self._dig(d, "smart", "trackingCode", default=None)
                or self._dig(d, "trackingCode", default="N/A")
            )
            billing_lines = []
            if billing_entries:
                total_fees = 0.0
                total_credits = 0.0
                for e in billing_entries:
                    amount = float((e.get("value") or {}).get("amount", 0) or 0)
                    desc = (e.get("type") or {}).get("description", "Inne")
                    offer_name = (e.get("offer") or {}).get("name", "")
                    occurred = e.get("occurredAt", "")[:10]
                    offer_part = f" — {offer_name}" if offer_name else ""
                    sign = "+" if amount > 0 else "-"
                    billing_lines.append(f"  - {occurred} | {desc}{offer_part} | {sign}{abs(amount):.2f} PLN")
                    if amount < 0:
                        total_fees += abs(amount)
                    else:
                        total_credits += amount
                net = order.total_price - total_fees + total_credits
                billing_lines.append(
                    f"  - Suma opłat: -{total_fees:.2f} PLN"
                    + (f" | Zwroty: +{total_credits:.2f} PLN" if total_credits else "")
                )
                billing_lines.append(f"  - Zysk netto: {net:.2f} PLN")

            # Final, ready-to-display plain-text bullet list — built here
            # instead of handed to the interpret LLM as raw data, because
            # _TOOL_SPECIFIC_INSTRUCTIONS for this tool already fully
            # prescribes the shape (exact fields, exact bullet order, "use
            # ONLY the data above, never invent") — there was never any real
            # judgment left for the LLM to apply, just mechanical field-
            # copying it was doing worse (slower, and with a nonzero chance
            # of skipping a billing row) than Python can. See
            # _PASSTHROUGH_TOOLS for how this reaches the user with zero
            # LLM calls.
            product_lines = [
                f"  - {li.offer_name} (ID: {li.offer_id}): {li.quantity} × "
                f"{self._format_price(li.price, li.currency)}"
                for li in order.line_items
            ]
            lines = [
                f"- Zamówienie: `{order.order_id}`",
                f"- Kupujący: {order.buyer_login}",
                f"- Status: {self._fulfillment_pl(order.fulfillment_status)}",
                f"- Wysyłka do: {self._dispatch_deadline_pl(order)}",
                f"- Wartość: {self._format_price(order.total_price, order.currency)}",
                f"- Faktura: {invoice_str}",
                "- Produkty:",
                *product_lines,
                "- Dostawa:",
                f"  - Metoda: {method_name}",
                f"  - Tracking: {tracking}",
            ]
            if billing_lines:
                lines.append("- Rozliczenie:")
                lines.extend(billing_lines)
            return "\n".join(lines)

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
            raw = await self._get_stock_relevant_offers(name_filter)
            logger.info("get_active_offers: %d raw offers fetched", len(raw))
            if not raw:
                return (
                    f"Brak aktywnych ofert pasujących do „{name_filter}”."
                    if name_filter else "Brak aktywnych ofert."
                )
            aggregated = self._aggregate_offers_by_name(raw)
            logger.info("get_active_offers: %d unique products after name aggregation", len(aggregated))
            return self._render_offers_table(aggregated, raw, name_filter)

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
            return self._render_dashboard(
                title="Podsumowanie ofert",
                # non-breaking-space thousands: "1 234 szt.", not the English "1,234"
                lead=(
                    f"Aktywnych ofert: **{total}**, łączny stan magazynowy: "
                    f"**{format(total_stock, ',').replace(',', chr(160))} szt.**"
                ),
                sections=[
                    ("Stany magazynowe", stock_buckets, "Liczba ofert wg stanu"),
                    ("Ceny", price_buckets, "Liczba ofert wg ceny"),
                ],
            )

        if tool_name == "query_offers_by_stock":
            name_filter = tool_input.get("name")
            max_stock = tool_input.get("max_stock")
            min_stock = tool_input.get("min_stock")
            offers = await self._get_stock_relevant_offers(name_filter)
            logger.info(
                "query_offers_by_stock: %d raw offers, name=%r, max_stock=%s min_stock=%s",
                len(offers), name_filter, max_stock, min_stock,
            )
            aggregated = self._aggregate_offers_by_name(offers)
            logger.info("query_offers_by_stock: %d unique products after aggregation", len(aggregated))
            results = []
            for g in aggregated:  # already sorted ascending by stock
                s = g["total_stock"]
                if max_stock is not None and s > max_stock:
                    continue
                if min_stock is not None and s < min_stock:
                    continue
                results.append(g)
            logger.info("query_offers_by_stock: %d products match the stock filter", len(results))
            if not results:
                scope = f" w „{name_filter}”" if name_filter else ""
                return f"Brak produktów{scope} spełniających podane kryteria stanów magazynowych."
            label = []
            if max_stock is not None:
                label.append(f"≤ {max_stock} szt.")
            if min_stock is not None:
                label.append(f"≥ {min_stock} szt.")
            criteria = ", ".join(label) or "wszystkie stany"
            scope = f", filtr nazwy: „{name_filter}”" if name_filter else ""
            rows = self._md_table(
                self._OFFER_COLUMNS, [self._offer_row(g) for g in results], align="lrrl"
            )
            total_stock = sum(g["total_stock"] for g in results)
            summary = (
                f"Znaleziono **{len(results)}** "
                f"{self._plural_pl(len(results), 'produkt', 'produkty', 'produktów')} "
                f"({criteria}{scope}) — łącznie **{total_stock} szt.** na stanie, "
                "posortowane rosnąco wg stanu."
            )
            return "\n".join([f"# Stany magazynowe ({criteria})", "", *rows, "", summary])

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
                label.append(f"od {self._format_price(min_price)}")
            if max_price is not None:
                label.append(f"do {self._format_price(max_price)}")
            criteria = " ".join(label) or "wszystkie ceny"
            rows = self._md_table(
                ["Nazwa", "Cena", "Stan (szt.)", "ID oferty"],
                [[name, self._format_price(price, currency), stock, f"`{oid}`"]
                 for oid, name, price, currency, stock in results],
                align="lrrl",
            )
            summary = (
                f"Znaleziono **{len(results)}** "
                f"{self._plural_pl(len(results), 'ofertę', 'oferty', 'ofert')} "
                f"w cenie {criteria}, posortowane rosnąco wg ceny."
            )
            return "\n".join([f"# Oferty wg ceny ({criteria})", "", *rows, "", summary])

        if tool_name == "get_products_to_reorder":
            assortment = tool_input.get("assortment")
            max_stock = tool_input.get("max_stock", 5)
            offers = await self._get_stock_relevant_offers(assortment)
            logger.info(
                "get_products_to_reorder: %d raw offers (assortment=%r, max_stock=%s)",
                len(offers), assortment, max_stock,
            )
            aggregated = self._aggregate_offers_by_name(offers)
            results = [g for g in aggregated if g["total_stock"] <= max_stock]  # already sorted ascending by stock
            logger.info("get_products_to_reorder: %d products at or below threshold", len(results))
            if not results:
                scope = f" w asortymencie „{assortment}”" if assortment else ""
                return f"Brak produktów{scope} wymagających uzupełnienia (próg: {max_stock} szt.)."
            rows = self._md_table(
                ["Nazwa", "Stan obecny (szt.)", "Cena sprzedaży", "ID ofert"],
                [self._offer_row(g) for g in results],
                align="lrrl",
            )
            sold_out = sum(1 for g in results if g["total_stock"] == 0)
            # Document format: the chat bubble previews the heading plus the line
            # right after it (web/js/app.js _docPreview), so that line has to be a
            # sentence — a table row there reads as garbled pipes.
            lead = (
                f"Do uzupełnienia: **{len(results)}** "
                f"{self._plural_pl(len(results), 'produkt', 'produkty', 'produktów')} "
                f"ze stanem ≤ {max_stock} szt."
                + (f" Asortyment: „{assortment}”." if assortment else "")
                + (f" W tym **{sold_out}** całkowicie wyprzedanych (0 szt.)." if sold_out else "")
            )
            return "\n".join([
                "# Zamówienie uzupełniające",
                "",
                lead,
                "",
                *rows,
                "",
                "Kolejność wierszy: od najniższego stanu magazynowego. Oferty oznaczone "
                "„(zakończona — wyprzedana)” Allegro zakończyło automatycznie po zejściu "
                "stanu do zera — po dostawie towaru trzeba je wznowić.",
            ])

        if tool_name == "get_sales_summary":
            date_from, date_to = self._local_day_bounds_to_utc(
                tool_input["date_from_local"], tool_input["date_to_local"]
            )
            period_label = f"{tool_input['date_from_local']} – {tool_input['date_to_local']}"
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
                return f"Brak opłaconych zamówień w okresie {period_label}."
            total_revenue = sum(o.total_price for o in orders)
            order_count = len(orders)
            avg_value = total_revenue / order_count if order_count else 0
            # Top products by revenue
            product_revenue: dict[str, float] = {}
            for o in orders:
                for li in o.line_items:
                    product_revenue[li.offer_name] = product_revenue.get(li.offer_name, 0) + li.price * li.quantity
            top = sorted(product_revenue.items(), key=lambda x: x[1], reverse=True)[:10]

            # Four blocks, assembled at the end in reading order: what came in
            # (totals, then month by month), what Allegro took out, what sold,
            # and only then the long per-order listing. Building them separately
            # is what lets the costs — the figure the seller actually looks
            # for — sit above the product table while the per-order table it is
            # computed from stays at the bottom of the document.
            month_sections: list[str] = self._monthly_sales_section(
                orders, tool_input["date_from_local"], tool_input["date_to_local"]
            )
            product_sections: list[str] = []
            cost_sections: list[str] = []
            order_sections: list[str] = []
            if top:
                product_sections += [
                    "",
                    "## Top produkty wg przychodu",
                    "",
                    *self._md_table(
                        ["#", "Produkt", "Przychód"],
                        [[i + 1, name, self._format_price(rev)] for i, (name, rev) in enumerate(top)],
                        align="rlr",
                    ),
                    "",
                    "```chart",
                    json.dumps(
                        {
                            "type": "bar",
                            "title": "Przychód wg produktu",
                            "labels": [name for name, _ in top],
                            "series": [{"name": "Przychód (PLN)",
                                        "data": [round(rev, 2) for _, rev in top]}],
                        },
                        ensure_ascii=False,
                    ),
                    "```",
                ]

            if billing_entries:
                # order_id → {"fees": total costs, "refunds": total credits}, both as
                # positive values. Built once from the batch billing fetch above — no
                # per-order API calls.
                by_order: dict[str, dict[str, float]] = defaultdict(lambda: {"fees": 0.0, "refunds": 0.0})
                fee_by_type: dict[str, float] = {}
                refund_by_type: dict[str, float] = {}
                total_fees = 0.0
                total_refunds = 0.0
                for e in billing_entries:
                    if self._is_balance_transfer_entry(e):
                        continue
                    amount = float((e.get("value") or {}).get("amount", 0) or 0)
                    type_desc = (e.get("type") or {}).get("description", "Inne")
                    order_id = (e.get("order") or {}).get("id", "")
                    if amount < 0:
                        total_fees += abs(amount)
                        fee_by_type[type_desc] = fee_by_type.get(type_desc, 0) + abs(amount)
                        if order_id:
                            by_order[order_id]["fees"] += abs(amount)
                    elif amount > 0:
                        total_refunds += amount
                        refund_by_type[type_desc] = refund_by_type.get(type_desc, 0) + amount
                        if order_id:
                            by_order[order_id]["refunds"] += amount
                # Revenue minus what Allegro charged, plus what it credited back. This is
                # NOT net profit — there's no data anywhere in this app about the seller's
                # own cost of goods, packaging, or other expenses, so it can't be computed.
                # Keep the name honest about what it actually is: revenue net of marketplace fees.
                revenue_after_fees = total_revenue - total_fees + total_refunds
                # Fees/refunds not tied to any single order (account subscriptions, listing
                # fees) are still real costs and belong in the total above, but they can't
                # appear in any order's row below — track them so the per-order table's sum
                # doesn't silently disagree with the total.
                unattributed_fees = total_fees - sum(v["fees"] for v in by_order.values())
                unattributed_refunds = total_refunds - sum(v["refunds"] for v in by_order.values())

                cost_sections += ["", f"## Koszty Allegro ({period_label})", ""]
                cost_sections.append(f"- Łączne opłaty: **{self._format_price(total_fees)}**")
                cost_sections += [
                    f"  - {desc}: {self._format_price(amt)}"
                    for desc, amt in sorted(fee_by_type.items(), key=lambda x: x[1], reverse=True)
                ]
                if total_refunds > 0:
                    cost_sections.append(f"- Zwroty/rabaty: **+{self._format_price(total_refunds)}**")
                    cost_sections += [
                        f"  - {desc}: +{self._format_price(amt)}"
                        for desc, amt in sorted(refund_by_type.items(), key=lambda x: x[1], reverse=True)
                    ]
                if abs(unattributed_fees) > 0.01 or abs(unattributed_refunds) > 0.01:
                    cost_sections.append(
                        "- w tym nieprzypisane do żadnego zamówienia (abonament, inne): "
                        f"opłaty **{self._format_price(unattributed_fees)}**"
                        + (f", zwroty/rabaty **+{self._format_price(unattributed_refunds)}**"
                           if unattributed_refunds > 0.01 else "")
                    )
                cost_sections += [
                    "",
                    f"**Przychód po opłatach Allegro (przychód − opłaty + zwroty): "
                    f"{self._format_price(revenue_after_fees)}**",
                    "",
                    "*To NIE jest zysk netto — nie obejmuje kosztu zakupu towaru ani innych "
                    "kosztów własnych, których ta aplikacja nie zna.*",
                ]

                order_rows = []
                for o in sorted(orders, key=lambda x: x.paid_at or x.created_at):
                    oid = o.order_id
                    fees = by_order[oid]["fees"] if oid in by_order else 0.0
                    refunds = by_order[oid]["refunds"] if oid in by_order else 0.0
                    items_short = ", ".join(
                        f"{li.offer_name}×{li.quantity}" for li in o.line_items[:2]
                    ) + ("…" if len(o.line_items) > 2 else "")
                    order_rows.append([
                        (o.paid_at or o.created_at)[:10],
                        o.buyer_login or "—",
                        self._format_price(o.total_price),
                        self._format_price(fees),
                        self._format_price(o.total_price - fees + refunds),
                        items_short or "—",
                    ])
                if order_rows:
                    order_sections += [
                        "",
                        "## Zestawienie per zamówienie",
                        "",
                        "Opłaty przypisane do konkretnego zamówienia — bez opłat kontowych "
                        "(np. abonamentu).",
                        "",
                        *self._md_table(
                            ["Data", "Kupujący", "Przychód", "Opłaty", "Po opłatach", "Produkty"],
                            order_rows,
                            align="llrrrl",
                        ),
                    ]
                    if abs(unattributed_fees) > 0.01 or abs(unattributed_refunds) > 0.01:
                        order_sections += [
                            "",
                            "Suma z tabeli nie zejdzie się z „Przychodem po opłatach Allegro” — "
                            "różnicę stanowią opłaty/zwroty nieprzypisane do żadnego zamówienia "
                            f"(opłaty {self._format_price(unattributed_fees)}"
                            + (f", zwroty +{self._format_price(unattributed_refunds)}"
                               if unattributed_refunds > 0.01 else "")
                            + ").",
                        ]
            elif billing_error:
                cost_sections += [
                    "",
                    "## Koszty Allegro",
                    "",
                    "⚠️ Dane o kosztach Allegro niedostępne (brak uprawnień do rozliczeń). "
                    "Zaloguj się ponownie przez /allegro/login, aby uzyskać dostęp do billing.",
                ]

            return "\n".join([
                f"## Podsumowanie sprzedaży ({period_label})",
                "",
                f"Zamówień: **{order_count}**, łączny przychód: "
                f"**{self._format_price(total_revenue)}**, średnia wartość zamówienia: "
                f"**{self._format_price(avg_value)}**.",
                *month_sections,
                *cost_sections,
                *product_sections,
                *order_sections,
            ])

        if tool_name == "get_offer_details":
            offer = await self._allegro.get_offer(tool_input["offer_id"])
            oid, name, price, currency, stock = self._offer_fields(offer)
            lines = [
                f"**{name}**",
                f"- ID oferty: `{oid}`",
                f"- Cena: **{self._format_price(price, currency)}**",
                f"- Stan magazynowy: **{stock} szt.**",
            ]
            sold = self._dig(offer, "stock", "sold")
            if sold is not None:
                lines.append(f"- Sprzedano: {sold} szt.")
            status = self._dig(offer, "publication", "status")
            if status:
                lines.append(f"- Status publikacji: **{self._publication_pl(status)}**")
            started = self._dig(offer, "publication", "startingAt", default="")
            if started:
                lines.append(f"- Wystawiona: {self._format_dt_pl(started)}")
            ended = self._dig(offer, "publication", "endingAt", default="")
            if ended:
                lines.append(f"- Zakończenie: {self._format_dt_pl(ended)}")
            ean = self._dig(offer, "external", "id")
            if ean:
                lines.append(f"- Sygnatura własna: {ean}")
            lines.append(f"- Link: https://allegro.pl/oferta/{oid}")
            return "\n".join(lines)

        # The three write actions below answer in Polish, finished — the user
        # reads this confirmation as-is (see _PASSTHROUGH_TOOLS), so it must not
        # depend on an LLM to translate an "Error: ..." string or an API dump.
        if tool_name == "update_offer_price":
            offer_id = tool_input["offer_id"]
            price = float(tool_input["price"])
            if price <= 0:
                return f"Nie zmieniłem ceny oferty `{offer_id}` — cena musi być większa od zera."
            await self._allegro.update_offer_price(offer_id, price)
            return f"✅ Cena oferty `{offer_id}` zmieniona na **{self._format_price(price)}**."

        if tool_name == "update_offer_stock":
            offer_id = tool_input["offer_id"]
            available = int(tool_input["available"])
            if available < 0:
                return f"Nie zmieniłem stanu oferty `{offer_id}` — stan nie może być ujemny."
            await self._allegro.update_offer_stock(offer_id, available)
            return f"✅ Stan oferty `{offer_id}` ustawiony na **{available} szt.**"

        if tool_name == "send_message_to_buyer":
            result = await self._allegro.send_message(tool_input["thread_id"], tool_input["text"])
            msg_id = result.get("id", "—")
            return f"✅ Wiadomość wysłana (wątek `{tool_input['thread_id']}`, ID wiadomości: `{msg_id}`)."

        if tool_name == "get_message_threads":
            threads = await self._allegro.get_message_threads(
                limit=min(int(tool_input.get("limit", 10)), 20)
            )
            if not threads:
                return (
                    "Nie masz nowych wiadomości." if tool_input.get("count_only")
                    else "Brak wątków wiadomości."
                )
            unread = [t for t in threads if is_thread_unread(t)]
            if tool_input.get("count_only"):
                if not unread:
                    return "Nie masz nowych wiadomości."
                buyers = ", ".join(
                    (t.get("interlocutor") or {}).get("login", "—") for t in unread[:5]
                )
                return self._count_sentence(
                    len(unread), "Masz",
                    ("nową wiadomość", "nowe wiadomości", "nowych wiadomości"),
                    none="Nie masz nowych wiadomości.", scope=f" (od: {buyers})",
                ) + " Pokazać szczegóły?"
            # The thread ID stays in the line: send_message_to_buyer needs it, and
            # it is read back from this very text on the next turn.
            return "\n".join(
                f"- **{(t.get('interlocutor') or {}).get('login', '—')}** — "
                f"{'🔴 nieprzeczytana' if is_thread_unread(t) else 'przeczytana'} — "
                f"ostatnia wiadomość: {self._format_dt_pl(thread_last_message_at(t))} — "
                f"wątek `{t.get('id')}`"
                for t in threads
            )

        if tool_name == "get_thread_messages":
            thread_id = tool_input.get("thread_id")
            matched_buyer = None
            if not thread_id:
                threads = await self._allegro.get_message_threads(limit=20)
                candidates = threads
                buyer_login = tool_input.get("buyer_login")
                if buyer_login:
                    candidates = [
                        t for t in candidates
                        if (t.get("interlocutor") or {}).get("login", "").lower() == buyer_login.lower()
                    ]
                date_filter = tool_input.get("date")
                if date_filter:
                    target_date = (
                        self._today_warsaw()
                        if date_filter.strip().lower() in ("dzisiaj", "today")
                        else date_filter[:10]
                    )
                    candidates = [
                        t for t in candidates
                        if self._to_warsaw_date(thread_last_message_at(t)) == target_date
                    ]
                if not candidates:
                    return "Nie znalazłem wątku pasującego do podanego kupującego / daty."
                thread_id = candidates[0].get("id")
                matched_buyer = (candidates[0].get("interlocutor") or {}).get("login", "N/A")
            messages = await self._allegro.get_thread_messages(
                thread_id, limit=min(int(tool_input.get("limit", 10)), 20)
            )
            if not messages:
                return f"Wątek `{thread_id}` nie zawiera wiadomości."
            header = f"Wątek `{thread_id}`" + (f" — kupujący **{matched_buyer}**" if matched_buyer else "")
            lines = [header, ""]
            for m in messages:
                author = m.get("author") or {}
                who = "Kupujący" if author.get("isInterlocutor") else "Sprzedawca"
                lines.append(
                    f"**{who}** ({author.get('login', '—')}) — "
                    f"{self._format_dt_pl(m.get('createdAt', ''))}"
                )
                # Verbatim, never summarized — the seller is reading what the
                # buyer actually wrote.
                lines.append(f"„{m.get('text', '')}”")
                lines.append("")
            return "\n".join(lines).rstrip()

        if tool_name == "get_account_info":
            info = await self._allegro.get_user_info()
            company = (info.get("company") or {}).get("name")
            return "\n".join([
                f"- Login: **{info.get('login', '—')}**",
                f"- E-mail: {info.get('email', '—')}",
                f"- Konto: {company or 'sprzedawca indywidualny'}",
                f"- Zarejestrowane: {self._format_dt_pl(info.get('registeredAt', ''))}",
            ])

        if tool_name == "get_billing_summary":
            date_from_local = tool_input.get("date_from_local")
            date_to_local = tool_input.get("date_to_local")
            date_from, date_to = (
                self._local_day_bounds_to_utc(date_from_local, date_to_local)
                if date_from_local and date_to_local
                else (None, None)
            )
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
                    period_label = f"{date_from_local} – {date_to_local}"
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
            rows = []
            skipped_transfers = 0
            for e in entries:
                amount_val = float((e.get("value") or {}).get("amount", 0) or 0)
                currency = (e.get("value") or {}).get("currency", "PLN")
                type_desc = (e.get("type") or {}).get("description", "Inne")
                occurred = e.get("occurredAt", "")[:10]
                order_id = (e.get("order") or {}).get("id", "")
                is_transfer = self._is_balance_transfer_entry(e)
                sign = "+" if amount_val > 0 else ""
                rows.append([
                    occurred,
                    type_desc + (" *(transfer wewnętrzny, poza sumami)*" if is_transfer else ""),
                    f"`{order_id}`" if order_id else "—",
                    f"{sign}{amount_val:.2f} {currency}".replace(".", ","),
                ])
                if is_transfer:
                    skipped_transfers += 1
                    continue
                if amount_val < 0:
                    total_fees += abs(amount_val)
                    fee_by_type[type_desc] = fee_by_type.get(type_desc, 0) + abs(amount_val)
                elif amount_val > 0:
                    total_refunds += amount_val
            net_cost = total_fees - total_refunds
            # Same 50-row cap the old bullet listing had — a busy month is
            # hundreds of entries and the totals above are the answer.
            shown, hidden = rows[:50], max(len(rows) - 50, 0)
            breakdown = [
                f"- {desc}: **{self._format_price(amt)}**"
                for desc, amt in sorted(fee_by_type.items(), key=lambda x: x[1], reverse=True)
            ]
            summary = (
                f"Łączne opłaty: **{self._format_price(total_fees)}**"
                + (f", zwroty/rabaty: **+{self._format_price(total_refunds)}**" if total_refunds > 0 else "")
                + f", koszt netto: **{self._format_price(net_cost)}** "
                f"({len(entries)} {self._plural_pl(len(entries), 'operacja', 'operacje', 'operacji')}, "
                f"{period_label})."
                + (f" W tabeli pokazano {len(shown)} najnowszych." if hidden else "")
                + (f" Transferów wewnętrznych pominiętych w sumach: {skipped_transfers}." if skipped_transfers else "")
            )
            return "\n".join([
                f"# Koszty Allegro ({period_label})",
                "",
                *(["**Podział wg rodzaju opłaty:**", "", *breakdown, ""] if breakdown else []),
                *self._md_table(["Data", "Rodzaj", "Zamówienie", "Kwota"], shown, align="lllr"),
                "",
                summary,
            ])

        if tool_name == "get_buyers":
            return await self._buyers_report(tool_input)

        if tool_name == "get_orders_pending_invoice":
            orders = await self._allegro.get_orders_needing_invoice(
                month=tool_input.get("month"),
                year=tool_input.get("year"),
            )
            if not orders:
                return "Brak zamówień wymagających wystawienia faktury." + "\n\n" + await self._invoice_reminder_status_block()
            # Fetch invoice address data for all orders in parallel
            inv_results = await asyncio.gather(
                *[self._allegro.get_order_invoice_data(o.order_id) for o in orders],
                return_exceptions=True,
            )
            header = f"**Zamówień bez faktury: {len(orders)}**\n"
            blocks = []
            for o, inv in zip(orders, inv_results):
                items_str = ", ".join(f"{li.offer_name} ×{li.quantity}" for li in o.line_items[:3])
                # No status/date lines here any more — _order_bullet already
                # prints both (and in Warsaw time), so repeating them made the
                # same order carry two "when" fields in two formats.
                extra = [
                    f"E-mail: {o.buyer_email}",
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
                blocks.append(self._order_bullet(o, extra_lines=extra))
            return header + "\n\n".join(blocks) + "\n\n" + await self._invoice_reminder_status_block()

        if tool_name == "preview_pending_invoices":
            return await self._preview_pending_invoices(
                month=tool_input.get("month"),
                year=tool_input.get("year"),
            )

        if tool_name == "issue_invoice_for_order":
            return await self._issue_invoice_for_order(tool_input["order_id"])

        if tool_name == "unblock_invoice_for_order":
            return await self._unblock_invoice_for_order(
                tool_input["order_id"], bool(tool_input.get("seller_confirmed")),
            )

        if tool_name == "attach_invoice_to_allegro_order":
            return await self._attach_invoice_to_allegro_order(
                tool_input["order_id"], tool_input["invoice_uuid"]
            )

        if tool_name == "send_invoice_to_ksef":
            return await self._send_invoice_to_ksef(tool_input["invoice_uuid"])

        # Both the "suggest" and "disable" tool for each monitor type resolve to the
        # same deterministic status block — the model only picks WHICH tool to call
        # based on user intent, but the actual reply/button always reflects the real
        # stored flag, never the model's guess of the current state (see
        # _monitoring_status_block's docstring for why: a model guess here caused a
        # reply/button contradiction identical to the earlier invoice-issuance bug).
        if tool_name in ("suggest_order_monitoring", "disable_order_monitoring"):
            return await self._monitoring_status_block()

        if tool_name in ("suggest_invoice_reminder", "disable_invoice_reminder"):
            return await self._invoice_reminder_status_block()

        if tool_name in ("suggest_message_monitoring", "disable_message_monitoring"):
            return await self._message_monitoring_status_block()

        if tool_name == "get_new_returns":
            date_from, date_to, period_label = self._optional_period(tool_input)
            returns = await self._allegro.get_customer_returns(
                limit=50, date_from=date_from, date_to=date_to
            )
            suffix = f" ({period_label})" if period_label else ""
            if tool_input.get("count_only"):
                body = self._count_sentence(
                    len(returns), "Masz", ("zwrot", "zwroty", "zwrotów"),
                    none=f"Brak zwrotów{suffix}.", scope=suffix,
                )
            else:
                body = (
                    f"Brak zwrotów{suffix}." if not returns
                    else self._returns_listing(returns, f"Zwroty{suffix}")
                )
            return body + "\n\n" + await self._returns_monitoring_status_block()

        if tool_name == "get_returns_to_process":
            date_from, date_to, period_label = self._optional_period(tool_input)
            returns = await self._allegro.get_customer_returns(
                limit=50, status="DELIVERED", date_from=date_from, date_to=date_to
            )
            suffix = f" ({period_label})" if period_label else ""
            if tool_input.get("count_only"):
                body = self._count_sentence(
                    len(returns), "Do obsłużenia masz", ("zwrot", "zwroty", "zwrotów"),
                    none=(
                        f"Brak zwrotów do obsłużenia{suffix} — żaden zwrócony towar nie dotarł "
                        "jeszcze z powrotem."
                    ),
                    scope=suffix,
                )
            else:
                body = (
                    f"Brak zwrotów do obsłużenia{suffix} — żaden zwrócony towar nie dotarł jeszcze "
                    "z powrotem."
                    if not returns else
                    self._returns_listing(returns, f"Zwroty do obsłużenia{suffix}")
                )
            return body + "\n\n" + await self._returns_monitoring_status_block()

        if tool_name == "get_new_complaints":
            date_from, date_to, period_label = self._optional_period(tool_input)
            issues = await self._allegro.get_issues(limit=50, date_from=date_from, date_to=date_to)
            suffix = f" ({period_label})" if period_label else ""
            if tool_input.get("count_only"):
                body = self._count_sentence(
                    len(issues), "Masz", ("reklamację", "reklamacje", "reklamacji"),
                    none=f"Brak reklamacji{suffix}.", scope=suffix,
                )
            else:
                body = (
                    f"Brak reklamacji{suffix}." if not issues
                    else self._complaints_listing(issues, f"Reklamacje{suffix}")
                )
            return body + "\n\n" + await self._returns_monitoring_status_block()

        if tool_name in ("suggest_returns_monitoring", "disable_returns_monitoring"):
            return await self._returns_monitoring_status_block()

        return f"Unknown tool: {tool_name}"
