"""Tool definitions for the Allegro agent (OpenAI/Gemini function-calling format)."""

import re

# ── Zamówienia: trzy intencje, jedna implementacja ──────────────────────────
# get_new_orders / get_orders / get_orders_delivery stay three separate tool
# schemas on purpose: intent routing is far more reliable against named
# presets ("nowe zamówienia", "zamówienia do wysłania") than against one
# generic tool hiding a dozen optional filters. What they must NOT be is three
# different features — each is only a set of default arguments over the SAME
# listing call (AllegroAgent._orders_listing), they share the parameter
# definitions below, they render with the same order bullet, and all three are
# "chat" in TOOL_OUTPUT_FORMAT. Before this, "nowe zamówienia" came back as
# chat text while "zamówienia do wysłania" came back as a markdown table the
# frontend then rendered as a document artifact — the same question answered
# in two different shapes depending on which preset the model happened to pick.
# Every fulfillment stage an Allegro order can be at, in the order it moves
# through them. One list, because three different tools now let the seller
# scope a question to a stage (get_orders, get_orders_delivery and — for
# "faktury do wysłania w zamówieniach nie nowych" — get_orders_pending_invoice)
# and a stage missing from one of their enums is a stage the model cannot name
# there, whatever the seller asked.
_FULFILLMENT_STATUSES: tuple[str, ...] = (
    "NEW", "PROCESSING", "READY_FOR_SHIPMENT", "SENT", "IN_TRANSIT",
    "READY_FOR_PICKUP", "PICKED_UP", "CANCELLED", "SUSPENDED",
)

_ORDER_PARAMS: dict[str, dict] = {
    "status": {
        "type": "string",
        "description": (
            "Checkout-form status. LEAVE IT OUT for every normal question: the listing then "
            "covers exactly the orders that exist for the seller, cash-on-delivery included. "
            "Pass CANCELLED only for a question that explicitly asks about cancelled orders "
            "('pokaż anulowane zamówienia'). Baskets a buyer started but never paid for are "
            "not orders and are never listed, whatever else the question asks for."
        ),
        "enum": ["READY_FOR_PROCESSING", "CANCELLED"],
    },
    "fulfillment_status": {
        "type": "string",
        "description": (
            "Filter by fulfillment status — ONE positive stage: NEW = not packed yet "
            "('do spakowania'), READY_FOR_SHIPMENT = packed, awaiting carrier handoff "
            "('do wysłania', 'gotowe do wysyłki', 'zapakowane'), SENT = already handed over. "
            "NEVER use it for a NEGATED question ('niewysłane', 'jeszcze nie wysłane', 'nie "
            "odebrane'): 'not sent' covers every stage before the handoff, not just the packed "
            "one, so it is exclude_fulfillment_status that answers it."
        ),
        "enum": ["NEW", "PROCESSING", "READY_FOR_SHIPMENT", "SENT", "PICKED_UP", "CANCELLED", "SUSPENDED"],
    },
    "exclude_fulfillment_status": {
        "type": "array",
        "items": {"type": "string", "enum": list(_FULFILLMENT_STATUSES)},
        "description": (
            "NEGATED stage filter — return every order whose fulfillment status is NOT one of "
            "these. A negated question is never one status, it is everything except one: "
            "'niewysłane' / 'jeszcze nie wysłane' / 'które nie zostały wysłane' means every "
            "order that has not left yet (NEW, PROCESSING, READY_FOR_SHIPMENT alike), NOT only "
            "the packed ones — so pass exclude_fulfillment_status=['SENT', 'IN_TRANSIT', "
            "'READY_FOR_PICKUP', 'PICKED_UP'] rather than fulfillment_status=READY_FOR_SHIPMENT, "
            "which silently drops everything nobody has packed yet. Same for any other negation: "
            "'nieodebrane' → exclude ['PICKED_UP'], 'niespakowane' → exclude "
            "['READY_FOR_SHIPMENT', 'SENT', 'IN_TRANSIT', 'READY_FOR_PICKUP', 'PICKED_UP']. "
            "Combine with fulfillment_status only when the question really names both sides."
        ),
    },
    "min_value": {
        "type": "number",
        "description": (
            "Only orders worth AT LEAST this much (the order total the buyer paid, delivery "
            "included, in the order's own currency). Pass it whenever the question names a "
            "floor: 'zamówienia powyżej 400 zł', 'ponad 1000 zł', 'od 250 zł w górę', "
            "'droższe niż 99,99'. Without it the amount the user just said is silently "
            "ignored and the listing comes back unfiltered, which reads like an answer."
        ),
    },
    "max_value": {
        "type": "number",
        "description": (
            "Only orders worth AT MOST this much (same figure as min_value). For a ceiling: "
            "'zamówienia poniżej 50 zł', 'do 100 zł', 'tańsze niż 20 zł'. Pass both bounds "
            "for a range ('od 100 do 300 zł')."
        ),
    },
    "buyer_login": {
        "type": "string",
        "description": (
            "Filter by ONE buyer's Allegro login. This is the only filter in the whole tool "
            "list that can answer a question about a NAMED buyer account — 'czy w tym roku "
            "kupował ode mnie ktoś z konta np1988', 'co kupił użytkownik anna.kowalska88', "
            "'ile zamówień złożył kasia.w' — so pass it whenever the question names one, "
            "together with the period filters if a period is named. get_buyers CANNOT do this: "
            "it has no login parameter and would answer with every customer of the period "
            "instead. Always the Allegro LOGIN, exactly as the user wrote it — never a company "
            "or person's name (Allegro matches it exactly); if the user gave a NAME instead, "
            "ask for the login rather than guessing it."
        ),
    },
    "min_value": {
        "type": "number",
        "description": (
            "Return only orders whose VALUE (the total the buyer paid, delivery included) is "
            "AT LEAST this many PLN. This is the ONLY way to answer a question that names an "
            "order amount — 'zamówienie na kwotę ponad 2000 zł', 'zamówienia powyżej 500 zł', "
            "'najdroższe zamówienie z tego tygodnia', 'czy było coś za więcej niż 1000 zł'. "
            "Without it the amount is silently dropped and the reply is the whole unfiltered "
            "list, which reads like an answer to a question nobody asked."
        ),
    },
    "max_value": {
        "type": "number",
        "description": (
            "Return only orders whose VALUE (the total the buyer paid, delivery included) is "
            "AT MOST this many PLN — 'zamówienia poniżej 100 zł', 'drobne zamówienia do 50 zł'. "
            "Combine with min_value for a range ('między 500 a 1000 zł')."
        ),
    },
    "product_names": {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            "Only orders CONTAINING one of these products — the ONLY way to answer a question "
            "that names what was inside the order: 'zamówienie z wczoraj z włóczką yarnart "
            "jeans', 'pokaż zamówienia z jeans plus', 'kto kupił kordonek'. One entry per model "
            "the user named, written as they wrote it but WITHOUT the generic category word: "
            "'włóczkę yarnart jeans' → ['yarnart jeans'], 'jeans i jeans plus' → ['jeans', "
            "'jeans plus'] (two models, never one merged entry). A name matches an offer title "
            "on whole words and the most specific name wins, so 'jeans' never swallows 'jeans "
            "plus'. Without it the product is silently dropped and the whole period's listing "
            "comes back, which reads like an answer to a question nobody asked."
        ),
    },
    "product_match": {
        "type": "string",
        "enum": ["any", "only"],
        "description": (
            "How product_names has to match the order's contents. 'any' (default) — the order "
            "contains at least one of the named products, next to anything else. 'only' — the "
            "order contains NOTHING BUT the named products: this is what 'tylko' / 'wyłącznie' / "
            "'same' / 'jedynie' mean ('zamówienie, które miało tylko włóczkę yarnart jeans'), "
            "and answering such a question with 'any' returns every mixed order too, which is a "
            "different question. Ignored when product_names is empty."
        ),
        "default": "any",
    },
    "line_items_sent": {
        "type": "array",
        "items": {"type": "string", "enum": ["NONE", "SOME", "ALL"]},
        "description": "Filter by shipment state. Multiple values allowed (OR logic).",
    },
    "bought_after_local": {
        "type": "string",
        "description": (
            "Order CREATION time filter — return only orders placed AT OR AFTER this local "
            "Polish time. Format: 'HH:MM' for today (e.g. '12:00'), or 'YYYY-MM-DD HH:MM' "
            "for a specific date (e.g. '2026-06-16 18:00'). Conversion to UTC is automatic."
        ),
    },
    "bought_before_local": {
        "type": "string",
        "description": (
            "Order CREATION time filter — return only orders placed AT OR BEFORE this local "
            "Polish time. Same format as bought_after_local: 'HH:MM' or 'YYYY-MM-DD HH:MM'."
        ),
    },
    "paid_after_local": {
        "type": "string",
        "description": (
            "PAYMENT time filter — return only orders paid AT OR AFTER this local Polish time. "
            "Use for queries with 'opłacone', 'zapłacone'. Format: 'HH:MM' or 'YYYY-MM-DD HH:MM'."
        ),
    },
    "paid_before_local": {
        "type": "string",
        "description": (
            "PAYMENT time filter — return only orders paid AT OR BEFORE this local Polish time. "
            "Same format as paid_after_local."
        ),
    },
    "dispatch_after_local": {
        "type": "string",
        "description": (
            "DISPATCH DEADLINE filter ('Wysyłka do' — when the parcel must be handed to the "
            "carrier): only orders whose deadline falls AT OR AFTER this local Polish time. "
            "Format: 'HH:MM' or 'YYYY-MM-DD HH:MM'."
        ),
    },
    "dispatch_before_local": {
        "type": "string",
        "description": (
            "DISPATCH DEADLINE filter ('Wysyłka do'): only orders that must be handed to the "
            "carrier AT OR BEFORE this local Polish time — use for 'co muszę wysłać dzisiaj', "
            "'które zamówienia mają termin do jutra', 'zamówienia po terminie' (pass the "
            "current time). Format: 'HH:MM' for today or 'YYYY-MM-DD HH:MM'. Orders for which "
            "Allegro returned no deadline are excluded when this filter is used."
        ),
    },
    "include_delivery": {
        "type": "boolean",
        "description": (
            "Add courier details to every order (carrier name, tracking number and link, pickup "
            "point) plus a per-courier count summary. Set true for any question combining orders "
            "with couriers/delivery providers/tracking."
        ),
        "default": False,
    },
    "count_only": {
        "type": "boolean",
        "description": (
            "Set true when the user only wants the NUMBER of orders ('ile zamówień', 'ile mam "
            "nowych', 'liczba zamówień') — the reply states just the count, without listing the "
            "orders. Do NOT set it when the user also wants to see the orders themselves "
            "('pokaż', 'jakie zamówienia mam' with no 'ile')."
        ),
        "default": False,
    },
    "limit": {
        "type": "integer",
        "description": "Max orders to return (1–100).",
        "default": 50,
    },
}


def _order_params(*names: str, **overrides: dict) -> dict:
    """Parameter schema for one order-listing tool: the shared definitions in
    `_ORDER_PARAMS` narrowed to `names`, with per-tool tweaks (a different
    default `limit`, a preset-specific description) merged on top — so the
    three order tools can never drift into meaning different things by the
    same argument name."""
    props: dict[str, dict] = {}
    for name in names:
        prop = dict(_ORDER_PARAMS[name])
        prop.update(overrides.get(name, {}))
        props[name] = prop
    return {"type": "object", "properties": props}


ALLEGRO_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_new_orders",
            "description": (
                "List new (unprocessed) orders waiting to be fulfilled — the get_orders listing "
                "with status=READY_FOR_PROCESSING + fulfillment_status=NEW already applied. "
                "Use for any question about 'nowe zamówienia', 'nowe', 'oczekujące', "
                "'co nowego', 'jakie zamówienia mam', 'ile zamówień', new/pending orders. "
                "Results are sorted newest-first. "
                "SINGULAR vs PLURAL: for 'ostatnie zamówienie' / 'ostatnie nowe zamówienie' / "
                "'najnowsze zamówienie' / 'last order' (singular — asking about ONE order), "
                "set limit=1. Only use the default (all orders) for plural phrasing like "
                "'nowe zamówienia' / 'ostatnie zamówienia' / 'jakie zamówienia mam'. "
                "COUNT-ONLY: for 'ile zamówień', 'ile mam nowych', 'ile jest wszystkich nowych', "
                "'liczba nowych zamówień' (the user wants a NUMBER, not the order details) — set "
                "count_only=true. "
                "For a different status, a date range, or courier details use get_orders — same "
                "listing, same reply shape, just other filters. "
                "Returns order IDs, buyer info, current status, dispatch deadline "
                "('Wysyłka do' — when the parcel must be handed to the carrier), and totals."
            ),
            "parameters": _order_params(
                "buyer_login", "dispatch_before_local", "min_value", "max_value",
                "product_names", "product_match", "count_only", "limit",
                limit={
                    "description": (
                        "Max orders to return (1–100). Set to 1 when the user asks about "
                        "THE LAST/newest order in the singular ('ostatnie zamówienie', "
                        "'ostatnie nowe zamówienie'). Leave at default for plural questions."
                    ),
                    "default": 100,
                },
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_orders",
            "description": (
                "List Allegro orders with arbitrary filters — the general form of the order "
                "listing (get_new_orders and get_orders_delivery are the same call with preset "
                "filters, and all three reply in the same plain-text format). "
                "USE THIS for a plain LIST of orders, incl. for a date range — 'lista zamówień', "
                "'pokaż wszystkie zamówienia z tego miesiąca/tygodnia', 'zamówienia z okresu X'. "
                "Do NOT use get_sales_summary for these — that tool is only for earnings/profit/fee "
                "questions ('ile zarobiłem', 'jakie opłaty'), not for listing orders. "
                "For new/pending orders use get_new_orders instead. "
                "This is also the FALLBACK for any order question no more specific tool covers — "
                "reach for it when the user names no fulfillment stage at all. "
                "ONE NAMED BUYER — 'czy w tym roku kupował ode mnie ktoś z konta np1988', "
                "'co kupował użytkownik X', 'ile zamówień złożyło konto Y': pass buyer_login "
                "(plus the period as bought_after_local/bought_before_local, and count_only=true "
                "for a 'czy'/'ile' question). NOT get_buyers for that — it has no login filter, "
                "so it answers with the whole customer list of the period. "
                "STAGE FILTERS only this tool can serve: 'w trakcie (realizacji)' / 'przetwarzane' / "
                "'w toku' / 'kompletowane' / 'co mam w robocie' / 'nieskończone' / 'do dokończenia' "
                "→ fulfillment_status=PROCESSING; 'odebrane' / 'dostarczone' / 'zrealizowane' / "
                "'zakończone' / 'co już dotarło' / 'co klient odebrał' → fulfillment_status=PICKED_UP. "
                "For the NOWE stage use get_new_orders and for DO WYSŁANIA / WYSŁANE use "
                "get_orders_delivery (it adds the courier and tracking details those questions want). "
                "NEGATED STAGE — only this tool can serve it, via exclude_fulfillment_status: "
                "'niewysłane' / 'jeszcze nie wysłane' / 'które nie zostały wysłane' → exclude "
                "['SENT', 'IN_TRANSIT', 'READY_FOR_PICKUP', 'PICKED_UP'] (every order still on your "
                "side, packed or not); 'nieodebrane' → exclude ['PICKED_UP']; 'niespakowane' → "
                "exclude ['READY_FOR_SHIPMENT', 'SENT', 'IN_TRANSIT', 'READY_FOR_PICKUP', "
                "'PICKED_UP']. A negation is never one positive status: answering 'niewysłane' with "
                "fulfillment_status=READY_FOR_SHIPMENT hides every order nobody has packed yet. "
                "CANCELLED ORDERS are never listed by any of these tools — there is nothing to "
                "pack, send or invoice — so they need no filtering on your side; ask for them only "
                "when the user explicitly wants them ('pokaż anulowane zamówienia' → "
                "fulfillment_status=CANCELLED), which is the one case they are shown. "
                "UNPAID BASKETS — a buyer who clicked buy but never paid — are not orders "
                "either and never reach a listing or a count, again with nothing to filter on "
                "your side; a cash-on-delivery order, paid on receipt and therefore carrying no "
                "payment date, IS an ordinary order and is always listed. "
                "PRODUCT FILTERS: product_names is the ONLY way to answer a question naming what "
                "was INSIDE the order — 'pokaż zamówienie z wczoraj, które miało włóczkę yarnart "
                "jeans', 'zamówienia z kordonkiem z tego tygodnia', 'kto kupił jeans plus'. Pass "
                "the model name without the category word ('włóczkę yarnart jeans' → "
                "product_names=['yarnart jeans']) together with the period the question names, and "
                "add product_match='only' when the question says the order held NOTHING ELSE "
                "('tylko', 'wyłącznie', 'same', 'jedynie'). Never drop the product and return the "
                "whole period's listing — it is handed to the seller as the answer. This is also "
                "NOT get_sold_quantities: that one counts PIECES over a period and never shows "
                "which orders they came from. "
                "VALUE FILTERS: min_value/max_value are the ONLY way to answer a question that names "
                "an amount — 'zamówienia powyżej 400 zł' → min_value=400, 'poniżej 50 zł' → "
                "max_value=50, 'od 100 do 300 zł' → both. Never answer such a question without them: "
                "the listing would come back unfiltered and read as if it were the answer. "
                "TIME FILTERS: bought_after/before_local = order PLACEMENT time; "
                "paid_after/before_local = PAYMENT time ('opłacone po X', 'zapłacone po X'); "
                "dispatch_after/before_local = DISPATCH DEADLINE ('do kiedy trzeba wysłać'). "
                "NEVER use this tool when the user names or already gave a SPECIFIC order_id "
                "(a UUID, e.g. '0c4854a0-9646-11f1-8028-338c43adc37a') — this tool has NO order_id "
                "parameter and cannot filter to a single order; it will silently ignore the ID and "
                "return an unrelated list of unrelated orders. Use get_order_details instead for any "
                "question (status, contents, invoice, cost) about one already-identified order. "
                "Every order returned carries its current status and its dispatch deadline "
                "('Wysyłka do' — when the parcel must be handed to the carrier). "
                "ORDER VALUE: min_value/max_value filter by the amount the buyer paid, and are "
                "the ONLY way to answer a question that names one — 'zamówienie na kwotę ponad "
                "2000 zł', 'zamówienia powyżej 500 zł z tego tygodnia', 'najdroższe zamówienie z "
                "ostatnich dni', 'coś poniżej 100 zł'. Pass them together with the period filters "
                "the question names, and add include_delivery=true when the question is about "
                "that order's DELIVERY (courier, tracking, koszt dostawy) — then this one call "
                "answers it. Never drop the amount and return an unfiltered list."
            ),
            "parameters": _order_params(
                "status", "fulfillment_status", "exclude_fulfillment_status",
                "buyer_login", "line_items_sent", "product_names", "product_match",
                "bought_after_local", "bought_before_local",
                "paid_after_local", "paid_before_local",
                "dispatch_after_local", "dispatch_before_local",
                "min_value", "max_value",
                "include_delivery", "count_only", "limit",
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": (
                "Get full details of a specific Allegro order: status, dispatch deadline "
                "('Wysyłka do' — when the parcel must be handed to the carrier), items, buyer address, "
                "delivery info, payment status, AND all Allegro billing entries for that order "
                "(individual commission per item, delivery fees, any credits). "
                "DELIVERY COSTS for one order come from THIS tool and nowhere else: it reports both "
                "sides — what the buyer paid for delivery (already included in the order value) and "
                "what Allegro charged you for the shipment, plus the balance between them. So "
                "'ile kosztowała dostawa', 'jaki był koszt dostawy tego zamówienia', 'ile zapłacił "
                "kupujący za wysyłkę', 'czy dostawa była darmowa', 'ile mnie kosztowała wysyłka tej "
                "paczki' → get_order_details with that order_id, NEVER get_orders_delivery (that one "
                "lists many orders and cannot filter by order_id). "
                "USE THIS for ANY question about ONE already-identified order — not just costs: "
                "'jaki jest status tego zamówienia', 'co się dzieje z zamówieniem X', 'sprawdź "
                "zamówienie <id>', 'jakie koszty miałem przy tym zamówieniu', 'podaj wpisy billing "
                "dla zamówienia X', 'ile prowizji zapłaciłem za to zamówienie'. The order_id may be "
                "given directly in the message (a UUID) or already established earlier in the "
                "conversation (e.g. the assistant just listed/described this exact order) — reuse "
                "that ID, do not ask the user to repeat it if it's already in context. "
                "Uses order.id filter so results are exact — never mixes entries from other orders. "
                "get_orders CANNOT do this — it has no order_id filter."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Allegro order (checkout form) UUID."},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_order_profit",
            "description": (
                "Profit of ONE order for a cost of goods THE USER states in the conversation: "
                "order value − Allegro's fees for that order + its credits − (purchase cost per "
                "unit × quantity) for every item. "
                "USE THIS whenever a question about one order names a purchase cost: 'dla tego "
                "zamówienia policz zysk zakładając koszt 1 szt. na poziomie 8,10 zł', 'ile na tym "
                "zarobiłem, jeśli kupiłem po 8 zł za sztukę', 'jaka marża przy koszcie 12 zł/szt'. "
                "get_order_details CANNOT answer it — it knows nothing about purchase costs, and "
                "its 'Zysk netto' line is only the order value minus Allegro fees, so answering "
                "with it silently ignores the cost the user just gave. get_sales_summary is for "
                "whole PERIODS and has neither an order_id nor a cost parameter. "
                "The order_id may come from the user's message or from earlier in this "
                "conversation ('dla TEGO zamówienia' right after one was listed or described) — "
                "reuse it, never ask for an ID that is already in context. "
                "NEVER invent the cost: if no purchase cost was given anywhere in the "
                "conversation, call ask_clarifying_question and ask for it instead of guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Allegro order (checkout form) UUID."},
                    "unit_cost": {
                        "type": "number",
                        "description": (
                            "Purchase cost of ONE unit, in PLN, exactly as the user stated it "
                            "('koszt 1 szt. na poziomie 8,10 zł' → 8.1). Applied to every item of "
                            "the order that item_costs does not cover — pass it alone whenever a "
                            "single cost applies to the whole order."
                        ),
                    },
                    "item_costs": {
                        "type": "array",
                        "description": (
                            "Per-product unit costs — only when the user gave DIFFERENT costs for "
                            "different products of the same order. Omit entirely otherwise."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "offer_id": {
                                    "type": "string",
                                    "description": "Allegro offer ID this cost applies to, if known.",
                                },
                                "offer_name": {
                                    "type": "string",
                                    "description": (
                                        "Product name (or a distinctive part of it) this cost "
                                        "applies to — used when no offer_id is known."
                                    ),
                                },
                                "unit_cost": {
                                    "type": "number",
                                    "description": "Purchase cost of ONE unit of this product, in PLN.",
                                },
                            },
                            "required": ["unit_cost"],
                        },
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_offers",
            "description": (
                "List ALL active Allegro offers (paginated, no limit), plus any ended offer that "
                "sold out to zero stock (Allegro auto-ends offers once stock hits zero, so those "
                "are still relevant — an ended offer that still has stock left was stopped "
                "deliberately and is excluded). Offers with the same name are aggregated — stock "
                "summed, sorted ascending by stock (lowest/most urgent first). "
                "Use for general 'show me my offers' questions. "
                "For stock or price filtering use query_offers_by_stock / query_offers_by_price instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Optional: filter by offer name (partial match, e.g. 'Himalaya "
                            "Dolphin', 'YarnArt Jeans'). Set this whenever the user names a "
                            "specific product, brand or collection — never leave it empty and "
                            "return the whole catalogue when only one was asked about."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_offers_summary",
            "description": (
                "Return statistics for all active offers: total count, total stock, "
                "stock distribution (out-of-stock / low / medium / high), "
                "and price distribution buckets. Use for overview/summary questions."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_offers_by_stock",
            "description": (
                "Filter offers by stock quantity — active offers plus any ended offer that sold "
                "out to zero stock (Allegro auto-ends offers once stock hits zero; an ended offer "
                "that still has stock left was stopped deliberately and is excluded). "
                "Offers with the same name are aggregated — stock is summed across all listings of the same product, "
                "and results are sorted ascending by stock (lowest first). "
                "Use for questions like 'offers with less than 10 items', 'out of stock offers', 'high stock offers'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Optional: filter to one product/brand/collection by partial name "
                            "match (e.g. 'Himalaya Dolphin', 'YarnArt Jeans'). Set this whenever "
                            "the user names a specific product, brand or collection — never leave "
                            "it empty and return every product's stock when only one was asked about."
                        ),
                    },
                    "max_stock": {
                        "type": "integer",
                        "description": (
                            "Return products with total stock ≤ this value (inclusive). "
                            "Convert the user's wording to the correct boundary number: "
                            "'poniżej N' / 'mniej niż N' / 'less than N' is EXCLUSIVE of N → pass N-1. "
                            "'do N' / 'maksymalnie N' / 'N lub mniej' / 'at most N' is inclusive → pass N as-is."
                        ),
                    },
                    "min_stock": {
                        "type": "integer",
                        "description": (
                            "Return products with total stock ≥ this value (inclusive). "
                            "Convert the user's wording to the correct boundary number: "
                            "'powyżej N' / 'więcej niż N' / 'more than N' is EXCLUSIVE of N → pass N+1. "
                            "'od N' / 'co najmniej N' / 'minimum N' / 'at least N' is inclusive → pass N as-is."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_offers_by_price",
            "description": (
                "Filter active offers by price (PLN). "
                "Use for questions like 'offers below 50 zł', 'most expensive offers', 'offers above 500 zł'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_price": {
                        "type": "number",
                        "description": "Return offers with price ≤ this value (PLN).",
                    },
                    "min_price": {
                        "type": "number",
                        "description": "Return offers with price ≥ this value (PLN).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_products_to_reorder",
            "description": (
                "Find products that are low on stock and need reordering/restocking from a "
                "supplier, optionally narrowed to one assortment/category by partial name "
                "match (e.g. 'włóczka', 'guziki', 'tkanina'). Considers active offers plus any "
                "ended offer that sold out to zero stock (Allegro auto-ends offers once stock "
                "hits zero, so those are exactly what needs reordering; an ended offer that "
                "still has stock left was stopped deliberately and is excluded). Returns each "
                "matching product's name, current total stock, and price, aggregated across all "
                "listings of the same product, sorted by stock ascending (lowest stock — most "
                "urgent — first). "
                "USE THIS TOOL when the user asks to prepare a reorder/restock email or list "
                "to send to a supplier: 'wygeneruj mail z zamówieniem do dostawcy', "
                "'przygotuj zamówienie uzupełniające', 'napisz do dostawcy o brakujące "
                "produkty', 'lista produktów do zamówienia', 'jakie produkty trzeba zamówić'. "
                "Do NOT use this for a plain stock-level question — use query_offers_by_stock "
                "for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "assortment": {
                        "type": "string",
                        "description": (
                            "Optional: filter to one product category/assortment by partial "
                            "name match (e.g. 'włóczka'). Leave empty to consider all products."
                        ),
                    },
                    "max_stock": {
                        "type": "integer",
                        "description": (
                            "Only include products with total stock at or below this level "
                            "(i.e. needing restock), inclusive. Defaults to 5 unless the user "
                            "specifies a different threshold. Convert the user's wording to the "
                            "correct boundary number: 'poniżej N szt.' / 'mniej niż N' is EXCLUSIVE "
                            "of N → pass N-1 (e.g. 'poniżej 10 sztuk' → pass 9). 'do N' / "
                            "'maksymalnie N' / 'N lub mniej' is inclusive → pass N as-is."
                        ),
                        "default": 5,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_offer_details",
            "description": "Get full details of a specific offer by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "offer_id": {"type": "string", "description": "Allegro offer ID."},
                },
                "required": ["offer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_offer_price",
            "description": "Update the price of an Allegro offer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "offer_id": {"type": "string", "description": "Allegro offer ID."},
                    "price": {"type": "number", "description": "New price in PLN (must be > 0)."},
                },
                "required": ["offer_id", "price"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_offer_stock",
            "description": "Update available stock quantity for an Allegro offer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "offer_id": {"type": "string", "description": "Allegro offer ID."},
                    "available": {"type": "integer", "description": "New available quantity (>= 0)."},
                },
                "required": ["offer_id", "available"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message_to_buyer",
            "description": "Send a message to a buyer via Allegro messaging (reply to an existing thread).",
            "parameters": {
                "type": "object",
                "properties": {
                    "thread_id": {"type": "string", "description": "Allegro messaging thread ID."},
                    "text": {"type": "string", "description": "Message text to send to the buyer."},
                },
                "required": ["thread_id", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_message_threads",
            "description": (
                "List recent buyer message threads with unread status. "
                "Use for any question about 'wiadomości', 'nowe wiadomości', new/unread buyer messages. "
                "COUNT-ONLY: for 'czy mam nowe wiadomości', 'czy są jakieś nowe wiadomości', "
                "'ile mam nowych wiadomości' (the user wants YES/NO or a NUMBER, not the thread "
                "list) — set count_only=true. Do NOT set count_only when the user also wants to see "
                "the messages/threads themselves (e.g. 'pokaż wiadomości', 'jakie mam wiadomości', "
                "'pokaż szczegóły' after being told the count)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max threads to return (1–20).", "default": 10},
                    "count_only": {
                        "type": "boolean",
                        "description": (
                            "True when the user only wants to know WHETHER there are new messages "
                            "or HOW MANY — not the list itself."
                        ),
                        "default": False,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_thread_messages",
            "description": (
                "Get the actual TEXT content of a buyer message thread — use when the user wants to "
                "READ what a message says ('pokaż mi wiadomość od X', 'co napisał kupujący', "
                "'wiadomość z dzisiaj', 'treść wiadomości', 'przeczytaj wiadomość'), not just "
                "metadata. get_message_threads does NOT include message text, only buyer/read-status/"
                "date — use this tool instead whenever the user wants to read a message. "
                "If thread_id isn't already known from earlier in the conversation, provide "
                "buyer_login and/or date to find the matching thread automatically — no need to call "
                "get_message_threads first. "
                "The result also names the ORDER the message is about (the checkout-form id Allegro "
                "attached to it, or the buyer's single order) — a buyer writing 'faktura do tej transakcji' "
                "never names the order themselves, so take the id from here and pass it straight to "
                "get_order_details / get_order_invoice_data / issue_invoice_for_order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thread_id": {"type": "string", "description": "Allegro messaging thread ID, if already known."},
                    "buyer_login": {"type": "string", "description": "Buyer's Allegro login, to find their thread."},
                    "date": {
                        "type": "string",
                        "description": (
                            "Find the thread whose last message is on this date. "
                            "'dzisiaj'/'today' for today, or 'YYYY-MM-DD' for a specific date."
                        ),
                    },
                    "limit": {"type": "integer", "description": "Max messages to return (1–20).", "default": 10},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_account_info",
            "description": "Get basic profile information about the seller's Allegro account (login, rating, subscription). Do NOT use for questions about orders, delivery, or couriers — use get_orders_delivery for those.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_billing_summary",
            "description": (
                "Get Allegro billing entries aggregated across a TIME PERIOD (all orders combined). "
                "Use ONLY for period-level questions: 'jakie koszty miałem w tym miesiącu', "
                "'ile prowizji zapłaciłem w czerwcu', 'ostatnie opłaty na koncie'. "
                "DO NOT use for a specific order — use get_order_details instead "
                "(it filters by order.id and shows exact per-item entries). "
                "When a period is given, pass date_from_local/date_to_local as Warsaw-local calendar "
                "dates — do NOT convert to UTC yourself, that conversion happens automatically and "
                "getting it wrong drops entries near local midnight. Without dates, returns recent entries. "
                "Returns: total fees, refunds/credits, net cost, breakdown by fee type, individual entries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_from_local": {
                        "type": "string",
                        "description": (
                            "Start of period as a Warsaw-local calendar date, 'YYYY-MM-DD'. Optional. "
                            "'dziś/today' → today's date; 'wczoraj/yesterday' → yesterday's date; "
                            "'ten tydzień/this week' → Monday of current week; "
                            "'ten miesiąc/this month' → 1st of current month; "
                            "'ostatni miesiąc/last month' → 1st of previous month."
                        ),
                    },
                    "date_to_local": {
                        "type": "string",
                        "description": (
                            "End of period as a Warsaw-local calendar date, 'YYYY-MM-DD' (inclusive). Optional. "
                            "'dziś/today' or 'ten tydzień/ten miesiąc' → today's date; "
                            "'wczoraj/yesterday' → yesterday's date; "
                            "'ostatni miesiąc/last month' → last day of previous month."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max recent entries to return when no date range given (1–100).",
                        "default": 50,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_orders_delivery",
            "description": (
                "Show which courier / delivery method the buyer chose for each order — the "
                "get_orders listing with fulfillment_status=READY_FOR_SHIPMENT and courier "
                "details switched on, so the reply is the same plain-text order list as the "
                "other order tools, plus a per-courier count summary and the total delivery "
                "cost the buyers paid. "
                "Use whenever the user asks: which couriers are in pending orders, "
                "which delivery methods were selected, tracking numbers, delivery costs across "
                "SEVERAL orders ('ile kosztowały dostawy w tych zamówieniach'), or any question "
                "combining orders with shipping/courier/delivery. "
                "For the delivery cost of ONE already-identified order use get_order_details "
                "instead — this tool has no order_id filter. "
                "For the delivery cost of ONE order the user describes by its AMOUNT or by WHEN "
                "it was placed ('dostawa zamówienia z ostatnich dni na kwotę ponad 2000 zł') use "
                "get_orders with min_value/bought_after_local + include_delivery=true instead: "
                "this tool defaults to the packed-and-waiting stage, so an order already sent — "
                "or not yet packed — would be silently excluded and the seller would get the "
                "whole courier list instead of their order (a real bug seen in production). "
                "Default (no filters): orders with fulfillment_status=READY_FOR_SHIPMENT "
                "(packed and awaiting carrier handoff). "
                "STAGE 'DO WYSŁANIA' — leave fulfillment_status EMPTY for any wording meaning the "
                "parcel is PACKED and still has to go out: 'do wysłania', 'gotowe do wysyłki', "
                "'czekają/oczekujące na wysyłkę', 'do nadania', 'przygotowane do nadania', "
                "'zapakowane', 'co czeka na kuriera', 'gotowe do wywózki', 'ile paczek do nadania'. "
                "NEGATION 'NIEWYSŁANE' IS NOT THIS STAGE — 'niewysłane', 'jeszcze nie wysłane', "
                "'które nie zostały wysłane' mean every order that has not left yet, including the "
                "ones nobody has packed: that is get_orders with "
                "exclude_fulfillment_status=['SENT', 'IN_TRANSIT', 'READY_FOR_PICKUP', 'PICKED_UP']. "
                "Answering it with this preset silently hides every unpacked order. "
                "STAGE 'WYSŁANE' — set fulfillment_status=SENT for wording meaning it already left: "
                "'wysłane', 'nadane', 'w transporcie', 'przekazane przewoźnikowi', 'co już poszło', "
                "'co odebrał kurier', 'ile dziś wysłałem', 'ile już wyjechało'. "
                "NOTE 'do spakowania'/'co mam spakować'/'niespakowane' is the PREVIOUS stage (not yet "
                "packed) — that is get_new_orders, not this tool."
            ),
            "parameters": _order_params(
                "status", "fulfillment_status", "buyer_login",
                "bought_after_local", "bought_before_local",
                "dispatch_after_local", "dispatch_before_local",
                "min_value", "max_value", "product_names", "product_match",
                "count_only", "limit",
                status={"description": "Order status filter. Default: READY_FOR_PROCESSING."},
                fulfillment_status={
                    "description": (
                        "Fulfillment status filter. Leave empty for READY_FOR_SHIPMENT orders "
                        "(packed, awaiting carrier handoff). Use SENT only when explicitly "
                        "asking about already-shipped orders."
                    ),
                },
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_orders_due_today",
            "description": (
                "The DEADLINE view: orders that still have to be handed to the carrier and whose "
                "dispatch deadline ('Wysyłka do' — delivery.time.dispatch.to) falls at or before "
                "the END OF TODAY, overdue ones included, earliest deadline first, with courier "
                "details on. Anything already handed over or no longer to be handed over (SENT, "
                "PICKED_UP, CANCELLED, SUSPENDED) is excluded, whatever its deadline was. "
                "USE THIS for 'co muszę wysłać dzisiaj', 'co mam dziś do wysyłki', 'co się pali', "
                "'jakie mam dzisiaj terminy', 'ile paczek muszę dziś nadać', 'co muszę dziś zdążyć "
                "nadać' — any question tying orders to TODAY'S dispatch deadline. "
                "NOT the same as get_orders_delivery: that one is the PACKING-STATE view "
                "(everything READY_FOR_SHIPMENT, whatever its deadline, including parcels due next "
                "week); this one is the DEADLINE view (everything due today, including orders "
                "nobody has packed yet). A question naming a day or a deadline is this tool; a "
                "question naming the packing state ('gotowe do wysyłki', 'zapakowane') is that one. "
                "OTHER HORIZONS: pass dispatch_before_local for a different cut-off — 'do jutra' → "
                "tomorrow 23:59, 'do piątku' → that date 23:59. For ONLY the already-overdue ones "
                "('co jest po terminie') pass dispatch_before_local with the CURRENT time instead."
            ),
            "parameters": _order_params(
                "dispatch_after_local", "dispatch_before_local", "buyer_login", "count_only", "limit",
                dispatch_before_local={
                    "description": (
                        "Deadline cut-off. Defaults to the end of today ('23:59'), which is what "
                        "'co muszę wysłać dzisiaj' means — pass a value only for a different "
                        "horizon ('YYYY-MM-DD 23:59' for 'do jutra'/'do piątku', or the current "
                        "time 'HH:MM' to see only what is already overdue)."
                    ),
                },
                dispatch_after_local={
                    "description": (
                        "Lower bound on the deadline. Leave EMPTY by default: 'co muszę wysłać "
                        "dzisiaj' has to include orders whose deadline already passed — those are "
                        "the most urgent ones, not the ones to hide."
                    ),
                },
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_orders_pending_invoice",
            "description": (
                "Find all paid orders for a given month where the buyer requested a VAT invoice "
                "but the seller has not yet uploaded one. Defaults to the current month. "
                "Use when asked about missing invoices or invoice obligations — 'jakie mam "
                "faktury do wystawienia', 'jakie faktury mam do wysłania', 'brakujące faktury', "
                "'zaległe faktury', 'do których zamówień muszę wystawić fakturę', 'komu jeszcze "
                "nie wysłałem faktury'. "
                "'FAKTURY DO WYSŁANIA' IS THIS TOOL, NOT A SHIPPING QUESTION: 'do wysłania' names "
                "the DOCUMENT the seller still owes the buyer, so it is the invoice listing — "
                "never get_orders_delivery, whose 'do wysłania' is about PARCELS waiting for the "
                "courier and which knows nothing about invoices. "
                "SCOPED TO AN ORDER STAGE — fulfillment_status / exclude_fulfillment_status: a "
                "seller very often asks only about part of their orders ('faktury do wysłania w "
                "zamówieniach nie nowych', 'jakie faktury muszę wystawić do wysłanych zamówień', "
                "'brakujące faktury w zamówieniach, których jeszcze nie wysłałem'). That stage is "
                "a FILTER and must be passed on — dropped, the reply lists every pending invoice "
                "of the month, which reads like a real answer to a question nobody asked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "month": {
                        "type": "integer",
                        "description": "Month number (1–12). Defaults to current month.",
                    },
                    "year": {
                        "type": "integer",
                        "description": "4-digit year. Defaults to current year.",
                    },
                    "fulfillment_status": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(_FULFILLMENT_STATUSES)},
                        "description": (
                            "Keep only orders at one of these fulfillment stages — the POSITIVE "
                            "stage scope ('faktury w nowych zamówieniach' → ['NEW'], 'w "
                            "zamówieniach do wysłania / spakowanych' → ['READY_FOR_SHIPMENT'], "
                            "'w zamówieniach w realizacji' → ['PROCESSING']). "
                            "A LIST, not one status, because a stage the seller names is often a "
                            "family: 'w wysłanych zamówieniach' means every parcel that has left "
                            "— ['SENT', 'IN_TRANSIT', 'READY_FOR_PICKUP', 'PICKED_UP'] — and "
                            "answering it with SENT alone silently drops the orders already "
                            "delivered, whose invoice is the most overdue of all. "
                            "For a NEGATED scope use exclude_fulfillment_status instead."
                        ),
                    },
                    "exclude_fulfillment_status": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(_FULFILLMENT_STATUSES)},
                        "description": (
                            "Drop orders at these fulfillment stages — the NEGATED stage scope. "
                            "A negation is never one status, it is everything except one: "
                            "'w zamówieniach nie nowych' / 'poza nowymi' → exclude ['NEW'], "
                            "'w zamówieniach, których nie wysłałem' → exclude ['SENT', "
                            "'IN_TRANSIT', 'READY_FOR_PICKUP', 'PICKED_UP'], 'w nieodebranych' → "
                            "exclude ['PICKED_UP']. Never answer a negated scope with "
                            "fulfillment_status: naming one stage where the seller excluded one "
                            "hides every other stage they did ask about."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sales_summary",
            "description": (
                "Return full earnings summary for a specific time period, with per-order cost breakdown. "
                "USE THIS TOOL for ANY question about: earnings, revenue, profit, Allegro fees/commissions, "
                "costs per order, net profit, 'ile zarobiłem', 'jakie mam koszty', 'prowizja per zamówienie', "
                "'koszty dla każdego zamówienia', 'pokaż prowizje', 'ile Allegro wzięło', "
                "'zarobek', 'przychód', 'zysk', 'koszty Allegro', 'opłaty per zamówienie'. "
                "Returns: total revenue, Allegro fees, revenue after Allegro fees (revenue minus fees plus "
                "refunds/rebates — NOT net profit: this app has no data on the seller's cost of goods, "
                "packaging or other own expenses, so never call this figure 'zysk netto'/'net profit' in the "
                "reply, only 'przychód po opłatach Allegro'), order count, average order value, top-selling "
                "products, breakdown of fee types (commission, listing, etc.), a per-order table showing "
                "revenue + Allegro cost + revenue-after-fees for each individual order, AND — whenever the "
                "period spans more than one calendar month — a month-by-month breakdown of orders and revenue. "
                "Uses payment.finishedAt (actual payment date) for order filtering — Allegro operates on UTC, "
                "but date_from_local/date_to_local are Warsaw-local calendar dates converted to the correct "
                "UTC window automatically; do NOT build the UTC boundaries yourself, that has previously "
                "caused orders placed near local midnight to be dropped from the wrong day's period. "
                "ALWAYS resolve common time expressions automatically — do NOT ask the user for clarification: "
                "'dziś/today' → today's date for both; "
                "'wczoraj/yesterday' → yesterday's date for both; "
                "'przedwczoraj/day before yesterday' → two days ago for both; "
                "'ostatni tydzień/last week/last 7 days' → 7 days ago to today; "
                "'ten tydzień/this week' → Monday of current week to today; "
                "'ten miesiąc/this month' → first day of current month to today; "
                "'ostatni miesiąc/last month' → first day of previous calendar month to last day of that month; "
                "'ten rok/w tym roku/z tego roku/cały rok/od początku roku/this year/YTD' → 1 JANUARY OF THE "
                "CURRENT YEAR through today — NEVER January alone, and never any other single month; "
                "'zeszły rok/ubiegły rok/poprzedni rok/last year' → 1 January to 31 December of the previous year; "
                "'ostatnie N miesięcy/last N months' → the first day of the month N-1 months back through today; "
                "'ten kwartał/this quarter' → first day of the current calendar quarter through today. "
                "MULTI-MONTH PERIODS — CRITICAL: call this tool ONCE for the WHOLE period, with "
                "date_from_local on its first day and date_to_local on its last. NEVER slice a longer period "
                "into one call per month and NEVER answer a year/quarter question with a single month: the "
                "report ALREADY breaks any period longer than one month down month by month ('Sprzedaż wg "
                "miesięcy' — orders, revenue and average order value per month, plus a chart), so 'sprzedaż z "
                "tego roku z podziałem na miesiące' is ONE call with date_from_local='<current year>-01-01' "
                "and date_to_local=today. "
                "Only ask the user if the period is truly ambiguous (e.g. no period mentioned at all). "
                "date_from_local and date_to_local must be 'YYYY-MM-DD' Warsaw-local calendar dates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_from_local": {
                        "type": "string",
                        "description": "Start of period as a Warsaw-local calendar date, 'YYYY-MM-DD'.",
                    },
                    "date_to_local": {
                        "type": "string",
                        "description": "End of period as a Warsaw-local calendar date, 'YYYY-MM-DD' (inclusive).",
                    },
                },
                "required": ["date_from_local", "date_to_local"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sold_quantities",
            "description": (
                "HOW MANY UNITS of a product were SOLD in a period — the quantity question, not the "
                "money one. USE THIS for: 'ile sztuk sprzedałem', 'ile sztuk sprzedanych dla <produkt>', "
                "'ile poszło <produkt>', 'ile sztuk <produkt> zeszło w tym miesiącu', 'co się najlepiej "
                "sprzedawało', 'ile zeszło włóczki jeans'. "
                "Counts units from PAID orders in the period, cancelled ones excluded. Returns are NOT "
                "subtracted — a returned item still counts as sold here, so say so if the number matters "
                "to the seller. "
                "NOT get_sales_summary: that one answers how much money came in and ranks products by "
                "REVENUE; this one counts PIECES and can be narrowed to named products. "
                "NOT get_active_offers/query_offers_by_stock: those report what is IN STOCK right now, "
                "which is a different number from what was sold. "
                "NOT for LISTING the orders a product was in ('pokaż zamówienie z wczoraj z włóczką "
                "yarnart jeans', 'które zamówienia miały jeans plus') — this tool answers with a "
                "units total and never names an order; that is get_orders with product_names. "
                "PRODUCT NAMES — pass every model the user names as a SEPARATE entry in `names`, exactly "
                "as they wrote it: 'włóczki jeans i jeans plus' is names=['jeans', 'jeans plus'], NOT "
                "['jeans'] and NOT ['jeans i jeans plus']. They are different models and the tool keeps "
                "them apart; merging them into one term is what makes the answer wrong. "
                "Omit `names` entirely only when the user named no product at all ('ile sztuk sprzedałem "
                "w maju') — then every product sold in the period is listed, most units first. "
                "Same period rules as get_sales_summary: resolve 'ostatnie 3 miesiące', 'w tym roku' etc. "
                "yourself into date_from_local/date_to_local and call this ONCE for the whole period."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Product/model names to count, one entry per model the user named. A name "
                            "matches an offer title on whole words, and the most specific name wins, so "
                            "'jeans' and 'jeans plus' never absorb each other's sales."
                        ),
                    },
                    "date_from_local": {
                        "type": "string",
                        "description": "Start of period as a Warsaw-local calendar date, 'YYYY-MM-DD'.",
                    },
                    "date_to_local": {
                        "type": "string",
                        "description": "End of period as a Warsaw-local calendar date, 'YYYY-MM-DD' (inclusive).",
                    },
                },
                "required": ["date_from_local", "date_to_local"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_buyers",
            "description": (
                "The BUYER view of a period: one row per CUSTOMER instead of one row per order — "
                "who bought, how many orders, for how much in total, when they last bought, and "
                "how many of their orders already have a VAT invoice. "
                "USE THIS for any question about the buyers themselves: 'lista kupujących', "
                "'lista klientów', 'kto u mnie kupował', 'ilu miałem klientów', 'moi najlepsi "
                "klienci', 'stali klienci', 'kto kupuje najwięcej', 'jakie firmy u mnie kupowały', "
                "'lista kupujących, dla których wystawiłem faktury VAT', 'klienci z NIP-em', "
                "'zestawienie kontrahentów'. "
                "FIRMA vs OSOBA PRYWATNA: the ONLY place Allegro states this is the VAT-invoice "
                "address on the order (company name + NIP), so buyer_type='company' means exactly "
                "'gave company invoice details on at least one order in the period' — a business "
                "that never asked for an invoice is indistinguishable from a private person and "
                "counts as 'person'. "
                "GROUPING: orders are grouped by NIP where there is one, otherwise by company "
                "name, otherwise by the buyer's Allegro login. Each row NAMES the buyer with their "
                "company name or first and last name (from the invoice address, else from the "
                "delivery address) — the Allegro login is a separate column, never the name. "
                "PERIOD: date_from_local/date_to_local are Warsaw-local 'YYYY-MM-DD' calendar "
                "dates; resolve the period yourself from the current date exactly as for "
                "get_sales_summary ('w tym roku' → 1 January of the current year through today, "
                "'w tym miesiącu' → the 1st of the current month through today, 'w zeszłym roku' → "
                "1 January to 31 December of the previous year). Omit BOTH to get the current "
                "calendar year, which is what a period-less 'lista kupujących' means — never ask "
                "the user for a period just to call this tool. Orders are counted by payment date, "
                "the same basis as get_sales_summary, so the two agree on the same period. "
                "NOT get_sales_summary (that answers 'ile zarobiłem' — revenue, Allegro fees and "
                "top products, and never names a buyer) and NOT get_orders (one bullet per order, "
                "no per-buyer totals). "
                "ONE CUSTOMER IDENTIFIED BY CONTACT DETAILS IS A DIFFERENT TOOL: this one has no "
                "phone, e-mail, name or NIP filter, so 'czy mam klienta z takim nr telefonu "
                "+48 880 197 834' or 'czy mam w bazie Jana Kowalskiego' must go to "
                "find_buyer_by_contact — asked here, the detail is silently dropped and the reply "
                "is the whole period's customer list. "
                "ONE NAMED ACCOUNT IS THE OTHER WAY ROUND: this tool describes the buyer "
                "POPULATION of a period and has NO buyer_login parameter, so a question about "
                "ONE named account — 'czy w tym roku kupował ode mnie ktoś z konta np1988', "
                "'co kupił użytkownik anna.kowalska88' — must go to get_orders with "
                "buyer_login=<that login>. Calling this tool for such a question drops the login "
                "silently and answers with every customer of the period, which reads like a real "
                "answer to a question nobody asked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_from_local": {
                        "type": "string",
                        "description": (
                            "Start of period as a Warsaw-local calendar date, 'YYYY-MM-DD'. "
                            "Defaults, together with date_to_local, to the current calendar year."
                        ),
                    },
                    "date_to_local": {
                        "type": "string",
                        "description": (
                            "End of period as a Warsaw-local calendar date, 'YYYY-MM-DD' "
                            "(inclusive). Defaults to today."
                        ),
                    },
                    "buyer_type": {
                        "type": "string",
                        "description": (
                            "Which buyers to keep: 'company' = only those who gave company invoice "
                            "details (firmy, 'na firmę', 'z NIP-em', B2B), 'person' = everyone else "
                            "(osoby prywatne), 'any' = both. Pass 'company'/'person' whenever the "
                            "question names one of them."
                        ),
                        "enum": ["any", "company", "person"],
                        "default": "any",
                    },
                    "invoice_status": {
                        "type": "string",
                        "description": (
                            "Filter on the VAT invoice: 'issued' = only buyers who already have an "
                            "invoice attached to at least one order in the period ('dla których "
                            "wystawiłem faktury', 'komu wystawiłem fakturę VAT'), 'missing' = only "
                            "those who asked for one that has NOT been issued yet, 'requested' = "
                            "those who asked for an invoice regardless of whether it exists, 'any' "
                            "= no invoice filter."
                        ),
                        "enum": ["any", "issued", "missing", "requested"],
                        "default": "any",
                    },
                    "sort_by": {
                        "type": "string",
                        "description": (
                            "Row order: 'value' = highest total spend first (default, the 'najlepsi "
                            "klienci' order), 'orders' = most orders first ('stali klienci', 'kto "
                            "kupuje najczęściej'), 'recent' = most recent purchase first."
                        ),
                        "enum": ["value", "orders", "recent"],
                        "default": "value",
                    },
                    "count_only": {
                        "type": "boolean",
                        "description": (
                            "Set true when the user only wants the NUMBER of buyers ('ilu miałem "
                            "klientów', 'ile firm u mnie kupowało') — the reply is one sentence "
                            "with the count and no table. Do NOT set it when they want to see the "
                            "buyers themselves ('lista', 'pokaż', 'kto')."
                        ),
                        "default": False,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max buyers listed in the table (1–200).",
                        "default": 100,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_buyer_by_contact",
            "description": (
                "LOOK UP ONE CUSTOMER in the store's customer base by their CONTACT DETAILS — "
                "phone number, e-mail, name/company or NIP — and answer whether that person has "
                "ever bought anything, and if so what. "
                "USE THIS whenever the question is 'do I know this contact?': 'czy mam klienta z "
                "takim nr telefonu +48 880 197 834', 'czy ten numer telefonu coś u mnie kupował', "
                "'kto to jest 880197834', 'sprawdź numer 880 197 834', 'czy mam w bazie klienta "
                "Jan Kowalski', 'czy kupował ode mnie ktoś z adresu jan@example.com', 'czy mam "
                "klienta z NIP 7792445588', 'co zamawiał klient o tym numerze'. A phone number in "
                "the message is almost always this tool. "
                "Pass ONLY the detail(s) the user actually gave, exactly as they wrote them — the "
                "phone may be in any format ('+48 880 197 834', '880-197-834', '880197834'), it is "
                "normalized here; never reformat it and never invent the missing digits of a "
                "partial number. At least one of phone/email/name/nip is REQUIRED: if the user "
                "named nobody at all, call ask_clarifying_question instead. "
                "Returns, for each matching customer: their name, phone, e-mail, Allegro login, "
                "how many orders they placed and for how much, when they last bought, and the "
                "list of those orders. "
                "NOT get_buyers (that lists the whole customer POPULATION of a period and cannot "
                "filter by a phone, e-mail or name — asked this question it would answer with "
                "every customer of the period, which reads like a real answer) and NOT get_orders "
                "(its buyer_login filter is the Allegro LOGIN, not a phone, an e-mail or a "
                "person's name). When the user names an Allegro LOGIN instead of contact details "
                "('z konta np1988'), that IS get_orders with buyer_login."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {
                        "type": "string",
                        "description": (
                            "The phone number to look for, exactly as the user wrote it — any "
                            "format ('+48 880 197 834', '880-197-834', '880197834'). Matched "
                            "against both the buyer's Allegro phone and the delivery address "
                            "phone, ignoring spaces, dashes and the +48 country prefix."
                        ),
                    },
                    "email": {
                        "type": "string",
                        "description": "The buyer's e-mail address to look for (exact, case-insensitive).",
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "A buyer or company name to look for ('Jan Kowalski', 'Kawa i "
                            "Spółka') — matched as a fragment of the invoice buyer's name, the "
                            "company name or the parcel recipient's name, ignoring case."
                        ),
                    },
                    "nip": {
                        "type": "string",
                        "description": "The company's NIP to look for; dashes and spaces are ignored.",
                    },
                    "date_from_local": {
                        "type": "string",
                        "description": (
                            "Start of the period to search, Warsaw-local 'YYYY-MM-DD'. Omit "
                            "unless the user names a period — the search then covers the last "
                            "24 months, which is what 'czy mam takiego klienta' means. Pass an "
                            "explicitly earlier date only when the user asks to look further back."
                        ),
                    },
                    "date_to_local": {
                        "type": "string",
                        "description": (
                            "End of the period to search, Warsaw-local 'YYYY-MM-DD' (inclusive). "
                            "Defaults to today."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_order_monitoring",
            "description": (
                "Present the user with a button to enable automatic background order monitoring. "
                "get_new_orders already appends this automatically, so do NOT call this tool right "
                "after get_new_orders. Only use this when the user brings up monitoring/notifications "
                "on its own, with no get_new_orders call in the same turn (e.g. 'chcę dostawać "
                "powiadomienia o zamówieniach')."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disable_order_monitoring",
            "description": (
                "Show a button to disable automatic order monitoring in the browser. "
                "get_new_orders already offers this button when monitoring is on, so only call this "
                "tool when the user asks to turn off/stop/disable order monitoring outside of an "
                "order query (no get_new_orders call in the same turn)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_invoice_data",
            "description": (
                "Fetch the full invoice/billing address for a specific order — "
                "company name, NIP/VAT ID, buyer's full name (for private persons), "
                "street, city, ZIP code, and country. "
                "Use this when the user asks for data needed to issue a VAT invoice (faktura VAT): "
                "'dane do faktury', 'NIP nabywcy', 'adres do faktury', 'dane firmy z zamówienia', "
                "'wystaw fakturę dla zamówienia X'. "
                "Always call this BEFORE drafting or describing invoice data for a specific order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Allegro order (checkout form) UUID."},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "preview_pending_invoices",
            "description": (
                "Build the exact VAT invoice data (as JSON) that WOULD be sent to inFakt for MULTIPLE "
                "orders — all orders that need an invoice and don't have one yet, for the given month "
                "(defaults to current). IMPORTANT: this does NOT create or send anything — bulk issuance "
                "is intentionally preview-only, so this just shows the data for manual review. "
                "Use for a BATCH request with no single specific order named — "
                "'wystaw faktury', 'wystaw brakujące faktury', 'utwórz faktury za ten miesiąc'. "
                "If the user names ONE specific order, use issue_invoice_for_order instead — that one "
                "actually issues it. For a read-only list use get_orders_pending_invoice, and for just "
                "the billing address of one order use get_order_invoice_data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "month": {
                        "type": "integer",
                        "description": "Month number (1–12). Defaults to current month.",
                    },
                    "year": {
                        "type": "integer",
                        "description": "4-digit year. Defaults to current year.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_invoice_for_order",
            "description": (
                "Actually CREATE/ISSUE a real VAT invoice in inFakt for exactly ONE named Allegro order. "
                "Requires an explicit order_id — never call this for a batch/month-wide request (use "
                "preview_pending_invoices for that, which only previews, never sends). "
                "Use when the user names a specific order and uses an issuance verb: "
                "'wystaw fakturę dla zamówienia <id>', 'wystaw fakturę do tego zamówienia', "
                "'utwórz fakturę dla <id>' — where <id> is a concrete order ID (from this conversation "
                "or given directly by the user). If you don't have a concrete order_id in context, ask "
                "the user for it or look it up first — never guess or invent one. "
                "This creates a real, numbered invoice in inFakt — it is not easily reversible. "
                "It STOPS at inFakt: it does NOT attach the invoice to the Allegro order and does NOT "
                "send it to KSeF, so the seller can check it first. Returns a share link for that "
                "review PLUS the invoice_uuid needed for the follow-up delivery tools "
                "(attach_invoice_to_allegro_order, send_invoice_to_ksef) — never call either of them "
                "in the same turn as this one, even if the user asked for both at once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Allegro order (checkout form) UUID."},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "attach_invoice_to_allegro_order",
            "description": (
                "Download the invoice PDF from inFakt and attach it to the corresponding Allegro order, "
                "so the buyer can see/download it directly from their Allegro order page. "
                "IRREVERSIBLE and visible to the buyer immediately, so call it ONLY when the user asks "
                "for it in the CURRENT message ('dołącz fakturę do zamówienia X') or confirms your own "
                "question about attaching ('ok', 'wygląda dobrze'). Never on the same turn that issued "
                "the invoice — the user has not read it yet — and never on your own initiative. "
                "Requires the Allegro order_id; invoice_uuid is optional — pass the one an earlier "
                "issue_invoice_for_order returned in this conversation, or leave it out and the invoice "
                "recorded for that order is used. Never guess a UUID. "
                "Allegro allows only ONE PDF invoice per order — calling this twice for the same order "
                "will fail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Allegro order (checkout form) UUID."},
                    "invoice_uuid": {
                        "type": "string",
                        "description": (
                            "inFakt invoice UUID from issue_invoice_for_order. Optional — omit it "
                            "rather than guessing; the invoice recorded for this order is used."
                        ),
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_invoice_to_ksef",
            "description": (
                "Submit an already-issued inFakt invoice to KSeF (Krajowy System e-Faktur), Poland's "
                "mandatory national e-invoicing system. Requires the inFakt invoice_uuid returned by an "
                "earlier issue_invoice_for_order call in this conversation — never guess it. "
                "Submission is asynchronous — this only confirms the request was accepted, final "
                "processing must be checked in the inFakt panel. "
                "ONLY for a COMPANY (B2B) buyer, identified by a NIP. An invoice for a PRIVATE "
                "PERSON must NEVER be sent to KSeF — KSeF addresses the buyer by NIP and a private "
                "person has none, so the filing would be wrong and cannot be withdrawn. This is not a "
                "default the user can override: if they ask for it anyway, say why it is impossible "
                "instead of calling this tool. Whether the buyer is a company is decided from "
                "ALLEGRO's invoice data for the ORDER (get_order_invoice_data: company_name + "
                "vat_id), never from what is in inFakt — so pass order_id whenever you know it; "
                "without it the order is looked up from the invoice we issued, and if that fails "
                "the call is refused rather than sent unchecked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_uuid": {"type": "string", "description": "inFakt invoice UUID from issue_invoice_for_order."},
                    "order_id": {
                        "type": "string",
                        "description": (
                            "Allegro order (checkout form) UUID this invoice was issued for. Pass it "
                            "whenever it is in context — it is what Allegro is asked about to confirm "
                            "the buyer is a company with a NIP. Omit rather than guessing."
                        ),
                    },
                },
                "required": ["invoice_uuid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_invoice_reminder",
            "description": (
                "Present the user with a button to enable the automatic invoice REMINDER — a "
                "scheduled check (every 2h, 7:00-20:00) for unissued VAT invoices on already-"
                "SHIPPED orders that asks the seller IN CHAT whether to issue them, and adapts to "
                "the reply: issues them right away, reschedules itself if the seller asks to be "
                "reminded again later (and remembers for how long), or nags again unchanged if the "
                "seller never answers. "
                "This is the ONLY automatic invoice notification the assistant has — call it "
                "whenever the user wants to be notified, asked or nagged about pending invoices. "
                "get_orders_pending_invoice ALREADY appends this automatically, so do NOT "
                "call this tool right after it. Only call this when the user brings up invoice "
                "reminders/nagging on its own, with no get_orders_pending_invoice call in the same "
                "turn. Do NOT call multiple times in one conversation."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disable_invoice_reminder",
            "description": (
                "Show a button to disable the automatic invoice REMINDER in the browser — the one "
                "that asks in chat whether to issue pending invoices and adapts to the reply. "
                "Call it whenever the user asks to stop any invoice reminders/notifications. "
                "get_orders_pending_invoice ALREADY offers this button when the reminder is on, so "
                "only call this tool when the user asks outside of an invoice query."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_message_reminder",
            "description": (
                "Present the user with a button to enable the automatic unread-MESSAGE REMINDER — "
                "a scheduled check (every 2h, 7:00-20:00) that asks the seller IN CHAT whenever a "
                "buyer message is still UNREAD, lists those threads on request, and reschedules "
                "itself if the seller asks to be reminded later (remembering for how long). "
                "Different from suggest_message_monitoring: the MONITOR pushes once the moment a "
                "message ARRIVES, this REMINDER keeps asking for as long as something STAYS "
                "unread. Call it when the user wants to be nagged/reminded about messages they "
                "have not answered yet, or is worried about missing or forgetting messages. "
                "Do NOT call multiple times in one conversation."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disable_message_reminder",
            "description": (
                "Show a button to disable the automatic unread-message REMINDER in the browser — "
                "the one that asks in chat about messages that are still unread. Call it when the "
                "user asks to stop being reminded or nagged about unanswered messages."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_sales_record_reminder",
            "description": (
                "Present the user with a button to enable the monthly SALES-RECORD REMINDER "
                "(ewidencja sprzedaży bezrachunkowej) — the nag to issue the record of "
                "non-invoiced sales for the PREVIOUS month, which is due by the 5th. It starts "
                "on the 1st at 8:00, asks twice a day (8:00 and 20:00) over the 1st-3rd and four "
                "times a day (8:00, 12:00, 16:00, 20:00) from the 4th on, and keeps going past "
                "the deadline until the seller says it is done (\"już wystawiłem\"). "
                "Unlike the other reminders it checks nothing in Allegro — the ewidencja lives "
                "in the seller's own accounting, so their answer is the only thing that stops it. "
                "Call it when the user brings up the ewidencja, the monthly accounting deadline "
                "of the 5th, or wants to be reminded about non-invoiced/receipt-less sales. "
                "Do NOT call multiple times in one conversation."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disable_sales_record_reminder",
            "description": (
                "Show a button to disable the monthly SALES-RECORD REMINDER (ewidencja sprzedaży "
                "bezrachunkowej) in the browser. Call it when the user asks to stop being "
                "reminded about the ewidencja or the monthly deadline of the 5th."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_message_monitoring",
            "description": (
                "Present the user with a button to enable automatic monitoring of new/unread buyer "
                "messages. Call this AFTER get_message_threads when the user asks about messages, "
                "unread messages, or wants to be notified when buyers write new messages. "
                "Do NOT call multiple times in one conversation."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disable_message_monitoring",
            "description": (
                "Show a button to disable automatic message monitoring in the browser. "
                "Call when the user asks to turn off, stop, or disable message monitoring/notifications."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_new_returns",
            "description": (
                "List recent customer returns (zwroty) reported by buyers. "
                "Use for 'nowe zwroty', 'jakie mam zwroty', 'czy są jakieś zwroty', 'ile zwrotów'. "
                "COUNT-ONLY: for 'ile zwrotów', 'ile mam zwrotów' (user wants a NUMBER, not the "
                "list) — set count_only=true. "
                "PERIOD: whenever the user names a time window — 'w tym miesiącu', 'w zeszłym "
                "miesiącu', 'w marcu', 'w tym tygodniu', 'dzisiaj', 'od 1 do 15 maja' — you MUST "
                "pass date_from_local/date_to_local, computed from the current date in your "
                "context. Without them the tool answers about the most recent returns regardless "
                "of date, which silently turns 'ile zwrotów w tym miesiącu' into a wrong number. "
                "Do NOT use this for complaints/disputes ('reklamacje', 'spory') — use "
                "get_new_complaints for those; a return and a complaint are different Allegro "
                "processes even though the user may casually call either one a 'zwrot'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count_only": {
                        "type": "boolean",
                        "description": (
                            "Set true when the user only wants the NUMBER of returns "
                            "('ile zwrotów', 'ile mam zwrotów') — the reply will state just the "
                            "count, not list individual returns."
                        ),
                        "default": False,
                    },
                    "date_from_local": {
                        "type": "string",
                        "description": (
                            "Start of the period as a Warsaw-local calendar date, 'YYYY-MM-DD'. "
                            "Optional — pass it together with date_to_local whenever the user "
                            "names a period ('w tym miesiącu', 'w zeszłym tygodniu', 'w marcu', "
                            "'dzisiaj'), computed from the current date given in your context. "
                            "Omitted (both dates) = most recent returns regardless of date."
                        ),
                    },
                    "date_to_local": {
                        "type": "string",
                        "description": (
                            "End of the period as a Warsaw-local calendar date, 'YYYY-MM-DD' "
                            "(inclusive). Pass together with date_from_local."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_returns_to_process",
            "description": (
                "List customer returns that need SELLER ACTION right now — status DELIVERED, "
                "meaning the returned parcel has physically arrived back and is awaiting the "
                "seller's decision (accept/refund or reject). "
                "Use for 'zwroty do obsłużenia', 'zwroty do rozpatrzenia', 'zwroty czekające na "
                "decyzję', 'czy mam jakieś zwroty do obsłużenia', 'zwroty gotowe do zwrotu "
                "pieniędzy' — i.e. any question about returns that require doing something now, "
                "not just ones that recently arrived as a request. "
                "Do NOT use get_new_returns for this — that tool lists ALL recently reported "
                "returns regardless of whether the parcel has arrived yet (most haven't, so it "
                "answers a different, broader question than 'what needs my attention'). "
                "COUNT-ONLY: for 'ile zwrotów do obsłużenia', 'ile mam zwrotów czekających' "
                "(user wants a NUMBER, not the list) — set count_only=true. "
                "PERIOD: if the user names a time window ('w tym miesiącu', 'w marcu', 'dzisiaj'), "
                "pass date_from_local/date_to_local — without them the answer covers all recent "
                "returns awaiting action, not the period asked about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count_only": {
                        "type": "boolean",
                        "description": (
                            "Set true when the user only wants the NUMBER of returns awaiting "
                            "action — the reply will state just the count, not list individual "
                            "returns."
                        ),
                        "default": False,
                    },
                    "date_from_local": {
                        "type": "string",
                        "description": (
                            "Start of the period as a Warsaw-local calendar date, 'YYYY-MM-DD'. "
                            "Optional — pass it together with date_to_local whenever the user "
                            "names a period ('w tym miesiącu', 'w zeszłym tygodniu', 'w marcu', "
                            "'dzisiaj'), computed from the current date given in your context. "
                            "Omitted (both dates) = most recent returns awaiting action regardless of date."
                        ),
                    },
                    "date_to_local": {
                        "type": "string",
                        "description": (
                            "End of the period as a Warsaw-local calendar date, 'YYYY-MM-DD' "
                            "(inclusive). Pass together with date_from_local."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_new_complaints",
            "description": (
                "List recent disputes/claims (reklamacje) opened by buyers. "
                "Use for 'nowe reklamacje', 'jakie mam reklamacje', 'czy są jakieś spory', "
                "'reklamacje kupujących', 'ile reklamacji'. "
                "COUNT-ONLY: for 'ile reklamacji', 'ile mam reklamacji' (user wants a NUMBER, not "
                "the list) — set count_only=true. "
                "PERIOD: whenever the user names a time window ('w tym miesiącu', 'w marcu', "
                "'dzisiaj'), you MUST pass date_from_local/date_to_local, computed from the "
                "current date in your context — without them the count covers all recent "
                "complaints, not the period asked about. "
                "Do NOT use this for product returns with no dispute ('zwroty') — use "
                "get_new_returns for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count_only": {
                        "type": "boolean",
                        "description": (
                            "Set true when the user only wants the NUMBER of complaints "
                            "('ile reklamacji', 'ile mam reklamacji') — the reply will state just "
                            "the count, not list individual complaints."
                        ),
                        "default": False,
                    },
                    "date_from_local": {
                        "type": "string",
                        "description": (
                            "Start of the period as a Warsaw-local calendar date, 'YYYY-MM-DD'. "
                            "Optional — pass it together with date_to_local whenever the user "
                            "names a period ('w tym miesiącu', 'w zeszłym tygodniu', 'w marcu', "
                            "'dzisiaj'), computed from the current date given in your context. "
                            "Omitted (both dates) = most recent complaints regardless of date."
                        ),
                    },
                    "date_to_local": {
                        "type": "string",
                        "description": (
                            "End of the period as a Warsaw-local calendar date, 'YYYY-MM-DD' "
                            "(inclusive). Pass together with date_from_local."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_returns_monitoring",
            "description": (
                "Present the user with a button to enable automatic background monitoring of new "
                "returns (zwroty) AND complaints/disputes (reklamacje) together — one shared toggle. "
                "get_new_returns, get_returns_to_process, and get_new_complaints already append this "
                "automatically, so do NOT call this tool right after any of them. Only use this when "
                "the user brings up monitoring/notifications for returns/complaints on its own, with "
                "no get_new_returns/get_returns_to_process/get_new_complaints call in the same turn "
                "(e.g. 'chcę dostawać powiadomienia o zwrotach i reklamacjach')."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disable_returns_monitoring",
            "description": (
                "Show a button to disable automatic returns/complaints monitoring in the browser. "
                "get_new_returns, get_returns_to_process, and get_new_complaints already offer this "
                "button when monitoring is on, so only call this tool when the user asks to turn "
                "off/stop/disable returns/complaints monitoring outside of a returns/complaints "
                "query (no get_new_returns/get_returns_to_process/get_new_complaints call in the "
                "same turn)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_clarifying_question",
            "description": (
                "Call this INSTEAD of any data tool when the user's message (plus conversation "
                "history) does NOT give you enough information to pick the right tool or to fill "
                "one of its REQUIRED parameters — e.g. you would have to invent/guess an order_id, "
                "invoice_uuid, buyer_login, date, product name, or price that was never given "
                "directly and was never established earlier in this conversation. Also use this "
                "when the request is genuinely ambiguous between two different tools/intents and "
                "the wording alone doesn't tell you which one the user means. "
                "Ask ONE short, specific question, in the same language as the user's message, "
                "naming exactly what you need (e.g. which order — ID or buyer login; which date "
                "or period; which product). Do NOT call any other tool in the same turn — this is "
                "a stop-and-ask, not a guess-and-verify. "
                "ALSO call this when NO tool here can answer the question AT ALL — not a missing "
                "parameter, but a missing capability: a figure none of these tools computes, a "
                "breakdown none of them produces. In that case do not ask a question — state "
                "plainly, in one sentence and in the user's language, that you cannot answer this "
                "one and what would be needed. Reaching for the nearest listing instead is the "
                "worst available answer: the user reads it as the figure they asked for, and "
                "nothing in the reply tells them it is not."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The clarifying question to show the user, in their language.",
                    },
                },
                "required": ["question"],
            },
        },
    },
]


# ── Output-format routing ────────────────────────────────────────────────────
# Each tool's result shape is known in advance, so the reply's presentation is
# decided by WHICH TOOL WAS CALLED — not guessed from the user's wording before
# any tool ran. The old approach classified "chat/table/document/dashboard" from
# free text up front, which was unreliable: a yes/no question naming a plural
# entity ("czy mam nowe wiadomości?") could get misclassified as "table" and
# produce an empty table for what should've been a one-sentence answer.
#
# This mapping is the source of truth for that decision — see AllegroAgent.run(),
# which resolves it from the tool(s) it actually called and asks the model to
# format the final answer accordingly.
TOOL_OUTPUT_FORMAT: dict[str, str] = {
    # Zamówienia — wszystkie listowania zamówień zwracają ZWYKŁY TEKST.
    # One implementation (AllegroAgent._orders_listing) renders all three, so
    # they must also share one format: with get_orders/get_orders_delivery as
    # "table" the same question answered by a different preset came back as a
    # markdown table, which web/js/app.js (_isDoc) hides behind the document
    # viewer — "nowe zamówienia" gave chat text, "zamówienia do wysłania" gave
    # a document. The listing text is already final; nothing here reshapes it.
    "get_new_orders": "chat",
    "get_orders": "chat",
    # "chat" (not "document") — a single order's details are a short, factual
    # answer to a question the user just asked; wrapping them in a document
    # artifact hid them behind a "zobacz pełną odpowiedź" click for no gain.
    "get_order_details": "chat",
    # Same reasoning as get_order_details: one order's profit calculation is a
    # short, factual answer to the question just asked, not a document.
    "calculate_order_profit": "chat",
    "get_orders_delivery": "chat",
    "get_orders_due_today": "chat",
    # Oferty
    "get_active_offers": "table",
    "get_offers_summary": "dashboard",
    "query_offers_by_stock": "table",
    "query_offers_by_price": "table",
    "get_products_to_reorder": "document",
    "get_offer_details": "chat",
    "update_offer_price": "action",
    "update_offer_stock": "action",
    # Wiadomości
    # "chat" (not "table") — like get_new_orders, the list itself is small and
    # often just a yes/no/count, so a fixed-shape conversational reply (see
    # _TOOL_SPECIFIC_INSTRUCTIONS in allegro_agent.py) beats a markdown table
    # that gets collapsed into a "zobacz pełną odpowiedź" link for short answers.
    "get_message_threads": "chat",
    "get_thread_messages": "chat",
    "send_message_to_buyer": "action",
    # Konto / rozliczenia / sprzedaż
    "get_account_info": "chat",
    "get_billing_summary": "table",
    "get_sales_summary": "dashboard",
    # "chat": a handful of product rows answering the question just asked,
    # not a document — same reasoning as get_order_details above.
    "get_sold_quantities": "chat",
    "get_buyers": "table",
    # "chat" (not "table") — a contact lookup answers a yes/no question about
    # ONE person, usually with a single match; a one-row table hidden behind
    # the document viewer would bury the answer.
    "find_buyer_by_contact": "chat",
    # Faktury
    "get_orders_pending_invoice": "chat",
    "get_order_invoice_data": "chat",
    "preview_pending_invoices": "action",
    "issue_invoice_for_order": "action",
    "attach_invoice_to_allegro_order": "action",
    "send_invoice_to_ksef": "action",
    # Zwroty i reklamacje
    "get_new_returns": "chat",
    "get_returns_to_process": "chat",
    "get_new_complaints": "chat",
    # Monitoring (przyciski w UI)
    "suggest_order_monitoring": "action",
    "suggest_invoice_reminder": "action",
    "suggest_message_reminder": "action",
    "disable_message_reminder": "action",
    "suggest_sales_record_reminder": "action",
    "disable_sales_record_reminder": "action",
    "suggest_message_monitoring": "action",
    "suggest_returns_monitoring": "action",
    "disable_order_monitoring": "action",
    "disable_invoice_reminder": "action",
    "disable_message_monitoring": "action",
    "disable_returns_monitoring": "action",
    # Dopytanie użytkownika, gdy brak narzędzia/parametru da się jednoznacznie ustalić
    "ask_clarifying_question": "chat",
}

# When several tools are called in the same turn, the most "structured" format
# wins — a UI-action button (e.g. a monitoring toggle riding along with
# get_orders_pending_invoice) never downgrades a real data reply.
_FORMAT_PRIORITY = ["dashboard", "document", "table", "chat", "action"]


def resolve_output_format(tool_names: list[str]) -> str:
    """Resolve the reply's output format from the tool(s) called in one turn."""
    formats = {TOOL_OUTPUT_FORMAT.get(name, "chat") for name in tool_names}
    for fmt in _FORMAT_PRIORITY:
        if fmt in formats:
            return fmt
    return "chat"


# ── Tool-select context filter ──────────────────────────────────────────────
# Every turn's tool-selection call sends all ~37 tool schemas regardless of
# what the query is actually about — most of that is dead weight the model
# has to read (and pay tokens for) just to conclude it doesn't apply. Each
# tool belongs to one topic; if the conversation text names that topic, the
# tool stays a candidate, otherwise it's dropped before the LLM ever sees it.
#
# Matching is stem-prefix based rather than whole-word, specifically to
# survive Polish declension without enumerating every inflected form:
# "zamówienie/zamówienia/zamówień/zamówieniem/zamówieniu" all start with the
# same 6-letter stem. Diacritics are stripped on both sides first so e.g.
# "zamowienia" (no diacritics, common in quick typing) still matches.
_TOOL_LABELS: dict[str, str] = {
    # zamowienia
    "get_new_orders":                  "zamowienia",
    "get_orders":                      "zamowienia",
    "get_order_details":               "zamowienia",
    "get_orders_delivery":             "zamowienia",
    "get_orders_due_today":            "zamowienia",
    # oferty
    "get_active_offers":               "oferty",
    "get_offers_summary":              "oferty",
    "query_offers_by_stock":           "oferty",
    "query_offers_by_price":           "oferty",
    "get_products_to_reorder":         "oferty",
    "get_offer_details":               "oferty",
    "update_offer_price":              "oferty",
    "update_offer_stock":              "oferty",
    # wiadomosci
    "send_message_to_buyer":           "wiadomosci",
    "get_message_threads":             "wiadomosci",
    "get_thread_messages":             "wiadomosci",
    # konto
    "get_account_info":                "konto",
    # finanse
    "get_billing_summary":             "finanse",
    "get_sales_summary":               "finanse",
    # A quantity-sold question names the PRODUCT, so it usually matches
    # "oferty" too — but what it asks for is a sales figure, and the
    # selling verbs ("sprzeda", "zarob") are what reliably fire here.
    "get_sold_quantities":             "finanse",
    # "finanse", not "zamowienia", even though it takes an order_id: what makes
    # a query reach for it is the MONEY vocabulary ("zysk", "koszt", "marża"),
    # and a follow-up often names no order at all ("a jaki zysk przy 8 zł za
    # sztukę?" right after the order was shown) — under the "zamowienia" label
    # that phrasing would drop the tool from the candidate list entirely.
    "calculate_order_profit":          "finanse",
    # kupujacy
    "get_buyers":                      "kupujacy",
    "find_buyer_by_contact":           "kupujacy",
    # faktury
    "get_orders_pending_invoice":      "faktury",
    "get_order_invoice_data":          "faktury",
    "preview_pending_invoices":        "faktury",
    "issue_invoice_for_order":         "faktury",
    "attach_invoice_to_allegro_order": "faktury",
    "send_invoice_to_ksef":            "faktury",
    # zwroty (incl. reklamacje — Allegro treats them as related but distinct
    # processes, see get_new_returns/get_new_complaints descriptions above,
    # but they share one monitoring toggle and one query-label here)
    "get_new_returns":                 "zwroty",
    "get_returns_to_process":          "zwroty",
    "get_new_complaints":              "zwroty",
    # monitoring (background checks + chat-based reminders, all UI-action
    # tools — kept as their own label rather than under each domain so a
    # plain domain question, e.g. "nowe zamówienia", doesn't drag in 10
    # near-identical toggle schemas it has no reason to call; get_new_orders
    # et al. already append their own monitoring suggestion automatically)
    "suggest_order_monitoring":        "monitoring",
    "disable_order_monitoring":        "monitoring",
    "suggest_invoice_reminder":        "monitoring",
    "disable_invoice_reminder":        "monitoring",
    "suggest_message_reminder":        "monitoring",
    "disable_message_reminder":        "monitoring",
    "suggest_sales_record_reminder":   "monitoring",
    "disable_sales_record_reminder":   "monitoring",
    "suggest_message_monitoring":      "monitoring",
    "disable_message_monitoring":      "monitoring",
    "suggest_returns_monitoring":      "monitoring",
    "disable_returns_monitoring":      "monitoring",
}

# Stems are deliberately generous (biased toward recall over precision):
# including an irrelevant tool costs a bit of prompt size, but missing a
# relevant one means select_tools_for_context() finds no label at all and
# AllegroAgent.run() asks the user to clarify instead of answering — the
# more expensive mistake by far. Mined from each tool's own description
# above (the words already given there as the phrases that should trigger
# it) plus the label's own name and its common alternate spelling.
_LABEL_STEMS: dict[str, tuple[str, ...]] = {
    # The trailing block is the order-STAGE vocabulary (see
    # deterministic_dispatch._ORDER_STAGE_SIGNALS): a seller often names the
    # stage without naming orders at all — "co czeka na kuriera?", "co mam w
    # robocie?", "ile dostarczonych?" — and without these stems Layer 1 finds
    # no label and falls back to all ~37 schemas. Words that name a stage but
    # are equally at home in another domain ("do obsłużenia", "czekające" —
    # both also returns wording) are deliberately left out: two labels would
    # only disable the deterministic layer for BOTH domains.
    "zamowienia": ("zamow", "zamaw", "order", "paczk", "przesyl", "wysylk", "kurier", "dostaw",
                   "wysla", "nadan", "spakow", "zapakow", "wpad", "nietkni", "kolejc", "rozpocz",
                   "trakci", "realizacj", "zrealizow", "przetwarz", "kompletu", "kompletow",
                   "robocie", "nieskoncz", "nieukoncz", "dokoncz", "wywoz", "transporcie",
                   "przewozn", "poszl", "wyjecha", "dotar", "odebr", "odbior", "dostarcz",
                   "zakonczon", "zamkni", "odhaczy", "termin", "paczek", "nadac", "nadaj"),
    # The assortment words are what a seller actually names instead of the
    # generic "produkt"/"oferta" — "ile zostało włóczek", "jakie tkaniny mam".
    # Without them such a query matched no label at all and fell back to the
    # full ~37-tool list. Diacritics folded (see _normalize), and stems cut
    # short of the fill vowel Polish inserts in the genitive plural:
    # "włóczka" → "włóczek" ("wloczek"), so the stem has to be "wlocz".
    "oferty":     ("ofert", "produkt", "cen", "stan", "magazyn", "zapas", "sklad", "dostawc", "uzupelni", "brakuj",
                   "wlocz", "tkanin", "przedz", "motk"),
    "wiadomosci": ("wiadomo", "watk", "napisa", "napisz", "pisz", "przeczyt", "tresc", "message", "odpisz", "odpowiedz"),
    "konto":      ("konto", "kont", "profil", "subskryp", "ocen", "rating", "account"),
    # "marz" is the margin vocabulary calculate_order_profit answers to
    # ("marża", "marżę", "marzy mi się" is not a store question). It also
    # prefixes the month "marzec" — a cheap miss: such a query keeps every
    # label it already had, it only loses the deterministic layer, which is
    # exactly the recall-over-precision trade this map is built on.
    # "sprzeda", not "sprzedaz": the noun is "sprzedaż" but the seller asks with
    # the PARTICIPLE — "ile sztuk sprzedanych", "co się sprzedało", "ile
    # sprzedałem" — and none of those contain the "ż". The longer stem matched
    # only the noun, which is the form that shows up least.
    "finanse":    ("prowizj", "oplat", "zarob", "przychod", "zysk", "koszt", "rozliczen", "sprzeda",
                   "bilans", "marz", "rentown", "narzut"),
    "faktury":    ("faktur", "nip", "ksef", "vat"),
    # A buyer question names the person, not the order: "lista kupujących",
    # "jakie firmy u mnie kupowały", "zestawienie kontrahentów". "nip"/"firm"
    # are shared with the invoice vocabulary on purpose — "kupujący z NIP-em"
    # is both, and two labels only mean both tool sets stay candidates.
    #
    # "klient" is deliberately NOT here, for the reason given at the top of this
    # map: a seller says it just as often ABOUT an order ("co klient odebrał",
    # "klient czeka na paczkę"), and a second label on those queries would
    # disable the deterministic layer for the order stage they name. A
    # klient-phrased buyer question simply matches no label and falls back to
    # the full tool list, where get_buyers is still there to be picked.
    #
    # The contact stems ("telefon", "mail", ...) are what routes "czy mam
    # klienta z takim nr telefonu" to find_buyer_by_contact. They can co-fire
    # with "zamowienia" on the rare order question that names a phone or an
    # e-mail ("podaj numer telefonu do tego zamówienia"), which only costs
    # that query the deterministic layer — the recall-over-precision trade
    # this whole map is built on.
    "kupujacy":   ("kupuj", "kupowa", "nabywc", "kontrahent", "firm",
                   "buyer", "customer", "nip",
                   "telefon", "tel", "komork", "phone", "mail", "e-mail"),
    "zwroty":     ("zwrot", "reklamacj", "spor"),
    "monitoring": ("monitor", "powiad", "notyfikacj", "przypomn", "wlacz", "wylacz",
                   "ewidencj", "bezrachunkow"),
}

_DIACRITICS = str.maketrans("ąćęłńóśźż", "acelnoszz")


def _normalize(text: str) -> str:
    return text.lower().translate(_DIACRITICS)


# ── A NAMED buyer account: "z konta np1988" ─────────────────────────────────
# A seller says "konto" about their OWN Allegro account ("moje konto", "dane
# konta") — that is get_account_info, the "konto" label above. Followed by a
# name, the very same word means somebody ELSE'S account: the buyer's login,
# and the ONLY tool that can filter by it is the order listing
# (get_orders' buyer_login).
#
# Without this, "czy w tym roku kupował ode mnie ktoś z konta np1988" matched
# {konto, kupujacy}, so the model never even SAW get_orders and answered with
# get_buyers — the whole year's customer list, with the login silently dropped.
# That is the worst failure shape this pipeline has: a filtered question
# answered by an unfiltered list, which reads like a real answer.
_ACCOUNT_WORD_RE = re.compile(
    r"^(?:kont(?:o|a|u|em|cie)|kontrahent\w*|login\w*|nick\w*|uzytkownik\w*|"
    r"kupujac\w*|buyer|user|account)$"
)
# Words that may sit between the account word and the name itself —
# "z konta o nazwie X", "konto allegro X", "login użytkownika X".
_LOGIN_FILLER_WORDS = frozenset({
    "o", "nazwie", "nazwa", "login", "loginie", "loginem", "nick", "nicku",
    "uzytkownika", "uzytkownik", "allegro", "kupujacego", "klienta",
})
# Punctuation and quoting that can wrap a login in a real message
# ("z konta 'np1988'", "z konta np1988?").
_TOKEN_TRIM = "\"'„”»«`([{)]}.,!?:;"
_LOGIN_TOKEN_RE = re.compile(r"^@?[A-Za-z0-9][A-Za-z0-9._-]{2,}$")
# What separates a login from an ordinary Polish word standing right after
# "konta"/"login" ("moje konto allegro jest zawieszone"): a digit or one of
# the separators logins are built from. A login made purely of letters is
# missed on purpose — see named_buyer_login's docstring.
_LOOKS_LIKE_LOGIN_RE = re.compile(r"[0-9._-]")


def named_buyer_login(text: str) -> str | None:
    """The BUYER's Allegro login when `text` names one ("z konta np1988",
    "od użytkownika anna.kowalska88", "login: sklep-abc"), else None.

    Deliberately narrow: the name has to LOOK like a login (a digit or a
    ._- separator in it) rather than an ordinary word, so "moje konto
    allegro" and "dane tego konta" never resolve to one. A miss costs
    nothing — every caller then behaves exactly as it did before this
    existed — while a false positive would put an invented login into a
    tool call, so recall is traded away deliberately.
    """
    raw = text.split()
    # _normalize only lowercases and folds diacritics, so the two lists are
    # token-for-token aligned and the login can be returned in its original
    # spelling (Allegro logins are matched exactly by the API).
    norm = _normalize(text).split()
    for i, word in enumerate(norm):
        if not _ACCOUNT_WORD_RE.match(word.strip(_TOKEN_TRIM)):
            continue
        # Up to three filler words may sit in between ("z konta o nazwie X").
        for j in range(i + 1, min(i + 4, len(norm))):
            if norm[j].strip(_TOKEN_TRIM) in _LOGIN_FILLER_WORDS:
                continue
            login = raw[j].strip(_TOKEN_TRIM).lstrip("@")
            if not (_LOGIN_TOKEN_RE.match(login) and _LOOKS_LIKE_LOGIN_RE.search(login)):
                break
            return login
    return None


# ── A phone number: "czy mam klienta z nr telefonu +48 880 197 834" ─────────
# A phone number is the one contact detail a seller pastes in raw, with no word
# around it that any stem above would catch ("sprawdź 880 197 834"), so the
# NUMBER ITSELF has to route the query — exactly like named_buyer_login above.
#
# Precision matters more than recall here, because a store message is full of
# other long digit runs: offer IDs (11 digits, e.g. 14587236901), NIPs (10),
# REGONs (9!), tracking codes (20+), order UUIDs. So a candidate is only read as
# a phone when it looks like one is written by a human:
#   • an explicit +48 / 0048 country prefix, or
#   • nine digits grouped with spaces or dashes ("880 197 834", "88-019-78-34"),
#   • or nine bare digits WITH a phone word somewhere in the message.
# A bare, unseparated, context-less digit run is never a phone here — that is
# the offer-ID/REGON shape, and mistaking one for a phone would send a seller
# a confident "nie masz takiego klienta" for a question they never asked.
_PHONE_CONTEXT_RE = re.compile(r"telefon|\btel\b|\bnr\b|numer|komórk|komork|phone|dzwoni", re.IGNORECASE)
# Preceded by one of these, nine digits are an identifier, not a phone.
_NOT_A_PHONE_BEFORE_RE = re.compile(r"(?:nip|regon|krs|pesel|id|oferty?|oferta|zamówieni\w*|zamowieni\w*)\W*$", re.IGNORECASE)
_PHONE_CANDIDATE_RE = re.compile(
    r"(?<![\d/-])"                      # not in the middle of a longer number/date
    r"(\+?\s?(?:48|0048)[\s-]?)?"       # optional country prefix
    r"(\d(?:[\s-]?\d){8})"              # nine digits, optionally separated
    r"(?![\d-])"                        # and no more digits after them
)


def phone_digits(raw: str) -> str:
    """A phone number reduced to the digits that identify it: no spaces, no
    dashes, no country prefix — '+48 880 197 834', '0048880197834' and
    '880-197-834' all become '880197834'.

    A number that isn't Polish keeps its own digits (nothing is stripped unless
    what remains is a full nine-digit national number), so a foreign buyer's
    phone still compares equal to itself.
    """
    digits = re.sub(r"\D", "", raw or "")
    for prefix in ("0048", "48", "0"):
        if digits.startswith(prefix) and len(digits) - len(prefix) == 9:
            return digits[len(prefix):]
    return digits


def named_phone_number(text: str) -> str | None:
    """The phone number `text` names, in its original spelling, or None.

    Returns the number as the user typed it (so the reply can echo it back
    unchanged); use phone_digits() for comparing.
    """
    has_context = bool(_PHONE_CONTEXT_RE.search(text))
    for match in _PHONE_CANDIDATE_RE.finditer(text):
        prefix, digits = match.group(1), match.group(2)
        if _NOT_A_PHONE_BEFORE_RE.search(text[: match.start()]):
            continue
        separated = bool(re.search(r"[\s-]", digits))
        if not (prefix or separated or has_context):
            continue
        return match.group(0).strip()
    return None


def matched_labels(text: str) -> set[str]:
    """Labels whose stems appear as a word-prefix anywhere in `text`."""
    words = _normalize(text).split()
    found: set[str] = set()
    for label, stems in _LABEL_STEMS.items():
        if any(word.startswith(stem) for word in words for stem in stems):
            found.add(label)
    # A named buyer account is an ORDER question whatever else it mentions:
    # get_orders' buyer_login is the only filter in the whole tool list that
    # can answer "did THIS account buy from me". Added, never substituted —
    # the stems above stay in force (recall over precision, as everywhere in
    # this map), so "czy np1988 kupował ode mnie" keeps get_buyers as a
    # candidate too and the model picks between them on the descriptions.
    if named_buyer_login(text):
        found.add("zamowienia")
    # A phone number is a CUSTOMER question however it is phrased — "sprawdź
    # 880 197 834" carries no stem at all, and without this it would fall back
    # to the full ~40-schema list with nothing pointing at the lookup tool.
    if named_phone_number(text):
        found.add("kupujacy")
    return found


def tools_for_labels(labels: set[str]) -> list[dict]:
    """Subset of ALLEGRO_TOOLS belonging to any of `labels`, plus the universal
    ask_clarifying_question escape hatch. It carries no domain label of its own
    (see _TOOL_LABELS) because it must stay reachable no matter which topic the
    query matched — it's the fallback for a missing/ambiguous parameter WITHIN
    whatever domain got matched, not a domain in itself."""
    return [
        t for t in ALLEGRO_TOOLS
        if _TOOL_LABELS.get(t["function"]["name"]) in labels
        or t["function"]["name"] == "ask_clarifying_question"
    ]


def select_tools_for_context(text: str) -> list[dict] | None:
    """Subset of ALLEGRO_TOOLS whose label was found in `text`, or None if no
    label matched at all — the caller's cue to fall back to the full,
    unfiltered tool list rather than guessing from a wrongly-narrowed one."""
    labels = matched_labels(text)
    if not labels:
        return None
    return tools_for_labels(labels)


# ── Matching a product the seller named against real offer titles ───────────
# "jeans" and "jeans plus" are two different yarns, and an Allegro title
# carries far more than the model name ("Włóczka Jeans Plus 100g kolor 05").
# A naive `term in title` therefore fails in BOTH directions: it counts every
# Jeans Plus sale towards "jeans", and it matches "jeans" inside an unrelated
# word. Two rules fix that:
#
#   1. Compare TOKENS, not characters. "jeans" matches the title token "Jeans",
#      never the middle of "jeanswear", and a multi-word term has to appear as
#      consecutive tokens.
#   2. Most specific term wins. A title matching both "jeans" and "jeans plus"
#      belongs to "jeans plus" — the longer term is the more precise claim
#      about which model it is.
#
# Rule 2 only separates models the seller actually named. When one term alone
# matches several different titles, nothing here decides that they are the
# same model — the caller reports each title on its own line instead of
# silently summing them, since the distinction it cannot make is exactly the
# one the seller can read off the names.
#
# "+" becomes the token "plus" so "Jeans+" and "Jeans Plus" are one model,
# which is how the seller writes them interchangeably.
_TOKEN_SPLIT_RE = re.compile(r"[^0-9a-z]+")


def product_tokens(text: str) -> list[str]:
    """Offer title or search term as comparable tokens (diacritics folded)."""
    return [t for t in _TOKEN_SPLIT_RE.split(_normalize(text).replace("+", " plus ")) if t]


def _contains_run(haystack: list[str], needle: list[str]) -> bool:
    """True when `needle` appears as consecutive items of `haystack`."""
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[i:i + len(needle)] == needle
        for i in range(len(haystack) - len(needle) + 1)
    )


# A seller names the model, but says the category out loud first — "włóczkę
# yarnart jeans", "przędza jeans plus". Allegro titles carry that word too,
# yet not always in front ("YarnArt Jeans 50g włóczka bawełniana"), and
# match_product_term compares CONSECUTIVE tokens — so the category word the
# seller prepended would break a match against a title that puts it elsewhere.
# It is dropped from the front of a search term, never from the title, and
# never when it is the whole term: "ile zeszło włóczki" names no model, and an
# empty term would match every offer in the store.
_PRODUCT_CATEGORY_PREFIXES = ("wloczk", "przedz", "tkanin", "motek", "motk")


def product_filter_terms(names: list[str]) -> list[str]:
    """Search terms as they should be matched against offer titles: normalized,
    blank entries dropped, and a leading category word ("włóczka") removed when
    the term names a model beyond it."""
    terms: list[str] = []
    for name in names:
        toks = product_tokens(name)
        while len(toks) > 1 and any(toks[0].startswith(p) for p in _PRODUCT_CATEGORY_PREFIXES):
            toks = toks[1:]
        if toks:
            terms.append(" ".join(toks))
    return terms


def match_product_term(offer_name: str, terms: list[str]) -> str | None:
    """Which of `terms` this offer title belongs to — the most specific one.

    Returns the matching term as the caller passed it (so it can be echoed back
    in the seller's own words), or None when the title matches none of them.
    """
    name_toks = product_tokens(offer_name)
    best: str | None = None
    best_len = 0
    for term in terms:
        term_toks = product_tokens(term)
        if len(term_toks) > best_len and _contains_run(name_toks, term_toks):
            best, best_len = term, len(term_toks)
    return best
