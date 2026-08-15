"""Tool definitions for the Allegro agent (OpenAI/Gemini function-calling format)."""

ALLEGRO_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_new_orders",
            "description": (
                "List new (unprocessed) orders waiting to be fulfilled. "
                "Use for any question about 'nowe zamówienia', 'nowe', 'oczekujące', "
                "'co nowego', 'jakie zamówienia mam', 'ile zamówień', new/pending orders. "
                "Automatically filters for READY_FOR_PROCESSING + fulfillment=NEW orders. "
                "Results are sorted newest-first. "
                "SINGULAR vs PLURAL: for 'ostatnie zamówienie' / 'ostatnie nowe zamówienie' / "
                "'najnowsze zamówienie' / 'last order' (singular — asking about ONE order), "
                "set limit=1. Only use the default (all orders) for plural phrasing like "
                "'nowe zamówienia' / 'ostatnie zamówienia' / 'jakie zamówienia mam'. "
                "COUNT-ONLY: for 'ile zamówień', 'ile mam nowych', 'ile jest wszystkich nowych', "
                "'liczba nowych zamówień' (the user wants a NUMBER, not the order details) — set "
                "count_only=true. Do NOT set count_only when the user also wants to see the orders "
                "themselves (e.g. 'pokaż nowe zamówienia', 'jakie zamówienia mam' with no 'ile'). "
                "Returns order IDs, buyer info, fulfillment status, and totals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "buyer_login": {
                        "type": "string",
                        "description": "Optionally filter by buyer's Allegro login.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Max orders to return (1–100). Set to 1 when the user asks about "
                            "THE LAST/newest order in the singular ('ostatnie zamówienie', "
                            "'ostatnie nowe zamówienie'). Leave at default for plural questions."
                        ),
                        "default": 100,
                    },
                    "count_only": {
                        "type": "boolean",
                        "description": (
                            "Set true when the user only wants the NUMBER of new orders "
                            "('ile zamówień', 'ile mam nowych', 'liczba nowych zamówień') — the "
                            "reply will state just the count, not list individual orders."
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
            "name": "get_orders",
            "description": (
                "List Allegro orders with arbitrary filters. "
                "USE THIS for a plain LIST of orders, incl. for a date range — 'lista zamówień', "
                "'pokaż wszystkie zamówienia z tego miesiąca/tygodnia', 'zamówienia z okresu X'. "
                "Do NOT use get_sales_summary for these — that tool is only for earnings/profit/fee "
                "questions ('ile zarobiłem', 'jakie opłaty'), not for listing orders. "
                "For new/pending orders use get_new_orders instead. "
                "TIME FILTERS: use bought_after/before_local for order PLACEMENT time; "
                "use paid_after/before_local for PAYMENT time ('opłacone po X', 'zapłacone po X'). "
                "NEVER use this tool when the user names or already gave a SPECIFIC order_id "
                "(a UUID, e.g. '0c4854a0-9646-11f1-8028-338c43adc37a') — this tool has NO order_id "
                "parameter and cannot filter to a single order; it will silently ignore the ID and "
                "return an unrelated list of unrelated orders. Use get_order_details instead for any "
                "question (status, contents, invoice, cost) about one already-identified order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by order status.",
                        "enum": ["BOUGHT", "FILLED_IN", "READY_FOR_PROCESSING", "CANCELLED"],
                    },
                    "buyer_login": {
                        "type": "string",
                        "description": "Filter orders by buyer's Allegro login.",
                    },
                    "fulfillment_status": {
                        "type": "string",
                        "description": "Filter by fulfillment status.",
                        "enum": ["NEW", "PROCESSING", "READY_FOR_SHIPMENT", "SENT", "PICKED_UP", "CANCELLED", "SUSPENDED"],
                    },
                    "line_items_sent": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["NONE", "SOME", "ALL"]},
                        "description": "Filter by shipment state. Multiple values allowed (OR logic).",
                    },
                    "bought_after_local": {
                        "type": "string",
                        "description": (
                            "Return only orders placed AT OR AFTER this local Polish time. "
                            "Format: 'HH:MM' for today (e.g. '12:00'), "
                            "or 'YYYY-MM-DD HH:MM' for a specific date (e.g. '2026-06-16 18:00'). "
                            "Timezone conversion to UTC is handled automatically."
                        ),
                    },
                    "bought_before_local": {
                        "type": "string",
                        "description": (
                            "Return only orders placed AT OR BEFORE this local Polish time. "
                            "Same format as bought_after_local: 'HH:MM' or 'YYYY-MM-DD HH:MM'."
                        ),
                    },
                    "paid_after_local": {
                        "type": "string",
                        "description": (
                            "Return only orders whose PAYMENT was completed AT OR AFTER this "
                            "local Polish time. Use for queries with 'opłacone', 'zapłacone'. "
                            "Format: 'HH:MM' for today or 'YYYY-MM-DD HH:MM' for a specific date."
                        ),
                    },
                    "paid_before_local": {
                        "type": "string",
                        "description": (
                            "Return only orders whose PAYMENT was completed AT OR BEFORE this "
                            "local Polish time. Same format as paid_after_local."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max orders to return (1–100).",
                        "default": 50,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": (
                "Get full details of a specific Allegro order: status, items, buyer address, "
                "delivery info, payment status, AND all Allegro billing entries for that order "
                "(individual commission per item, delivery fees, any credits). "
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
                "get_message_threads first."
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
                "Show which courier / delivery method the buyer chose for each order. "
                "Use whenever the user asks: which couriers are in pending orders, "
                "which delivery methods were selected, tracking numbers, or any question "
                "combining orders with shipping/courier/delivery. "
                "Default (no filters): returns orders with fulfillment_status=READY_FOR_SHIPMENT "
                "(packed and awaiting carrier handoff). "
                "For 'orders to send' / 'do wysłania' leave fulfillment_status empty."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Order status filter. Default: READY_FOR_PROCESSING.",
                        "enum": ["BOUGHT", "FILLED_IN", "READY_FOR_PROCESSING", "CANCELLED"],
                    },
                    "fulfillment_status": {
                        "type": "string",
                        "description": (
                            "Fulfillment status filter. Leave empty to get READY_FOR_SHIPMENT orders. "
                            "Use SENT only when explicitly asking about already-shipped orders."
                        ),
                        "enum": ["NEW", "PROCESSING", "READY_FOR_SHIPMENT", "SENT", "PICKED_UP", "CANCELLED", "SUSPENDED"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max orders to return (1–50).",
                        "default": 50,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_orders_pending_invoice",
            "description": (
                "Find all paid orders for a given month where the buyer requested a VAT invoice "
                "but the seller has not yet uploaded one. Defaults to the current month. "
                "Use when asked about missing invoices or invoice obligations."
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
                "products, breakdown of fee types (commission, listing, etc.), AND a per-order table showing "
                "revenue + Allegro cost + revenue-after-fees for each individual order. "
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
                "'ostatni miesiąc/last month' → first day of previous calendar month to last day of that month. "
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
            "name": "suggest_invoice_monitoring",
            "description": (
                "Present the user with a button to enable automatic VAT invoice monitoring. "
                "Call this AFTER get_orders_pending_invoice when the user asks about invoices, "
                "missing invoices, or wants to be notified when new orders require a VAT invoice. "
                "Do NOT call multiple times in one conversation."
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
                "Returns a share link for manual review PLUS the invoice_uuid needed for the follow-up "
                "delivery tools (attach_invoice_to_allegro_order, send_invoice_to_ksef)."
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
                "Requires BOTH the Allegro order_id and the inFakt invoice_uuid returned by an earlier "
                "issue_invoice_for_order call in this conversation — never guess either ID; ask or look "
                "it up if missing. Allegro allows only ONE PDF invoice per order — calling this twice "
                "for the same order will fail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Allegro order (checkout form) UUID."},
                    "invoice_uuid": {"type": "string", "description": "inFakt invoice UUID from issue_invoice_for_order."},
                },
                "required": ["order_id", "invoice_uuid"],
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
                "Typically relevant for company (B2B) buyers; don't call it for a private-person buyer "
                "unless the user explicitly asks for it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_uuid": {"type": "string", "description": "inFakt invoice UUID from issue_invoice_for_order."},
                },
                "required": ["invoice_uuid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disable_invoice_monitoring",
            "description": (
                "Show a button to disable automatic VAT invoice monitoring in the browser. "
                "Call when the user asks to turn off, stop, or disable invoice monitoring/notifications."
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
    # Zamówienia
    "get_new_orders": "chat",
    "get_orders": "table",
    "get_order_details": "document",
    "get_orders_delivery": "table",
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
    # Faktury
    "get_orders_pending_invoice": "chat",
    "get_order_invoice_data": "chat",
    "preview_pending_invoices": "action",
    "issue_invoice_for_order": "action",
    "attach_invoice_to_allegro_order": "action",
    "send_invoice_to_ksef": "action",
    # Monitoring (przyciski w UI)
    "suggest_order_monitoring": "action",
    "suggest_invoice_monitoring": "action",
    "suggest_message_monitoring": "action",
    "disable_order_monitoring": "action",
    "disable_invoice_monitoring": "action",
    "disable_message_monitoring": "action",
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
