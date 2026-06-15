"""Tool definitions for the Allegro agent (OpenAI/Gemini function-calling format)."""

ALLEGRO_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_orders",
            "description": (
                "List orders from Allegro. "
                "MANDATORY for 'nowe zamówienia' / new/pending orders: "
                "status=READY_FOR_PROCESSING AND fulfillment_status=NEW AND line_items_sent=[NONE]. "
                "NEVER omit fulfillment_status=NEW when the user asks about new orders. "
                "Always use limit=50 or higher unless user explicitly asks for fewer. "
                "Returns order IDs, buyer info, fulfillment status, and totals."
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
                    "limit": {
                        "type": "integer",
                        "description": "Max orders to return (1–100). Default 50 — always fetch at least 50 to avoid missing orders.",
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
                "Get full details of a specific Allegro order: items, buyer address, "
                "delivery info, payment status, AND all Allegro billing entries for that order "
                "(individual commission per item, delivery fees, any credits). "
                "USE THIS for any question about costs/fees/profit of a SPECIFIC order — "
                "e.g. 'jakie koszty miałem przy tym zamówieniu', 'podaj wpisy billing dla zamówienia X', "
                "'ile prowizji zapłaciłem za to zamówienie'. "
                "Uses order.id filter so results are exact — never mixes entries from other orders."
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
                "List ALL active Allegro offers (paginated, no limit). "
                "Use for general 'show me my offers' questions. "
                "For stock or price filtering use query_offers_by_stock / query_offers_by_price instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Optional: filter by offer name (partial match)."},
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
                "Filter active offers by stock quantity. "
                "Offers with the same name are aggregated — stock is summed across all listings of the same product. "
                "Use for questions like 'offers with less than 10 items', 'out of stock offers', 'high stock offers'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_stock": {
                        "type": "integer",
                        "description": "Return products with total stock ≤ this value (inclusive).",
                    },
                    "min_stock": {
                        "type": "integer",
                        "description": "Return products with total stock ≥ this value (inclusive).",
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
            "description": "List recent buyer message threads with unread status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max threads to return (1–50).", "default": 10},
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
                "When a period is given, pass date_from/date_to. Without dates, returns recent entries. "
                "Returns: total fees, refunds/credits, net cost, breakdown by fee type, individual entries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_from": {
                        "type": "string",
                        "description": "Start of period in UTC ISO 8601, e.g. '2026-06-01T00:00:00Z'. Optional.",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End of period in UTC ISO 8601, e.g. '2026-06-30T23:59:59Z'. Optional.",
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
                "Default (no filters): returns all unsent orders (status=READY_FOR_PROCESSING, "
                "fulfillment_status not SENT/PICKED_UP). "
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
                            "Fulfillment status filter. Leave empty to get all unsent orders. "
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
                "Returns: total revenue, Allegro fees, net profit (revenue minus fees), order count, "
                "average order value, top-selling products, breakdown of fee types (commission, listing, etc.), "
                "AND a per-order table showing revenue + Allegro cost + net profit for each individual order. "
                "Uses payment.finishedAt (actual payment date) for order filtering — Allegro operates on UTC. "
                "ALWAYS resolve common time expressions automatically — do NOT ask the user for clarification: "
                "'dziś/today' → today 00:00:00Z–23:59:59Z; "
                "'wczoraj/yesterday' → yesterday 00:00:00Z–23:59:59Z; "
                "'przedwczoraj/day before yesterday' → two days ago 00:00:00Z–23:59:59Z; "
                "'ostatni tydzień/last week/last 7 days' → 7 days ago 00:00:00Z to today 23:59:59Z; "
                "'ten tydzień/this week' → Monday of current week 00:00:00Z to today 23:59:59Z; "
                "'ten miesiąc/this month' → first day of current month 00:00:00Z to last day 23:59:59Z; "
                "'ostatni miesiąc/last month' → first day of previous calendar month 00:00:00Z to last day 23:59:59Z. "
                "Only ask the user if the period is truly ambiguous (e.g. no period mentioned at all). "
                "date_from and date_to must be ISO 8601 UTC strings, e.g. '2026-06-01T00:00:00Z'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_from": {
                        "type": "string",
                        "description": "Start of period in UTC, ISO 8601, e.g. '2026-06-01T00:00:00Z'.",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End of period in UTC, ISO 8601, e.g. '2026-06-30T23:59:59Z'.",
                    },
                },
                "required": ["date_from", "date_to"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_order_monitoring",
            "description": (
                "Present the user with a button to enable automatic background order monitoring. "
                "Call this AFTER get_orders or get_orders_delivery when the user is asking about "
                "new/pending orders or wants to be notified about incoming orders. "
                "Do NOT call multiple times in one conversation — only the first time orders are shown."
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
                "Call when the user asks to turn off, stop, or disable order monitoring/notifications."
            ),
            "parameters": {"type": "object", "properties": {}},
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
]
