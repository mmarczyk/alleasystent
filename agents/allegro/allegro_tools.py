"""Tool definitions for the Allegro agent (OpenAI/Gemini function-calling format)."""

ALLEGRO_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_orders",
            "description": (
                "List recent orders from Allegro. "
                "For new/pending orders use status=READY_FOR_PROCESSING, fulfillment_status=NEW, "
                "line_items_sent=[NONE]. "
                "Returns order IDs, buyer info, status, and totals."
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
                        "description": "Max orders to return (1–50).",
                        "default": 10,
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
                "delivery info, and payment status."
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
            "description": "List the store's active Allegro offers with prices and stock levels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Optional: filter by offer name (partial match)."},
                    "limit": {"type": "integer", "description": "Max offers to return (1–50).", "default": 10},
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
            "description": "Get basic information about the seller's Allegro account.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_billing_summary",
            "description": "Get recent billing/fee entries from Allegro.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max entries to return.", "default": 10},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_orders_delivery",
            "description": (
                "List orders with their delivery method and tracking info. "
                "Use when the user asks about delivery methods, carriers, or tracking numbers across orders."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by order status (default: READY_FOR_PROCESSING).",
                        "enum": ["BOUGHT", "FILLED_IN", "READY_FOR_PROCESSING", "CANCELLED"],
                    },
                    "fulfillment_status": {
                        "type": "string",
                        "description": "Filter by fulfillment status.",
                        "enum": ["NEW", "PROCESSING", "READY_FOR_SHIPMENT", "SENT", "PICKED_UP", "CANCELLED", "SUSPENDED"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max orders to return (1–50).",
                        "default": 20,
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
]
