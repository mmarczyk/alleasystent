"""Tool definitions for the Allegro agent (Anthropic tool schema format)."""

ALLEGRO_TOOLS: list[dict] = [
    {
        "name": "get_orders",
        "description": (
            "List recent orders from Allegro. "
            "For new/pending orders use status=READY_FOR_PROCESSING, fulfillment_status=NEW, "
            "line_items_sent=[NONE]. "
            "Returns order IDs, buyer info, status, and totals."
        ),
        "input_schema": {
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
                    "items": {
                        "type": "string",
                        "enum": ["NONE", "SOME", "ALL"],
                    },
                    "description": (
                        "Filter by how many line items have been shipped. "
                        "Multiple values allowed (OR logic). "
                        "E.g. [\"SOME\", \"ALL\"] for partially or fully shipped orders."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max orders to return (1–50).",
                    "default": 10,
                },
            },
        },
    },
    {
        "name": "get_order_details",
        "description": (
            "Get full details of a specific Allegro order: items, buyer address, "
            "delivery info, and payment status."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Allegro order (checkout form) UUID.",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "get_active_offers",
        "description": "List the store's active Allegro offers with prices and stock levels.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Optional: filter by offer name (partial match).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max offers to return (1–50).",
                    "default": 10,
                },
            },
        },
    },
    {
        "name": "get_offer_details",
        "description": "Get full details of a specific offer by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "offer_id": {"type": "string", "description": "Allegro offer ID."}
            },
            "required": ["offer_id"],
        },
    },
    {
        "name": "update_offer_price",
        "description": "Update the price of an Allegro offer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "offer_id": {"type": "string", "description": "Allegro offer ID."},
                "price": {
                    "type": "number",
                    "description": "New price in PLN (must be > 0).",
                },
            },
            "required": ["offer_id", "price"],
        },
    },
    {
        "name": "update_offer_stock",
        "description": "Update available stock quantity for an Allegro offer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "offer_id": {"type": "string", "description": "Allegro offer ID."},
                "available": {
                    "type": "integer",
                    "description": "New available quantity (>= 0).",
                },
            },
            "required": ["offer_id", "available"],
        },
    },
    {
        "name": "send_message_to_buyer",
        "description": (
            "Send a message to a buyer via Allegro messaging "
            "(reply to an existing thread)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "Allegro messaging thread ID.",
                },
                "text": {
                    "type": "string",
                    "description": "Message text to send to the buyer.",
                },
            },
            "required": ["thread_id", "text"],
        },
    },
    {
        "name": "get_message_threads",
        "description": "List recent buyer message threads with unread status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max threads to return (1–50).",
                    "default": 10,
                }
            },
        },
    },
    {
        "name": "get_account_info",
        "description": "Get basic information about the seller's Allegro account.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_billing_summary",
        "description": "Get recent billing/fee entries from Allegro.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max entries to return.",
                    "default": 10,
                }
            },
        },
    },
]
