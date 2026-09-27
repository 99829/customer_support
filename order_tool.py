"""
STEP 2: Tool-Calling Layer
===========================
Fetches real order data given an order_id extracted by the classifier.

Concept: Tool Use / Function Calling
--------------------------------------
This is what separates an "agent" from a plain chatbot. A chatbot can only
generate text from what it already knows (or hallucinate). An agent can
call external functions/APIs/databases to fetch REAL, current data and
ground its response in facts.

In a real company, get_order_details() would query an actual database
(PostgreSQL/MongoDB) or call an internal REST API. Here, orders.json
simulates that backend - same concept, smaller scale.
"""

import json
import os
from typing import Optional


ORDERS_FILE = os.path.join(os.path.dirname(__file__), "data", "orders.json")


def _load_orders() -> dict:
    """Loads the mock order database from disk."""
    with open(ORDERS_FILE, "r") as f:
        return json.load(f)


def get_order_details(order_id: Optional[str]) -> dict:
    """
    Fetches order details for a given order_id.

    This is the 'tool' the agent calls. It always returns a dict with a
    'found' key so the calling code can branch cleanly - no exceptions
    to catch, no None checks scattered everywhere.

    Args:
        order_id: e.g. "ORD12345". Can be None if the classifier didn't
                  extract one from the ticket.

    Returns:
        dict with:
          - found (bool): whether the order exists
          - data (dict | None): the order details if found
          - message (str): human-readable status, useful for logging/debugging
    """
    if not order_id:
        return {
            "found": False,
            "data": None,
            "message": "No order ID was provided or extracted from the ticket."
        }

    orders = _load_orders()
    order_id = order_id.strip().upper()  # normalize input (e.g. "ord12345" -> "ORD12345")

    if order_id not in orders:
        return {
            "found": False,
            "data": None,
            "message": f"Order ID '{order_id}' was not found in our records."
        }

    return {
        "found": True,
        "data": orders[order_id],
        "message": f"Order '{order_id}' found successfully."
    }


def format_order_for_context(order_result: dict) -> str:
    """
    Converts the tool's dict output into a clean text summary that can be
    injected into an LLM prompt later (Step 4: Response Generation).

    Why this matters: LLMs work better with clean, readable context rather
    than raw JSON/dicts dumped into a prompt.
    """
    if not order_result["found"]:
        return order_result["message"]

    d = order_result["data"]
    delivery_line = f"Expected delivery: {d['expected_delivery']}" if d["expected_delivery"] else "No delivery date (order not shipped)"

    return (
        f"Order Status: {d['status']}\n"
        f"Product: {d['product']}\n"
        f"Order Date: {d['order_date']}\n"
        f"{delivery_line}\n"
        f"Payment Status: {d['payment_status']} (via {d['payment_method']})\n"
        f"Amount: Rs. {d['amount']}"
    )


# ---------------------------------------------------------------------------
# CLI TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_order_ids = ["ORD12345", "ord12346", "ORD99999", None]

    for oid in test_order_ids:
        print(f"\n--- Looking up: {oid} ---")
        result = get_order_details(oid)
        print(f"Found: {result['found']}")
        print(f"Message: {result['message']}")
        if result["found"]:
            print("\nFormatted for LLM context:")
            print(format_order_for_context(result))