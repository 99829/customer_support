
"""
STEP 1: Classifier Agent
=========================
Takes raw ticket text -> returns structured category + urgency.

Concept: Structured Output
--------------------------
Instead of asking the LLM to freely describe the ticket (which gives
inconsistent text like "payment issue" vs "Payment" vs "billing problem"),
we force it to pick from a FIXED set of values using a Pydantic schema.
This makes the output reliable enough for our code to branch on (if/else
logic later) instead of trying to parse messy free text.
"""

import os
from enum import Enum
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# SCHEMA DEFINITION
# ---------------------------------------------------------------------------
class Category(str, Enum):
    ORDER_STATUS = "Order Status"
    PAYMENT = "Payment"
    RETURNS_REFUND = "Returns/Refund"
    DELIVERY_PROBLEM = "Delivery Problem"
    CANCELLATION = "Cancellation"
    ACCOUNT_ADDRESS = "Account/Address"
    PRODUCT_INQUIRY = "Product Inquiry"
    GENERAL = "General Inquiry"


class Urgency(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class TicketClassification(BaseModel):
    """The structured output we force the LLM to produce."""
    category: Category = Field(description="The single best-fitting category for this ticket")
    urgency: Urgency = Field(description="How urgently this ticket needs attention")
    order_id: str | None = Field(
        default=None,
        description="Order ID mentioned in the ticket, if any (e.g. 'ORD12345'). Null if none mentioned."
    )
    reasoning: str = Field(description="One-sentence explanation for the category and urgency chosen")


# ---------------------------------------------------------------------------
# CLASSIFIER AGENT
# ---------------------------------------------------------------------------
CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a support ticket classifier for an e-commerce company.
Analyze the customer's ticket and classify it accurately.

Category clarification (based on eval error analysis):
- "Payment" is ONLY for transaction issues: failed payments, double charges,
  money deducted but order failed, refund not received. If the customer is
  asking a general PRE-PURCHASE question like "what payment methods do you
  accept" with no actual transaction problem, that is "General Inquiry", not
  "Payment".

Urgency guidelines:
- Critical: security issues, fraud, account compromise
- High: payment problems with money deducted, damaged/wrong items, angry/frustrated tone
- Medium: delivery delays, order modification requests
- Low: general questions, policy inquiries

Extract the order ID only if explicitly mentioned (format like ORD12345)."""),
    ("human", "{ticket_text}")
])


def classify_ticket(ticket_text: str, api_key: str) -> TicketClassification:
    """
    Classifies a support ticket into category + urgency using structured output.

    Why .with_structured_output():
    This tells the LLM provider to constrain its response to match our
    Pydantic schema exactly - no need to manually parse JSON or handle
    the LLM saying something slightly different each time.
    """
    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=api_key, temperature=0)
    structured_llm = llm.with_structured_output(TicketClassification)

    chain = CLASSIFICATION_PROMPT | structured_llm
    result = chain.invoke({"ticket_text": ticket_text})
    return result


# ---------------------------------------------------------------------------
# CLI TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY in a .env file first. Get a free key at https://console.groq.com")
        exit(1)

    test_tickets = [
        "My order ORD12345 hasn't arrived yet, it's already 3 days late",
        "Paid Rs 499 for a case but order ORD12346 shows Payment Failed, money got deducted",
        "URGENT - I think someone accessed my account without permission",
        "Can you tell me the return window for electronics?",
    ]

    for ticket in test_tickets:
        print(f"\nTicket: {ticket}")
        result = classify_ticket(ticket, api_key)
        print(f"  Category: {result.category.value}")
        print(f"  Urgency: {result.urgency.value}")
        print(f"  Order ID: {result.order_id}")
        print(f"  Reasoning: {result.reasoning}")