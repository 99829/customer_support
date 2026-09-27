"""
STEP 4: Response Generation
=============================
Combines everything from Steps 1-3 into one grounded customer response:
  - Classification (category, urgency) -> tells us HOW to respond
  - Order details (if order_id found)   -> gives REAL facts about their order
  - Policy context (RAG)                -> gives REAL company rules

Concept: Grounded Generation
------------------------------
The LLM is NOT asked to "write a helpful response" in isolation - that's
how hallucination happens. Instead, we hand it ONLY verified facts (order
data + policy text) and instruct it to use ONLY those facts. This is the
same anti-hallucination principle from our RAG chatbot project, now
extended with real-time order data too.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from classifier import TicketClassification
from order_tool import get_order_details, format_order_for_context
from rag_retrival import retrieve_policy_context, build_knowledge_base


# ---------------------------------------------------------------------------
# OUTPUT SCHEMA
# ---------------------------------------------------------------------------
class GeneratedResponse(BaseModel):
    """Structured output so we always get a response + a confidence score together."""
    response_text: str = Field(description="The draft reply to send to the customer")
    confidence: int = Field(
        description="1-10 score: how confident are you this response fully resolves the issue "
                    "using ONLY the provided facts. Low if facts were missing or ticket was ambiguous."
    )


# ---------------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------------
RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a customer support agent for an e-commerce company.
Draft a helpful, empathetic reply to the customer's ticket.

STRICT RULES:
- Use ONLY the facts given below (order details + policy information).
- Do NOT invent order statuses, dates, or policy terms not shown here.
- If the order details say "not found", acknowledge that and ask the
  customer to double check their order ID - do not guess their order status.
- Keep the tone empathetic but professional. Keep it concise (3-5 sentences).
- Rate your own confidence honestly - if information is missing or the
  ticket is ambiguous, give a LOW confidence score."""),
    ("human", """Ticket category: {category}
Ticket urgency: {urgency}

Customer's message:
{ticket_text}

Order details (from our system):
{order_context}

Relevant company policy:
{policy_context}

Draft a response using only the above facts.""")
])


# ---------------------------------------------------------------------------
# MAIN GENERATION FUNCTION
# ---------------------------------------------------------------------------
def generate_response_from_context(
    ticket_text: str,
    classification: TicketClassification,
    order_context: str,
    policy_context: str,
    api_key: str,
) -> GeneratedResponse:
    """Pure generation step - assumes context already fetched."""
    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=api_key, temperature=0.3)
    structured_llm = llm.with_structured_output(GeneratedResponse)
    chain = RESPONSE_PROMPT | structured_llm

    return chain.invoke({
        "category": classification.category.value,
        "urgency": classification.urgency.value,
        "ticket_text": ticket_text,
        "order_context": order_context,
        "policy_context": policy_context,
    })


def generate_response(
    ticket_text: str,
    classification: TicketClassification,
    api_key: str,
    vector_store=None,
) -> GeneratedResponse:
    """Bundled version - fetches context itself, then calls the pure function above."""
    order_result = get_order_details(classification.order_id)
    order_context = format_order_for_context(order_result)

    if vector_store is None:
        vector_store = build_knowledge_base()
    policy_context = retrieve_policy_context(ticket_text, vector_store=vector_store)

    return generate_response_from_context(
        ticket_text, classification, order_context, policy_context, api_key
    )
# ---------------------------------------------------------------------------
# CLI TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    from classifier import classify_ticket

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY in a .env file first.")
        exit(1)

    print("Loading knowledge base...")
    vs = build_knowledge_base()

    test_tickets = [
        "My order ORD12345 hasn't arrived yet, it's already 3 days late",
        "Paid Rs 499 for a case but order ORD12346 shows Payment Failed, money got deducted",
    ]

    for ticket in test_tickets:
        print(f"\n{'='*60}")
        print(f"TICKET: {ticket}")
        print('='*60)

        classification = classify_ticket(ticket, api_key)
        print(f"Category: {classification.category.value} | Urgency: {classification.urgency.value}")

        response = generate_response(ticket, classification, api_key, vector_store=vs)
        print(f"\nDraft Response:\n{response.response_text}")
        print(f"\nConfidence: {response.confidence}/10")