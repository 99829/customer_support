"""
STEP 5: Decision/Escalation Logic
====================================
Takes the classification + generated response and decides the final
routing action: auto-respond, flag for human review, or escalate.

Concept: Graduated Autonomy (Risk-Based Routing)
---------------------------------------------------
A production support agent should NOT have the same level of autonomy
for every ticket. A "how do I track my order" question is low-risk and
safe to auto-answer. A security or payment dispute is higher-risk and
should have a human checkpoint before anything is sent to the customer.

This module encodes that risk-based decision-making as explicit,
readable rules - not buried inside a single giant LLM prompt - so the
logic is auditable and easy to explain/justify in an interview.
"""

from enum import Enum
from pydantic import BaseModel

from classifier import TicketClassification, Category, Urgency
from response_generator import GeneratedResponse


# ---------------------------------------------------------------------------
# ROUTING DECISION TYPES
# ---------------------------------------------------------------------------
class RoutingAction(str, Enum):
    AUTO_RESPOND = "Auto-Respond"          # send directly to customer, no human needed
    FLAG_FOR_REVIEW = "Flag for Review"    # draft ready, human approves before sending
    ESCALATE = "Escalate to Human"         # too risky/uncertain for AI to draft alone


class RoutingDecision(BaseModel):
    action: RoutingAction
    reason: str  # human-readable explanation of WHY this path was chosen


# ---------------------------------------------------------------------------
# SENSITIVE CATEGORIES (get extra caution regardless of confidence)
# ---------------------------------------------------------------------------
SENSITIVE_CATEGORIES = {Category.PAYMENT, Category.RETURNS_REFUND, Category.CANCELLATION}

# Confidence thresholds
MIN_CONFIDENCE_TO_RESPOND = 6   # below this -> always escalate, AI itself is unsure
HIGH_CONFIDENCE_THRESHOLD = 8   # sensitive categories need this much confidence to auto-respond


# ---------------------------------------------------------------------------
# ROUTING LOGIC
# ---------------------------------------------------------------------------
def decide_routing(
    classification: TicketClassification,
    response: GeneratedResponse,
) -> RoutingDecision:
    """
    Applies risk-based rules to decide what happens to this ticket next.

    Rules are checked in order of severity - most dangerous condition
    first, so a Critical+low-confidence ticket doesn't accidentally
    fall through to a less strict rule below it.
    """

    # Rule 1: Critical urgency ALWAYS goes to a human, no exceptions.
    # Security/fraud tickets are never something an AI should decide alone.
    if classification.urgency == Urgency.CRITICAL:
        return RoutingDecision(
            action=RoutingAction.ESCALATE,
            reason="Critical urgency tickets (security/fraud related) always require "
                   "immediate human attention, regardless of AI confidence."
        )

    # Rule 2: If the AI itself is not confident, don't let it act autonomously.
    if response.confidence < MIN_CONFIDENCE_TO_RESPOND:
        return RoutingDecision(
            action=RoutingAction.ESCALATE,
            reason=f"AI confidence ({response.confidence}/10) is below the minimum threshold "
                   f"({MIN_CONFIDENCE_TO_RESPOND}) - likely missing information or an ambiguous ticket."
        )

    # Rule 3: Sensitive categories (money-related) need a higher confidence bar
    # AND still get a human check before sending, even if confidence is decent.
    if classification.category in SENSITIVE_CATEGORIES:
        if response.confidence < HIGH_CONFIDENCE_THRESHOLD:
            return RoutingDecision(
                action=RoutingAction.ESCALATE,
                reason=f"'{classification.category.value}' is a sensitive category and confidence "
                       f"({response.confidence}/10) didn't meet the higher bar "
                       f"({HIGH_CONFIDENCE_THRESHOLD}) required for it."
            )
        return RoutingDecision(
            action=RoutingAction.FLAG_FOR_REVIEW,
            reason=f"'{classification.category.value}' involves money/refunds - draft is ready "
                   f"and confident, but a human approves before it's sent, as a safety net."
        )

    # Rule 4: High urgency (non-critical) still gets a human glance before sending.
    if classification.urgency == Urgency.HIGH:
        return RoutingDecision(
            action=RoutingAction.FLAG_FOR_REVIEW,
            reason="High urgency ticket - draft is ready but flagged for a quick human check "
                   "before sending, given the urgency."
        )

    # Rule 5: Everything else (Low/Medium urgency, non-sensitive category,
    # good confidence) is safe to auto-respond.
    return RoutingDecision(
        action=RoutingAction.AUTO_RESPOND,
        reason=f"Low-risk ticket ({classification.category.value}, {classification.urgency.value} "
               f"urgency) with high confidence ({response.confidence}/10) - safe to send directly."
    )


# ---------------------------------------------------------------------------
# CLI TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from classifier import classify_ticket
    from response_generator import generate_response, build_knowledge_base

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
        "URGENT - I think someone accessed my account without my permission and changed my password",
        "Can you tell me the return window for electronics?",
        "I want a refund for order ORD99999 immediately, this is unacceptable",
    ]

    for ticket in test_tickets:
        print(f"\n{'='*70}")
        print(f"TICKET: {ticket}")
        print('='*70)

        classification = classify_ticket(ticket, api_key)
        print(f"Category: {classification.category.value} | Urgency: {classification.urgency.value}")

        response = generate_response(ticket, classification, api_key, vector_store=vs)
        print(f"Confidence: {response.confidence}/10")

        decision = decide_routing(classification, response)
        print(f"\n>>> ROUTING DECISION: {decision.action.value}")
        print(f">>> REASON: {decision.reason}")

        if decision.action != RoutingAction.ESCALATE:
            print(f"\nDraft response:\n{response.response_text}")