"""
STEP 7: LangGraph Orchestration
==================================
Rewires Steps 1-5 (classify -> tool-call -> RAG -> generate -> route) as an
explicit stateful graph instead of a plain sequential function-call chain.

Concept: Why a graph instead of plain function calls?
---------------------------------------------------------
Our earlier code worked fine for a straight-line pipeline. But real agent
flows often need:
  - Conditional branching (skip steps based on a decision)
  - Shared state across steps (without passing 5 parameters everywhere)
  - Easy visualization/debugging of the flow
  - A natural place to later add loops (e.g. "retry generation if
    confidence too low") without the code turning into nested if-else soup

LangGraph models the agent as:
  - Nodes: each processing step (functions that read/update shared state)
  - Edges: connections between nodes (can be conditional)
  - State: one shared dict-like object every node can read and update

Optimization built into this graph: if a ticket is Critical urgency, we
skip the response-generation LLM call entirely (since Rule 1 in routing
escalates Critical tickets regardless of the drafted response) - this
saves one LLM call and some latency for the highest-urgency tickets.
"""

import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from classifier import classify_ticket, TicketClassification, Urgency
from order_tool import get_order_details, format_order_for_context
from rag_retrival import retrieve_policy_context, build_knowledge_base
from response_generator import generate_response_from_context, GeneratedResponse
from routing_logic import decide_routing, RoutingDecision


# ---------------------------------------------------------------------------
# STATE SCHEMA
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """
    Shared state that flows through every node. Each node reads what it
    needs and returns a dict of fields to update - LangGraph merges that
    into the overall state automatically.
    """
    ticket_text: str
    api_key: str
    vector_store: object  # cached FAISS store, passed in once at invocation

    classification: Optional[TicketClassification]
    order_context: Optional[str]
    policy_context: Optional[str]
    response: Optional[GeneratedResponse]
    routing_decision: Optional[RoutingDecision]


# ---------------------------------------------------------------------------
# NODES (each one wraps a function we already built in Steps 1-5)
# ---------------------------------------------------------------------------
def classify_node(state: AgentState) -> dict:
    classification = classify_ticket(state["ticket_text"], state["api_key"])
    return {"classification": classification}


def tool_call_node(state: AgentState) -> dict:
    order_result = get_order_details(state["classification"].order_id)
    return {"order_context": format_order_for_context(order_result)}


def rag_node(state: AgentState) -> dict:
    context = retrieve_policy_context(state["ticket_text"], vector_store=state["vector_store"])
    return {"policy_context": context}


def generate_node(state: AgentState) -> dict:
    response = generate_response_from_context(
        ticket_text=state["ticket_text"],
        classification=state["classification"],
        order_context=state["order_context"],
        policy_context=state["policy_context"],
        api_key=state["api_key"],
    )
    return {"response": response}


def escalate_shortcut_node(state: AgentState) -> dict:
    """
    Used ONLY for Critical tickets. Skips the generation LLM call entirely
    since we already know (Rule 1 in routing_logic) that Critical tickets
    always escalate - drafting a response would be wasted cost/latency.
    """
    placeholder = GeneratedResponse(
        response_text="[No draft generated - critical ticket routed directly to a human agent]",
        confidence=0,
    )
    return {"response": placeholder}


def route_node(state: AgentState) -> dict:
    decision = decide_routing(state["classification"], state["response"])
    return {"routing_decision": decision}


# ---------------------------------------------------------------------------
# CONDITIONAL EDGE LOGIC
# ---------------------------------------------------------------------------
def check_urgency(state: AgentState) -> str:
    """
    Decides which path to take after classification.
    Returns the NAME of the next node (LangGraph uses this string to
    pick the edge).
    """
    if state["classification"].urgency == Urgency.CRITICAL:
        return "escalate_shortcut"
    return "tool_call"


# ---------------------------------------------------------------------------
# BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("classify", classify_node)
    graph.add_node("tool_call", tool_call_node)
    graph.add_node("rag", rag_node)
    graph.add_node("generate", generate_node)
    graph.add_node("escalate_shortcut", escalate_shortcut_node)
    graph.add_node("route", route_node)

    # Entry point
    graph.set_entry_point("classify")

    # Conditional branch: Critical -> shortcut, else -> normal pipeline
    graph.add_conditional_edges(
        "classify",
        check_urgency,
        {
            "escalate_shortcut": "escalate_shortcut",
            "tool_call": "tool_call",
        },
    )

    # Normal pipeline edges (sequential)
    graph.add_edge("tool_call", "rag")
    graph.add_edge("rag", "generate")
    graph.add_edge("generate", "route")

    # Shortcut path also ends at routing
    graph.add_edge("escalate_shortcut", "route")

    # Routing is always the last step
    graph.add_edge("route", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# CLI TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY in a .env file first.")
        exit(1)

    print("Loading knowledge base...")
    vs = build_knowledge_base()

    app = build_graph()
    print(app.get_graph().draw_ascii())
    print(app.get_graph().draw_mermaid())

    test_tickets = [
        "My order ORD12345 hasn't arrived yet, it's already 3 days late",
        "URGENT - I think someone accessed my account without my permission and changed my password",
    ]

    for ticket in test_tickets:
        print(f"\n{'='*70}")
        print(f"TICKET: {ticket}")
        print('='*70)

        initial_state = {
            "ticket_text": ticket,
            "api_key": api_key,
            "vector_store": vs,
            "classification": None,
            "order_context": None,
            "policy_context": None,
            "response": None,
            "routing_decision": None,
        }

        final_state = app.invoke(initial_state)

        cls = final_state["classification"]
        resp = final_state["response"]
        decision = final_state["routing_decision"]

        print(f"Category: {cls.category.value} | Urgency: {cls.urgency.value}")
        print(f"Confidence: {resp.confidence}/10")
        print(f"Routing: {decision.action.value}")
        print(f"Reason: {decision.reason}")
        print(f"\nResponse:\n{resp.response_text}")