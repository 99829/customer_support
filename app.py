"""
Ticket Triage Agent - Streamlit UI
=====================================
Wraps the LangGraph pipeline (graph.py) in a web interface so the full
flow - classification, routing decision, and drafted response - can be
demoed visually instead of only through CLI output.

Run with: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from graph import build_graph
from rag_retrival import build_knowledge_base
from routing_logic import RoutingAction

load_dotenv()

st.set_page_config(page_title="Ticket Triage Agent", page_icon="🎫", layout="centered")
st.title("🎫 E-commerce Support Ticket Triage Agent")
st.caption("Agentic pipeline: Classify → Tool-call → RAG → Generate → Route (built with LangGraph)")

# --- API key ---
with st.sidebar:
    st.header("Setup")
    api_key_input = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get a free key at https://console.groq.com",
    )
    st.markdown("---")
    st.markdown(
        "**Pipeline steps:**\n"
        "1. Classify ticket (category + urgency)\n"
        "2. If Critical → shortcut straight to escalation\n"
        "3. Otherwise: look up order (tool call) → retrieve policy (RAG) → draft response\n"
        "4. Route: Auto-Respond / Flag for Review / Escalate"
    )
    st.markdown("---")
    st.markdown("**Sample order IDs to try:** ORD12345, ORD12346, ORD12350, ORD12352")

if not api_key_input:
    st.warning("Enter your Groq API key in the sidebar to get started (free at console.groq.com).")
    st.stop()


# --- Cache the graph + knowledge base so they don't rebuild every interaction ---
@st.cache_resource(show_spinner="Setting up agent pipeline...")
def get_pipeline():
    vs = build_knowledge_base()
    app = build_graph()
    return app, vs


app, vector_store = get_pipeline()

# --- Ticket input ---
example_tickets = {
    "-- Select an example --": "",
    "Delivery delay": "My order ORD12345 hasn't arrived yet, it's already 3 days late",
    "Damaged product": "I received my order ORD12351 today but the gaming mouse box was crushed and the mouse doesn't work",
    "Wrong item delivered": "I ordered a Smart Watch (ORD12350) but received a completely different product in the box",
    "Payment failed": "Paid Rs 499 for a case but order ORD12346 shows Payment Failed, money got deducted",
    "Security issue (Critical)": "URGENT - I think someone accessed my account without my permission and changed my password",
    "Invalid order refund": "I want a refund for order ORD99999 immediately, this is unacceptable",
    "General question": "Can you tell me the return window for electronics?",
}

selected_example = st.selectbox("Try an example, or type your own below:", list(example_tickets.keys()))

ticket_text = st.text_area(
    "Customer ticket text",
    value=example_tickets[selected_example],
    height=100,
    placeholder="e.g. My order ORD12345 hasn't arrived yet...",
)

process_clicked = st.button("Process Ticket", type="primary")

# --- Routing action visual styling ---
ACTION_STYLE = {
    RoutingAction.AUTO_RESPOND: ("✅", "green"),
    RoutingAction.FLAG_FOR_REVIEW: ("🟡", "orange"),
    RoutingAction.ESCALATE: ("🔴", "red"),
}

if process_clicked:
    if not ticket_text.strip():
        st.error("Please enter a ticket or select an example.")
        st.stop()

    with st.spinner("Running pipeline (classify → tool-call/RAG → generate → route)..."):
        initial_state = {
            "ticket_text": ticket_text,
            "api_key": api_key_input,
            "vector_store": vector_store,
            "classification": None,
            "order_context": None,
            "policy_context": None,
            "response": None,
            "routing_decision": None,
        }
        final_state = app.invoke(initial_state)

    classification = final_state["classification"]
    response = final_state["response"]
    decision = final_state["routing_decision"]

    st.markdown("---")

    # --- Classification result ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Category", classification.category.value)
    col2.metric("Urgency", classification.urgency.value)
    col3.metric("Confidence", f"{response.confidence}/10")

    # --- Routing decision (highlighted) ---
    emoji, color = ACTION_STYLE[decision.action]
    st.markdown(f"### {emoji} Routing Decision: :{color}[{decision.action.value}]")
    st.caption(decision.reason)

    # --- Draft response ---
    st.markdown("### Draft Response")
    if decision.action == RoutingAction.ESCALATE and response.confidence == 0:
        st.info(response.response_text)
    else:
        st.write(response.response_text)

    # --- Debug/transparency expander ---
    with st.expander("🔍 See what the agent used to generate this (order data + policy context)"):
        st.markdown(f"**Extracted Order ID:** {classification.order_id or 'None'}")
        st.markdown(f"**Classifier reasoning:** {classification.reasoning}")