---
title: Ticket Triage Agent
emoji: 🎫
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---

# 🎫 E-commerce Support Ticket Triage Agent

An agentic AI pipeline that classifies customer support tickets, looks up real order data, retrieves relevant company policy via RAG, drafts a grounded response, and applies risk-based routing (auto-respond / flag for review / escalate to human) - built with LangChain, LangGraph, and Groq.

## Tech Stack

- LangChain + LangGraph for agent orchestration
- Groq (Llama 3.1) for fast LLM inference
- FAISS + HuggingFace embeddings for RAG over policy documents
- Pydantic for structured LLM outputs
- Streamlit for the UI
- LangSmith for tracing/observability (optional)

## Project Structure

- classifier.py - Ticket classification (category + urgency)
- order_tool.py - Mock order database tool-calling
- rag_retrieval.py - Policy document RAG retrieval
- response_generator.py - Grounded response generation
- routing_logic.py - Risk-based escalation routing
- graph.py - LangGraph orchestration of the full pipeline
- evaluate.py - Classifier accuracy evaluation
- app.py - Streamlit UI