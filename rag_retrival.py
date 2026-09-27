"""
STEP 3: RAG Retrieval
=======================
Searches the knowledge_base/ policy documents to find chunks relevant to
the customer's ticket, so the response generator (Step 4) can ground its
answer in real company policy instead of guessing.

Concept recap (same as our earlier RAG chatbot project):
  Documents -> Chunk -> Embed -> FAISS vector store
  Query -> Embed -> Similarity search -> Top-K relevant chunks

What's different here: RAG is no longer a standalone chatbot - it's one
component inside a bigger agent pipeline. So this module is built as a
reusable function that Step 4 will call, not a full app by itself.
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


KB_PATH = os.path.join(os.path.dirname(__file__), "know_base")
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "kb_vector_store")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 2  # fewer chunks than a full chatbot since this feeds into a larger prompt later
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_knowledge_base(force_rebuild: bool = False):
    """
    Builds (or loads) the FAISS vector store from policy documents.

    force_rebuild=False by default so we don't re-embed every single run -
    embedding is the slowest part of RAG, so we cache it to disk.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if not force_rebuild and os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(
            VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
        )

    loader = DirectoryLoader(KB_PATH, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store


def retrieve_policy_context(query: str, vector_store=None) -> str:
    """
    Given a ticket/query, returns the most relevant policy text as a
    single formatted string, ready to inject into an LLM prompt.

    Args:
        query: the customer's ticket text (or a reformulated search query)
        vector_store: optionally pass an already-loaded vector store to
                       avoid reloading it on every call (important for
                       performance when processing many tickets)

    Returns:
        A string combining the top-K relevant chunks with their source
        file named, so the generator later can also mention which policy
        it used.
    """
    if vector_store is None:
        vector_store = build_knowledge_base()

    results = vector_store.similarity_search(query, k=TOP_K)

    if not results:
        return "No relevant policy information found."

    formatted_chunks = []
    for doc in results:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        formatted_chunks.append(f"[From {source}]\n{doc.page_content}")

    return "\n\n".join(formatted_chunks)


# ---------------------------------------------------------------------------
# CLI TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building/loading knowledge base...")
    vs = build_knowledge_base(force_rebuild=True)

    test_queries = [
        "My order hasn't arrived yet, it's 3 days late",
        "Payment failed but money got deducted",
        "Can I return a defective product?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        context = retrieve_policy_context(query, vector_store=vs)
        print(context)