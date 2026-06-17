"""
Layer 3 — RAG as a tool

Wraps the existing LlamaIndex RAG pipeline as a LangChain tool
so LangGraph agents can call it like any other tool.

Agents call rag_query(question, document_filter) and get back
a grounded answer with source citations.
"""

import os
from langchain_core.tools import tool
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DOCS_DIR  = "loan_docs"
INDEX_DIR = "stored_index"

# Global query engine — initialised once, reused by all agents
_query_engine = None

SYSTEM_PROMPT = """You are an expert credit underwriter assistant.
Answer questions based strictly on the documents provided.
For every factual claim, cite the source document in brackets.
If information is not in the documents, say: NOT_FOUND.
Be concise and precise — underwriters need facts, not narratives."""


def get_query_engine():
    """Initialise or return existing query engine."""
    global _query_engine
    if _query_engine is not None:
        return _query_engine

    Settings.llm = Anthropic(
        model="claude-sonnet-4-6",
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        system_prompt=SYSTEM_PROMPT,
        temperature=0,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader(DOCS_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=INDEX_DIR)

    _query_engine = index.as_query_engine(
        similarity_top_k=6,
    )
    return _query_engine


@tool
def rag_query(question: str) -> str:
    """
    Query the RAG pipeline over all loan documents.
    Returns a grounded answer with source citations.
    Use for extracting facts from loan documents.
    """
    try:
        engine = get_query_engine()
        response = engine.query(question)
        return str(response)
    except Exception as e:
        return f"RAG query failed: {str(e)}"


@tool
def rag_query_financial(question: str) -> str:
    """
    Query financial documents specifically.
    Best for: revenue, profit, ratios, balance sheet questions.
    """
    financial_question = (
        f"From the financial statements, ITR, and GST returns only: {question}"
    )
    return rag_query.invoke(financial_question)


@tool
def rag_query_bureau(question: str) -> str:
    """
    Query bureau report specifically.
    Best for: credit score, DPD history, adverse markers, enquiries.
    """
    bureau_question = f"From the bureau report only: {question}"
    return rag_query.invoke(bureau_question)


@tool
def rag_query_policy(question: str) -> str:
    """
    Query lending policy document.
    Best for: checking if borrower meets policy requirements,
    finding relevant policy rules, approval authority.
    """
    policy_question = f"From the lending policy document only: {question}"
    return rag_query.invoke(policy_question)


@tool
def rag_query_collateral(question: str) -> str:
    """
    Query collateral and valuation documents.
    Best for: property values, LTV, existing charges, machinery details.
    """
    collateral_question = (
        f"From the valuation report and loan application collateral section: {question}"
    )
    return rag_query.invoke(collateral_question)


@tool
def rag_cross_document_compare(field: str) -> str:
    """
    Compare how a specific field appears across ALL documents.
    Best for: finding contradictions, verifying consistency.
    Example: rag_cross_document_compare('annual revenue')
    """
    question = (
        f"Search across ALL documents and list every mention of '{field}' "
        f"with the exact figure, the document it appears in, and the section. "
        f"Format as a comparison table."
    )
    return rag_query.invoke(question)
