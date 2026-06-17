import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── Configuration ──────────────────────────────────────────────────────────────
DOCS_DIR   = "loan_docs"
INDEX_DIR  = "stored_index"
MODEL      = "claude-sonnet-4-6"
API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "your-api-key-here")
DEBUG_MODE = False   # toggle to True to see retrieved chunks + full prompt

SYSTEM_PROMPT = """You are an expert credit underwriter assistant working inside
a loan origination system. Your job is to help underwriters quickly understand
a loan application by answering their questions based strictly on the documents
loaded for this case.

Rules you must always follow:
1. Base every answer only on the documents provided. Never speculate.
2. For every factual claim, cite the exact source in brackets e.g.
   [bureau_report.txt — Adverse Markers] or [loan_application.txt — Financial Summary].
   Never state a fact without a citation.
3. If information is not in the documents, respond only with:
   'This information is not available in the loaded documents.'
   Do not speculate or elaborate beyond this.
4. Flag any inconsistencies you notice across documents.
5. Use concise, professional language — the underwriter is an expert.
6. For numerical figures, always state the time period they relate to."""

# ── Setup ──────────────────────────────────────────────────────────────────────
def setup():
    Settings.llm = Anthropic(
        model=MODEL,
        api_key=API_KEY,
        system_prompt=SYSTEM_PROMPT,
        temperature=0,        # deterministic — same question = same answer
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

# ── Index ──────────────────────────────────────────────────────────────────────
def build_or_load_index():
    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        print("Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage_context)

    print("Building index from documents...")
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        print(f"\n ERROR: No documents found in '{DOCS_DIR}/'")
        print(" Add PDF or text files to that folder and run again.\n")
        exit(1)

    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print(f" Indexed {len(documents)} document chunk(s) from '{DOCS_DIR}/'.")

    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(" Index saved. Future runs will load instantly.\n")
    return index

# ── Debug query ────────────────────────────────────────────────────────────────
def query_with_debug(question, retriever, query_engine):
    """
    Shows exactly what happens between your question and Claude's answer:
    1. Which chunks were retrieved from the documents
    2. The relevance score for each chunk
    3. The full prompt that gets sent to Claude
    """
    # Step 1: retrieve chunks
    nodes = retriever.retrieve(question)

    print("\n" + "─" * 55)
    print("DEBUG — RETRIEVED CHUNKS")
    print("─" * 55)
    for i, node in enumerate(nodes, 1):
        source = node.metadata.get("file_name", "unknown")
        score  = node.score if node.score is not None else 0
        print(f"\nChunk {i} | Source: {source} | Relevance: {score:.3f}")
        print(f"{node.text[:300]}{'...' if len(node.text) > 300 else ''}")

    # Step 2: show what gets assembled and sent to Claude
    context_blocks = []
    for node in nodes:
        source = node.metadata.get("file_name", "unknown")
        context_blocks.append(f"[From {source}]\n{node.text}")
    full_context = "\n\n---\n\n".join(context_blocks)

    print("\n" + "─" * 55)
    print("DEBUG — FULL PROMPT SENT TO CLAUDE")
    print("─" * 55)
    print(f"SYSTEM:\n{SYSTEM_PROMPT}")
    print(f"\nCONTEXT:\n{full_context}")
    print(f"\nQUESTION:\n{question}")
    print("─" * 55 + "\n")

    # Step 3: get Claude's response
    response = query_engine.query(question)
    return response

# ── Commands ───────────────────────────────────────────────────────────────────
MEMO_QUESTIONS = [
    ("BORROWER PROFILE",
     "Summarize the borrower's business — name, type of entity, years in "
     "operation, industry, and primary business activity."),

    ("FINANCIAL SUMMARY",
     "What is the borrower's annual revenue, EBITDA, net profit, and net "
     "profit margin for the latest financial year? Compare to prior year if available."),

    ("DEBT AND OBLIGATIONS",
     "List all existing loans, credit facilities, and repayment obligations "
     "mentioned across the documents including outstanding amounts and lenders."),

    ("BUREAU HIGHLIGHTS",
     "Summarize the bureau report — credit score, repayment track record, "
     "any delays, defaults, or adverse markers."),

    ("KEY RISKS",
     "Based on all documents, what are the top 3 credit risks for this "
     "borrower? Be specific with supporting figures."),

    ("RECOMMENDATION BASIS",
     "What factors support approval and what conditions or covenants would "
     "you recommend be attached to any sanction?"),
]

REPORT_SECTIONS = [
    ("1. BORROWER PROFILE",
     "Provide full borrower details: company name, PAN, GST, entity type, "
     "year of incorporation, registered address, industry, and primary business activity. "
     "Include promoter names, shareholding, experience, and personal net worth if available."),

    ("2. LOAN REQUEST",
     "List all loan facilities requested: type, amount, tenure, and stated purpose "
     "for each facility. State the total exposure clearly."),

    ("3. FINANCIAL ANALYSIS",
     "Provide a 3-year financial summary table covering revenue, EBITDA, EBITDA margin, "
     "net profit, net profit margin, total debt, and debt-to-equity ratio. "
     "Identify the key trend and explain what is driving any deterioration."),

    ("4. BANKING BEHAVIOUR",
     "Summarize the borrower's banking relationship: primary bank, existing facilities "
     "with outstanding amounts, average bank balance, and annual bank turnover. "
     "Note whether bank turnover is consistent with stated revenue."),

    ("5. BUREAU ASSESSMENT",
     "Provide the commercial credit score, risk category, and full DPD history "
     "for all facilities. List any adverse markers, unanswered enquiries, "
     "and promoter personal bureau scores."),

    ("6. COLLATERAL",
     "List all collateral offered: type, estimated value, and any existing charges. "
     "Include personal guarantees provided."),

    ("7. KEY RISKS",
     "Identify the top 3-5 credit risks for this borrower. For each risk: "
     "name it, quantify it with specific figures, and state which document it comes from."),

    ("8. MITIGANTS",
     "For each risk identified, what mitigating factors exist in the documents? "
     "What covenants or conditions would address these risks?"),

    ("9. RECOMMENDATION",
     "Based strictly on the documents, state whether the case merits approval, "
     "conditional approval, or decline. List specific conditions that should be "
     "attached to any sanction. Flag what additional information is needed."),
]

# Keywords that trigger the structured report
REPORT_KEYWORDS = [
    "case report", "generate report", "detailed report",
    "credit report", "appraisal report", "full report",
    "case appraisal", "generate case"
]

def generate_report(query_engine):
    print("\n" + "=" * 55)
    print("  CREDIT APPRAISAL REPORT — AUTO-GENERATED DRAFT")
    print("  Structured format — for underwriter review before use")
    print("=" * 55 + "\n")

    for section_title, question in REPORT_SECTIONS:
        print(f"\n{'─' * 55}")
        print(f"  {section_title}")
        print(f"{'─' * 55}")
        response = query_engine.query(question)
        print(f"{response}\n")

    print("=" * 55)
    print("  END OF REPORT — Review all sections before submission")
    print("=" * 55 + "\n")

def generate_memo(query_engine):
    print("\n" + "=" * 55)
    print("  CREDIT MEMO — AUTO-GENERATED DRAFT")
    print("  For underwriter review and edit before submission")
    print("=" * 55 + "\n")

    for section_title, question in MEMO_QUESTIONS:
        print(f"── {section_title} " + "─" * (45 - len(section_title)))
        response = query_engine.query(question)
        print(f"{response}\n")

    print("=" * 55)
    print("  END OF DRAFT — Please review all sections before use")
    print("=" * 55 + "\n")

def show_help():
    print("""
  Commands:
    memo      — 6-section credit memo draft
    report    — 9-section full credit appraisal report
    debug     — toggle debug mode on/off
    help      — show this list
    quit      — end the session

  Or type any free-text question about the loan application.
  Examples:
    What are the top risks?
    Does bank turnover match stated revenue?
    What is the customer concentration risk?
""")

# ── Chat loop ──────────────────────────────────────────────────────────────────
def run_chat(index):
    global DEBUG_MODE

    # Build retriever and query engine separately so debug mode can use both
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    synthesizer = get_response_synthesizer(response_mode="compact")
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
    )

    print("\n" + "=" * 55)
    print("  UNDERWRITER COPILOT")
    print("=" * 55)
    print("  Documents loaded and ready.")
    print("  Type a question, 'memo' for a full draft, or 'help'.")
    print(f"  Debug mode: {'ON' if DEBUG_MODE else 'OFF'} (type 'debug' to toggle)\n")

    while True:
        try:
            question = input("Underwriter: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not question:
            continue

        if question.lower() in ["quit", "exit", "q"]:
            print("Session ended.")
            break

        if question.lower() == "help":
            show_help()
            continue

        # Detect report keywords — any phrasing that means full report
        q_lower = question.lower()
        if any(kw in q_lower for kw in REPORT_KEYWORDS):
            generate_report(query_engine)
            continue

        if q_lower == "report":
            generate_report(query_engine)
            continue

        if q_lower == "memo":
            generate_memo(query_engine)
            continue

        if question.lower() == "debug":
            DEBUG_MODE = not DEBUG_MODE
            print(f"  Debug mode: {'ON — you will see retrieved chunks and full prompt' if DEBUG_MODE else 'OFF'}\n")
            continue

        print("\nCopilot:", end=" ", flush=True)

        if DEBUG_MODE:
            response = query_with_debug(question, retriever, query_engine)
        else:
            response = query_engine.query(question)

        print(f"{response}\n")

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    setup()
    index = build_or_load_index()
    run_chat(index)

if __name__ == "__main__":
    main()
