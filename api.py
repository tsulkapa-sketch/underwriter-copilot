import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from main import setup, build_or_load_index, REPORT_SECTIONS, REPORT_KEYWORDS
from agent import build_graph, run_full_analysis, resume_analysis, FullAnalysisState

app = FastAPI(title="Underwriter Copilot API", version="2.0")

# Allow the browser UI to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the UI files at /ui
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# ── Startup ────────────────────────────────────────────────────────────────────
setup()
index = build_or_load_index()
query_engine = index.as_query_engine(similarity_top_k=5)

# Session store for in-flight multi-agent analyses (keyed by case_id)
# In production: use Redis or a persistent store
_active_sessions: dict = {}   # { case_id: {"graph": ..., "config": ..., "state": ...} }

# ── Request / Response Models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    case_id: str = "SME-2024-00891"

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

class DecisionRequest(BaseModel):
    case_id: str = "SME-2024-00891"
    decision: str   # "approve" | "conditional_approve" | "decline" | "escalate"


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0", "case": "SME-2024-00891"}


# ── RAG query ──────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    response = query_engine.query(request.question)
    sources = list(set([
        node.metadata.get("file_name", "unknown")
        for node in response.source_nodes
    ]))
    return QueryResponse(answer=str(response), sources=sources)


# ── Memo and report (legacy RAG-only endpoints) ────────────────────────────────
MEMO_SECTIONS = [
    ("Borrower profile",
     "Summarize the borrower's business — name, entity type, years in operation, industry."),
    ("Financial summary",
     "Revenue, EBITDA, net profit and margin for FY2024 vs FY2023."),
    ("Debt and obligations",
     "All existing loans, outstanding amounts and lenders."),
    ("Bureau highlights",
     "Credit score, repayment history, adverse markers."),
    ("Key risks",
     "Top 3 credit risks with supporting figures."),
    ("Recommendation basis",
     "Factors supporting approval and recommended covenants."),
]

@app.post("/memo")
def memo():
    result = {}
    for title, question in MEMO_SECTIONS:
        response = query_engine.query(question)
        result[title] = str(response)
    return result

@app.post("/report")
def report():
    result = {}
    for title, question in REPORT_SECTIONS:
        response = query_engine.query(question)
        result[title] = str(response)
    return result


# ── Detect query type ──────────────────────────────────────────────────────────
@app.post("/detect")
def detect(request: QueryRequest):
    q = request.question.lower()
    if any(kw in q for kw in REPORT_KEYWORDS) or q == "report":
        return {"type": "report"}
    if q in ["memo", "generate memo", "credit memo"]:
        return {"type": "memo"}
    return {"type": "query"}


# ── Legacy agent endpoints (contradiction-only) ────────────────────────────────
@app.post("/agent/scan")
async def agent_scan(request: QueryRequest):
    """
    Legacy endpoint: contradiction detection only.
    Kept for backward compatibility with the existing UI.
    """
    from agents.contradiction import run_contradiction_analysis
    import os

    DOCS_DIR = "loan_docs"
    documents = {}
    if os.path.exists(DOCS_DIR):
        for filename in sorted(os.listdir(DOCS_DIR)):
            if filename.endswith((".txt", ".pdf")):
                with open(os.path.join(DOCS_DIR, filename), "r", encoding="utf-8") as f:
                    documents[filename] = f.read()

    result = run_contradiction_analysis(documents)

    return {
        "case_id":          request.case_id,
        "documents_scanned": len(documents),
        "contradictions":   result["contradictions"],
        "severities":       result["severities"],
        "summary":          result["key_metrics"],
    }

@app.post("/agent/decide")
def agent_decide(request: QueryRequest):
    """Legacy: record underwriter decision from the UI."""
    return {
        "case_id": request.case_id,
        "decision": request.question,
        "status": "recorded",
        "message": f"Decision '{request.question}' recorded for case {request.case_id}",
    }


# ── NEW: Full multi-agent analysis ────────────────────────────────────────────
@app.post("/agent/full-analysis")
async def full_analysis(request: QueryRequest):
    """
    Start the full 5-agent analysis for a case.

    Runs: financial → bureau → contradiction → policy → collateral (parallel)
    Then: aggregates → pauses for human review.

    Returns:
      - All agent findings
      - System recommendation
      - interrupt_summary: text to show the underwriter for their decision
      - session active (call /agent/full-analysis/resume with your decision)
    """
    case_id = request.case_id

    result, graph, config = run_full_analysis(case_id)

    # Extract interrupt value if graph paused
    # LangGraph returns Interrupt objects with a .value attribute (not subscriptable dicts)
    interrupt_summary = None
    if result.get("__interrupt__"):
        interrupt_summary = result["__interrupt__"][0].value

    # Store session for resume
    _active_sessions[case_id] = {
        "graph":  graph,
        "config": config,
        "state":  result,
    }

    aggregated = result.get("aggregated", {})

    return {
        "case_id":              case_id,
        "status":               "awaiting_decision",
        "system_recommendation": aggregated.get("system_recommendation"),
        "overall_risk":         aggregated.get("overall_risk"),
        "agent_ratings":        aggregated.get("agent_ratings", {}),
        "key_metrics":          aggregated.get("key_metrics", {}),
        "executive_summary":    aggregated.get("executive_summary"),
        "conditions":           aggregated.get("conditions", []),
        "warnings":             aggregated.get("warnings", []),
        "contradictions": {
            "total":  result.get("contradiction", {}).get("key_metrics", {}).get("total", 0),
            "high":   result.get("contradiction", {}).get("key_metrics", {}).get("high", 0),
            "medium": result.get("contradiction", {}).get("key_metrics", {}).get("medium", 0),
            "low":    result.get("contradiction", {}).get("key_metrics", {}).get("low", 0),
            "items":  result.get("contradiction", {}).get("contradictions", []),
            "severities": result.get("contradiction", {}).get("severities", []),
        },
        "interrupt_summary":    interrupt_summary,
        "message": (
            "Analysis complete. Review findings and POST to "
            "/agent/full-analysis/resume with your decision."
        ),
    }


@app.post("/agent/full-analysis/resume")
def full_analysis_resume(request: DecisionRequest):
    """
    Resume the full analysis after underwriter provides their decision.

    Body:
        case_id  : str  — same case_id used in /agent/full-analysis
        decision : str  — "approve" | "conditional_approve" | "decline" | "escalate"

    Returns:
        final_report: the complete structured credit decision report
    """
    case_id  = request.case_id
    decision = request.decision.lower()

    if case_id not in _active_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"No active analysis session found for case_id '{case_id}'. "
                   f"Run /agent/full-analysis first."
        )

    session = _active_sessions[case_id]
    graph   = session["graph"]
    config  = session["config"]

    final_state = resume_analysis(graph, config, decision)

    # Clean up session
    del _active_sessions[case_id]

    return {
        "case_id":      case_id,
        "decision":     decision.upper(),
        "status":       "completed",
        "final_report": final_state.get("final_report", "No report generated."),
        "overall_risk": final_state.get("aggregated", {}).get("overall_risk"),
    }
