"""
Multi-Agent Orchestrator
------------------------
LangGraph graph that fans out to all 5 specialist agents in parallel,
aggregates findings, pauses for human review, then compiles a final
credit recommendation.

Flow:
  load_case_data
      ↓  (fan-out to 5 parallel agents)
  run_financial ──┐
  run_bureau    ──┤
  run_contradiction ──┤  → aggregate_findings → human_review → compile_report
  run_policy    ──┤
  run_collateral──┘

Human-in-the-loop:
  - Graph pauses at human_review via LangGraph interrupt()
  - api.py resumes it with underwriter's proceed / escalate decision
  - Final report is compiled with decision recorded for audit trail
"""

import os
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_anthropic import ChatAnthropic

from agents import (
    run_financial_analysis,
    run_bureau_analysis,
    run_contradiction_analysis,
    run_policy_analysis,
    run_collateral_analysis,
)

# ── Hardcoded case data for SME-2024-00891 (Meridian Textile Exports) ─────────
# In production: pulled from LOS by case_id
MERIDIAN_CASE_DATA = {
    "case_id":              "SME-2024-00891",
    "pan":                  "AABCM7823Q",
    "promoter_pans":        ["BBBPM4412R", "CCCPM9921S"],
    "loan_amount_lakhs":    270,
    "property_address":     "Plot 14, MIDC Pune",
    # Financial inputs (all in lakhs)
    "revenue":              394.82,
    "ebitda":               58.46,
    "net_profit":           32.20,
    "total_debt":           34.10,
    "total_equity":         174.80,
    "current_assets":       146.14,
    "current_liabilities":  92.20,
    "interest_expense":     8.42,
    "annual_debt_repayment": 14.40,
    "depreciation":         9.20,
    # Bank statement (3 months)
    "monthly_credits":          [29.00, 26.00, 63.20],
    "monthly_debits":           [28.59, 28.74, 59.85],
    "monthly_closing_balances": [8.84, 6.09, 9.44],
    "stated_annual_turnover":   680,
    "cc_limit":                 40,
    # Bureau / scorecard inputs
    "bureau_score":                    74,
    "dpd_instances_last_24_months":    1,
    "customer_concentration_pct":      67,
    "years_in_operation":              10,
    "revenue_trend":                   "GROWING",
    # Fraud / policy inputs
    "cc_cycling_detected":             True,
    "revenue_variance_pct":            6.5,
    "bank_turnover_variance_pct":      30.4,
    "unanswered_enquiries":            2,
    "undisclosed_contingent":          True,
    # Policy engine inputs
    # DSCR = (net_profit + depreciation + interest) / (repayment + interest)
    # = (32.20 + 9.20 + 8.42) / (14.40 + 8.42) = 49.82 / 22.82 ≈ 2.18
    "dscr":                            2.18,
    "blended_ltv":                     0.517,
    "property_ltv":                    0.865,
    "bureau_score_commercial":         74,
    "dpd_60_plus_last_36_months":      False,
    "npa_last_60_months":              False,
    "wilful_defaulter":                False,
    "revenue_overstatement_pct":       6.5,
    "undisclosed_contingent_liability": True,
    "epcg_obligation_lakhs":           42,
    "sector":                          "textile",
    "promoter_net_worth_lakhs":        380,
    # Collateral inputs
    "property_market_value":           312,
    "machinery_value":                 210,
    "existing_charge_outstanding":     22,
    "property_type":                   "industrial",
}

DOCS_DIR = "loan_docs"

# ── LLM for aggregation ────────────────────────────────────────────────────────
_model = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    temperature=0,
)

# ── State ──────────────────────────────────────────────────────────────────────
class FullAnalysisState(TypedDict):
    case_id:              str
    case_data:            dict
    documents:            dict          # raw document content
    # Specialist agent outputs
    financial:            dict
    bureau:               dict
    contradiction:        dict
    policy:               dict
    collateral:           dict
    # Aggregated
    aggregated:           dict
    # Human review
    underwriter_decision: str
    final_report:         str


# ── Node 1: Load case data and documents ──────────────────────────────────────
def load_case_data(state: FullAnalysisState) -> FullAnalysisState:
    """Load loan documents and resolve case data."""
    print(f"\n[Orchestrator] Loading case {state['case_id']}...")

    # Load documents from loan_docs/
    documents = {}
    if os.path.exists(DOCS_DIR):
        for filename in sorted(os.listdir(DOCS_DIR)):
            if filename.endswith((".txt", ".pdf")):
                with open(os.path.join(DOCS_DIR, filename), "r", encoding="utf-8") as f:
                    documents[filename] = f.read()
        print(f"[Orchestrator] Loaded {len(documents)} documents.")

    # Use hardcoded case data for this simulation
    # In production: fetch from LOS by case_id
    case_data = {**MERIDIAN_CASE_DATA, "case_id": state["case_id"]}

    return {"case_data": case_data, "documents": documents}


# ── Nodes 2a-2e: Specialist agents (run in parallel by LangGraph) ─────────────

def run_financial(state: FullAnalysisState) -> dict:
    print("[Orchestrator] → Financial agent starting")
    result = run_financial_analysis(state["case_data"])
    return {"financial": result}


def run_bureau(state: FullAnalysisState) -> dict:
    print("[Orchestrator] → Bureau agent starting")
    result = run_bureau_analysis(state["case_data"])
    return {"bureau": result}


def run_contradiction(state: FullAnalysisState) -> dict:
    print("[Orchestrator] → Contradiction agent starting")
    result = run_contradiction_analysis(state["documents"])
    return {"contradiction": result}


def run_policy(state: FullAnalysisState) -> dict:
    print("[Orchestrator] → Policy agent starting")
    result = run_policy_analysis(state["case_data"])
    return {"policy": result}


def run_collateral(state: FullAnalysisState) -> dict:
    print("[Orchestrator] → Collateral agent starting")
    result = run_collateral_analysis(state["case_data"])
    return {"collateral": result}


# ── Node 3: Aggregate all agent outputs ───────────────────────────────────────
def aggregate_findings(state: FullAnalysisState) -> FullAnalysisState:
    """Merge all agent results into a structured aggregated summary."""
    print("\n[Orchestrator] Aggregating findings from all agents...")

    fin  = state.get("financial", {})
    bur  = state.get("bureau", {})
    con  = state.get("contradiction", {})
    pol  = state.get("policy", {})
    col  = state.get("collateral", {})

    # Collect risk ratings
    ratings = {
        "financial":     fin.get("risk_rating", "UNKNOWN"),
        "bureau":        bur.get("risk_rating", "UNKNOWN"),
        "contradiction": con.get("risk_rating", "UNKNOWN"),
        "policy":        pol.get("risk_rating", "UNKNOWN"),
        "collateral":    col.get("risk_rating", "UNKNOWN"),
    }

    # Overall risk = worst of all agents
    risk_priority = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}
    overall_risk = max(ratings.values(), key=lambda r: risk_priority.get(r, 0))

    # Auto-decline if hard stops triggered
    auto_decline = (
        pol.get("policy_result", {}).get("auto_decline", False)
        or bool(bur.get("hard_stop_flags"))
    )

    # System recommendation
    if auto_decline:
        system_recommendation = "DECLINE"
    elif overall_risk == "HIGH":
        system_recommendation = "CONDITIONAL_APPROVE"
    elif overall_risk == "MEDIUM":
        system_recommendation = "CONDITIONAL_APPROVE"
    else:
        system_recommendation = "APPROVE"

    # Build aggregation prompt for Claude
    agg_prompt = f"""You are a Chief Credit Officer reviewing a full multi-agent credit analysis.
Synthesize the findings below into an executive summary (max 200 words).

CASE: {state['case_id']}
LOAN: ₹{state['case_data'].get('loan_amount_lakhs')}L

AGENT RISK RATINGS:
  Financial     : {ratings['financial']}
  Bureau        : {ratings['bureau']}
  Contradictions: {ratings['contradiction']}
  Policy        : {ratings['policy']}
  Collateral    : {ratings['collateral']}
  Overall       : {overall_risk}

KEY METRICS:
  Credit scorecard score : {fin.get('key_metrics', {}).get('credit_score', 'N/A')}/100 ({fin.get('key_metrics', {}).get('grade', 'N/A')})
  DSCR                   : {fin.get('key_metrics', {}).get('dscr', 'N/A')}x
  Bureau score           : {bur.get('key_metrics', {}).get('commercial_score', 'N/A')}/100
  Fraud score            : {bur.get('key_metrics', {}).get('fraud_score', 'N/A')}/100
  Policy pass            : {pol.get('key_metrics', {}).get('policy_pass', 'N/A')}
  Hard stops             : {pol.get('key_metrics', {}).get('hard_stops_count', 0)}
  Conditions             : {pol.get('key_metrics', {}).get('conditions_count', 0)}
  Blended LTV            : {col.get('key_metrics', {}).get('blended_ltv_pct', 'N/A')}%
  Coverage ratio         : {col.get('key_metrics', {}).get('collateral_coverage_ratio', 'N/A')}x
  Contradictions (HIGH)  : {con.get('key_metrics', {}).get('high', 0)}

SYSTEM RECOMMENDATION: {system_recommendation}

Provide:
1. Executive summary (3 sentences)
2. Top 3 risks
3. Recommended conditions if approved
4. Rationale for system recommendation"""

    agg_response = _model.invoke(agg_prompt)

    aggregated = {
        "case_id":                state["case_id"],
        "overall_risk":           overall_risk,
        "system_recommendation":  system_recommendation,
        "auto_decline":           auto_decline,
        "agent_ratings":          ratings,
        "hard_stop_flags":        bur.get("hard_stop_flags", []) + (
            pol.get("policy_result", {}).get("hard_stops", [])
        ),
        "conditions":             pol.get("policy_result", {}).get("conditions", []),
        "warnings":               pol.get("policy_result", {}).get("warnings", []),
        "key_metrics": {
            "credit_score":        fin.get("key_metrics", {}).get("credit_score"),
            "grade":               fin.get("key_metrics", {}).get("grade"),
            "dscr":                fin.get("key_metrics", {}).get("dscr"),
            "commercial_score":    bur.get("key_metrics", {}).get("commercial_score"),
            "fraud_score":         bur.get("key_metrics", {}).get("fraud_score"),
            "blended_ltv":         col.get("key_metrics", {}).get("blended_ltv_pct"),
            "coverage_ratio":      col.get("key_metrics", {}).get("collateral_coverage_ratio"),
            "contradictions_high": con.get("key_metrics", {}).get("high"),
            "policy_pass":         pol.get("key_metrics", {}).get("policy_pass"),
        },
        "executive_summary": agg_response.content,
    }

    print(f"[Orchestrator] Aggregated — overall_risk={overall_risk}, "
          f"recommendation={system_recommendation}")

    return {"aggregated": aggregated}


# ── Node 4: Human review (LangGraph interrupt) ─────────────────────────────────
def human_review(state: FullAnalysisState) -> FullAnalysisState:
    """Pause for underwriter decision. Resumes when api.py supplies the decision."""
    print("\n[Orchestrator] Pausing for underwriter review...")

    agg = state.get("aggregated", {})
    km  = agg.get("key_metrics", {})

    summary = f"""
{'='*60}
FULL CREDIT ANALYSIS COMPLETE — Case {state['case_id']}
{'='*60}

SYSTEM RECOMMENDATION : {agg.get('system_recommendation')}
OVERALL RISK          : {agg.get('overall_risk')}
AUTO DECLINE          : {agg.get('auto_decline')}

AGENT RISK RATINGS:
{''.join(f"  {k:<16}: {v}" + chr(10) for k, v in agg.get('agent_ratings', {}).items())}
KEY METRICS:
  Credit score   : {km.get('credit_score')}/100 (Grade {km.get('grade')})
  DSCR           : {km.get('dscr')}x
  Bureau score   : {km.get('commercial_score')}/100
  Fraud score    : {km.get('fraud_score')}/100
  Blended LTV    : {km.get('blended_ltv')}%
  Coverage ratio : {km.get('coverage_ratio')}x
  HIGH contradictions: {km.get('contradictions_high')}
  Policy pass    : {km.get('policy_pass')}

EXECUTIVE SUMMARY:
{agg.get('executive_summary', '')}

{'='*60}
Enter underwriter decision: 'approve' | 'conditional_approve' | 'decline' | 'escalate'
"""

    decision = interrupt(summary)
    return {"underwriter_decision": decision}


# ── Node 5: Compile final report ───────────────────────────────────────────────
def compile_report(state: FullAnalysisState) -> FullAnalysisState:
    """Build the final structured credit decision report."""
    print(f"\n[Orchestrator] Compiling final report "
          f"(decision: {state.get('underwriter_decision', 'N/A')})...")

    agg = state.get("aggregated", {})
    fin = state.get("financial", {})
    bur = state.get("bureau", {})
    con = state.get("contradiction", {})
    pol = state.get("policy", {})
    col = state.get("collateral", {})

    decision = str(state.get("underwriter_decision") or "").upper()

    # Format conditions
    conditions_text = ""
    for c in agg.get("conditions", []):
        conditions_text += f"\n- [{c.get('rule')}] {c.get('condition', c.get('finding'))}"

    # Format hard stops
    hard_stops_text = ""
    for h in agg.get("hard_stop_flags", []):
        if isinstance(h, dict):
            hard_stops_text += f"\n- [{h.get('rule')}] {h.get('finding')}"
        else:
            hard_stops_text += f"\n- {h}"

    # Format contradictions
    contradictions_text = ""
    contras = con.get("contradictions", [])
    sevs    = con.get("severities", [])
    for i, (c, s) in enumerate(zip(contras, sevs), 1):
        contradictions_text += (
            f"\n{i}. [{s.get('severity')}] {c.get('field', 'Unknown')}"
            f"\n   Values  : {c.get('values', 'N/A')}"
            f"\n   Variance: {c.get('variance', 'N/A')}"
            f"\n   Finding : {c.get('finding', 'N/A')}\n"
        )

    report = f"""# CREDIT DECISION REPORT
{'='*60}
Case Reference     : {state['case_id']}
System Rec.        : {agg.get('system_recommendation')}
UW Decision        : {decision}
Overall Risk       : {agg.get('overall_risk')}
{'='*60}

## 1. EXECUTIVE SUMMARY

{agg.get('executive_summary', 'N/A')}

## 2. FINANCIAL ANALYSIS
  Credit score    : {fin.get('key_metrics', {}).get('credit_score')}/100  Grade {fin.get('key_metrics', {}).get('grade')}
  DSCR            : {fin.get('key_metrics', {}).get('dscr')}x
  Net margin      : {fin.get('key_metrics', {}).get('net_profit_margin')}%
  Bank turnover gap: {fin.get('key_metrics', {}).get('bank_variance_pct')}%

{fin.get('synthesis', '')}

## 3. BUREAU & FRAUD ASSESSMENT
  Commercial score: {bur.get('key_metrics', {}).get('commercial_score')}/100
  Fraud score     : {bur.get('key_metrics', {}).get('fraud_score')}/100 ({bur.get('key_metrics', {}).get('fraud_risk_level')})
  Adverse markers : {bur.get('key_metrics', {}).get('adverse_markers')}

{bur.get('synthesis', '')}

## 4. CROSS-DOCUMENT CONTRADICTIONS
  Total found : {con.get('key_metrics', {}).get('total')}
  HIGH        : {con.get('key_metrics', {}).get('high')}
  MEDIUM      : {con.get('key_metrics', {}).get('medium')}
  LOW         : {con.get('key_metrics', {}).get('low')}
{contradictions_text if contradictions_text else "  None detected."}

## 5. POLICY COMPLIANCE
  Policy pass : {pol.get('key_metrics', {}).get('policy_pass')}
  Hard stops  : {pol.get('key_metrics', {}).get('hard_stops_count')}
  Conditions  : {pol.get('key_metrics', {}).get('conditions_count')}
  Warnings    : {pol.get('key_metrics', {}).get('warnings_count')}

{pol.get('synthesis', '')}

## 6. COLLATERAL ASSESSMENT
  Blended LTV : {col.get('key_metrics', {}).get('blended_ltv_pct')}%
  Coverage    : {col.get('key_metrics', {}).get('collateral_coverage_ratio')}x
  LTV in policy: {col.get('key_metrics', {}).get('blended_ltv_within_policy')}

{col.get('synthesis', '')}

## 7. CONDITIONS TO ATTACH
{conditions_text if conditions_text else "  None — clean approval."}

## 8. HARD STOPS / DECLINE REASONS
{hard_stops_text if hard_stops_text else "  None triggered."}

## 9. UNDERWRITER DECISION & RATIONALE
Decision  : {decision}
Recorded by: {state.get('case_data', {}).get('case_id', 'N/A')} — all findings above must be addressed in the credit memo before final sanction.
"""

    print("[Orchestrator] Final report compiled.")
    return {"final_report": report}


# ── Build the graph ────────────────────────────────────────────────────────────
def build_graph():
    """
    Construct the multi-agent LangGraph.

    Fan-out: load_case_data → [5 parallel agents]
    Fan-in : all 5 agents   → aggregate_findings
    Then   : aggregate      → human_review → compile_report → END
    """
    builder = StateGraph(FullAnalysisState)

    # Nodes
    builder.add_node("load_case_data",     load_case_data)
    builder.add_node("run_financial",      run_financial)
    builder.add_node("run_bureau",         run_bureau)
    builder.add_node("run_contradiction",  run_contradiction)
    builder.add_node("run_policy",         run_policy)
    builder.add_node("run_collateral",     run_collateral)
    builder.add_node("aggregate_findings", aggregate_findings)
    builder.add_node("human_review",       human_review)
    builder.add_node("compile_report",     compile_report)

    # Entry
    builder.add_edge(START, "load_case_data")

    # Fan-out to all 5 agents in parallel
    for agent_node in [
        "run_financial", "run_bureau", "run_contradiction",
        "run_policy", "run_collateral",
    ]:
        builder.add_edge("load_case_data", agent_node)

    # Fan-in: all agents must complete before aggregation
    for agent_node in [
        "run_financial", "run_bureau", "run_contradiction",
        "run_policy", "run_collateral",
    ]:
        builder.add_edge(agent_node, "aggregate_findings")

    # Linear tail
    builder.add_edge("aggregate_findings", "human_review")
    builder.add_edge("human_review",       "compile_report")
    builder.add_edge("compile_report",      END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ── Public API used by api.py ──────────────────────────────────────────────────
def run_full_analysis(case_id: str = "SME-2024-00891"):
    """
    Run the full multi-agent analysis.
    Blocks at human_review interrupt — returns state dict with __interrupt__.
    Call resume_analysis() with a decision to complete.
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": case_id}}

    initial_state = FullAnalysisState(
        case_id=case_id,
        case_data={},
        documents={},
        financial={},
        bureau={},
        contradiction={},
        policy={},
        collateral={},
        aggregated={},
        underwriter_decision="",
        final_report="",
    )

    print(f"\n{'='*60}")
    print(f"  UNDERWRITER COPILOT — FULL MULTI-AGENT ANALYSIS")
    print(f"  Case: {case_id}")
    print(f"{'='*60}")

    result = graph.invoke(initial_state, config=config)
    return result, graph, config


def resume_analysis(graph, config: dict, decision: str):
    """
    Resume the graph after human review with underwriter's decision.
    Uses Command(resume=decision) so that interrupt() returns the decision
    string directly — not a state-update dict.
    Returns final state dict with completed final_report.
    """
    result = graph.invoke(
        Command(resume=decision),
        config=config,
    )
    return result


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result, graph, config = run_full_analysis()

    # Handle interrupt
    if result.get("__interrupt__"):
        interrupt_value = result["__interrupt__"][0]["value"]
        print(interrupt_value)
        decision = input("\nYour decision: ").strip().lower()
        result = resume_analysis(graph, config, decision)

    print("\n" + "="*60)
    print(result.get("final_report", "No report generated."))
    print("="*60)
