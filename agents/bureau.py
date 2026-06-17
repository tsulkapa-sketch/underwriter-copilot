"""
Bureau Agent
------------
Fetches and interprets the commercial credit bureau report and
CRILC / RBI registry checks.

Tools used:
  - call_bureau_api   (Layer 1)
  - call_crilc_check  (Layer 1)
  - call_fraud_model  (Layer 1)
  - rag_query_bureau  (Layer 3)

Returns bureau score, adverse markers, fraud risk, CRILC status,
and a Claude synthesis with an overall bureau risk rating.
"""

import os
from langchain_anthropic import ChatAnthropic
from tools import (
    call_bureau_api,
    call_crilc_check,
    call_fraud_model,
    rag_query_bureau,
)

_model = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    temperature=0,
)


def run_bureau_analysis(case_data: dict) -> dict:
    """
    Run credit bureau and fraud analysis for a loan case.

    Args:
        case_data: dict with keys:
            pan, case_id, promoter_pans,
            cc_cycling_detected, revenue_variance_pct,
            bank_turnover_variance_pct, unanswered_enquiries,
            undisclosed_contingent

    Returns:
        dict with bureau_report, crilc, fraud, synthesis, risk_rating
    """
    print("[BureauAgent] Starting analysis...")

    # ── Step 1: Pull bureau data from RAG (document perspective) ─────────────
    rag_context = rag_query_bureau.invoke({
        "question": (
            "What is the credit score, DPD history, adverse markers, "
            "and enquiry details from the bureau report?"
        )
    })

    # ── Step 2: Call bureau API ───────────────────────────────────────────────
    bureau_report = call_bureau_api.invoke({
        "pan":     case_data["pan"],
        "case_id": case_data["case_id"],
    })

    # ── Step 3: CRILC / RBI check ─────────────────────────────────────────────
    crilc = call_crilc_check.invoke({
        "entity_pan":    case_data["pan"],
        "promoter_pans": case_data["promoter_pans"],
    })

    # ── Step 4: Fraud model ───────────────────────────────────────────────────
    fraud = call_fraud_model.invoke({
        "case_id": case_data["case_id"],
        "application_data": {
            "cc_cycling_detected":       case_data.get("cc_cycling_detected", False),
            "revenue_variance_pct":      case_data.get("revenue_variance_pct", 0),
            "bank_turnover_variance_pct": case_data.get("bank_turnover_variance_pct", 0),
            "unanswered_enquiries":      case_data.get("unanswered_enquiries", 0),
            "undisclosed_contingent":    case_data.get("undisclosed_contingent", False),
        },
    })

    # ── Step 5: Claude synthesis ───────────────────────────────────────────────
    adverse_markers = bureau_report.get("adverse_markers", [])
    promoter_scores = bureau_report.get("promoter_scores", {})
    promoter_summary = ", ".join(
        f"{name}: {data['score']}/{data['out_of']}"
        for name, data in promoter_scores.items()
    )

    synthesis_prompt = f"""You are a senior credit analyst reviewing bureau and fraud findings.
Write a concise 3-paragraph assessment (max 150 words). Be direct.

BUREAU REPORT:
  Commercial score    : {bureau_report.get('commercial_score')}/100
  Risk category       : {bureau_report.get('risk_category')}
  Adverse markers     : {len(adverse_markers)} ({[m['type'] for m in adverse_markers]})
  Unanswered enquiries: {bureau_report.get('unanswered_enquiries')}
  NPA history         : {bureau_report.get('npa_history')}
  Wilful defaulter    : {bureau_report.get('wilful_defaulter')}
  Promoter scores     : {promoter_summary}

CRILC / RBI STATUS:
  Entity status       : {crilc.get('entity_crilc_status')}
  Entity NPA          : {crilc.get('entity_npa')}
  Wilful defaulter    : {crilc.get('wilful_defaulter_entity')}
  Fraud registry      : {crilc.get('fraud_registry')}

FRAUD MODEL:
  Fraud score         : {fraud.get('fraud_score')}/100
  Risk level          : {fraud.get('risk_level')}
  Signals             : {fraud.get('total_signals')} ({fraud.get('flags')})
  Recommendation      : {fraud.get('recommendation')}

RAG CONTEXT FROM BUREAU DOCUMENT:
{rag_context[:400]}

Paragraph 1: Credit track record summary.
Paragraph 2: Fraud risk assessment.
Paragraph 3: Overall bureau risk rating (LOW / MEDIUM / HIGH) with key reason."""

    synthesis_response = _model.invoke(synthesis_prompt)
    synthesis_text = synthesis_response.content

    risk_rating = "MEDIUM"
    for level in ["HIGH", "MEDIUM", "LOW"]:
        if level in synthesis_text.upper():
            risk_rating = level
            break

    # Hard stops from bureau
    hard_stop_flags = []
    if bureau_report.get("wilful_defaulter"):
        hard_stop_flags.append("WILFUL_DEFAULTER")
    if bureau_report.get("npa_history"):
        hard_stop_flags.append("NPA_HISTORY")
    if crilc.get("wilful_defaulter_entity"):
        hard_stop_flags.append("CRILC_WILFUL_DEFAULTER")
    if fraud.get("risk_level") == "HIGH":
        hard_stop_flags.append("HIGH_FRAUD_RISK")

    print(f"[BureauAgent] Done — bureau_score={bureau_report.get('commercial_score')}, "
          f"fraud_score={fraud.get('fraud_score')}, risk={risk_rating}")

    return {
        "agent":         "bureau",
        "status":        "completed",
        "bureau_report": bureau_report,
        "crilc":         crilc,
        "fraud":         fraud,
        "rag_context":   rag_context,
        "synthesis":     synthesis_text,
        "risk_rating":   risk_rating,
        "hard_stop_flags": hard_stop_flags,
        "key_metrics": {
            "commercial_score":   bureau_report.get("commercial_score"),
            "fraud_score":        fraud.get("fraud_score"),
            "fraud_risk_level":   fraud.get("risk_level"),
            "adverse_markers":    len(adverse_markers),
            "unanswered_enquiries": bureau_report.get("unanswered_enquiries"),
        },
    }
