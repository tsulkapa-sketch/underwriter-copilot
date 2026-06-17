"""
Policy Agent
------------
Runs the policy rules engine and checks the case against lending policy.
Produces hard stops, conditions, warnings, and a Claude synthesis.

Tools used:
  - run_policy_rules_engine  (Layer 2)
  - rag_query_policy         (Layer 3)

Returns policy pass/fail status, all rule violations, and a synthesis
so the orchestrator knows exactly what conditions to attach to the sanction.
"""

import os
from langchain_anthropic import ChatAnthropic
from tools import run_policy_rules_engine, rag_query_policy

_model = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    temperature=0,
)


def run_policy_analysis(case_data: dict) -> dict:
    """
    Run policy compliance check for a loan case.

    Args:
        case_data: dict with keys:
            loan_amount_lakhs, dscr, blended_ltv, property_ltv,
            customer_concentration_pct, bureau_score_commercial,
            dpd_60_plus_last_36_months, npa_last_60_months,
            wilful_defaulter, unanswered_enquiries,
            cc_cycling_detected, revenue_overstatement_pct,
            undisclosed_contingent_liability, epcg_obligation_lakhs,
            annual_revenue_lakhs (same as revenue), sector,
            years_in_operation, promoter_net_worth_lakhs

    Returns:
        dict with policy_result, synthesis, risk_rating
    """
    print("[PolicyAgent] Starting analysis...")

    # ── Step 1: RAG — check policy document for specific rules ─────────────────
    rag_context = rag_query_policy.invoke({
        "question": (
            "What are the minimum DSCR, LTV, bureau score, and customer "
            "concentration thresholds for SME term loans? What are the "
            "hard stop conditions?"
        )
    })

    # ── Step 2: Run deterministic policy rules engine ─────────────────────────
    policy_result = run_policy_rules_engine.invoke({
        "loan_amount_lakhs":                case_data["loan_amount_lakhs"],
        "dscr":                             case_data["dscr"],
        "blended_ltv":                      case_data["blended_ltv"],
        "property_ltv":                     case_data["property_ltv"],
        "customer_concentration_pct":       case_data["customer_concentration_pct"],
        "bureau_score_commercial":          case_data["bureau_score_commercial"],
        "dpd_60_plus_last_36_months":       case_data["dpd_60_plus_last_36_months"],
        "npa_last_60_months":               case_data["npa_last_60_months"],
        "wilful_defaulter":                 case_data["wilful_defaulter"],
        "unanswered_enquiries":             case_data["unanswered_enquiries"],
        "cc_cycling_detected":              case_data["cc_cycling_detected"],
        "revenue_overstatement_pct":        case_data["revenue_overstatement_pct"],
        "undisclosed_contingent_liability": case_data["undisclosed_contingent_liability"],
        "epcg_obligation_lakhs":            case_data["epcg_obligation_lakhs"],
        "annual_revenue_lakhs":             case_data["revenue"],
        "sector":                           case_data["sector"],
        "years_in_operation":               case_data["years_in_operation"],
        "promoter_net_worth_lakhs":         case_data["promoter_net_worth_lakhs"],
    })

    # ── Step 3: Claude synthesis ───────────────────────────────────────────────
    hard_stops_text = (
        "\n".join(
            f"  [{h['rule']}] {h['finding']}"
            for h in policy_result["hard_stops"]
        ) or "  None"
    )
    conditions_text = (
        "\n".join(
            f"  [{c['rule']}] {c['finding']}\n  → {c['condition']}"
            for c in policy_result["conditions"]
        ) or "  None"
    )
    warnings_text = (
        "\n".join(
            f"  [{w['rule']}] {w['finding']}"
            for w in policy_result["warnings"]
        ) or "  None"
    )

    synthesis_prompt = f"""You are a senior credit policy officer. Write a concise 3-paragraph
policy assessment (max 150 words). Be direct and specific.

POLICY ENGINE RESULT:
  Policy pass   : {policy_result['policy_pass']}
  Auto decline  : {policy_result['auto_decline']}
  Hard stops    : {policy_result['summary']['hard_stops_count']}
  Conditions    : {policy_result['summary']['conditions_count']}
  Warnings      : {policy_result['summary']['warnings_count']}

HARD STOPS:
{hard_stops_text}

CONDITIONS TO ATTACH:
{conditions_text}

WARNINGS:
{warnings_text}

Paragraph 1: Overall policy compliance status.
Paragraph 2: Key conditions that must be attached if approved.
Paragraph 3: Overall policy risk rating (LOW / MEDIUM / HIGH) and recommendation."""

    synthesis_response = _model.invoke(synthesis_prompt)
    synthesis_text = synthesis_response.content

    risk_rating = "HIGH" if not policy_result["policy_pass"] else "MEDIUM"
    for level in ["HIGH", "MEDIUM", "LOW"]:
        if level in synthesis_text.upper():
            risk_rating = level
            break

    print(f"[PolicyAgent] Done — policy_pass={policy_result['policy_pass']}, "
          f"hard_stops={policy_result['summary']['hard_stops_count']}, "
          f"conditions={policy_result['summary']['conditions_count']}")

    return {
        "agent":         "policy",
        "status":        "completed",
        "policy_result": policy_result,
        "rag_context":   rag_context,
        "synthesis":     synthesis_text,
        "risk_rating":   risk_rating,
        "key_metrics": {
            "policy_pass":       policy_result["policy_pass"],
            "auto_decline":      policy_result["auto_decline"],
            "hard_stops_count":  policy_result["summary"]["hard_stops_count"],
            "conditions_count":  policy_result["summary"]["conditions_count"],
            "warnings_count":    policy_result["summary"]["warnings_count"],
        },
    }
