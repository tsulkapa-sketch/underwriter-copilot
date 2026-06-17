"""
Collateral Agent
----------------
Assesses collateral adequacy, LTV ratios, and coverage.

Tools used:
  - call_valuation_system   (Layer 1)
  - calculate_ltv           (Layer 2)
  - rag_query_collateral    (Layer 3)

Returns LTV calculations, coverage ratios, existing charges,
and a Claude synthesis with overall collateral adequacy rating.
"""

import os
from langchain_anthropic import ChatAnthropic
from tools import call_valuation_system, calculate_ltv, rag_query_collateral

_model = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    temperature=0,
)


def run_collateral_analysis(case_data: dict) -> dict:
    """
    Run collateral assessment for a loan case.

    Args:
        case_data: dict with keys:
            case_id, property_address, loan_amount_lakhs,
            property_market_value, machinery_value,
            existing_charge_outstanding, property_type

    Returns:
        dict with valuation, ltv, synthesis, risk_rating
    """
    print("[CollateralAgent] Starting analysis...")

    # ── Step 1: RAG — collateral details from documents ───────────────────────
    rag_context = rag_query_collateral.invoke({
        "question": (
            "What collateral is offered? Provide property type, market value, "
            "any existing charges, and machinery details."
        )
    })

    # ── Step 2: Fetch valuation from valuation system ─────────────────────────
    valuation = call_valuation_system.invoke({
        "property_address": case_data["property_address"],
        "case_id":          case_data["case_id"],
    })

    # ── Step 3: Calculate LTV ─────────────────────────────────────────────────
    ltv = calculate_ltv.invoke({
        "loan_amount":                case_data["loan_amount_lakhs"],
        "property_market_value":      case_data["property_market_value"],
        "machinery_value":            case_data["machinery_value"],
        "existing_charge_outstanding": case_data["existing_charge_outstanding"],
        "property_type":              case_data["property_type"],
    })

    # ── Step 4: Claude synthesis ───────────────────────────────────────────────
    synthesis_prompt = f"""You are a senior credit analyst assessing collateral.
Write a concise 3-paragraph assessment (max 150 words). Be specific.

VALUATION:
  Total market value    : ₹{valuation['total_collateral_market_value']/100000:.1f}L
  Total forced sale     : ₹{valuation['total_collateral_forced_sale']/100000:.1f}L
  Blended LTV           : {valuation['blended_ltv']:.1%}
  Coverage ratio        : {valuation['collateral_coverage_ratio']:.2f}x
  Existing charge (SBI) : ₹{valuation['immovable_property']['existing_charge']['outstanding']/100000:.1f}L
  Valuation valid until : {valuation['valuation_valid_until']}

LTV ANALYSIS:
  Loan amount           : ₹{ltv['loan_amount']}L
  Property LTV          : {ltv['property_ltv_pct']}% (policy limit: {ltv['policy_ltv_limit']*100:.0f}%)
  Property LTV in policy: {ltv['property_ltv_within_policy']}
  Blended LTV           : {ltv['blended_ltv_pct']}%
  Blended LTV in policy : {ltv['blended_ltv_within_policy']}
  Coverage ratio        : {ltv['collateral_coverage_ratio']}x
  Min coverage met (1.25x): {ltv['minimum_coverage_met']}

RAG CONTEXT FROM DOCUMENTS:
{rag_context[:400]}

Paragraph 1: Collateral adequacy summary.
Paragraph 2: Key risks (LTV breach, existing charge, valuation concerns).
Paragraph 3: Overall collateral risk rating (LOW / MEDIUM / HIGH) with rationale."""

    synthesis_response = _model.invoke(synthesis_prompt)
    synthesis_text = synthesis_response.content

    # Risk rating logic — primary LTV breach is a strong signal
    if not ltv["property_ltv_within_policy"] and not ltv["blended_ltv_within_policy"]:
        risk_rating = "HIGH"
    elif not ltv["property_ltv_within_policy"] or not ltv["minimum_coverage_met"]:
        risk_rating = "MEDIUM"
    else:
        risk_rating = "LOW"

    print(f"[CollateralAgent] Done — property_ltv={ltv['property_ltv_pct']}%, "
          f"blended_ltv={ltv['blended_ltv_pct']}%, "
          f"coverage={ltv['collateral_coverage_ratio']}x, risk={risk_rating}")

    return {
        "agent":      "collateral",
        "status":     "completed",
        "valuation":  valuation,
        "ltv":        ltv,
        "rag_context": rag_context,
        "synthesis":  synthesis_text,
        "risk_rating": risk_rating,
        "key_metrics": {
            "property_ltv_pct":          ltv["property_ltv_pct"],
            "blended_ltv_pct":           ltv["blended_ltv_pct"],
            "collateral_coverage_ratio":  ltv["collateral_coverage_ratio"],
            "property_ltv_within_policy": ltv["property_ltv_within_policy"],
            "blended_ltv_within_policy":  ltv["blended_ltv_within_policy"],
            "minimum_coverage_met":       ltv["minimum_coverage_met"],
        },
    }
