"""
Financial Agent
---------------
Analyzes borrower financials: income statement ratios, debt service,
credit scorecard, and bank statement behaviour.

Tools used:
  - calculate_financial_ratios  (Layer 2)
  - run_credit_scorecard        (Layer 2)
  - analyze_bank_statement      (Layer 2)
  - rag_query_financial         (Layer 3)

Returns a structured dict with ratios, scorecard, bank findings, and a
one-paragraph Claude synthesis — ready for the orchestrator to aggregate.
"""

import os
from langchain_anthropic import ChatAnthropic
from tools import (
    calculate_financial_ratios,
    run_credit_scorecard,
    analyze_bank_statement,
    rag_query_financial,
)

_model = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    temperature=0,
)


def run_financial_analysis(case_data: dict) -> dict:
    """
    Run full financial analysis for a loan case.

    Args:
        case_data: dict with keys:
            revenue, ebitda, net_profit, total_debt, total_equity,
            current_assets, current_liabilities, interest_expense,
            annual_debt_repayment, depreciation,
            monthly_credits, monthly_debits, monthly_closing_balances,
            stated_annual_turnover, cc_limit,
            bureau_score, dpd_instances_last_24_months,
            customer_concentration_pct, years_in_operation, revenue_trend

    Returns:
        dict with ratios, scorecard, bank_statement, synthesis, risk_rating
    """
    print("[FinancialAgent] Starting analysis...")

    # ── Step 1: Pull financial context from documents via RAG ──────────────────
    rag_context = rag_query_financial.invoke({
        "question": (
            "Provide the borrower's revenue, EBITDA, net profit, total debt, "
            "equity, and key financial ratios for FY2024 and FY2023."
        )
    })

    # ── Step 2: Calculate all ratios ───────────────────────────────────────────
    ratios = calculate_financial_ratios.invoke({
        "revenue":               case_data["revenue"],
        "ebitda":                case_data["ebitda"],
        "net_profit":            case_data["net_profit"],
        "total_debt":            case_data["total_debt"],
        "total_equity":          case_data["total_equity"],
        "current_assets":        case_data["current_assets"],
        "current_liabilities":   case_data["current_liabilities"],
        "interest_expense":      case_data["interest_expense"],
        "annual_debt_repayment": case_data["annual_debt_repayment"],
        "depreciation":          case_data["depreciation"],
    })

    # ── Step 3: Run credit scorecard ───────────────────────────────────────────
    scorecard = run_credit_scorecard.invoke({
        "net_profit_margin_pct":        ratios["net_profit_margin_pct"],
        "dscr":                         ratios["dscr"],
        "debt_equity_ratio":            ratios["debt_equity_ratio"],
        "bureau_score":                 case_data["bureau_score"],
        "dpd_instances_last_24_months": case_data["dpd_instances_last_24_months"],
        "customer_concentration_pct":   case_data["customer_concentration_pct"],
        "years_in_operation":           case_data["years_in_operation"],
        "revenue_trend":                case_data["revenue_trend"],
    })

    # ── Step 4: Bank statement analysis ───────────────────────────────────────
    bank = analyze_bank_statement.invoke({
        "monthly_credits":          case_data["monthly_credits"],
        "monthly_debits":           case_data["monthly_debits"],
        "monthly_closing_balances": case_data["monthly_closing_balances"],
        "stated_annual_turnover":   case_data["stated_annual_turnover"],
        "cc_limit":                 case_data["cc_limit"],
    })

    # ── Step 5: Claude synthesis ───────────────────────────────────────────────
    synthesis_prompt = f"""You are a senior credit analyst. Write a concise 3-paragraph financial
assessment (max 150 words total) based on the data below. Be direct — no filler.

FINANCIAL RATIOS:
  Net profit margin : {ratios['net_profit_margin_pct']}%
  DSCR              : {ratios['dscr']}x
  Debt / Equity     : {ratios['debt_equity_ratio']}x
  Current ratio     : {ratios['current_ratio']}x
  ICR               : {ratios['interest_coverage_ratio']}x
  Flags             : {ratios['flags']}

CREDIT SCORECARD:
  Score      : {scorecard['total_score']}/100
  Grade      : {scorecard['grade']}
  Risk band  : {scorecard['risk_band']}
  Result     : {scorecard['recommendation']}

BANK STATEMENT (3 months):
  Annualised credits : ₹{bank['annualised_credits_lakhs']}L
  Stated turnover    : ₹{bank['stated_turnover_lakhs']}L
  Variance           : {bank['turnover_variance_pct']}%
  Bank flags         : {bank['flags']}

RAG CONTEXT FROM DOCUMENTS:
{rag_context[:500]}

Paragraph 1: Financial health summary.
Paragraph 2: Key strengths.
Paragraph 3: Key concerns and risk rating (LOW / MEDIUM / HIGH)."""

    synthesis_response = _model.invoke(synthesis_prompt)

    # Extract risk rating from synthesis
    synthesis_text = synthesis_response.content
    risk_rating = "MEDIUM"
    for level in ["HIGH", "MEDIUM", "LOW"]:
        if level in synthesis_text.upper():
            risk_rating = level
            break

    print(f"[FinancialAgent] Done — score={scorecard['total_score']}/100, "
          f"grade={scorecard['grade']}, risk={risk_rating}")

    return {
        "agent":         "financial",
        "status":        "completed",
        "ratios":        ratios,
        "scorecard":     scorecard,
        "bank_statement": bank,
        "rag_context":   rag_context,
        "synthesis":     synthesis_text,
        "risk_rating":   risk_rating,
        "key_metrics": {
            "dscr":                ratios["dscr"],
            "net_profit_margin":   ratios["net_profit_margin_pct"],
            "credit_score":        scorecard["total_score"],
            "grade":               scorecard["grade"],
            "bank_variance_pct":   bank["turnover_variance_pct"],
        },
    }
