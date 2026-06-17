"""
Ground truth and test inputs for the eval suite.
Based on Meridian Textile Exports Pvt. Ltd. — Case SME-2024-00891.

All expected values are derived from the actual tool implementations
and the known Meridian financials — not assumed.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Full case data (mirrors MERIDIAN_CASE_DATA in agent.py) ──────────────────
CASE_DATA = {
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
    # Bank statement (3-month sample)
    "monthly_credits":          [29.00, 26.00, 63.20],
    "monthly_debits":           [28.59, 28.74, 59.85],
    "monthly_closing_balances": [8.84, 6.09, 9.44],
    "stated_annual_turnover":   680,
    "cc_limit":                 40,
    # Bureau / scorecard
    "bureau_score":                    74,
    "bureau_score_commercial":         74,
    "dpd_instances_last_24_months":    1,
    "customer_concentration_pct":      67,
    "years_in_operation":              10,
    "revenue_trend":                   "GROWING",
    "net_profit_margin_pct":           8.1,
    # Fraud / risk flags
    "cc_cycling_detected":             True,
    "revenue_variance_pct":            6.5,
    "bank_turnover_variance_pct":      30.4,
    "unanswered_enquiries":            2,
    "undisclosed_contingent":          True,
    # Policy engine inputs
    "dscr":                            2.18,   # pre-computed to avoid race condition
    "blended_ltv":                     0.517,
    "property_ltv":                    0.865,
    "dpd_60_plus_last_36_months":      False,
    "npa_last_60_months":              False,
    "wilful_defaulter":                False,
    "revenue_overstatement_pct":       6.5,
    "undisclosed_contingent_liability": True,
    "epcg_obligation_lakhs":           42,
    "sector":                          "textile",
    "promoter_net_worth_lakhs":        380,
    # Collateral
    "property_market_value":           312,
    "machinery_value":                 210,
    "existing_charge_outstanding":     22,
    "property_type":                   "industrial",
}

# ── Expected outputs (ground truth for assertions) ────────────────────────────
GROUND_TRUTH = {
    # Financial ratios
    "dscr_min":         1.8,
    "dscr_max":         2.6,
    "net_margin_min":   6.0,
    "net_margin_max":  12.0,
    "current_ratio_min": 1.0,
    "icr_min":          3.0,

    # Credit scorecard
    "credit_score_min":  45,
    "credit_score_max":  75,
    "grade_options":    ["B", "C", "D"],   # given the risk profile

    # Layer 1 — Bureau
    "commercial_score":         74,
    "unanswered_enquiries":      2,
    "wilful_defaulter":         False,

    # Layer 2 — Policy: these rules MUST be flagged as conditions or worse
    "required_conditions_keywords": [
        "customer_concentration",   # 67% exposure to one buyer
        "cc_cycling",               # CC cycling detected
    ],
    "required_warnings_keywords": [
        "property_ltv",             # 86.5% vs ~65% policy limit
    ],

    # Layer 2 — LTV
    "property_ltv_pct_min":   80.0,     # known to be ~86.5% — a breach
    "blended_ltv_pct_min":    45.0,
    "blended_ltv_pct_max":    60.0,
    "coverage_ratio_min":      1.5,

    # Layer 2 — Bank statement
    "turnover_variance_min":  25.0,     # known ~30%+ gap between bank credits and stated
    "annualised_credits_min": 350,      # 3-month sample annualised

    # Orchestrator E2E
    "overall_risk":           "HIGH",
    "system_recommendation":  "DECLINE",   # auto_decline expected from policy hard stops

    # RAG keyword maps — fragments that MUST appear in a good answer
    "rag_revenue":    ["421", "394", "lakh", "revenue"],
    "rag_score":      ["74", "score", "bureau", "experian"],
    "rag_collateral": ["property", "machinery", "midc", "collateral"],
    "rag_policy_dscr": ["1.25", "dscr", "minimum", "debt service"],
}

# ── RAG test cases ────────────────────────────────────────────────────────────
RAG_TESTS = [
    {
        "id":        "revenue",
        "question":  "What is the borrower's annual revenue for FY2024?",
        "tool":      "general",
        "keywords":  GROUND_TRUTH["rag_revenue"],
        "min_score": 0.4,
        "hint":      "Documents state ₹394–421 lakh revenue for FY2024",
    },
    {
        "id":        "credit_score",
        "question":  "What is the commercial credit score?",
        "tool":      "general",
        "keywords":  GROUND_TRUTH["rag_score"],
        "min_score": 0.4,
        "hint":      "Experian commercial credit score is 74 out of 100",
    },
    {
        "id":        "collateral",
        "question":  "What collateral is offered and what is its approximate value?",
        "tool":      "collateral",
        "keywords":  GROUND_TRUTH["rag_collateral"],
        "min_score": 0.4,
        "hint":      "Industrial property at MIDC Pune and machinery offered as collateral",
    },
    {
        "id":        "policy_dscr",
        "question":  "What is the minimum DSCR required for SME term loans?",
        "tool":      "policy",
        "keywords":  GROUND_TRUTH["rag_policy_dscr"],
        "min_score": 0.4,
        "hint":      "Policy requires minimum DSCR of 1.25x for SME loans",
    },
]

# LLM-as-judge test cases (first 2 from RAG_TESTS)
LLM_JUDGE_TESTS = RAG_TESTS[:2]


def load_documents() -> dict:
    """Load all .txt/.pdf files from loan_docs/ for the contradiction agent eval."""
    docs = {}
    docs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "loan_docs"
    )
    if not os.path.exists(docs_dir):
        return docs
    for fname in sorted(os.listdir(docs_dir)):
        if fname.endswith((".txt", ".pdf")):
            try:
                with open(os.path.join(docs_dir, fname), "r", encoding="utf-8", errors="ignore") as f:
                    docs[fname] = f.read()
            except Exception:
                pass
    return docs
