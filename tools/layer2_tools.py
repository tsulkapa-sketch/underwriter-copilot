"""
Layer 2 — Deterministic tools

Pure Python functions — no LLM involved.
Same inputs always produce identical outputs.
Fully auditable and reproducible.
These are the tools agents call to compute scores and check rules.

In production these would connect to:
  - Core credit scoring engine
  - Policy rules management system
  - Financial ratio calculation service
"""

from langchain_core.tools import tool


# ── Financial Ratio Calculator ─────────────────────────────────────────────────

@tool
def calculate_financial_ratios(
    revenue: float,
    ebitda: float,
    net_profit: float,
    total_debt: float,
    total_equity: float,
    current_assets: float,
    current_liabilities: float,
    interest_expense: float,
    annual_debt_repayment: float,
    depreciation: float
) -> dict:
    """
    Calculate all standard credit ratios from financial inputs.
    All inputs in same currency unit (lakhs).
    Returns ratios used for credit assessment.
    """
    # Profitability
    net_profit_margin    = round(net_profit / revenue * 100, 2) if revenue else 0
    ebitda_margin        = round(ebitda / revenue * 100, 2) if revenue else 0

    # Leverage
    debt_equity_ratio    = round(total_debt / total_equity, 2) if total_equity else 0

    # Liquidity
    current_ratio        = round(current_assets / current_liabilities, 2) if current_liabilities else 0

    # Debt service
    # DSCR = (Net Profit + Depreciation + Interest) / (Annual Repayment + Interest)
    numerator            = net_profit + depreciation + interest_expense
    denominator          = annual_debt_repayment + interest_expense
    dscr                 = round(numerator / denominator, 2) if denominator else 0

    # Interest coverage
    icr                  = round(ebitda / interest_expense, 2) if interest_expense else 0

    # Return on equity
    roe                  = round(net_profit / total_equity * 100, 2) if total_equity else 0

    return {
        "net_profit_margin_pct":  net_profit_margin,
        "ebitda_margin_pct":      ebitda_margin,
        "debt_equity_ratio":      debt_equity_ratio,
        "current_ratio":          current_ratio,
        "dscr":                   dscr,
        "interest_coverage_ratio": icr,
        "return_on_equity_pct":   roe,
        "flags": {
            "dscr_below_125":     dscr < 1.25,
            "dscr_below_100":     dscr < 1.00,
            "current_ratio_low":  current_ratio < 1.20,
            "icr_low":            icr < 2.0,
            "high_leverage":      debt_equity_ratio > 2.0,
        }
    }


# ── Credit Scorecard ───────────────────────────────────────────────────────────

@tool
def run_credit_scorecard(
    net_profit_margin_pct: float,
    dscr: float,
    debt_equity_ratio: float,
    bureau_score: int,
    dpd_instances_last_24_months: int,
    customer_concentration_pct: float,
    years_in_operation: int,
    revenue_trend: str
) -> dict:
    """
    Run deterministic credit scorecard.
    Returns weighted score, grade, and section breakdown.
    Same inputs always produce same score — fully auditable.

    revenue_trend: 'GROWING' / 'STABLE' / 'DECLINING'
    """
    score = 0
    breakdown = {}

    # ── Section 1: Profitability (25 points) ──
    if net_profit_margin_pct >= 15:       pts = 25
    elif net_profit_margin_pct >= 10:     pts = 20
    elif net_profit_margin_pct >= 7:      pts = 15
    elif net_profit_margin_pct >= 5:      pts = 10
    else:                                 pts = 3
    score += pts
    breakdown["profitability"] = {
        "score": pts, "max": 25,
        "input": f"Net margin {net_profit_margin_pct}%"
    }

    # ── Section 2: Debt Service (25 points) ──
    if dscr >= 2.0:       pts = 25
    elif dscr >= 1.75:    pts = 22
    elif dscr >= 1.50:    pts = 18
    elif dscr >= 1.25:    pts = 12
    elif dscr >= 1.00:    pts = 5
    else:                 pts = 0
    score += pts
    breakdown["debt_service"] = {
        "score": pts, "max": 25,
        "input": f"DSCR {dscr}x"
    }

    # ── Section 3: Bureau and Track Record (25 points) ──
    if bureau_score >= 85:      pts = 25
    elif bureau_score >= 75:    pts = 20
    elif bureau_score >= 65:    pts = 14
    elif bureau_score >= 55:    pts = 8
    else:                       pts = 0
    # Penalise DPD instances
    dpd_penalty = dpd_instances_last_24_months * 5
    pts = max(0, pts - dpd_penalty)
    score += pts
    breakdown["bureau_track_record"] = {
        "score": pts, "max": 25,
        "input": f"Bureau score {bureau_score}, DPD instances {dpd_instances_last_24_months}",
        "dpd_penalty": dpd_penalty
    }

    # ── Section 4: Leverage (15 points) ──
    if debt_equity_ratio <= 0.5:      pts = 15
    elif debt_equity_ratio <= 1.0:    pts = 12
    elif debt_equity_ratio <= 1.5:    pts = 8
    elif debt_equity_ratio <= 2.0:    pts = 4
    elif debt_equity_ratio <= 3.0:    pts = 1
    else:                             pts = 0
    score += pts
    breakdown["leverage"] = {
        "score": pts, "max": 15,
        "input": f"D/E ratio {debt_equity_ratio}x"
    }

    # ── Section 5: Concentration Risk (10 points) ──
    if customer_concentration_pct <= 30:      pts = 10
    elif customer_concentration_pct <= 40:    pts = 8
    elif customer_concentration_pct <= 50:    pts = 5
    elif customer_concentration_pct <= 60:    pts = 2
    else:                                     pts = 0
    score += pts
    breakdown["concentration_risk"] = {
        "score": pts, "max": 10,
        "input": f"Top customer concentration {customer_concentration_pct}%"
    }

    # Determine grade and recommendation
    if score >= 85:
        grade = "A"
        recommendation = "APPROVE"
        risk_band = "LOW RISK"
    elif score >= 70:
        grade = "B+"
        recommendation = "APPROVE"
        risk_band = "LOW-MEDIUM RISK"
    elif score >= 60:
        grade = "B"
        recommendation = "CONDITIONAL_APPROVE"
        risk_band = "MEDIUM RISK"
    elif score >= 45:
        grade = "C"
        recommendation = "CONDITIONAL_APPROVE"
        risk_band = "MEDIUM-HIGH RISK"
    else:
        grade = "D"
        recommendation = "DECLINE"
        risk_band = "HIGH RISK"

    return {
        "total_score":      score,
        "max_score":        100,
        "grade":            grade,
        "risk_band":        risk_band,
        "recommendation":   recommendation,
        "breakdown":        breakdown,
        "scorecard_version": "ACB-SC-SME-v2.1"
    }


# ── Policy Rules Engine ────────────────────────────────────────────────────────

@tool
def run_policy_rules_engine(
    loan_amount_lakhs: float,
    dscr: float,
    blended_ltv: float,
    property_ltv: float,
    customer_concentration_pct: float,
    bureau_score_commercial: int,
    dpd_60_plus_last_36_months: bool,
    npa_last_60_months: bool,
    wilful_defaulter: bool,
    unanswered_enquiries: int,
    cc_cycling_detected: bool,
    revenue_overstatement_pct: float,
    undisclosed_contingent_liability: bool,
    epcg_obligation_lakhs: float,
    annual_revenue_lakhs: float,
    sector: str,
    years_in_operation: int,
    promoter_net_worth_lakhs: float
) -> dict:
    """
    Run complete policy rules engine.
    Returns hard stops, conditions, warnings, and policy compliance status.
    Every rule is binary — pass or fail. No LLM involved.
    """
    hard_stops  = []   # automatic decline — cannot be overridden
    conditions  = []   # sanction with specific conditions attached
    warnings    = []   # must be noted and acknowledged in credit memo
    info        = []   # informational — no action required

    # ── HARD STOPS ──────────────────────────────────────────────
    if wilful_defaulter:
        hard_stops.append({
            "rule": "WD001",
            "finding": "Promoter / entity on wilful defaulter list",
            "policy_ref": "Section 2.2(b)"
        })

    if npa_last_60_months:
        hard_stops.append({
            "rule": "NPA001",
            "finding": "NPA classification in last 60 months",
            "policy_ref": "Section 2.2(a)"
        })

    if dpd_60_plus_last_36_months:
        hard_stops.append({
            "rule": "DPD001",
            "finding": "60+ DPD instance in last 36 months",
            "policy_ref": "Section 7.2"
        })

    if dscr < 1.00:
        hard_stops.append({
            "rule": "DSCR001",
            "finding": f"DSCR of {dscr}x is below minimum 1.00x — loan cannot be serviced",
            "policy_ref": "Section 3.1"
        })

    if bureau_score_commercial < 55:
        hard_stops.append({
            "rule": "BURO001",
            "finding": f"Commercial bureau score {bureau_score_commercial} below minimum 55",
            "policy_ref": "Section 2.1(e)"
        })

    if customer_concentration_pct > 75 and loan_amount_lakhs > 100:
        hard_stops.append({
            "rule": "CONC001",
            "finding": f"Customer concentration {customer_concentration_pct}% exceeds 75% hard stop for facilities above ₹1 crore",
            "policy_ref": "Section 5.2"
        })

    if sector.lower() in ["tobacco", "gambling", "crypto", "arms"]:
        hard_stops.append({
            "rule": "SECT001",
            "finding": f"Sector '{sector}' is on the negative list",
            "policy_ref": "Section 8.4"
        })

    # ── CONDITIONS ───────────────────────────────────────────────
    if dscr < 1.25 and dscr >= 1.00:
        conditions.append({
            "rule": "DSCR002",
            "finding": f"DSCR of {dscr}x is below preferred 1.25x",
            "condition": "Additional collateral or personal guarantee required to cover shortfall",
            "policy_ref": "Section 3.1"
        })

    if customer_concentration_pct > 60 and customer_concentration_pct <= 75:
        conditions.append({
            "rule": "CONC002",
            "finding": f"Top 2 customers account for {customer_concentration_pct}% of revenue — exceeds 60% threshold",
            "condition": "Borrower must submit written diversification plan with 18-month milestones. Mandatory 12-month review.",
            "policy_ref": "Section 5.2"
        })

    if loan_amount_lakhs > 500:
        conditions.append({
            "rule": "EXP001",
            "finding": f"Exposure of ₹{loan_amount_lakhs} lakh exceeds ₹5 crore threshold",
            "condition": "Mandatory Credit Committee approval required before sanction",
            "policy_ref": "Section 4.1"
        })

    if property_ltv > 0.65:
        conditions.append({
            "rule": "LTV001",
            "finding": f"Primary property LTV of {property_ltv*100:.1f}% exceeds 65% policy limit for industrial property",
            "condition": "Acceptable on blended LTV basis — document justification in credit memo. Obtain senior credit officer sign-off.",
            "policy_ref": "Section 4.2"
        })

    if undisclosed_contingent_liability:
        conditions.append({
            "rule": "CONT001",
            "finding": "Material contingent liability (EPCG obligation) not disclosed in loan application",
            "condition": "Borrower must provide written explanation for non-disclosure. Escalate to credit committee regardless of amount per policy.",
            "policy_ref": "Section 9.3"
        })

    if unanswered_enquiries >= 2:
        conditions.append({
            "rule": "ENQ001",
            "finding": f"{unanswered_enquiries} unanswered credit enquiries in last 12 months",
            "condition": "Borrower must provide written explanation for each unanswered enquiry (Kotak Nov-23, SIDBI Jun-23)",
            "policy_ref": "Section 7.3"
        })

    # ── WARNINGS ─────────────────────────────────────────────────
    if cc_cycling_detected:
        warnings.append({
            "rule": "FRAU001",
            "finding": "CC cycling detected — full repayment and same-day redrawdown within 30 days of loan application",
            "warning": "Per Section 7.4, this is a mandatory fraud desk referral trigger. Fraud desk clearance required before sanction.",
            "policy_ref": "Section 7.4"
        })

    if revenue_overstatement_pct > 5:
        warnings.append({
            "rule": "FRAU002",
            "finding": f"Revenue overstated by {revenue_overstatement_pct:.1f}% in application vs GST-verified figures",
            "warning": "Use audited/GST revenue (₹395 lakh) for all calculations. If overstatement is intentional, treat as fraud indicator.",
            "policy_ref": "Sections 3.1, 7.4"
        })

    if epcg_obligation_lakhs > 0:
        epcg_pct_of_revenue = epcg_obligation_lakhs / annual_revenue_lakhs * 100
        warnings.append({
            "rule": "CONT002",
            "finding": f"EPCG export obligation of ₹{epcg_obligation_lakhs} lakh ({epcg_pct_of_revenue:.1f}% of revenue)",
            "warning": "Stress test repayment if EPCG obligation is triggered. Customs duty recovery could impact cash flows.",
            "policy_ref": "Section 9.3"
        })

    if years_in_operation < 5:
        warnings.append({
            "rule": "VIN001",
            "finding": f"Borrower has {years_in_operation} years of operation — relatively early stage",
            "warning": "Limited track record. Ensure minimum 3 years audited financials reviewed.",
            "policy_ref": "Section 2.1(a)"
        })

    # ── INFORMATIONAL ─────────────────────────────────────────────
    loan_to_nw = loan_amount_lakhs / promoter_net_worth_lakhs if promoter_net_worth_lakhs else 0
    if loan_to_nw > 0.5:
        info.append({
            "rule": "INFO001",
            "finding": f"Loan to promoter net worth ratio: {loan_to_nw:.2f}x",
            "note": "Within acceptable range but note in credit memo"
        })

    info.append({
        "rule": "INFO002",
        "finding": f"Sector: {sector} — Standard sector, normal policy applies",
        "note": "No sector-specific restrictions. Export-oriented manufacturing noted.",
        "policy_ref": "Section 8.2"
    })

    info.append({
        "rule": "INFO003",
        "finding": f"Approval authority: Exposure ₹{loan_amount_lakhs} lakh requires Zonal Credit Committee",
        "note": "Ensure correct approval authority signs off",
        "policy_ref": "Section 12"
    })

    return {
        "policy_pass":          len(hard_stops) == 0,
        "auto_decline":         len(hard_stops) > 0,
        "hard_stops":           hard_stops,
        "conditions":           conditions,
        "warnings":             warnings,
        "informational":        info,
        "summary": {
            "hard_stops_count": len(hard_stops),
            "conditions_count": len(conditions),
            "warnings_count":   len(warnings),
        },
        "policy_version":       "ACB-CP-SME-2024 v3.2"
    }


# ── LTV Calculator ────────────────────────────────────────────────────────────

@tool
def calculate_ltv(
    loan_amount: float,
    property_market_value: float,
    machinery_value: float,
    existing_charge_outstanding: float,
    property_type: str
) -> dict:
    """
    Calculate Loan-to-Value ratios for collateral assessment.
    All values in same unit (lakhs).
    property_type: 'residential' / 'commercial' / 'industrial'
    """
    policy_ltv_limits = {
        "residential": 0.75,
        "commercial":  0.65,
        "industrial":  0.65,
    }
    policy_limit = policy_ltv_limits.get(property_type.lower(), 0.65)

    # Net property value after existing charge
    net_property_value = property_market_value - existing_charge_outstanding
    total_collateral   = net_property_value + machinery_value

    # LTV calculations
    property_ltv = round(loan_amount / property_market_value, 3) if property_market_value else 0
    blended_ltv  = round(loan_amount / total_collateral, 3) if total_collateral else 0
    coverage     = round(total_collateral / loan_amount, 2) if loan_amount else 0

    return {
        "loan_amount":              loan_amount,
        "property_market_value":    property_market_value,
        "machinery_value":          machinery_value,
        "existing_charge":          existing_charge_outstanding,
        "net_property_value":       net_property_value,
        "total_net_collateral":     total_collateral,
        "property_ltv":             property_ltv,
        "property_ltv_pct":         round(property_ltv * 100, 1),
        "blended_ltv":              blended_ltv,
        "blended_ltv_pct":          round(blended_ltv * 100, 1),
        "collateral_coverage_ratio": coverage,
        "policy_ltv_limit":         policy_limit,
        "property_ltv_within_policy": property_ltv <= policy_limit,
        "blended_ltv_within_policy":  blended_ltv <= policy_limit,
        "minimum_coverage_met":     coverage >= 1.25
    }


# ── Bank Statement Analyzer ────────────────────────────────────────────────────

@tool
def analyze_bank_statement(
    monthly_credits: list,
    monthly_debits: list,
    monthly_closing_balances: list,
    stated_annual_turnover: float,
    cc_limit: float
) -> dict:
    """
    Analyze bank statement data for key underwriting signals.
    All values in lakhs.
    monthly_credits/debits/balances: list of 3 monthly values
    """
    # Annualised turnover from statement
    avg_monthly_credits   = sum(monthly_credits) / len(monthly_credits)
    annualised_credits    = avg_monthly_credits * 12

    # Turnover variance
    turnover_variance_pct = round(
        (stated_annual_turnover - annualised_credits) / stated_annual_turnover * 100, 1
    ) if stated_annual_turnover else 0

    # Balance analysis
    avg_balance     = sum(monthly_closing_balances) / len(monthly_closing_balances)
    min_balance     = min(monthly_closing_balances)
    max_balance     = max(monthly_closing_balances)

    # CC utilisation
    avg_monthly_debit     = sum(monthly_debits) / len(monthly_debits)

    # Flags
    flags = []

    if turnover_variance_pct > 20:
        flags.append({
            "flag": "TURNOVER_DISCREPANCY",
            "detail": f"Annualised credits ₹{annualised_credits:.1f} lakh vs stated ₹{stated_annual_turnover} lakh — {turnover_variance_pct}% gap"
        })

    if min_balance < cc_limit * 0.10:
        flags.append({
            "flag": "LOW_BALANCE",
            "detail": f"Minimum balance ₹{min_balance:.1f} lakh fell below 10% of CC limit (₹{cc_limit * 0.10:.1f} lakh)"
        })

    return {
        "annualised_credits_lakhs":   round(annualised_credits, 1),
        "stated_turnover_lakhs":      stated_annual_turnover,
        "turnover_variance_pct":      turnover_variance_pct,
        "avg_monthly_balance_lakhs":  round(avg_balance, 1),
        "min_balance_lakhs":          round(min_balance, 1),
        "max_balance_lakhs":          round(max_balance, 1),
        "avg_monthly_credits_lakhs":  round(avg_monthly_credits, 1),
        "avg_monthly_debits_lakhs":   round(avg_monthly_debit, 1),
        "flags":                      flags,
        "turnover_adequate":          turnover_variance_pct <= 20,
        "balance_behaviour":          "ADEQUATE" if min_balance > cc_limit * 0.10 else "BELOW_MINIMUM"
    }
