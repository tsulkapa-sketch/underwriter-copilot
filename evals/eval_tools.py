"""
Tier 1 — Tool correctness evals (10 tests).
Tests Layer 1 (External APIs) and Layer 2 (Deterministic Calculators).

No Claude calls needed — all deterministic.
Runs in ~5 seconds.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals._base import EvalResult, safe_run
from evals.fixtures import CASE_DATA, GROUND_TRUTH

CAT = "tools"


# ── Layer 1: External APIs ─────────────────────────────────────────────────────

def eval_bureau_api() -> EvalResult:
    from tools.layer1_apis import call_bureau_api
    r = call_bureau_api.invoke({"pan": CASE_DATA["pan"], "case_id": CASE_DATA["case_id"]})
    issues = []
    for key in ["commercial_score", "risk_category", "unanswered_enquiries", "wilful_defaulter"]:
        if key not in r:
            issues.append(f"missing key: {key}")
    if r.get("commercial_score") != GROUND_TRUTH["commercial_score"]:
        issues.append(f"score={r.get('commercial_score')} expected {GROUND_TRUTH['commercial_score']}")
    if r.get("unanswered_enquiries") != GROUND_TRUTH["unanswered_enquiries"]:
        issues.append(f"unanswered_enquiries={r.get('unanswered_enquiries')} expected {GROUND_TRUTH['unanswered_enquiries']}")
    if r.get("wilful_defaulter") is not False:
        issues.append(f"wilful_defaulter={r.get('wilful_defaulter')} expected False")
    passed = not issues
    details = (
        f"score={r.get('commercial_score')}, risk={r.get('risk_category')}, "
        f"enquiries={r.get('unanswered_enquiries')}"
        if passed else " | ".join(issues)
    )
    return EvalResult("Bureau API — score + enquiries + wilful defaulter", passed, 1.0 if passed else 0.0, details)


def eval_fraud_model() -> EvalResult:
    from tools.layer1_apis import call_fraud_model
    r = call_fraud_model.invoke({
        "case_id": CASE_DATA["case_id"],
        "application_data": {
            "cc_cycling_detected":        CASE_DATA["cc_cycling_detected"],
            "revenue_variance_pct":       CASE_DATA["revenue_variance_pct"],
            "bank_turnover_variance_pct": CASE_DATA["bank_turnover_variance_pct"],
            "unanswered_enquiries":       CASE_DATA["unanswered_enquiries"],
            "undisclosed_contingent":     CASE_DATA["undisclosed_contingent"],
        },
    })
    issues = []
    for key in ["fraud_score", "risk_level", "total_signals", "recommendation"]:
        if key not in r:
            issues.append(f"missing key: {key}")
    # 5 risk factors supplied → expect >= 3 signals
    if r.get("total_signals", 0) < 3:
        issues.append(f"total_signals={r.get('total_signals')} expected >= 3 (5 risk factors given)")
    if r.get("risk_level") not in ("MEDIUM", "HIGH"):
        issues.append(f"risk_level={r.get('risk_level')} expected MEDIUM or HIGH")
    passed = not issues
    details = (
        f"score={r.get('fraud_score')}, risk={r.get('risk_level')}, signals={r.get('total_signals')}"
        if passed else " | ".join(issues)
    )
    return EvalResult("Fraud Model — signal count + risk level", passed, 1.0 if passed else 0.0, details)


def eval_core_banking() -> EvalResult:
    from tools.layer1_apis import call_core_banking
    r = call_core_banking.invoke({"entity_pan": CASE_DATA["pan"], "case_id": CASE_DATA["case_id"]})
    issues = []
    for key in ["existing_relationship", "existing_exposure", "account_conduct"]:
        if key not in r:
            issues.append(f"missing key: {key}")
    if not isinstance(r.get("existing_relationship"), bool):
        issues.append("existing_relationship should be bool")
    passed = not issues
    details = (
        f"relationship={r.get('existing_relationship')}, exposure={r.get('existing_exposure')}, conduct={r.get('account_conduct')}"
        if passed else " | ".join(issues)
    )
    return EvalResult("Core Banking — schema valid", passed, 1.0 if passed else 0.0, details)


def eval_crilc() -> EvalResult:
    from tools.layer1_apis import call_crilc_check
    r = call_crilc_check.invoke({
        "entity_pan":    CASE_DATA["pan"],
        "promoter_pans": CASE_DATA["promoter_pans"],
    })
    issues = []
    for key in ["entity_crilc_status", "entity_npa", "wilful_defaulter_entity"]:
        if key not in r:
            issues.append(f"missing key: {key}")
    if r.get("wilful_defaulter_entity") is not False:
        issues.append(f"wilful_defaulter_entity={r.get('wilful_defaulter_entity')} expected False")
    if r.get("entity_npa") is not False:
        issues.append(f"entity_npa={r.get('entity_npa')} expected False")
    passed = not issues
    details = (
        f"status={r.get('entity_crilc_status')}, npa={r.get('entity_npa')}, wilful={r.get('wilful_defaulter_entity')}"
        if passed else " | ".join(issues)
    )
    return EvalResult("CRILC — no NPA, no wilful default", passed, 1.0 if passed else 0.0, details)


def eval_valuation_system() -> EvalResult:
    from tools.layer1_apis import call_valuation_system
    r = call_valuation_system.invoke({
        "property_address": CASE_DATA["property_address"],
        "case_id":          CASE_DATA["case_id"],
    })
    issues = []
    for key in ["total_collateral_market_value", "blended_ltv", "collateral_coverage_ratio"]:
        if key not in r:
            issues.append(f"missing key: {key}")
    if r.get("collateral_coverage_ratio", 0) < 1.0:
        issues.append(f"coverage={r.get('collateral_coverage_ratio'):.2f}x expected > 1.0")
    if r.get("total_collateral_market_value", 0) < 1_000_000:
        issues.append("total_collateral_market_value looks implausibly low")
    passed = not issues
    mv = r.get("total_collateral_market_value", 0) / 100_000
    details = (
        f"market_value=₹{mv:.1f}L, blended_ltv={r.get('blended_ltv'):.1%}, coverage={r.get('collateral_coverage_ratio'):.2f}x"
        if passed else " | ".join(issues)
    )
    return EvalResult("Valuation System — market value + coverage > 1.0x", passed, 1.0 if passed else 0.0, details)


# ── Layer 2: Deterministic Calculators ────────────────────────────────────────

def _compute_ratios():
    """Helper: compute financial ratios (reused by multiple evals)."""
    from tools.layer2_tools import calculate_financial_ratios
    return calculate_financial_ratios.invoke({
        "revenue":               CASE_DATA["revenue"],
        "ebitda":                CASE_DATA["ebitda"],
        "net_profit":            CASE_DATA["net_profit"],
        "total_debt":            CASE_DATA["total_debt"],
        "total_equity":          CASE_DATA["total_equity"],
        "current_assets":        CASE_DATA["current_assets"],
        "current_liabilities":   CASE_DATA["current_liabilities"],
        "interest_expense":      CASE_DATA["interest_expense"],
        "annual_debt_repayment": CASE_DATA["annual_debt_repayment"],
        "depreciation":          CASE_DATA["depreciation"],
    })


def eval_financial_ratios() -> EvalResult:
    r = _compute_ratios()
    gt = GROUND_TRUTH
    issues = []

    dscr = r.get("dscr", 0)
    if not (gt["dscr_min"] <= dscr <= gt["dscr_max"]):
        issues.append(f"DSCR={dscr} outside [{gt['dscr_min']}, {gt['dscr_max']}]")

    margin = r.get("net_profit_margin_pct", 0)
    if not (gt["net_margin_min"] <= margin <= gt["net_margin_max"]):
        issues.append(f"margin={margin}% outside [{gt['net_margin_min']}, {gt['net_margin_max']}]")

    cr = r.get("current_ratio", 0)
    if cr < gt["current_ratio_min"]:
        issues.append(f"current_ratio={cr} < {gt['current_ratio_min']}")

    icr = r.get("interest_coverage_ratio", 0)
    if icr < gt["icr_min"]:
        issues.append(f"ICR={icr} < {gt['icr_min']}")

    passed = not issues
    details = (
        f"DSCR={dscr}x, margin={margin}%, current_ratio={cr}x, ICR={icr}x"
        if passed else " | ".join(issues)
    )
    return EvalResult("Financial Ratios — DSCR + margin + CR + ICR in range", passed, 1.0 if passed else 0.0, details)


def eval_credit_scorecard() -> EvalResult:
    from tools.layer2_tools import run_credit_scorecard
    ratios = _compute_ratios()
    r = run_credit_scorecard.invoke({
        "net_profit_margin_pct":        CASE_DATA["net_profit_margin_pct"],
        "dscr":                         ratios["dscr"],
        "debt_equity_ratio":            ratios["debt_equity_ratio"],
        "bureau_score":                 CASE_DATA["bureau_score"],
        "dpd_instances_last_24_months": CASE_DATA["dpd_instances_last_24_months"],
        "customer_concentration_pct":   CASE_DATA["customer_concentration_pct"],
        "years_in_operation":           CASE_DATA["years_in_operation"],
        "revenue_trend":                CASE_DATA["revenue_trend"],
    })
    gt = GROUND_TRUTH
    issues = []

    score = r.get("total_score", 0)
    if not (gt["credit_score_min"] <= score <= gt["credit_score_max"]):
        issues.append(f"score={score} outside [{gt['credit_score_min']}, {gt['credit_score_max']}]")

    grade = r.get("grade", "")
    if grade not in gt["grade_options"]:
        issues.append(f"grade={grade} not in expected {gt['grade_options']}")

    if "breakdown" not in r:
        issues.append("missing 'breakdown' key")

    passed = not issues
    details = (
        f"score={score}/100, grade={grade}, recommendation={r.get('recommendation')}"
        if passed else " | ".join(issues)
    )
    return EvalResult("Credit Scorecard — score range + grade + breakdown", passed, 1.0 if passed else 0.0, details)


def eval_policy_rules_engine() -> EvalResult:
    from tools.layer2_tools import run_policy_rules_engine
    r = run_policy_rules_engine.invoke({
        "loan_amount_lakhs":                CASE_DATA["loan_amount_lakhs"],
        "dscr":                             CASE_DATA["dscr"],
        "blended_ltv":                      CASE_DATA["blended_ltv"],
        "property_ltv":                     CASE_DATA["property_ltv"],
        "customer_concentration_pct":       CASE_DATA["customer_concentration_pct"],
        "bureau_score_commercial":          CASE_DATA["bureau_score_commercial"],
        "dpd_60_plus_last_36_months":       CASE_DATA["dpd_60_plus_last_36_months"],
        "npa_last_60_months":               CASE_DATA["npa_last_60_months"],
        "wilful_defaulter":                 CASE_DATA["wilful_defaulter"],
        "unanswered_enquiries":             CASE_DATA["unanswered_enquiries"],
        "cc_cycling_detected":              CASE_DATA["cc_cycling_detected"],
        "revenue_overstatement_pct":        CASE_DATA["revenue_overstatement_pct"],
        "undisclosed_contingent_liability": CASE_DATA["undisclosed_contingent_liability"],
        "epcg_obligation_lakhs":            CASE_DATA["epcg_obligation_lakhs"],
        "annual_revenue_lakhs":             CASE_DATA["revenue"],
        "sector":                           CASE_DATA["sector"],
        "years_in_operation":               CASE_DATA["years_in_operation"],
        "promoter_net_worth_lakhs":         CASE_DATA["promoter_net_worth_lakhs"],
    })
    issues = []
    for key in ["policy_pass", "hard_stops", "conditions", "warnings", "summary"]:
        if key not in r:
            issues.append(f"missing key: {key}")

    # Required conditions must be flagged
    cond_rules = " ".join(c.get("rule", "").lower() for c in r.get("conditions", []))
    cond_findings = " ".join(c.get("finding", "").lower() for c in r.get("conditions", []))
    cond_text = cond_rules + " " + cond_findings

    for kw in GROUND_TRUTH["required_conditions_keywords"]:
        if kw.lower() not in cond_text:
            # Also check hard_stops (might be escalated there)
            stop_text = " ".join(
                (h.get("rule", "") + " " + h.get("finding", "")).lower()
                for h in r.get("hard_stops", [])
            )
            if kw.lower() not in stop_text:
                issues.append(f"required condition keyword '{kw}' not found in conditions or hard_stops")

    # DSCR = 2.18 — should NOT trigger a DSCR hard stop
    stop_rules = " ".join(h.get("rule", "").lower() for h in r.get("hard_stops", []))
    if "dscr" in stop_rules:
        issues.append("DSCR=2.18 should not trigger hard_stop (min 1.25)")

    passed = not issues
    cs = len(r.get("conditions", []))
    ws = len(r.get("warnings", []))
    hs = len(r.get("hard_stops", []))
    details = (
        f"hard_stops={hs}, conditions={cs}, warnings={ws}, policy_pass={r.get('policy_pass')}"
        if passed else " | ".join(issues)
    )
    return EvalResult("Policy Rules Engine — required conditions flagged, DSCR safe", passed, 1.0 if passed else 0.0, details)


def eval_ltv_calculator() -> EvalResult:
    from tools.layer2_tools import calculate_ltv
    r = calculate_ltv.invoke({
        "loan_amount":                CASE_DATA["loan_amount_lakhs"],
        "property_market_value":      CASE_DATA["property_market_value"],
        "machinery_value":            CASE_DATA["machinery_value"],
        "existing_charge_outstanding": CASE_DATA["existing_charge_outstanding"],
        "property_type":              CASE_DATA["property_type"],
    })
    gt = GROUND_TRUTH
    issues = []

    prop_ltv = r.get("property_ltv_pct", 0)
    blend_ltv = r.get("blended_ltv_pct", 0)
    coverage  = r.get("collateral_coverage_ratio", 0)

    if prop_ltv < gt["property_ltv_pct_min"]:
        issues.append(f"property_ltv_pct={prop_ltv}% should be >= {gt['property_ltv_pct_min']}% (known breach)")
    if not (gt["blended_ltv_pct_min"] <= blend_ltv <= gt["blended_ltv_pct_max"]):
        issues.append(f"blended_ltv_pct={blend_ltv}% outside [{gt['blended_ltv_pct_min']}, {gt['blended_ltv_pct_max']}]")
    if coverage < gt["coverage_ratio_min"]:
        issues.append(f"collateral_coverage_ratio={coverage}x < {gt['coverage_ratio_min']}x")
    if "property_ltv_within_policy" not in r:
        issues.append("missing property_ltv_within_policy flag")

    passed = not issues
    details = (
        f"property_ltv={prop_ltv}%, blended_ltv={blend_ltv}%, coverage={coverage}x, within_policy={r.get('property_ltv_within_policy')}"
        if passed else " | ".join(issues)
    )
    return EvalResult("LTV Calculator — property LTV breach + coverage", passed, 1.0 if passed else 0.0, details)


def eval_bank_statement() -> EvalResult:
    from tools.layer2_tools import analyze_bank_statement
    r = analyze_bank_statement.invoke({
        "monthly_credits":          CASE_DATA["monthly_credits"],
        "monthly_debits":           CASE_DATA["monthly_debits"],
        "monthly_closing_balances": CASE_DATA["monthly_closing_balances"],
        "stated_annual_turnover":   CASE_DATA["stated_annual_turnover"],
        "cc_limit":                 CASE_DATA["cc_limit"],
    })
    gt = GROUND_TRUTH
    issues = []

    variance = abs(r.get("turnover_variance_pct", 0))
    credits  = r.get("annualised_credits_lakhs", 0)

    if variance < gt["turnover_variance_min"]:
        issues.append(f"variance={variance:.1f}% < {gt['turnover_variance_min']}% (large gap expected)")
    if credits < gt["annualised_credits_min"]:
        issues.append(f"annualised_credits=₹{credits}L < ₹{gt['annualised_credits_min']}L")
    if "flags" not in r:
        issues.append("missing 'flags' key")
    if "stated_turnover_lakhs" not in r:
        issues.append("missing 'stated_turnover_lakhs' key")

    passed = not issues
    details = (
        f"annualised=₹{credits}L, stated=₹{r.get('stated_turnover_lakhs')}L, "
        f"variance={r.get('turnover_variance_pct')}%, flags={len(r.get('flags', []))}"
        if passed else " | ".join(issues)
    )
    return EvalResult("Bank Statement — variance detection + annualised credits", passed, 1.0 if passed else 0.0, details)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_tool_evals() -> list:
    tests = [
        ("Bureau API",           eval_bureau_api),
        ("Fraud Model",          eval_fraud_model),
        ("Core Banking",         eval_core_banking),
        ("CRILC Check",          eval_crilc),
        ("Valuation System",     eval_valuation_system),
        ("Financial Ratios",     eval_financial_ratios),
        ("Credit Scorecard",     eval_credit_scorecard),
        ("Policy Rules Engine",  eval_policy_rules_engine),
        ("LTV Calculator",       eval_ltv_calculator),
        ("Bank Statement",       eval_bank_statement),
    ]
    return [safe_run(name, fn, CAT) for name, fn in tests]
