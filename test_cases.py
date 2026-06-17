"""
test_cases.py
=============
Four scenario case dicts for testing conditional routing in the LangGraph orchestrator.

Each dict is a complete override of MERIDIAN_CASE_DATA targeting one specific routing path:

  WILFUL_DEFAULTER_CASE  → auto_decline_bureau   (bureau hard stop)
  LOW_DSCR_CASE          → auto_decline_policy    (policy hard stop: DSCR < 1.0)
  HIGH_CONTRADICTION_CASE→ escalate               (override_high_contradictions = 3)
  CLEAN_CASE             → fast_track_approve     (all LOW risk, no conditions)

The `expected_routing` field is used by eval_routing.py to assert correct behaviour.

Usage:
------
  from test_cases import WILFUL_DEFAULTER_CASE, LOW_DSCR_CASE, HIGH_CONTRADICTION_CASE, CLEAN_CASE
  result, graph, config = run_full_analysis("SME-2024-00001", case_data=WILFUL_DEFAULTER_CASE)
  assert result["routing_path"] == WILFUL_DEFAULTER_CASE["expected_routing"]
"""

from agent import MERIDIAN_CASE_DATA

# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: Bureau Hard Stop → auto_decline_bureau
# Trigger: wilful_defaulter = True
#          Route logic priority 1: bureau hard stop overrides everything
# ─────────────────────────────────────────────────────────────────────────────
WILFUL_DEFAULTER_CASE = {
    **MERIDIAN_CASE_DATA,

    # Case identity
    "case_id":  "SME-2024-00001",
    "pan":      "ZZZBM0001A",

    # Bureau hard stop trigger
    "wilful_defaulter":             True,
    "npa_last_60_months":           True,
    "dpd_60_plus_last_36_months":   True,
    "bureau_score":                 42,
    "bureau_score_commercial":      42,
    "dpd_instances_last_24_months": 8,

    # Everything else kept clean — routing should still hard-stop on bureau
    "dscr":                         2.50,
    "blended_ltv":                  0.45,
    "property_ltv":                 0.65,
    "cc_cycling_detected":          False,
    "undisclosed_contingent_liability": False,
    "undisclosed_contingent":       False,
    "revenue_overstatement_pct":    0,
    "unanswered_enquiries":         0,

    # Eval assertion
    "expected_routing": "auto_decline_bureau",
}


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: Policy Hard Stop → auto_decline_policy
# Trigger: dscr = 0.75 (below DSCR_HARD_STOP_THRESHOLD = 1.0)
#          run_policy_rules_engine sets auto_decline = True when hard stop exists
#          Priority 2: only fires if bureau is clean
# ─────────────────────────────────────────────────────────────────────────────
LOW_DSCR_CASE = {
    **MERIDIAN_CASE_DATA,

    # Case identity
    "case_id":  "SME-2024-00002",
    "pan":      "ZZZBM0002B",

    # Bureau is clean — do NOT trigger Scenario 1
    "wilful_defaulter":             False,
    "npa_last_60_months":           False,
    "dpd_60_plus_last_36_months":   False,
    "bureau_score":                 72,
    "bureau_score_commercial":      72,
    "dpd_instances_last_24_months": 0,

    # Policy hard stop trigger: DSCR below minimum (1.0)
    # DSCR = (net_profit + depreciation + interest) / (repayment + interest)
    # Set net_profit very low so DSCR < 1.0
    "dscr":             0.75,
    "net_profit":       5.00,    # very low — insufficient to service debt
    "ebitda":           14.00,

    # Keep blended LTV and property LTV clean
    "blended_ltv":      0.45,
    "property_ltv":     0.65,

    # No fraud triggers
    "cc_cycling_detected":              False,
    "undisclosed_contingent_liability": False,
    "undisclosed_contingent":           False,
    "revenue_overstatement_pct":        0,
    "unanswered_enquiries":             0,

    # Eval assertion
    "expected_routing": "auto_decline_policy",
}


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: High Contradictions → escalate
# Trigger: override_high_contradictions = 3 (≥ ESCALATION_HIGH_CONTRADICTION_THRESHOLD = 2)
#          This field short-circuits the actual contradiction count, letting us
#          test escalation without needing separate loan document sets.
#          Priority 3: only fires if bureau and policy are clean
# ─────────────────────────────────────────────────────────────────────────────
HIGH_CONTRADICTION_CASE = {
    **MERIDIAN_CASE_DATA,

    # Case identity
    "case_id":  "SME-2024-00003",
    "pan":      "ZZZBM0003C",

    # Bureau clean
    "wilful_defaulter":             False,
    "npa_last_60_months":           False,
    "dpd_60_plus_last_36_months":   False,
    "bureau_score":                 78,
    "bureau_score_commercial":      78,
    "dpd_instances_last_24_months": 0,

    # Policy clean: good DSCR, LTV within limits
    "dscr":         2.30,
    "blended_ltv":  0.45,
    "property_ltv": 0.65,

    # No hard fraud triggers
    "cc_cycling_detected":              False,
    "undisclosed_contingent_liability": False,
    "undisclosed_contingent":           False,
    "revenue_overstatement_pct":        0,
    "unanswered_enquiries":             0,

    # Escalation trigger: synthetic high contradiction count
    # route_after_aggregation checks this BEFORE contradiction agent's key_metrics
    "override_high_contradictions": 3,

    # Eval assertion
    "expected_routing": "escalate",
}


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4: Clean Case → fast_track_approve
# Trigger: all agent_ratings == "LOW", no conditions, no hard stop flags
#          Priority 4: fires only if bureau, policy, and escalation are clean
#          This is a best-case borrower profile — everything green
# ─────────────────────────────────────────────────────────────────────────────
CLEAN_CASE = {
    **MERIDIAN_CASE_DATA,

    # Case identity
    "case_id":  "SME-2024-00004",
    "pan":      "ZZZBM0004D",

    # Excellent bureau
    "wilful_defaulter":             False,
    "npa_last_60_months":           False,
    "dpd_60_plus_last_36_months":   False,
    "bureau_score":                 85,
    "bureau_score_commercial":      85,
    "dpd_instances_last_24_months": 0,

    # Excellent financials: DSCR = 3.20, low leverage
    "dscr":                         3.20,
    "net_profit":                   68.00,
    "ebitda":                       92.00,
    "revenue":                      520.00,
    "total_debt":                   18.00,
    "total_equity":                 240.00,

    # Conservative LTVs
    "blended_ltv":      0.35,
    "property_ltv":     0.50,

    # No fraud signals
    "cc_cycling_detected":              False,
    "undisclosed_contingent_liability": False,
    "undisclosed_contingent":           False,
    "revenue_overstatement_pct":        0.0,
    "revenue_variance_pct":             2.0,
    "bank_turnover_variance_pct":       5.0,
    "unanswered_enquiries":             0,
    "customer_concentration_pct":       35,   # well below 60% threshold

    # No override — rely on actual agent outputs being LOW
    "override_high_contradictions":     0,

    # Strong collateral
    "property_market_value":    380,
    "machinery_value":          260,
    "existing_charge_outstanding": 0,

    # Eval assertion
    "expected_routing": "fast_track_approve",
}


# ─────────────────────────────────────────────────────────────────────────────
# All cases in one list for batch evals
# ─────────────────────────────────────────────────────────────────────────────
ALL_TEST_CASES = [
    WILFUL_DEFAULTER_CASE,
    LOW_DSCR_CASE,
    HIGH_CONTRADICTION_CASE,
    CLEAN_CASE,
]


if __name__ == "__main__":
    print("Test cases loaded:")
    for c in ALL_TEST_CASES:
        print(f"  {c['case_id']:20s}  expected_routing={c['expected_routing']}")
