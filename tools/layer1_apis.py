"""
Layer 1 — Simulated external APIs

In production these would be real HTTP calls to:
  - Experian / CIBIL commercial bureau API
  - Internal fraud scoring model API
  - Core banking system API
  - RBI CRILC API
  - Property valuation system API

In this simulation they return realistic hardcoded data
for Meridian Textile Exports (Case SME-2024-00891).

The function signatures and return structures are identical
to what a real integration would look like.
When moving to production: change the function body only.
Everything that calls these functions stays the same.
"""

from langchain_core.tools import tool


# ── Bureau API ─────────────────────────────────────────────────────────────────

@tool
def call_bureau_api(pan: str, case_id: str) -> dict:
    """
    Fetch commercial bureau report from Experian.
    Returns credit score, DPD history, enquiries, and adverse markers.
    In production: POST to https://api.experian.com/v2/commercial/score
    """
    # Simulated response for Meridian Textile Exports (PAN: AABCM7823Q)
    return {
        "pan": pan,
        "case_id": case_id,
        "commercial_score": 74,
        "score_out_of": 100,
        "risk_category": "MEDIUM",
        "active_facilities": [
            {
                "lender": "State Bank of India",
                "type": "Cash Credit",
                "sanctioned": 4000000,
                "outstanding": 3100000,
                "status": "STANDARD",
                "dpd_last_6_months": [0, 0, 0, 0, 0, 0]
            },
            {
                "lender": "State Bank of India",
                "type": "Vehicle Loan",
                "sanctioned": 900000,
                "outstanding": 310000,
                "status": "STANDARD",
                "dpd_last_6_months": [0, 0, 0, 0, 0, 0]
            }
        ],
        "closed_facilities": [
            {
                "lender": "HDFC Bank",
                "type": "Working Capital",
                "sanctioned": 2500000,
                "outstanding": 0,
                "status": "CLOSED",
                "dpd_history": [0, 0, 30, 0, 0, 0],
                "closed_date": "2023-08"
            },
            {
                "lender": "Tata Capital",
                "type": "Equipment Loan",
                "sanctioned": 1800000,
                "outstanding": 0,
                "status": "CLOSED",
                "dpd_history": [0, 0, 0, 0, 0, 0]
            }
        ],
        "adverse_markers": [
            {
                "type": "DPD_30",
                "lender": "HDFC Bank",
                "date": "2023-03",
                "account_status": "Now closed"
            }
        ],
        "enquiries_last_12_months": [
            {"date": "2024-03-14", "lender": "Current lender", "purpose": "Term Loan + CC"},
            {"date": "2023-11-02", "lender": "Kotak Mahindra Bank", "purpose": "Working Capital", "sanction": False},
            {"date": "2023-06-15", "lender": "SIDBI", "purpose": "MSME Loan", "sanction": False}
        ],
        "unanswered_enquiries": 2,
        "npa_history": False,
        "wilful_defaulter": False,
        "suit_filed": False,
        "promoter_scores": {
            "Vikram Mehta": {"score": 761, "out_of": 900, "adverse": False},
            "Sunita Mehta": {"score": 748, "out_of": 900, "adverse": False}
        },
        "source": "Experian Commercial Bureau (Simulated)",
        "report_date": "2024-03-14"
    }


# ── Fraud Model API ────────────────────────────────────────────────────────────

@tool
def call_fraud_model(case_id: str, application_data: dict) -> dict:
    """
    Score fraud risk using internal ML fraud model.
    In production: POST to internal fraud scoring microservice.
    Returns probability score and specific fraud signal flags.
    """
    # Deterministic rule-based simulation
    # In production: trained XGBoost / neural network model
    fraud_score = 0
    flags = []
    signal_details = []

    # Signal 1: CC cycling before application (HIGH weight)
    if application_data.get("cc_cycling_detected", True):
        fraud_score += 35
        flags.append("CC_CYCLING")
        signal_details.append({
            "signal": "CC cycling detected",
            "detail": "Full CC repayment (₹31 lakh) and same-day redrawdown on 10-Mar-24 — 2 days before loan application",
            "weight": 35
        })

    # Signal 2: Revenue overstated vs GST (MEDIUM weight)
    revenue_variance = application_data.get("revenue_variance_pct", 6.5)
    if revenue_variance > 5:
        fraud_score += 25
        flags.append("REVENUE_OVERSTATEMENT")
        signal_details.append({
            "signal": "Revenue overstated vs GST returns",
            "detail": f"Application states ₹421 lakh vs GST-verified ₹395 lakh — variance of {revenue_variance}%",
            "weight": 25
        })

    # Signal 3: Bank turnover discrepancy (HIGH weight)
    turnover_variance = application_data.get("bank_turnover_variance_pct", 30.4)
    if turnover_variance > 20:
        fraud_score += 20
        flags.append("BANK_TURNOVER_DISCREPANCY")
        signal_details.append({
            "signal": "Bank turnover below stated figure",
            "detail": f"Annualised bank credits ₹4.73 crore vs stated ₹6.8 crore — variance of {turnover_variance:.1f}%",
            "weight": 20
        })

    # Signal 4: Unanswered enquiries (LOW-MEDIUM weight)
    unanswered = application_data.get("unanswered_enquiries", 2)
    if unanswered >= 2:
        fraud_score += 15
        flags.append("UNANSWERED_ENQUIRIES")
        signal_details.append({
            "signal": f"{unanswered} unanswered credit enquiries",
            "detail": "Kotak Mahindra (Nov 2023) and SIDBI (Jun 2023) enquiries with no sanctions — possible prior declines",
            "weight": 15
        })

    # Signal 5: Undisclosed contingent liability (MEDIUM weight)
    if application_data.get("undisclosed_contingent", True):
        fraud_score += 10
        flags.append("UNDISCLOSED_LIABILITY")
        signal_details.append({
            "signal": "Undisclosed contingent liability",
            "detail": "EPCG export obligation of ₹42 lakh found in financial statements — not declared in application",
            "weight": 10
        })

    fraud_score = min(fraud_score, 100)

    return {
        "case_id": case_id,
        "fraud_score": fraud_score,
        "risk_level": (
            "HIGH"   if fraud_score >= 60 else
            "MEDIUM" if fraud_score >= 30 else
            "LOW"
        ),
        "flags": flags,
        "signal_details": signal_details,
        "total_signals": len(flags),
        "recommendation": (
            "REFER_TO_FRAUD_DESK" if fraud_score >= 60 else
            "ENHANCED_DUE_DILIGENCE" if fraud_score >= 30 else
            "STANDARD_PROCESSING"
        ),
        "model_version": "fraud_model_mock_v1.0",
        "note": "Simulated fraud model — in production this calls trained ML model"
    }


# ── Core Banking API ───────────────────────────────────────────────────────────

@tool
def call_core_banking(entity_pan: str, case_id: str) -> dict:
    """
    Fetch existing exposure and relationship data from core banking system.
    In production: query internal CBS (Finacle / Temenos / BaNCS).
    Returns existing facilities, conduct, and relationship history.
    """
    return {
        "case_id": case_id,
        "entity_pan": entity_pan,
        "existing_relationship": False,
        "existing_exposure": 0,
        "existing_facilities": [],
        "relationship_since": None,
        "conduct_rating": "NOT_APPLICABLE",
        "note": "New to bank customer — no existing relationship",
        "group_entities": [],
        "group_exposure": 0,
        "source": "Core Banking System (Simulated)"
    }


# ── CRILC / RBI Check ─────────────────────────────────────────────────────────

@tool
def call_crilc_check(entity_pan: str, promoter_pans: list) -> dict:
    """
    Check RBI CRILC database for SMA / NPA / wilful defaulter status.
    In production: query via RBI CRILC API or CIBIL commercial report.
    """
    return {
        "entity_pan": entity_pan,
        "entity_crilc_status": "STANDARD",
        "entity_sma_status": None,
        "entity_npa": False,
        "wilful_defaulter_entity": False,
        "promoter_checks": [
            {"pan": pan, "wilful_defaulter": False, "guarantor_npa": False}
            for pan in promoter_pans
        ],
        "fraud_registry": False,
        "source": "RBI CRILC (Simulated)",
        "check_date": "2024-03-14"
    }


# ── Property Valuation API ────────────────────────────────────────────────────

@tool
def call_valuation_system(property_address: str, case_id: str) -> dict:
    """
    Fetch latest valuation from empanelled valuer system.
    In production: query property valuation management system.
    Returns market value, forced sale value, and LTV metrics.
    """
    return {
        "case_id": case_id,
        "property_address": property_address,
        "valuation_date": "2024-03-08",
        "valuer_name": "Ramesh Kulkarni",
        "valuer_registration": "IBBI/RV/06/2019/10892",
        "valuation_valid_until": "2024-09-08",
        "immovable_property": {
            "market_value": 31200000,
            "forced_sale_value": 24960000,
            "ltv_policy_limit": 0.65,
            "ltv_on_market_value": 0.865,
            "existing_charge": {
                "lender": "State Bank of India",
                "outstanding": 2200000
            },
            "net_available_market_value": 29000000
        },
        "machinery": {
            "existing_value": 4671300,
            "post_upgrade_value": 21000000,
            "forced_sale_value_post_upgrade": 12600000
        },
        "total_collateral_market_value": 52200000,
        "total_collateral_forced_sale": 37560000,
        "blended_ltv": 0.517,
        "collateral_coverage_ratio": 1.93,
        "source": "Property Valuation System (Simulated)"
    }
