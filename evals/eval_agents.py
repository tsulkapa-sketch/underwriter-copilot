"""
Tier 2 — Individual agent output validation (5 tests).
Each agent runs independently with Meridian case data.

Requires ANTHROPIC_API_KEY.
Each agent makes several Claude API calls — expect 3–5 minutes total.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals._base import EvalResult, safe_run
from evals.fixtures import CASE_DATA, GROUND_TRUTH, load_documents

CAT           = "agents"
VALID_RATINGS = {"HIGH", "MEDIUM", "LOW"}


def _missing_keys(result: dict, keys: list, label: str) -> list:
    return [f"missing '{k}' in {label}" for k in keys if k not in result]


# ── 1. Financial agent ─────────────────────────────────────────────────────────

def eval_financial_agent() -> EvalResult:
    from agents.financial import run_financial_analysis

    r = run_financial_analysis(CASE_DATA)

    issues = _missing_keys(r, ["ratios", "scorecard", "bank_statement", "synthesis", "risk_rating", "key_metrics"], "financial")

    if r.get("risk_rating") not in VALID_RATINGS:
        issues.append(f"invalid risk_rating: {r.get('risk_rating')}")

    km   = r.get("key_metrics", {})
    dscr = km.get("dscr", 0)
    if not (GROUND_TRUTH["dscr_min"] <= dscr <= GROUND_TRUTH["dscr_max"]):
        issues.append(f"key_metrics.dscr={dscr} outside [{GROUND_TRUTH['dscr_min']}, {GROUND_TRUTH['dscr_max']}]")

    synthesis = r.get("synthesis", "")
    if len(synthesis) < 100:
        issues.append(f"synthesis too short ({len(synthesis)} chars — expected > 100)")

    passed  = not issues
    details = (
        f"risk={r.get('risk_rating')}, dscr={dscr}x, grade={km.get('grade')}, synthesis={len(synthesis)} chars"
        if passed else " | ".join(issues)
    )
    return EvalResult("Financial Agent — schema + DSCR in range + synthesis", passed, 1.0 if passed else 0.0, details)


# ── 2. Bureau agent ────────────────────────────────────────────────────────────

def eval_bureau_agent() -> EvalResult:
    from agents.bureau import run_bureau_analysis

    r = run_bureau_analysis(CASE_DATA)

    issues = _missing_keys(r, ["bureau_report", "crilc", "fraud", "synthesis", "risk_rating", "hard_stop_flags", "key_metrics"], "bureau")

    if r.get("risk_rating") not in VALID_RATINGS:
        issues.append(f"invalid risk_rating: {r.get('risk_rating')}")

    if not isinstance(r.get("hard_stop_flags"), list):
        issues.append("hard_stop_flags should be a list")

    km = r.get("key_metrics", {})
    if km.get("commercial_score") != GROUND_TRUTH["commercial_score"]:
        issues.append(f"commercial_score={km.get('commercial_score')} expected {GROUND_TRUTH['commercial_score']}")

    synthesis = r.get("synthesis", "")
    if len(synthesis) < 100:
        issues.append(f"synthesis too short ({len(synthesis)} chars)")

    passed  = not issues
    details = (
        f"risk={r.get('risk_rating')}, score={km.get('commercial_score')}, "
        f"fraud_score={km.get('fraud_score')}, hard_stops={len(r.get('hard_stop_flags', []))}"
        if passed else " | ".join(issues)
    )
    return EvalResult("Bureau Agent — schema + bureau score + hard_stop_flags", passed, 1.0 if passed else 0.0, details)


# ── 3. Contradiction agent ─────────────────────────────────────────────────────

def eval_contradiction_agent() -> EvalResult:
    from agents.contradiction import run_contradiction_analysis

    docs = load_documents()
    if not docs:
        return EvalResult(
            "Contradiction Agent — schema + severity alignment",
            False, 0.0,
            "No documents found in loan_docs/ — cannot run eval",
        )

    r = run_contradiction_analysis(docs)

    issues = _missing_keys(r, ["contradictions", "severities", "risk_rating", "key_metrics"], "contradiction")

    if r.get("risk_rating") not in VALID_RATINGS:
        issues.append(f"invalid risk_rating: {r.get('risk_rating')}")

    contras = r.get("contradictions", [])
    sevs    = r.get("severities", [])
    if len(contras) != len(sevs):
        issues.append(f"contradictions({len(contras)}) vs severities({len(sevs)}) length mismatch")

    km = r.get("key_metrics", {})
    for k in ["total", "high", "medium", "low"]:
        if k not in km:
            issues.append(f"key_metrics missing '{k}'")

    if km.get("total") != len(contras):
        issues.append(f"key_metrics.total={km.get('total')} != len(contradictions)={len(contras)}")

    # Severity values must all be valid
    for s in sevs:
        if s.get("severity") not in VALID_RATINGS:
            issues.append(f"invalid severity value: {s.get('severity')}")
            break

    passed  = not issues
    details = (
        f"docs={len(docs)}, total={km.get('total')}, HIGH={km.get('high')}, MEDIUM={km.get('medium')}, LOW={km.get('low')}"
        if passed else " | ".join(issues)
    )
    return EvalResult("Contradiction Agent — schema + counts aligned + severities valid", passed, 1.0 if passed else 0.0, details)


# ── 4. Policy agent ────────────────────────────────────────────────────────────

def eval_policy_agent() -> EvalResult:
    from agents.policy import run_policy_analysis

    r = run_policy_analysis(CASE_DATA)

    issues = _missing_keys(r, ["policy_result", "synthesis", "risk_rating", "key_metrics"], "policy")

    if r.get("risk_rating") not in VALID_RATINGS:
        issues.append(f"invalid risk_rating: {r.get('risk_rating')}")

    pr = r.get("policy_result", {})
    for k in ["hard_stops", "conditions", "warnings"]:
        if k not in pr:
            issues.append(f"policy_result missing '{k}'")

    km = r.get("key_metrics", {})
    for k in ["policy_pass", "hard_stops_count", "conditions_count", "warnings_count"]:
        if k not in km:
            issues.append(f"key_metrics missing '{k}'")

    synthesis = r.get("synthesis", "")
    if len(synthesis) < 50:
        issues.append(f"synthesis too short ({len(synthesis)} chars)")

    # DSCR = 2.18 — should not appear as hard stop
    hs_text = " ".join(
        (h.get("rule", "") + " " + h.get("finding", "")).lower()
        for h in pr.get("hard_stops", [])
    )
    if "dscr" in hs_text and "below" in hs_text:
        issues.append("DSCR=2.18 incorrectly flagged as hard stop")

    passed  = not issues
    details = (
        f"risk={r.get('risk_rating')}, hard_stops={km.get('hard_stops_count')}, "
        f"conditions={km.get('conditions_count')}, warnings={km.get('warnings_count')}"
        if passed else " | ".join(issues)
    )
    return EvalResult("Policy Agent — schema + policy_result structure + DSCR safe", passed, 1.0 if passed else 0.0, details)


# ── 5. Collateral agent ────────────────────────────────────────────────────────

def eval_collateral_agent() -> EvalResult:
    from agents.collateral import run_collateral_analysis

    r = run_collateral_analysis(CASE_DATA)

    issues = _missing_keys(r, ["valuation", "ltv", "synthesis", "risk_rating", "key_metrics"], "collateral")

    if r.get("risk_rating") not in VALID_RATINGS:
        issues.append(f"invalid risk_rating: {r.get('risk_rating')}")

    km       = r.get("key_metrics", {})
    prop_ltv = km.get("property_ltv_pct", 0)
    coverage = km.get("collateral_coverage_ratio", 0)

    for k in ["property_ltv_pct", "blended_ltv_pct", "collateral_coverage_ratio",
              "property_ltv_within_policy", "blended_ltv_within_policy"]:
        if k not in km:
            issues.append(f"key_metrics missing '{k}'")

    if prop_ltv < GROUND_TRUTH["property_ltv_pct_min"]:
        issues.append(f"property_ltv_pct={prop_ltv}% should be >= {GROUND_TRUTH['property_ltv_pct_min']}% (known breach)")

    if coverage < GROUND_TRUTH["coverage_ratio_min"]:
        issues.append(f"coverage_ratio={coverage}x < {GROUND_TRUTH['coverage_ratio_min']}x")

    synthesis = r.get("synthesis", "")
    if len(synthesis) < 100:
        issues.append(f"synthesis too short ({len(synthesis)} chars)")

    passed  = not issues
    details = (
        f"risk={r.get('risk_rating')}, property_ltv={prop_ltv}%, "
        f"blended_ltv={km.get('blended_ltv_pct')}%, coverage={coverage}x"
        if passed else " | ".join(issues)
    )
    return EvalResult("Collateral Agent — schema + LTV breach detected + coverage", passed, 1.0 if passed else 0.0, details)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_agent_evals() -> list:
    tests = [
        ("Financial Agent",     eval_financial_agent),
        ("Bureau Agent",        eval_bureau_agent),
        ("Contradiction Agent", eval_contradiction_agent),
        ("Policy Agent",        eval_policy_agent),
        ("Collateral Agent",    eval_collateral_agent),
    ]
    return [safe_run(name, fn, CAT) for name, fn in tests]
