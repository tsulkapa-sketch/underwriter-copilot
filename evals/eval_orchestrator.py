"""
Tier 3 — End-to-end orchestrator evals (2 tests).
Runs the full multi-agent LangGraph for Meridian case.

EXPENSIVE: all 5 agents run in parallel → expect 60–90 seconds + API costs.
Use only with --e2e or --all flag.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals._base import EvalResult, safe_run
from evals.fixtures import GROUND_TRUTH

CAT = "e2e"


def _run_full_analysis():
    """Run the full multi-agent analysis and return (result, graph, config)."""
    from agent import run_full_analysis
    return run_full_analysis("SME-2024-00891")


def eval_full_analysis_structure(result: dict) -> EvalResult:
    """
    Check that the analysis result:
    - Has all 5 agent outputs
    - Agent ratings are valid values
    - Overall risk matches Meridian expectation (HIGH)
    - System recommendation matches expectation (DECLINE)
    - Key metrics are populated (not None)
    - Graph paused at human_review interrupt
    """
    issues = []

    # All 5 agent outputs must be present and non-empty
    for key in ["financial", "bureau", "contradiction", "policy", "collateral"]:
        if not result.get(key):
            issues.append(f"missing or empty: {key}")

    agg = result.get("aggregated", {})
    if not agg:
        issues.append("aggregated section is empty")

    # Agent ratings
    ratings = agg.get("agent_ratings", {})
    for agent in ["financial", "bureau", "contradiction", "policy", "collateral"]:
        if agent not in ratings:
            issues.append(f"agent_ratings missing: {agent}")
        elif ratings[agent] not in {"HIGH", "MEDIUM", "LOW"}:
            issues.append(f"invalid rating for {agent}: {ratings[agent]}")

    # Overall risk
    overall = agg.get("overall_risk")
    if overall != GROUND_TRUTH["overall_risk"]:
        issues.append(f"overall_risk={overall} expected {GROUND_TRUTH['overall_risk']}")

    # System recommendation
    rec = agg.get("system_recommendation")
    if rec != GROUND_TRUTH["system_recommendation"]:
        issues.append(f"system_recommendation={rec} expected {GROUND_TRUTH['system_recommendation']}")

    # Key metrics populated
    km = agg.get("key_metrics", {})
    for k in ["dscr", "credit_score", "blended_ltv"]:
        if km.get(k) is None:
            issues.append(f"key_metrics.{k} is None")

    # Human-in-loop interrupt fired
    if not result.get("__interrupt__"):
        issues.append("graph did not pause at human_review (interrupt missing)")

    # Executive summary written
    summary = agg.get("executive_summary", "")
    if len(summary) < 50:
        issues.append(f"executive_summary too short ({len(summary)} chars)")

    passed  = not issues
    details = (
        f"overall_risk={overall}, rec={rec}, ratings={list(ratings.values())}, "
        f"interrupted={bool(result.get('__interrupt__'))}, summary={len(summary)} chars"
        if passed else " | ".join(issues)
    )
    return EvalResult(
        "E2E — full analysis structure, risk=HIGH, rec=DECLINE, interrupt fired",
        passed, 1.0 if passed else 0.0, details,
    )


def eval_resume_flow(graph, config: dict) -> EvalResult:
    """
    Resume with 'decline' decision and verify the final report:
    - Contains all major report sections
    - Underwriter decision recorded correctly
    - Report is substantive (> 500 chars)
    """
    from agent import resume_analysis

    final_state = resume_analysis(graph, config, "decline")  # uses Command(resume=) internally

    issues  = []
    report  = final_state.get("final_report", "")
    decision = final_state.get("underwriter_decision", "")

    if len(report) < 500:
        issues.append(f"final_report too short ({len(report)} chars — expected > 500)")

    for section in [
        "CREDIT DECISION REPORT",
        "FINANCIAL ANALYSIS",
        "BUREAU",
        "COLLATERAL",
        "UNDERWRITER DECISION",
    ]:
        if section not in report.upper():
            issues.append(f"missing report section: '{section}'")

    if decision.lower() != "decline":
        issues.append(f"underwriter_decision='{decision}' expected 'decline'")

    passed  = not issues
    details = (
        f"report={len(report)} chars, decision={decision}, sections_present={not any('missing report section' in i for i in issues)}"
        if passed else " | ".join(issues)
    )
    return EvalResult("E2E — resume with 'decline' → final report", passed, 1.0 if passed else 0.0, details)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_orchestrator_evals() -> list:
    """
    Runs the full analysis ONCE and reuses the graph/config for the resume eval.
    Two evals total: structure check + resume flow.
    """
    results = []

    # Run full analysis
    try:
        result, graph, config = _run_full_analysis()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        results.append(EvalResult(
            "E2E — full analysis structure", False, 0.0,
            f"Exception running full analysis: {e}", error=tb, category=CAT,
        ))
        results.append(EvalResult(
            "E2E — resume flow", False, 0.0,
            "Skipped — full analysis failed", category=CAT,
        ))
        return results

    # Structure eval
    structure_result = eval_full_analysis_structure(result)
    structure_result.category = CAT
    results.append(structure_result)

    # Resume eval — only attempt if structure passed (valid graph state)
    if structure_result.passed:
        resume_result = safe_run(
            "E2E — resume flow",
            lambda g=graph, c=config: eval_resume_flow(g, c),
            CAT,
        )
    else:
        resume_result = EvalResult(
            "E2E — resume flow", False, 0.0,
            "Skipped — structure eval failed (graph state may be invalid)",
            category=CAT,
        )

    results.append(resume_result)
    return results
