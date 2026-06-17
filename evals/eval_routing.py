"""
evals/eval_routing.py
=====================
Routing correctness tests for the LangGraph conditional router.

Runs the full graph for each of the 4 scenario test cases and asserts that
`state["routing_path"]` matches the expected path. No LLM judge needed —
routing is deterministic given the case_data inputs.

Test matrix:
  WILFUL_DEFAULTER_CASE   → auto_decline_bureau   (bureau hard stop)
  LOW_DSCR_CASE           → auto_decline_policy   (DSCR < 1.0 → policy hard stop)
  HIGH_CONTRADICTION_CASE → escalate              (override_high_contradictions ≥ 2)
  CLEAN_CASE              → fast_track_approve    (all LOW, no conditions)

Run:
  python -m evals.eval_routing
  python -m evals.run_evals --routing

Each test calls run_full_analysis() with case_data override, then asserts:
  1. routing_path   == expected_routing
  2. underwriter_decision is non-empty and appropriate for the path
  3. final_report is populated (graph ran to completion without interrupt)
"""

import sys
import os
import time
from typing import Optional

# Ensure project root is on path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import run_full_analysis
from test_cases import (
    WILFUL_DEFAULTER_CASE,
    LOW_DSCR_CASE,
    HIGH_CONTRADICTION_CASE,
    CLEAN_CASE,
    ALL_TEST_CASES,
)

# ── Expected decision values per routing path ─────────────────────────────────
EXPECTED_DECISIONS = {
    "auto_decline_bureau":  "decline",
    "auto_decline_policy":  "decline",
    "fast_track_approve":   "approve",
    "escalate":             "escalate",
}

# ── Colours for terminal output ────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def _pass(msg: str) -> str:
    return f"{GREEN}✓ PASS{RESET}  {msg}"

def _fail(msg: str) -> str:
    return f"{RED}✗ FAIL{RESET}  {msg}"

def _warn(msg: str) -> str:
    return f"{YELLOW}⚠ WARN{RESET}  {msg}"


def run_routing_test(case_data: dict, verbose: bool = True) -> dict:
    """
    Run one routing test.

    Returns:
        {
            "case_id":          str,
            "expected_routing": str,
            "actual_routing":   str,
            "decision":         str,
            "passed":           bool,
            "failures":         list[str],
            "duration_s":       float,
        }
    """
    case_id          = case_data["case_id"]
    expected_routing = case_data["expected_routing"]
    failures         = []

    if verbose:
        print(f"\n{'─'*60}")
        print(f"{BOLD}Case: {case_id}  |  Expected: {expected_routing.upper()}{RESET}")
        print(f"{'─'*60}")

    t0 = time.time()
    try:
        result, graph, config = run_full_analysis(case_id, case_data=case_data)
    except Exception as e:
        return {
            "case_id":          case_id,
            "expected_routing": expected_routing,
            "actual_routing":   "ERROR",
            "decision":         "",
            "passed":           False,
            "failures":         [f"run_full_analysis raised: {e}"],
            "duration_s":       time.time() - t0,
        }
    duration = time.time() - t0

    actual_routing = result.get("routing_path", "")
    decision       = (result.get("underwriter_decision") or "").lower()
    final_report   = result.get("final_report", "")

    # ── Assertion 1: routing_path matches expected ────────────────────────────
    if actual_routing != expected_routing:
        failures.append(
            f"routing_path mismatch — expected '{expected_routing}', got '{actual_routing}'"
        )

    # ── Assertion 2: underwriter_decision is appropriate for this path ────────
    expected_decision = EXPECTED_DECISIONS.get(expected_routing, "")
    if expected_decision and decision != expected_decision:
        failures.append(
            f"decision mismatch — for path '{expected_routing}' expected "
            f"'{expected_decision}', got '{decision}'"
        )

    # ── Assertion 3: graph ran to completion (no interrupt for these paths) ───
    if result.get("__interrupt__"):
        failures.append(
            "graph returned __interrupt__ — auto-routed cases should not pause for human review"
        )

    # ── Assertion 4: final_report is populated ────────────────────────────────
    if not final_report or len(final_report.strip()) < 50:
        failures.append("final_report is empty or too short — compile_report may have failed")

    passed = len(failures) == 0

    if verbose:
        if passed:
            print(_pass(f"routing_path = {actual_routing.upper()}"))
            print(_pass(f"decision     = {decision.upper()}"))
            print(_pass(f"final_report = {len(final_report)} chars"))
        else:
            for f in failures:
                print(_fail(f))
            print(f"  routing_path  : {actual_routing}")
            print(f"  decision      : {decision}")
            print(f"  report_length : {len(final_report)}")
        print(f"  Duration: {duration:.1f}s")

    return {
        "case_id":          case_id,
        "expected_routing": expected_routing,
        "actual_routing":   actual_routing,
        "decision":         decision,
        "passed":           passed,
        "failures":         failures,
        "duration_s":       duration,
    }


def run_all_routing_tests(verbose: bool = True) -> dict:
    """
    Run all 4 routing tests and return a summary dict.

    Returns:
        {
            "passed":       int,
            "failed":       int,
            "total":        int,
            "results":      list[dict],
            "duration_s":   float,
        }
    """
    if verbose:
        print(f"\n{BOLD}{'='*60}")
        print("ROUTING EVAL — Conditional Routing Correctness")
        print(f"{'='*60}{RESET}")
        print(f"Running {len(ALL_TEST_CASES)} routing scenarios...")

    t0      = time.time()
    results = []

    for case in ALL_TEST_CASES:
        r = run_routing_test(case, verbose=verbose)
        results.append(r)

    total_duration = time.time() - t0
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    if verbose:
        print(f"\n{'='*60}")
        print(f"{BOLD}SUMMARY{RESET}")
        print(f"{'='*60}")
        for r in results:
            status = f"{GREEN}PASS{RESET}" if r["passed"] else f"{RED}FAIL{RESET}"
            print(
                f"  [{status}] {r['case_id']:22s}  "
                f"{r['expected_routing']:25s}  → {r['actual_routing']}"
            )
        print(f"\n  {passed}/{len(results)} passed  |  Total: {total_duration:.1f}s")
        if failed == 0:
            print(f"\n{GREEN}{BOLD}All routing tests passed.{RESET}")
        else:
            print(f"\n{RED}{BOLD}{failed} routing test(s) FAILED.{RESET}")

    return {
        "passed":     passed,
        "failed":     failed,
        "total":      len(results),
        "results":    results,
        "duration_s": total_duration,
    }


if __name__ == "__main__":
    summary = run_all_routing_tests(verbose=True)
    sys.exit(0 if summary["failed"] == 0 else 1)
