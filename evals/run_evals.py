#!/usr/bin/env python
"""
Underwriter Copilot — Eval Runner
==================================
Usage:
  py -3.11 evals/run_evals.py                   # Tools + RAG        (~30s, no API key needed)
  py -3.11 evals/run_evals.py --agents           # + Agent validation (~5 min, needs API key)
  py -3.11 evals/run_evals.py --e2e              # + Full E2E         (~2 min, needs API key)
  py -3.11 evals/run_evals.py --routing          # + Routing evals    (~5 min, needs API key)
  py -3.11 evals/run_evals.py --all              # Everything         (~10 min, needs API key)
  py -3.11 evals/run_evals.py --llm-judge        # + LLM-as-judge for RAG
  py -3.11 evals/run_evals.py --section tools    # Tools only
  py -3.11 evals/run_evals.py --section rag      # RAG only
  py -3.11 evals/run_evals.py --section routing  # Routing only

Exit code: 0 if all pass, 1 if any fail.
"""

import sys
import os
import time

# Allow running from repo root or from evals/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Parse args ───────────────────────────────────────────────────────────────
args         = set(sys.argv[1:])
RUN_AGENTS   = "--agents"   in args or "--all" in args
RUN_E2E      = "--e2e"      in args or "--all" in args
RUN_ROUTING  = "--routing"  in args or "--all" in args
LLM_JUDGE    = "--llm-judge" in args or "--all" in args
SECTION      = None
if "--section" in sys.argv:
    idx     = sys.argv.index("--section")
    SECTION = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None

# ── Display helpers ───────────────────────────────────────────────────────────
W     = 62
GREEN = "\033[32m"
RED   = "\033[31m"
DIM   = "\033[90m"
BOLD  = "\033[1m"
RESET = "\033[0m"

def bar(char="─"):
    return char * W

def section_header(title: str):
    print(f"\n{bar()}")
    print(f"  {BOLD}{title}{RESET}")
    print(bar())

def print_result(r):
    icon   = f"{GREEN}✓{RESET}" if r.passed else f"{RED}✗{RESET}"
    status = f"{GREEN}PASS{RESET}" if r.passed else f"{RED}FAIL{RESET}"
    # Truncate name to fit
    name = r.name[:50]
    print(f"  {icon}  {name:<52} {status}")
    if r.details:
        details = r.details[:110]
        print(f"      {DIM}{details}{RESET}")
    if not r.passed and r.error:
        # Show last 2 lines of traceback
        lines = [l for l in r.error.strip().split("\n") if l.strip()][-2:]
        for line in lines:
            print(f"      {DIM}{line}{RESET}")

def print_summary(all_results: list, elapsed: float):
    passed  = sum(1 for r in all_results if r.passed)
    total   = len(all_results)
    failed  = [r for r in all_results if not r.passed]
    avg     = sum(r.score for r in all_results) / total if total else 0

    # Pass rate colour
    pct    = passed / total if total else 0
    color  = GREEN if pct == 1.0 else RED if pct < 0.7 else "\033[33m"

    print(f"\n{bar('═')}")
    print(f"  {BOLD}RESULTS: {color}{passed}/{total} passed{RESET}  "
          f"│  Score: {avg:.0%}  │  Time: {elapsed:.1f}s")

    if failed:
        print(f"\n  {RED}FAILURES:{RESET}")
        for r in failed:
            print(f"    ✗ {r.name}")

    # Category breakdown
    cats = {}
    for r in all_results:
        cats.setdefault(r.category, []).append(r)

    print(f"\n  BREAKDOWN:")
    for cat, rs in cats.items():
        p   = sum(1 for r in rs if r.passed)
        col = GREEN if p == len(rs) else RED if p == 0 else "\033[33m"
        print(f"    {cat:<12} : {col}{p}/{len(rs)}{RESET} passed")

    print(bar("═"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_results = []
    start       = time.time()

    mode_label = (
        "All tiers (--all)"                    if (RUN_E2E and RUN_AGENTS and RUN_ROUTING) else
        "Tools + RAG + Agents + E2E (--all)"   if RUN_E2E and RUN_AGENTS else
        "Tools + RAG + Agents (--agents)"       if RUN_AGENTS else
        "Tools + RAG + E2E (--e2e)"             if RUN_E2E else
        "Routing scenarios (--routing)"         if RUN_ROUTING else
        f"Section: {SECTION}"                   if SECTION else
        "Tools + RAG  [use --agents/--e2e/--routing/--all for more]"
    )

    print(f"\n{bar('═')}")
    print(f"  {BOLD}UNDERWRITER COPILOT — EVAL SUITE{RESET}")
    print(f"  Case : SME-2024-00891 (Meridian Textile Exports)")
    print(f"  Mode : {mode_label}")
    print(bar("═"))

    # ── Tools ────────────────────────────────────────────────
    if not SECTION or SECTION == "tools":
        section_header("LAYER 1 & 2 — Tool Correctness  (10 tests, ~5s)")
        from evals.eval_tools import run_all_tool_evals
        results = run_all_tool_evals()
        for r in results:
            print_result(r)
        all_results.extend(results)

    # ── RAG ──────────────────────────────────────────────────
    if not SECTION or SECTION == "rag":
        n_rag = 4 + (2 if LLM_JUDGE else 0)
        section_header(f"LAYER 3 — RAG Quality  ({n_rag} tests, ~15s)")
        if LLM_JUDGE:
            print(f"  {DIM}LLM judge enabled — uses claude-haiku (2 extra tests){RESET}")
        from evals.eval_rag import run_all_rag_evals
        results = run_all_rag_evals(include_llm_judge=LLM_JUDGE)
        for r in results:
            print_result(r)
        all_results.extend(results)

    # ── Agents ───────────────────────────────────────────────
    if RUN_AGENTS and (not SECTION or SECTION == "agents"):
        section_header("AGENTS — Individual Agent Validation  (5 tests, ~5 min)")
        print(f"  {DIM}Each agent makes multiple Claude API calls{RESET}")
        from evals.eval_agents import run_all_agent_evals
        results = run_all_agent_evals()
        for r in results:
            print_result(r)
        all_results.extend(results)

    # ── E2E ──────────────────────────────────────────────────
    if RUN_E2E and (not SECTION or SECTION == "e2e"):
        section_header("E2E — Full Orchestrator + Resume  (2 tests, ~90s)")
        print(f"  {DIM}Runs all 5 agents in parallel — most expensive eval{RESET}")
        from evals.eval_orchestrator import run_all_orchestrator_evals
        results = run_all_orchestrator_evals()
        for r in results:
            print_result(r)
        all_results.extend(results)

    # ── Routing ──────────────────────────────────────────────
    if RUN_ROUTING and (not SECTION or SECTION == "routing"):
        section_header("ROUTING — Conditional Router Correctness  (4 tests, ~5 min)")
        print(f"  {DIM}Runs full graph for each scenario, asserts routing_path{RESET}")
        from evals.eval_routing import run_all_routing_tests
        summary = run_all_routing_tests(verbose=True)
        # Convert summary results to EvalResult format for unified reporting
        from evals._base import EvalResult  # reuse the dataclass
        for r in summary["results"]:
            failures_str = "; ".join(r["failures"]) if r["failures"] else ""
            all_results.append(EvalResult(
                name     = f"routing/{r['case_id']} → {r['expected_routing']}",
                passed   = r["passed"],
                score    = 1.0 if r["passed"] else 0.0,
                details  = f"actual={r['actual_routing']}  decision={r['decision']}",
                error    = failures_str,
                category = "routing",
            ))

    elapsed = time.time() - start
    print_summary(all_results, elapsed)

    # Exit code: 0 = all pass, 1 = any fail
    passed = sum(1 for r in all_results if r.passed)
    sys.exit(0 if passed == len(all_results) else 1)


if __name__ == "__main__":
    main()
