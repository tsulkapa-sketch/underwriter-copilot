"""
RAG quality evals (4–6 tests).

Two modes:
  1. Keyword presence  — fast, no LLM needed
  2. LLM-as-judge      — slower, uses claude-haiku, opt-in with --llm-judge

Scoring for keyword tests: fraction of expected keywords found.
LLM judge scores faithfulness (0-3), relevance (0-3), specificity (0-3),
passes if total >= 6 / 9.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals._base import EvalResult, safe_run, score_keywords
from evals.fixtures import RAG_TESTS

CAT = "rag"


def _run_rag_tool(tool_name: str, question: str) -> str:
    """Dispatch to the appropriate RAG tool based on tool_name."""
    if tool_name == "policy":
        from tools.layer3_rag import rag_query_policy
        return rag_query_policy.invoke({"question": question})
    elif tool_name == "collateral":
        from tools.layer3_rag import rag_query_collateral
        return rag_query_collateral.invoke({"question": question})
    else:
        from tools.layer3_rag import rag_query
        return rag_query.invoke({"question": question})


def eval_rag_keywords(tc: dict) -> EvalResult:
    """
    Check that the RAG answer contains the expected keywords.
    Score = fraction of keywords found (0.0 – 1.0).
    Pass threshold = tc['min_score'].
    """
    answer = _run_rag_tool(tc["tool"], tc["question"])

    if not answer or len(answer.strip()) < 20:
        return EvalResult(
            f"RAG keyword — {tc['id']}",
            False, 0.0,
            f"Empty/very short response: '{answer[:60]}'",
        )

    kw_score = score_keywords(answer, tc["keywords"])
    passed   = kw_score >= tc["min_score"]

    found = [kw for kw in tc["keywords"] if kw.lower() in answer.lower()]
    details = (
        f"score={kw_score:.0%}, found={found}, answer_len={len(answer)}"
    )
    return EvalResult(f"RAG keyword — {tc['id']}", passed, kw_score, details)


def eval_rag_llm_judge(tc: dict) -> EvalResult:
    """
    LLM-as-judge: use claude-haiku to rate faithfulness, relevance, specificity.
    Pass if total >= 6/9.
    """
    from langchain_anthropic import ChatAnthropic

    answer = _run_rag_tool(tc["tool"], tc["question"])

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",      # cheaper model for judging
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        temperature=0,
    )

    prompt = f"""You are evaluating a RAG system that answers questions about a credit loan application.

QUESTION : {tc["question"]}
ANSWER   : {answer}
EXPECTED : {tc["hint"]}

Rate on three dimensions:
  FAITHFULNESS (0-3) — grounded in documents, no hallucination
  RELEVANCE    (0-3) — directly addresses the question
  SPECIFICITY  (0-3) — includes concrete figures, names, or values

Reply ONLY in this exact format:
FAITHFULNESS: <0-3>
RELEVANCE: <0-3>
SPECIFICITY: <0-3>
TOTAL: <0-9>
VERDICT: PASS | FAIL

PASS if TOTAL >= 6."""

    response = llm.invoke(prompt)
    text     = response.content

    total, verdict = 0, "FAIL"
    faithfulness = relevance = specificity = "?"
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("FAITHFULNESS:"): faithfulness = line.split(":", 1)[1].strip()
        if line.startswith("RELEVANCE:"):    relevance    = line.split(":", 1)[1].strip()
        if line.startswith("SPECIFICITY:"):  specificity  = line.split(":", 1)[1].strip()
        if line.startswith("TOTAL:"):
            try: total = int(line.split(":", 1)[1].strip())
            except: pass
        if line.startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip()

    passed  = verdict == "PASS"
    score   = total / 9.0
    details = f"F={faithfulness} R={relevance} S={specificity} total={total}/9 → {verdict}"
    return EvalResult(f"RAG LLM judge — {tc['id']}", passed, score, details)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_rag_evals(include_llm_judge: bool = False) -> list:
    results = []

    # Keyword evals — all 4 test cases
    for tc in RAG_TESTS:
        results.append(safe_run(f"RAG keyword — {tc['id']}", lambda t=tc: eval_rag_keywords(t), CAT))

    # LLM judge — first 2 test cases only (cost control)
    if include_llm_judge:
        for tc in RAG_TESTS[:2]:
            results.append(safe_run(f"RAG LLM judge — {tc['id']}", lambda t=tc: eval_rag_llm_judge(t), CAT))

    return results
