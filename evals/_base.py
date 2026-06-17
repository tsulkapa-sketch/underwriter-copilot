"""
Base types and helpers for the eval suite.
"""

from dataclasses import dataclass, field
from typing import Optional
import traceback


@dataclass
class EvalResult:
    name:     str
    passed:   bool
    score:    float          # 0.0 – 1.0
    details:  str
    error:    Optional[str] = None
    category: str = "misc"


def safe_run(name: str, fn, category: str = "misc") -> EvalResult:
    """Run an eval function; catch all exceptions so the suite never crashes."""
    try:
        result = fn()
        result.category = category
        return result
    except Exception:
        tb = traceback.format_exc()
        short = tb.strip().split("\n")[-1]
        return EvalResult(
            name=name, passed=False, score=0.0,
            details=f"Exception: {short}",
            error=tb, category=category,
        )


def score_keywords(text: str, keywords: list) -> float:
    """Return fraction of keywords found in text (case-insensitive)."""
    if not text or not keywords:
        return 0.0
    text_lower = text.lower()
    found = sum(1 for kw in keywords if str(kw).lower() in text_lower)
    return found / len(keywords)
