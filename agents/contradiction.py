"""
Contradiction Agent
-------------------
Detects where the same figure appears differently across documents.
Adapted from the original agent.py into a standalone function so the
orchestrator can run it as one node in the multi-agent graph.

Tools used:
  - rag_cross_document_compare  (Layer 3)
  - rag_query                   (Layer 3)

Uses Claude to extract claims per document, compare them, and rate severity.
Returns a list of contradictions with severity ratings.
"""

import os
from langchain_anthropic import ChatAnthropic
from tools import rag_cross_document_compare

_model = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    temperature=0,
)

# Fields to cross-check across all documents
_FIELDS_TO_CHECK = [
    "annual revenue",
    "net profit",
    "bank turnover",
    "total debt",
    "credit score",
    "cash credit outstanding",
]


def run_contradiction_analysis(documents: dict) -> dict:
    """
    Detect cross-document contradictions for a loan case.

    Args:
        documents: dict of {filename: content} from loan_docs/

    Returns:
        dict with contradictions list, severities, summary, risk_rating
    """
    print("[ContradictionAgent] Starting cross-document comparison...")

    # ── Step 1: Extract key claims per document ────────────────────────────────
    claims = {}
    for filename, content in documents.items():
        prompt = f"""Extract verifiable financial figures from this credit document.
Document: {filename}

{content[:3000]}

Return ONLY these fields, one per line, using exact values from the document:
REVENUE: <figure or NOT_FOUND>
NET_PROFIT: <figure or NOT_FOUND>
BANK_TURNOVER: <figure or NOT_FOUND>
CREDIT_SCORE: <figure or NOT_FOUND>
TOTAL_DEBT: <figure or NOT_FOUND>
CC_OUTSTANDING: <figure or NOT_FOUND>
NET_PROFIT_MARGIN: <figure or NOT_FOUND>
EBITDA_MARGIN: <figure or NOT_FOUND>

Use exact figures only. Do not calculate or infer."""

        response = _model.invoke(prompt)
        claims[filename] = response.content
        print(f"[ContradictionAgent]   Claims extracted: {filename}")

    # ── Step 2: RAG cross-document comparison for key fields ──────────────────
    rag_comparisons = {}
    for field in _FIELDS_TO_CHECK:
        rag_comparisons[field] = rag_cross_document_compare.invoke({"field": field})

    # ── Step 3: Claude detects contradictions ─────────────────────────────────
    claims_text = "\n".join(
        f"\n--- {fname} ---\n{content}"
        for fname, content in claims.items()
    )

    detect_prompt = f"""You are a fraud detection specialist reviewing a loan application.
Compare these figures extracted from multiple documents for the same borrower.
Identify contradictions where the same figure differs materially (>2%) between sources.

EXTRACTED FIGURES BY DOCUMENT:
{claims_text}

Format each contradiction exactly as:

CONTRADICTION 1:
FIELD: <field name>
VALUES: <doc1>: <value> | <doc2>: <value> | <doc3>: <value>
VARIANCE: <amount and % difference>
FINDING: <one sentence — what this means for the credit assessment>

CONTRADICTION 2:
... and so on.

Rules:
- Only flag differences >2%. Minor rounding differences are NOT contradictions.
- If a field appears in only one document, skip it.
- If no contradiction exists, write: NO_CONTRADICTIONS_FOUND"""

    detect_response = _model.invoke(detect_prompt)
    contradictions = _parse_contradictions(detect_response.content)

    # ── Step 4: Rate severity of each contradiction ────────────────────────────
    severities = []
    for c in contradictions:
        sev_prompt = f"""Rate the severity of this financial contradiction in a loan application.

Field   : {c.get('field', 'Unknown')}
Values  : {c.get('values', 'Unknown')}
Variance: {c.get('variance', 'Unknown')}
Finding : {c.get('finding', 'Unknown')}

Rating criteria:
HIGH   — variance >10%, suggests deliberate misrepresentation, or impacts repayment capacity
MEDIUM — variance 5-10%, could be methodology difference but needs clarification
LOW    — variance 2-5%, likely timing or rounding

Return only:
SEVERITY: HIGH | MEDIUM | LOW
REASON: <one sentence>"""

        sev_response = _model.invoke(sev_prompt)
        severity, reason = "MEDIUM", ""
        for line in sev_response.content.split("\n"):
            if line.startswith("SEVERITY:"):
                raw = line.replace("SEVERITY:", "").strip().upper()
                severity = raw if raw in ("HIGH", "MEDIUM", "LOW") else "MEDIUM"
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()

        severities.append({
            "field":    c.get("field", "Unknown"),
            "severity": severity,
            "reason":   reason,
        })
        print(f"[ContradictionAgent]   {c.get('field')}: {severity}")

    # ── Step 5: Summary counts ─────────────────────────────────────────────────
    high   = sum(1 for s in severities if s["severity"] == "HIGH")
    medium = sum(1 for s in severities if s["severity"] == "MEDIUM")
    low    = sum(1 for s in severities if s["severity"] == "LOW")

    risk_rating = "HIGH" if high > 0 else ("MEDIUM" if medium > 0 else "LOW")

    print(f"[ContradictionAgent] Done — {len(contradictions)} contradiction(s): "
          f"{high} HIGH, {medium} MEDIUM, {low} LOW")

    return {
        "agent":           "contradiction",
        "status":          "completed",
        "claims":          claims,
        "rag_comparisons": rag_comparisons,
        "contradictions":  contradictions,
        "severities":      severities,
        "risk_rating":     risk_rating,
        "key_metrics": {
            "total":  len(contradictions),
            "high":   high,
            "medium": medium,
            "low":    low,
        },
    }


def _parse_contradictions(text: str) -> list:
    """Parse Claude's structured contradiction output into a list of dicts."""
    contradictions = []
    current = {}

    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith("NO_CONTRADICTIONS"):
            break
        if line.startswith("CONTRADICTION"):
            if current and "field" in current:
                contradictions.append(current)
            current = {}
        elif line.startswith("FIELD:"):
            current["field"] = line.replace("FIELD:", "").strip()
        elif line.startswith("VALUES:"):
            current["values"] = line.replace("VALUES:", "").strip()
        elif line.startswith("VARIANCE:"):
            current["variance"] = line.replace("VARIANCE:", "").strip()
        elif line.startswith("FINDING:"):
            current["finding"] = line.replace("FINDING:", "").strip()

    if current and "field" in current:
        contradictions.append(current)

    return contradictions
