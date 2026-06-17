# Underwriter Copilot

A multi-agent AI system for SME credit underwriting. Five specialist agents run in parallel — financial analysis, bureau & fraud, cross-document contradiction detection, policy compliance, and collateral assessment — then aggregate into a structured decision report with a human-in-the-loop review step.

Built as a prototype to demonstrate how GenAI can accelerate credit decisioning in financial services.

---

## What it does

1. **Deep Analysis** — One API call triggers 5 agents in parallel. Each agent pulls simulated bureau, fraud, CRILC, valuation, and bank data, then uses Claude to synthesise findings.
2. **Human-in-the-loop** — Graph pauses after aggregation. The underwriter reviews all agent ratings, key metrics, contradictions, and conditions, then submits their decision.
3. **Final Report** — On decision submission, a structured credit decision report is compiled with all agent findings, hard stops, conditions, and the underwriter's rationale.
4. **RAG Q&A** — Free-text questions over the loan documents (bureau report, financials, GST returns, bank statements, valuation report).

**Case used in this demo:** SME-2024-00891 — Meridian Textile Exports, ₹270L loan request.

---

## Architecture

```
Browser UI (ui/index.html)
        │  POST /agent/full-analysis
        ▼
FastAPI (api.py)
        │
        ▼
LangGraph Orchestrator (agent.py)
  ├── load_case_data
  ├── [parallel fan-out]
  │     ├── Financial Agent   → DSCR, scorecard, bank variance
  │     ├── Bureau Agent      → credit score, fraud model, CRILC
  │     ├── Contradiction Agent → cross-doc mismatch detection (RAG)
  │     ├── Policy Agent      → hard stops, conditions, warnings
  │     └── Collateral Agent  → LTV, coverage ratio, valuation
  ├── aggregate_findings      → overall risk, system recommendation
  ├── human_review            → interrupt() — waits for underwriter
  └── compile_report          → final structured report

Tool layers:
  Layer 1 — Simulated external APIs (bureau, fraud model, CRILC, valuation, core banking)
  Layer 2 — Deterministic calculators (DSCR, scorecard, policy rules, LTV, bank statement)
  Layer 3 — RAG via LlamaIndex (loan_docs/ → HuggingFace embeddings → Claude)
```

---

## Quickstart

### Requirements
- Python 3.11 (required — some dependencies not compatible with 3.12/3.13)
- Anthropic API key → [console.anthropic.com](https://console.anthropic.com)

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/underwriter-copilot.git
cd underwriter-copilot
```

### 2. Create virtual environment
```bash
python -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> First run also downloads the HuggingFace embedding model (~130MB). This is cached after the first download.

### 4. Set your API key

Copy the example file and fill it in:
```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Start the server
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

First start indexes the loan documents — takes ~30 seconds. Subsequent starts are instant (index cached in `stored_index/`).

### 6. Open the UI

```
http://localhost:8000/ui/index.html
```

---

## Running a full analysis

In the UI:

1. Click **"Run Deep Analysis"** in the chat panel (or the Approve/Decline/Conditional/Escalate buttons in the left panel).
2. Wait ~60–90 seconds while all 5 agents run in parallel. Progress messages cycle in the chat.
3. Review the findings panel — agent risk ratings, key metrics (DSCR, credit score, LTV), contradictions, conditions.
4. Click one of the decision buttons: **Approve / Conditional / Decline / Escalate**.
5. The final credit decision report appears in the chat.

You can also ask free-text questions about the case at any time:
```
What is the DSCR?
Are there any bureau hard stops?
What collateral has been offered?
Does bank turnover support stated revenue?
```

---

## Running the eval suite

The eval suite has four tiers:

```bash
# Fast — tool correctness only, no API key needed, ~30s
python -m evals.run_evals

# + Individual agent validation, needs API key, ~5 min
python -m evals.run_evals --agents

# + Full E2E orchestrator + resume, needs API key, ~2 min
python -m evals.run_evals --e2e

# Everything, including LLM-as-judge for RAG
python -m evals.run_evals --all --llm-judge
```

Exit code `0` = all pass, `1` = failures (CI-compatible).

---

## Project structure

```
underwriter_copilot/
├── loan_docs/                  ← Case documents (8 files, Meridian case)
│   ├── loan_application.txt
│   ├── financial_statements.txt
│   ├── bank_statements.txt
│   ├── bureau_report.txt
│   ├── gst_returns.txt
│   ├── itr_fy2024.txt
│   ├── valuation_report.txt
│   └── lending_policy.txt
│
├── agents/                     ← 5 specialist agents
│   ├── financial.py            → DSCR, scorecard, bank statement analysis
│   ├── bureau.py               → Bureau, CRILC, fraud model
│   ├── contradiction.py        → Cross-document mismatch detection
│   ├── policy.py               → Hard stops, conditions, warnings
│   └── collateral.py           → LTV, coverage, valuation
│
├── tools/                      ← Tool layer (called by agents)
│   ├── layer1_apis.py          → Simulated external APIs
│   ├── layer2_tools.py         → Deterministic calculators
│   └── layer3_rag.py           → RAG tools (LlamaIndex)
│
├── evals/                      ← Eval suite
│   ├── run_evals.py            → CLI runner
│   ├── eval_tools.py           → 10 tool correctness tests
│   ├── eval_agents.py          → 5 agent validation tests
│   ├── eval_rag.py             → RAG keyword + LLM judge tests
│   ├── eval_orchestrator.py    → E2E + resume flow tests
│   ├── fixtures.py             → Ground truth + test inputs
│   └── _base.py                → EvalResult, safe_run, helpers
│
├── ui/
│   └── index.html              ← Browser UI (single file, no build step)
│
├── agent.py                    ← LangGraph multi-agent orchestrator
├── api.py                      ← FastAPI server
├── main.py                     ← RAG pipeline setup
├── requirements.txt
├── .env.example                ← Copy to .env and add your API key
└── README.md
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/agent/full-analysis` | Start 5-agent parallel analysis |
| POST | `/agent/full-analysis/resume` | Submit underwriter decision, get final report |
| POST | `/query` | Free-text RAG question |
| POST | `/agent/scan` | Contradiction scan only (legacy) |

---

## Built with

- [LangGraph](https://github.com/langchain-ai/langgraph) — multi-agent orchestration, human-in-the-loop interrupt
- [Anthropic Claude](https://www.anthropic.com) — agent reasoning and report generation
- [LlamaIndex](https://www.llamaindex.ai) — RAG pipeline and document indexing
- [FastAPI](https://fastapi.tiangolo.com) — REST API
- HuggingFace `BAAI/bge-small-en-v1.5` — local embedding model

---

## Disclaimer

Prototype only. All case data (Meridian Textile Exports) is fictional. Not for use in actual credit decisions.
