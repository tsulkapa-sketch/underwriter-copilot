import json
import time
import statistics
from main import setup, build_or_load_index

# ── Test Cases ─────────────────────────────────────────────────────────────────
# These are the ground truth answers based on our sample documents.
# In production: underwriters define these from real reviewed cases.

TEST_CASES = [
    {
        "id": "TC001",
        "category": "financial",
        "question": "What is the borrower's net profit margin for FY2024?",
        "expected_facts": ["8.1%", "FY2024"],
        "expected_source": "loan_application",
        "must_not_contain": ["9.8%", "9.0%"],  # prior year figures
    },
    {
        "id": "TC002",
        "category": "financial",
        "question": "What is the total loan amount being requested?",
        "expected_facts": ["2.70 crore", "1.80 crore", "90 lakh"],
        "expected_source": "loan_application",
        "must_not_contain": [],
    },
    {
        "id": "TC003",
        "category": "bureau",
        "question": "What is the commercial credit score and risk category?",
        "expected_facts": ["74", "MEDIUM"],
        "expected_source": "bureau_report",
        "must_not_contain": ["HIGH", "LOW"],
    },
    {
        "id": "TC004",
        "category": "bureau",
        "question": "Were there any payment delays in the bureau report?",
        "expected_facts": ["30", "HDFC", "March 2023"],
        "expected_source": "bureau_report",
        "must_not_contain": ["no delays", "clean record"],
    },
    {
        "id": "TC005",
        "category": "risk",
        "question": "What is the customer concentration risk for this borrower?",
        "expected_facts": ["67%", "two", "EuroFashion", "Atlantic"],
        "expected_source": "loan_application",
        "must_not_contain": [],
    },
    {
        "id": "TC006",
        "category": "risk",
        "question": "What collateral has been offered against this loan?",
        "expected_facts": ["machinery", "factory", "MIDC", "3.4 crore"],
        "expected_source": "loan_application",
        "must_not_contain": [],
    },
    {
        "id": "TC007",
        "category": "hallucination",
        "question": "What is the borrower's export revenue from Japan?",
        "expected_facts": [],
        "expected_source": None,
        # For hallucination test: answer must say info is not available
        "must_contain_any": [
            "not mentioned", "not available", "no information",
            "not found", "does not mention", "cannot find"
        ],
        "must_not_contain": ["Japan", "¥", "JPY"],
    },
    {
        "id": "TC008",
        "category": "consistency",  # run 3 times, check same answer
        "question": "What is the borrower's annual revenue for FY2024?",
        "expected_facts": ["421", "lakh"],
        "expected_source": "loan_application",
        "must_not_contain": [],
        "runs": 3,  # consistency test — run multiple times
    },
]

# ── LLM Judge ──────────────────────────────────────────────────────────────────
def llm_judge(question, answer, expected_facts, must_not_contain,
              must_contain_any, source_cited, expected_source, anthropic_client):
    """
    Uses Claude as a judge to score the copilot's answer.
    Returns scores for each dimension.
    """
    judge_prompt = f"""You are evaluating an AI underwriter copilot's answer.
Score each dimension from 1-5. Return ONLY valid JSON, nothing else.

QUESTION: {question}

COPILOT ANSWER: {answer}

EVALUATION CRITERIA:
1. answer_correctness: Did the answer correctly state these facts: {expected_facts}?
   5 = all facts correct, 3 = some correct, 1 = wrong or missing
   
2. groundedness: Did the answer stay grounded in documents without hallucinating?
   Facts it must NOT contain (hallucination check): {must_not_contain}
   5 = fully grounded, 3 = minor speculation, 1 = clear hallucination
   
3. risk_completeness: For risk questions, did it identify all key risks?
   For non-risk questions, score 5 by default.
   5 = complete, 3 = partial, 1 = missed key risks
   
4. citation_accuracy: Did it cite the source document correctly?
   Expected source: {expected_source}
   5 = correct citation, 3 = vague citation, 1 = wrong or no citation
   
5. clarity: Is the answer clear and professional for an underwriter?
   5 = excellent, 3 = acceptable, 1 = confusing

Return this exact JSON structure:
{{
  "answer_correctness": <1-5>,
  "groundedness": <1-5>,
  "risk_completeness": <1-5>,
  "citation_accuracy": <1-5>,
  "clarity": <1-5>,
  "reasoning": "<one sentence explaining the scores>"
}}"""

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": judge_prompt}]
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

# ── Hallucination check ────────────────────────────────────────────────────────
def check_hallucination(answer, test_case):
    answer_lower = answer.lower()

    # Check must_not_contain
    hallucinated = [
        phrase for phrase in test_case.get("must_not_contain", [])
        if phrase.lower() in answer_lower
    ]

    # For hallucination test cases, check must_contain_any
    correctly_declined = True
    if "must_contain_any" in test_case:
        correctly_declined = any(
            phrase.lower() in answer_lower
            for phrase in test_case["must_contain_any"]
        )

    return hallucinated, correctly_declined

# ── Consistency check ──────────────────────────────────────────────────────────
def check_consistency(answers):
    """
    Checks if multiple runs produce consistent core facts.
    Uses simple overlap scoring — in production use embedding similarity.
    """
    if len(answers) < 2:
        return 1.0

    # Check if key terms appear consistently across all runs
    word_sets = [set(a.lower().split()) for a in answers]
    common_words = word_sets[0]
    for ws in word_sets[1:]:
        common_words = common_words.intersection(ws)

    # Remove stop words for meaningful comparison
    stop_words = {"the", "a", "an", "is", "in", "of", "and", "to",
                  "for", "with", "that", "this", "are", "was", "has"}
    meaningful_common = common_words - stop_words

    all_meaningful = set()
    for ws in word_sets:
        all_meaningful.update(ws - stop_words)

    consistency_score = (
        len(meaningful_common) / len(all_meaningful)
        if all_meaningful else 0
    )
    return round(consistency_score, 2)

# ── Run single test ────────────────────────────────────────────────────────────
def run_test(test_case, query_engine, anthropic_client):
    runs = test_case.get("runs", 1)
    answers = []

    for _ in range(runs):
        response = query_engine.query(test_case["question"])
        answers.append(str(response))
        if runs > 1:
            time.sleep(1)  # avoid rate limiting on multiple runs

    primary_answer = answers[0]
    hallucinated, correctly_declined = check_hallucination(
        primary_answer, test_case
    )

    # Check if source was cited
    source_cited = (
        test_case["expected_source"] is None or
        (test_case["expected_source"] and
         test_case["expected_source"].lower().replace("_", "") in
         primary_answer.lower().replace("_", "").replace(" ", ""))
    )

    # LLM judge scoring
    scores = llm_judge(
        question=test_case["question"],
        answer=primary_answer,
        expected_facts=test_case["expected_facts"],
        must_not_contain=test_case.get("must_not_contain", []),
        must_contain_any=test_case.get("must_contain_any", []),
        source_cited=source_cited,
        expected_source=test_case["expected_source"],
        anthropic_client=anthropic_client,
    )

    consistency = check_consistency(answers) if runs > 1 else None

    return {
        "id": test_case["id"],
        "category": test_case["category"],
        "question": test_case["question"],
        "answer": primary_answer,
        "all_answers": answers if runs > 1 else None,
        "scores": scores,
        "hallucinated_phrases": hallucinated,
        "correctly_declined": correctly_declined,
        "source_cited": source_cited,
        "consistency_score": consistency,
        "passed": (
            len(hallucinated) == 0 and
            scores["answer_correctness"] >= 3 and
            scores["groundedness"] >= 3
        )
    }

# ── Print result ───────────────────────────────────────────────────────────────
def print_result(result):
    status = "PASS" if result["passed"] else "FAIL"
    print(f"\n[{status}] {result['id']} — {result['category'].upper()}")
    print(f"Q: {result['question']}")
    print(f"A: {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")
    print(f"Scores: correctness={result['scores']['answer_correctness']} | "
          f"groundedness={result['scores']['groundedness']} | "
          f"risk_completeness={result['scores']['risk_completeness']} | "
          f"citation={result['scores']['citation_accuracy']} | "
          f"clarity={result['scores']['clarity']}")
    print(f"Reasoning: {result['scores']['reasoning']}")

    if result["hallucinated_phrases"]:
        print(f"HALLUCINATION DETECTED: {result['hallucinated_phrases']}")

    if result["consistency_score"] is not None:
        pct = result["consistency_score"] * 100
        print(f"Consistency across {len(result['all_answers'])} runs: {pct:.0f}%")

    if result["category"] == "hallucination":
        declined = result["correctly_declined"]
        print(f"Correctly declined unanswerable question: {declined}")

# ── Summary ────────────────────────────────────────────────────────────────────
def print_summary(results):
    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    all_scores = {
        "answer_correctness": [],
        "groundedness": [],
        "risk_completeness": [],
        "citation_accuracy": [],
        "clarity": [],
    }
    for r in results:
        for dim, scores in all_scores.items():
            scores.append(r["scores"][dim])

    print("\n" + "=" * 55)
    print("  EVAL SUMMARY")
    print("=" * 55)
    print(f"  Overall pass rate : {passed}/{total} ({passed/total*100:.0f}%)")
    print()
    for dim, scores in all_scores.items():
        avg = statistics.mean(scores)
        bar = "█" * int(avg) + "░" * (5 - int(avg))
        print(f"  {dim:<22} {bar} {avg:.1f}/5")

    consistency_results = [
        r for r in results if r["consistency_score"] is not None
    ]
    if consistency_results:
        avg_consistency = statistics.mean(
            r["consistency_score"] for r in consistency_results
        )
        print(f"\n  Consistency score  : {avg_consistency*100:.0f}%")

    hallucination_results = [
        r for r in results if r["category"] == "hallucination"
    ]
    if hallucination_results:
        declined_correctly = sum(
            1 for r in hallucination_results if r["correctly_declined"]
        )
        print(f"  Hallucination guard: "
              f"{declined_correctly}/{len(hallucination_results)} correctly declined")

    print("\n  By category:")
    categories = set(r["category"] for r in results)
    for cat in sorted(categories):
        cat_results = [r for r in results if r["category"] == cat]
        cat_passed = sum(1 for r in cat_results if r["passed"])
        print(f"    {cat:<20} {cat_passed}/{len(cat_results)} passed")

    print("=" * 55)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    import anthropic

    print("\n" + "=" * 55)
    print("  UNDERWRITER COPILOT — EVAL HARNESS")
    print("=" * 55)
    print(f"  Running {len(TEST_CASES)} test cases...\n")

    setup()
    index = build_or_load_index()
    query_engine = index.as_query_engine(similarity_top_k=5)
    anthropic_client = anthropic.Anthropic()

    results = []
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"  Running {i}/{len(TEST_CASES)}: {test_case['id']}...",
              end="", flush=True)
        result = run_test(test_case, query_engine, anthropic_client)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f" {status}")

    print("\n--- DETAILED RESULTS ---")
    for result in results:
        print_result(result)

    print_summary(results)

    # Save results to JSON for tracking over time
    with open("eval_results.json", "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %Human:%M:%S"),
            "results": results
        }, f, indent=2, default=str)
    print("\n  Full results saved to eval_results.json")

if __name__ == "__main__":
    main()
