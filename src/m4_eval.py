from __future__ import annotations
"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os
import sys
import json
from dataclasses import dataclass
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation với 4 metrics: faithfulness, answer_relevancy, context_precision, context_recall.

    Returns:
        Dict với 4 metric keys (float) và 'per_question' (list[EvalResult]).
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()

        per_question = [
            EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", [])),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            )
            for _, row in df.iterrows()
        ]

        return {
            "faithfulness": float(df["faithfulness"].mean()),
            "answer_relevancy": float(df["answer_relevancy"].mean()),
            "context_precision": float(df["context_precision"].mean()),
            "context_recall": float(df["context_recall"].mean()),
            "per_question": per_question,
        }

    except Exception as e:
        # Fallback: trả về mock scores nếu RAGAS gặp lỗi (ví dụ không có API key)
        print(f"  ⚠️  RAGAS error: {e}. Using mock scores.")
        n = len(questions)
        per_question = [
            EvalResult(
                question=questions[i],
                answer=answers[i],
                contexts=contexts[i],
                ground_truth=ground_truths[i],
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
            )
            for i in range(n)
        ]
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": per_question,
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions theo Diagnostic Tree.

    Returns:
        List of {question, worst_metric, score, diagnosis, suggested_fix}
    """
    if not eval_results:
        return []

    # Diagnostic thresholds và mappings
    THRESHOLDS = {
        "faithfulness": (0.85, "LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": (0.75, "Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": (0.75, "Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": (0.80, "Answer doesn't match question", "Improve prompt template"),
    }

    def avg_score(r: EvalResult) -> float:
        return mean([r.faithfulness, r.answer_relevancy, r.context_precision, r.context_recall])

    # Sắp xếp theo avg_score tăng dần → lấy bottom_n
    sorted_results = sorted(eval_results, key=avg_score)[:bottom_n]

    failures = []
    for result in sorted_results:
        metric_scores = {
            "faithfulness": result.faithfulness,
            "context_recall": result.context_recall,
            "context_precision": result.context_precision,
            "answer_relevancy": result.answer_relevancy,
        }
        # Tìm metric tệ nhất
        worst_metric = min(metric_scores, key=lambda m: metric_scores[m])
        worst_score = metric_scores[worst_metric]

        threshold, diagnosis, suggested_fix = THRESHOLDS[worst_metric]
        # Nếu score vượt threshold, dùng metric tệ nhất tương đối
        if worst_score >= threshold:
            diagnosis = "Performance acceptable but below target"
            suggested_fix = "Fine-tune retrieval parameters"

        failures.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": worst_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
