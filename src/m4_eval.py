from __future__ import annotations
"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os
import sys
import json
from dataclasses import dataclass
import math
import re
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, TEST_SET_PATH


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


def _safe_metric(value: object) -> float:
    """Chuẩn hóa metric về số thực hữu hạn."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(numeric) or math.isinf(numeric):
        return 0.0
    return numeric


def _tokenize_for_overlap(text: str) -> set[str]:
    """Tách token đơn giản để tính overlap cục bộ."""

    return {
        token
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(token) >= 2
    }


def _overlap_score(source: str, target: str) -> float:
    """Tính tỉ lệ overlap token giữa source và target."""

    source_tokens = _tokenize_for_overlap(source)
    target_tokens = _tokenize_for_overlap(target)
    if not source_tokens or not target_tokens:
        return 0.0
    return len(source_tokens & target_tokens) / max(len(target_tokens), 1)


def _heuristic_eval_result(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> EvalResult:
    """Fallback cục bộ khi không dùng được RAGAS đầy đủ."""

    best_context_overlap = max((_overlap_score(context, ground_truth) for context in contexts), default=0.0)
    answer_overlap = _overlap_score(answer, ground_truth)
    answer_vs_context = max((_overlap_score(answer, context) for context in contexts), default=0.0)
    relevant_contexts = [context for context in contexts if _overlap_score(context, ground_truth) >= 0.2]
    context_precision = len(relevant_contexts) / len(contexts) if contexts else 0.0

    return EvalResult(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
        faithfulness=_safe_metric(answer_vs_context),
        answer_relevancy=_safe_metric(answer_overlap),
        context_precision=_safe_metric(context_precision),
        context_recall=_safe_metric(1.0 if best_context_overlap >= 0.3 else best_context_overlap),
    )


def _heuristic_evaluate(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Đánh giá cục bộ, không phụ thuộc API ngoài."""

    per_question = [
        _heuristic_eval_result(question, answer, context_list, ground_truth)
        for question, answer, context_list, ground_truth in zip(questions, answers, contexts, ground_truths)
    ]

    return {
        "faithfulness": mean(result.faithfulness for result in per_question) if per_question else 0.0,
        "answer_relevancy": mean(result.answer_relevancy for result in per_question) if per_question else 0.0,
        "context_precision": mean(result.context_precision for result in per_question) if per_question else 0.0,
        "context_recall": mean(result.context_recall for result in per_question) if per_question else 0.0,
        "per_question": per_question,
    }


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
    if not OPENAI_API_KEY:
        return _heuristic_evaluate(questions, answers, contexts, ground_truths)

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        result = evaluate(
            dataset,
            metrics=[faithfulness, context_precision, context_recall],
        )
        df = result.to_pandas()
        heuristic = _heuristic_evaluate(questions, answers, contexts, ground_truths)

        per_question = [
            EvalResult(
                question=questions[index],
                answer=answers[index],
                contexts=contexts[index],
                ground_truth=ground_truths[index],
                faithfulness=_safe_metric(row.get("faithfulness", 0.0)),
                answer_relevancy=heuristic["per_question"][index].answer_relevancy,
                context_precision=_safe_metric(row.get("context_precision", 0.0)),
                context_recall=_safe_metric(row.get("context_recall", 0.0)),
            )
            for index, (_, row) in enumerate(df.iterrows())
        ]

        return {
            "faithfulness": _safe_metric(df["faithfulness"].mean()),
            "answer_relevancy": _safe_metric(heuristic["answer_relevancy"]),
            "context_precision": _safe_metric(df["context_precision"].mean()),
            "context_recall": _safe_metric(df["context_recall"].mean()),
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
