from __future__ import annotations
"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4+M5."""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY

# Latency tracking globals
_latency_log: list[dict] = []


def _get_llm_client():
    """Lấy OpenAI client nếu API key hợp lệ."""
    if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."):
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    return None


def generate_answer(query: str, contexts: list[str]) -> str:
    """Gọi LLM (gpt-4o-mini) để trả lời dựa trên context.

    Prompt tiếng Việt, strict: chỉ dùng context, không hallucinate.
    Fallback: trả về context đầu tiên nếu không có API key.
    """
    client = _get_llm_client()
    if not client:
        return contexts[0] if contexts else "Không tìm thấy thông tin."

    context_str = "\n\n".join(contexts)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý nội bộ. CHỈ trả lời dựa trên CONTEXT bên dưới. "
                        "Nếu CONTEXT không đủ → trả lời chính xác: 'Không tìm thấy thông tin.' "
                        "Trả lời ngắn gọn, đúng trọng tâm."
                    ),
                },
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context_str}\n\nCâu hỏi: {query}",
                },
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️  LLM error: {e}. Using context fallback.")
        return contexts[0] if contexts else "Không tìm thấy thông tin."


def build_pipeline():
    """Build production RAG pipeline (M1 → M5 enrichment → M2 index → M3 reranker)."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Load & Chunk (M1)
    t0 = time.perf_counter()
    print("\n[1/4] Chunking documents (hierarchical)...")
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": child.parent_id},
            })
    t_chunk = (time.perf_counter() - t0) * 1000
    print(f"  {len(all_chunks)} chunks from {len(docs)} documents ({t_chunk:.0f}ms)")

    # Step 2: Enrichment (M5) — Bonus +3
    print("\n[2/4] Enriching chunks (M5 — contextual prepend)...")
    enriched = enrich_chunks(all_chunks, methods=["contextual", "metadata"])
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  Enriched {len(enriched)} chunks")
    else:
        print("  ⚠️  M5 fallback — using raw chunks")

    # Step 3: Index (M2)
    t1 = time.perf_counter()
    print("\n[3/4] Indexing (BM25 + Dense bge-m3)...")
    search = HybridSearch()
    search.index(all_chunks)
    t_index = (time.perf_counter() - t1) * 1000
    print(f"  Indexing done ({t_index:.0f}ms)")

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker (bge-reranker-v2-m3)...")
    reranker = CrossEncoderReranker()

    return search, reranker


def run_query(query: str, search: HybridSearch,
              reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query — trả về (answer, contexts) với latency tracking."""
    latency: dict[str, float] = {}

    # Search
    t0 = time.perf_counter()
    results = search.search(query)
    latency["search_ms"] = (time.perf_counter() - t0) * 1000

    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]

    # Rerank
    t1 = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    latency["rerank_ms"] = (time.perf_counter() - t1) * 1000

    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    # Generate
    t2 = time.perf_counter()
    answer = generate_answer(query, contexts)
    latency["generate_ms"] = (time.perf_counter() - t2) * 1000

    latency["query"] = query
    _latency_log.append(latency)

    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation trên test set + RAGAS + latency report."""
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:55]}...")

    print("\n[Eval] Running RAGAS...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    # Latency breakdown — Bonus +2
    _save_latency_report()

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


def _save_latency_report() -> None:
    """Tính và lưu latency breakdown report (Bonus +2)."""
    if not _latency_log:
        return

    os.makedirs("reports", exist_ok=True)

    def _avg(key: str) -> float:
        vals = [e[key] for e in _latency_log if key in e]
        return sum(vals) / len(vals) if vals else 0.0

    report = {
        "num_queries": len(_latency_log),
        "avg_search_ms": _avg("search_ms"),
        "avg_rerank_ms": _avg("rerank_ms"),
        "avg_generate_ms": _avg("generate_ms"),
        "avg_total_ms": _avg("search_ms") + _avg("rerank_ms") + _avg("generate_ms"),
        "per_query": _latency_log,
    }

    with open("reports/latency_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n⏱️  Latency Breakdown (avg per query):")
    print(f"  Search  : {report['avg_search_ms']:>8.1f} ms")
    print(f"  Rerank  : {report['avg_rerank_ms']:>8.1f} ms")
    print(f"  Generate: {report['avg_generate_ms']:>8.1f} ms")
    print(f"  Total   : {report['avg_total_ms']:>8.1f} ms")
    print("  → Saved reports/latency_report.json")


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
