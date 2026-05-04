from __future__ import annotations
"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import sys
import time
from dataclasses import dataclass
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Load FlagReranker (bge-reranker-v2-m3), fallback về CrossEncoder."""
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(self.model_name, use_fp16=True)
                self._model_type = "flag"
            except (ImportError, Exception):
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                self._model_type = "cross"
        return self._model

    def rerank(self, query: str, documents: list[dict],
               top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k bằng cross-encoder."""
        if not documents:
            return []

        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]

        # Compute scores theo model type
        if self._model_type == "flag":
            scores = model.compute_score(pairs, normalize=True)
        else:
            scores = model.predict(pairs).tolist()

        # Đảm bảo scores là list
        if not isinstance(scores, list):
            scores = [scores]

        # Ghép score với doc và sort giảm dần
        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Fallback về CrossEncoderReranker nếu không có flashrank."""
    def __init__(self):
        self._model = None
        self._fallback = CrossEncoderReranker()

    def rerank(self, query: str, documents: list[dict],
               top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank dùng flashrank nếu có, không thì dùng CrossEncoder."""
        try:
            from flashrank import Ranker, RerankRequest
            if self._model is None:
                self._model = Ranker()
            passages = [{"id": i, "text": d["text"]} for i, d in enumerate(documents)]
            req = RerankRequest(query=query, passages=passages)
            results = self._model.rerank(req)
            scored = [(r["score"], documents[r["id"]]) for r in results]
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                RerankResult(
                    text=doc["text"],
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(score),
                    metadata=doc.get("metadata", {}),
                    rank=i,
                )
                for i, (score, doc) in enumerate(scored[:top_k])
            ]
        except ImportError:
            return self._fallback.rerank(query, documents, top_k=top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict],
                        n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs — trả về avg/min/max/p50 (ms)."""
    times: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "p50_ms": median(times),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
