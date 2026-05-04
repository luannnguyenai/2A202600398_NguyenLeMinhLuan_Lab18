from __future__ import annotations
"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text thành từ (word tokenization).

    Dùng underthesea — nhận biết cụm từ như 'nghỉ phép', 'thâm niên'.
    Fallback về text gốc nếu không import được.
    """
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except Exception:
        return text  # fallback an toàn


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index từ chunks đã segment."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        # Segment từng chunk và split thành tokens
        self.corpus_tokens = [
            segment_vietnamese(c["text"]).split()
            for c in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search bằng BM25 — trả về top_k kết quả."""
        if self.bm25 is None or not self.documents:
            return []

        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in top_indices
            if scores[i] > 0  # bỏ qua docs có score = 0
        ] or [
            # Nếu tất cả score = 0, vẫn trả về top_k
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in top_indices
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks vào Qdrant bằng bge-m3 embeddings."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        # Tạo lại collection
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        texts = [c["text"] for c in chunks]
        encoder = self._get_encoder()
        vectors = encoder.encode(texts, show_progress_bar=True, batch_size=32)

        points = [
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={**chunks[i].get("metadata", {}), "text": chunks[i]["text"]},
            )
            for i in range(len(chunks))
        ]

        # Upsert theo batch
        batch_size = 100
        for start in range(0, len(points), batch_size):
            self.client.upsert(collection_name=collection, points=points[start:start + batch_size])

    def search(self, query: str, top_k: int = DENSE_TOP_K,
               collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search bằng dense vector similarity."""
        encoder = self._get_encoder()
        query_vector = encoder.encode(query).tolist()

        try:
            hits = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k,
            )
        except Exception:
            # Fallback: dùng query_points nếu search deprecated
            result = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
            )
            hits = result.points

        return [
            SearchResult(
                text=hit.payload.get("text", ""),
                score=float(hit.score),
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
                method="dense",
            )
            for hit in hits
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists bằng RRF: score(d) = Σ 1/(k + rank + 1)."""
    rrf_scores: dict[str, dict] = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = result.text
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "result": result}
            rrf_scores[key]["score"] += 1.0 / (k + rank + 1)

    # Sắp xếp theo RRF score giảm dần
    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

    return [
        SearchResult(
            text=item["result"].text,
            score=item["score"],
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in sorted_items[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
