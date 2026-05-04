from __future__ import annotations
"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os
import sys
import glob
import re
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: Optional[str] = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: Optional[dict] = None) -> list:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}

    # 1. Tách câu theo dấu kết thúc câu và xuống dòng đôi
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "chunk_index": 0, "strategy": "semantic"})]

    # 2. Encode các câu bằng model nhẹ
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    # 3. Hàm tính cosine similarity
    def cosine_sim(a: "np.ndarray", b: "np.ndarray") -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # 4. Nhóm câu theo similarity, tách khi similarity < threshold
    chunks: list[Chunk] = []
    current_group: list[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            # Đóng group hiện tại
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
            ))
            current_group = []
        current_group.append(sentences[i])

    # Flush group cuối
    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group),
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
        ))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: Optional[dict] = None) -> tuple:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk (sliding window, 32-char overlap).
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    parents: list[Chunk] = []
    children: list[Chunk] = []

    # 1. Gom paragraph thành parents ≤ parent_size
    current_text = ""
    for para in paragraphs:
        if len(current_text) + len(para) > parent_size and current_text:
            pid = f"parent_{len(parents)}"
            parents.append(Chunk(
                text=current_text.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
                parent_id=pid,
            ))
            current_text = ""
        current_text += para + "\n\n"

    # Flush parent cuối
    if current_text.strip():
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=current_text.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
            parent_id=pid,
        ))

    # 2. Tách mỗi parent thành children với sliding window (32-char overlap)
    overlap = 32
    for parent in parents:
        pid = parent.metadata["parent_id"]
        p_text = parent.text
        start = 0
        while start < len(p_text):
            end = start + child_size
            child_text = p_text[start:end].strip()
            if child_text:
                children.append(Chunk(
                    text=child_text,
                    metadata={**metadata, "chunk_type": "child"},
                    parent_id=pid,
                ))
            if end >= len(p_text):
                break
            start = end - overlap  # 32-char overlap

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: Optional[dict] = None) -> list:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}

    # 1. Split by markdown headers (giữ lại headers trong kết quả)
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    # 2. Ghép header với content của nó
    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            # Flush section cũ nếu có content
            if current_content.strip():
                chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**metadata, "section": current_header.strip(), "strategy": "structure"},
                ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    # 3. Flush section cuối
    if current_content.strip():
        chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
        chunks.append(Chunk(
            text=chunk_text,
            metadata={**metadata, "section": current_header.strip(), "strategy": "structure"},
        ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    def _stats(chunks_list: list[Chunk]) -> dict:
        lengths = [len(c.text) for c in chunks_list]
        if not lengths:
            return {"num_chunks": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "num_chunks": len(lengths),
            "avg_len": int(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_basic: list[Chunk] = []
    all_semantic: list[Chunk] = []
    all_hierarchical: list[Chunk] = []  # dùng children để so sánh
    all_structure: list[Chunk] = []

    for doc in documents:
        text = doc["text"]
        meta = doc.get("metadata", {})
        all_basic.extend(chunk_basic(text, metadata=meta))
        all_semantic.extend(chunk_semantic(text, metadata=meta))
        _, children = chunk_hierarchical(text, metadata=meta)
        all_hierarchical.extend(children)
        all_structure.extend(chunk_structure_aware(text, metadata=meta))

    results = {
        "basic": _stats(all_basic),
        "semantic": _stats(all_semantic),
        "hierarchical": _stats(all_hierarchical),
        "structure": _stats(all_structure),
    }

    # In bảng so sánh
    print(f"\n{'Strategy':<16} | {'Chunks':>6} | {'Avg Len':>7} | {'Min':>5} | {'Max':>6}")
    print("-" * 50)
    for name, s in results.items():
        print(f"{name:<16} | {s['num_chunks']:>6} | {s['avg_len']:>7} | {s['min_len']:>5} | {s['max_len']:>6}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
