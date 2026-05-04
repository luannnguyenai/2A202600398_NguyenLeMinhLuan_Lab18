from __future__ import annotations
"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os
import sys
import json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _get_openai_client():
    """Lấy OpenAI client nếu API key hợp lệ."""
    if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."):
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    return None


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """Tạo summary ngắn cho chunk (2-3 câu tiếng Việt).

    Dùng gpt-4o-mini nếu có API key, fallback extractive (2 câu đầu).

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    # Extractive fallback: lấy 2 câu đầu
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    summary = ". ".join(sentences[:2])
    return summary + "." if summary and not summary.endswith(".") else summary


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate câu hỏi mà chunk có thể trả lời (HyQA).

    Bridge vocabulary gap: index cả questions lẫn chunk.

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                            "Mỗi câu hỏi trên 1 dòng. Chỉ trả về câu hỏi, không đánh số."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0.7,
            )
            raw = resp.choices[0].message.content.strip().split("\n")
            return [q.strip().lstrip("0123456789.-) ") for q in raw if q.strip()]
        except Exception:
            pass

    # Fallback: không có API → trả về list rỗng
    return []


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend context giải thích chunk nằm ở đâu trong document.

    Anthropic benchmark: giảm 49% retrieval failure.

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended (original text phải có mặt).
    """
    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Viết 1 câu ngắn (≤25 từ) mô tả đoạn văn này nằm ở đâu trong tài liệu "
                            "và nói về chủ đề gì. Chỉ trả về 1 câu, bằng tiếng Việt."
                        ),
                    },
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
                temperature=0,
            )
            context_sentence = resp.choices[0].message.content.strip()
            return f"{context_sentence}\n\n{text}"
        except Exception:
            pass

    # Fallback: tạo context đơn giản từ document title
    if document_title:
        context_sentence = f"Trích từ tài liệu: {document_title}."
        return f"{context_sentence}\n\n{text}"
    return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """LLM extract metadata tự động: topic, entities, category, language.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields (fallback: {language: "vi"}).
    """
    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            'Trích xuất metadata từ đoạn văn. Trả về JSON hợp lệ: '
                            '{"topic": "...", "entities": ["..."], '
                            '"category": "policy|hr|it|finance", "language": "vi|en"}'
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            # Xử lý JSON trong markdown code block nếu có
            if "```" in raw:
                raw = raw.split("```")[1].strip()
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            return json.loads(raw)
        except Exception:
            pass

    # Fallback metadata cơ bản
    return {"language": "vi"}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects (original_text luôn được giữ nguyên).
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched: list[EnrichedChunk] = []

    for chunk in chunks:
        original_text = chunk["text"]
        meta = chunk.get("metadata", {})
        doc_title = meta.get("source", "")

        use_summary = "summary" in methods or "full" in methods
        use_hyqa = "hyqa" in methods or "full" in methods
        use_contextual = "contextual" in methods or "full" in methods
        use_metadata = "metadata" in methods or "full" in methods

        # 1. Summary
        summary = summarize_chunk(original_text) if use_summary else ""

        # 2. HyQA — generate hypothesis questions
        questions = generate_hypothesis_questions(original_text) if use_hyqa else []

        # 3. Contextual prepend
        enriched_text = contextual_prepend(original_text, doc_title) if use_contextual else original_text

        # 4. Auto metadata extraction
        auto_meta = extract_metadata(original_text) if use_metadata else {}

        enriched.append(EnrichedChunk(
            original_text=original_text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**meta, **auto_meta},
            method="+".join(methods),
        ))

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
