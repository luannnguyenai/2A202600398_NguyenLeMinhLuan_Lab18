# Group Report — Lab 18: Production RAG System

**Nhóm:** Solo — 2A202600398  
**Ngày:** 2026-05-04  
**Sinh viên:** Nguyễn Lê Minh Luân

---

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------| 
| Nguyễn Lê Minh Luân | M1: Advanced Chunking | ✅ | 9/9 |
| Nguyễn Lê Minh Luân | M2: Hybrid Search | ✅ | 5/5 |
| Nguyễn Lê Minh Luân | M3: Reranking | ✅ | 5/5 |
| Nguyễn Lê Minh Luân | M4: RAGAS Evaluation | ✅ | 4/4 |
| Nguyễn Lê Minh Luân | M5: Enrichment (Bonus) | ✅ | 7/7 |
| Nguyễn Lê Minh Luân | Pipeline Integration | ✅ | — |

---

## Kiến trúc Pipeline

```
Documents (MD) 
    ↓ M1: Hierarchical Chunking (parent 2048 / child 256)
    ↓ M5: Contextual Prepend + Auto Metadata (Bonus +3)
    ↓ M2: BM25 (underthesea) + Dense (bge-m3) → RRF Fusion
    ↓ M3: Cross-Encoder Reranking (bge-reranker-v2-m3) top-20→3
    ↓ LLM: gpt-4o-mini (strict Vietnamese prompt, t=0)
    ↓ M4: RAGAS Evaluation (4 metrics)
```

---

## Kết quả RAGAS

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|-----------|---|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.9867 | 0.9923 | +0.0056 |
| Context Precision | 0.9167 | 0.8667 | -0.0500 |
| Context Recall | 1.0000 | 1.0000 | +0.0000 |

---

## Key Findings

### 1. Biggest Win — Hybrid Search + RRF Fusion
**Module:** M2 (Hybrid Search)  
**Tại sao:** Kết hợp BM25 (exact keyword match cho tiếng Việt, ví dụ "nghỉ phép") với Dense (semantic match cho paraphrase như "ngày nghỉ năm") giải quyết cả hai failure modes. RRF fusion cân bằng hai rankings mà không cần tune weights.

**Ví dụ cụ thể:** Query "nghỉ phép không lương bao nhiêu ngày?" — BM25 match "nghỉ phép không lương" chính xác, Dense match "leave without pay maximum duration". RRF kết hợp → top-1 luôn đúng.

### 2. Challenge — Vietnamese Tokenization
**Vấn đề:** "nghỉ phép" = 2 token BM25 riêng biệt nếu không segment → score thấp. Underthesea segment thành "nghỉ_phép" (1 token) → BM25 match chính xác hơn.

**Giải pháp:** `underthesea.word_tokenize(text, format="text")` trước khi index BM25.

### 3. Surprise — Cross-encoder Reranking Latency
**Phát hiện:** bge-reranker-v2-m3 mất nhiều thời gian nhất trong pipeline. Query đầu tiên cao do cold start, trung bình toàn bộ rerank khoảng 2257.8ms/query. Latency này vẫn chấp nhận được cho lab nhưng là bottleneck rõ nhất nếu triển khai production nội bộ.

---

## Latency Breakdown (avg per query)

| Bước | Thời gian (avg) |
|------|----------------|
| Search (BM25 + Dense + RRF) | 308.5ms |
| Rerank (bge-reranker-v2-m3) | 2257.8ms |
| Generate | 0.0ms |
| **Total** | **2566.4ms** |

> Chi tiết: `reports/latency_report.json`

---

## Presentation Notes (5 phút)

### 1. RAGAS Scores (1 phút)
- Bảng so sánh: Naive vs Production (4 metrics)
- Highlight: `answer_relevancy` tăng nhẹ (+0.0056), `context_precision` giảm do corpus mở rộng và retrieval còn nhiễu ở nhóm câu hỏi IT/HR cụ thể

### 2. Biggest Win (1 phút)
- **Module M2: Hybrid Search** — RRF fusion giải quyết cả exact match lẫn semantic match
- Chứng minh: "nghỉ phép" query → BM25 rank cao → hybrid top-1 luôn đúng
- Với corpus OCR mới, top-1 truy vấn pháp lý và tài chính đều rơi đúng vào file nguồn mới (`nghi_dinh_13_2023.md`, `bctc.md`)

### 3. Case Study (2 phút)
- Question: "Thời gian thử việc với vị trí quản lý là bao nhiêu ngày?"
- Error Tree: Context đúng tài liệu nhưng chứa cả “60 ngày nhân viên” và “90 ngày quản lý” → fail ở `context_precision`
- Fix: metadata-aware reranking hoặc chunk nhỏ hơn cho section thử việc

### 4. Next Step (1 phút)
- **Query rewrite** với LLM trước khi search → +5-10% recall
- **MMR (Maximal Marginal Relevance)** thay thế top-k simple → đa dạng context
- **Fine-tune semantic threshold** từ 0.85 → 0.75 cho Vietnamese text

---

## Nhận xét kỹ thuật

### Mở rộng corpus bằng OCR PDF
Pipeline ingest mới đã mở rộng corpus từ hai file markdown mẫu sang thêm hai PDF tiếng Việt thực tế: một văn bản pháp lý dài và một tài liệu tài chính nhiều số liệu. Việc OCR + chuẩn hóa markdown giúp tăng độ phủ ngữ nghĩa cho các truy vấn ngoài miền HR/IT ban đầu, đặc biệt là các câu hỏi về quyền của chủ thể dữ liệu và số liệu doanh thu/thuế. Về mặt RAGAS, thay đổi này kỳ vọng cải thiện rõ nhất ở `context_recall` vì hệ thống có thể truy xuất được các đoạn trước đây hoàn toàn không có trong corpus, đồng thời giữ được `context_precision` ở mức chấp nhận được nhờ chunking theo cấu trúc và index dense bằng `BAAI/bge-m3`.

### Chunking Strategy Comparison
| Strategy | Pros | Cons | Best For |
|----------|------|------|---------|
| Basic (paragraph) | Đơn giản, nhanh | Cắt giữa ý | Proof-of-concept |
| Semantic | Giữ nguyên ý | Cần model embedding | General docs |
| **Hierarchical** ★ | Precision (child) + Context (parent) | Phức tạp hơn | **Production RAG** |
| Structure-aware | Giữ nguyên section | Cần markdown format | Policy docs, manuals |

### Tại sao Hierarchical là default production choice?
- Index **children** (256 chars) → embedding chính xác, retrieve đúng passage
- Return **parent** (2048 chars) → LLM đủ context, giảm hallucination
- Parent-child link → có thể expand context khi cần
