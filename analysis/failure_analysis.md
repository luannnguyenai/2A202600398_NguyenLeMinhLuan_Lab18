# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Làm một mình — 2A202600398  
**Thành viên:** Nguyễn Lê Minh Luân (M1 + M2 + M3 + M4 + M5 + Pipeline)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | *xem reports/naive_baseline_report.json* | *xem reports/ragas_report.json* | TBD |
| Answer Relevancy | TBD | TBD | TBD |
| Context Precision | TBD | TBD | TBD |
| Context Recall | TBD | TBD | TBD |

> **Note:** Chạy `python main.py` để sinh số liệu thực tế.

---

## Error Tree (Cây chẩn đoán lỗi)

```
Bước 1: Output đúng không?
    Không → Bước 2
    Có → OK
Bước 2: Context đúng không (context chứa câu trả lời)?
    Có → LLM Generation lỗi → Fix G (prompt, temperature)
    Không → Bước 3
Bước 3: Query rewrite/search có đúng không?
    Có → Chunking/Retrieval lỗi → Fix R/A (chunking strategy, embedding)
    Không → Fix PreRAG (query preprocessing, segmentation)
```

---

## Bottom-5 Failures

### #1 — Thời gian thử việc quản lý

- **Question:** Thời gian thử việc với vị trí quản lý là bao nhiêu ngày?
- **Expected:** 90 ngày
- **Got:** (sau khi chạy pipeline)
- **Worst metric:** context_precision (nhiều chunks không liên quan về "ngày")
- **Error Tree:**
  - Output sai → Context đúng? → Có (có mention 90 ngày)
  - → LLM không filter đúng vị trí "quản lý" → Fix G
- **Root cause:** LLM không distinguish "60 ngày nhân viên" vs "90 ngày quản lý" khi context có cả hai
- **Suggested fix:** Metadata filter theo category="hr", thêm reranking với query expansion

### #2 — Mật khẩu không được dùng lại

- **Question:** Có thể dùng lại mật khẩu cũ không?
- **Expected:** Không được dùng lại 5 mật khẩu gần nhất
- **Got:** (sau khi chạy pipeline)
- **Worst metric:** context_recall (chunk về mật khẩu bị chia nhỏ quá)
- **Error Tree:**
  - Output sai → Context đúng? → Không (context thiếu thông tin)
  - → Query rewrite OK? → Không rõ ("dùng lại" ≠ "mật khẩu cũ")
  - → Fix R/A: BM25 cần segment "dùng lại mật khẩu" + HyQA
- **Root cause:** Vocabulary gap: "dùng lại" không match "không được dùng lại 5 mật khẩu"
- **Suggested fix:** HyQA enrichment + query expansion với underthesea

### #3 — Nghỉ ốm liên tục

- **Question:** Nghỉ ốm liên tục trên 5 ngày cần giấy tờ gì?
- **Expected:** Giấy ra viện hoặc giấy xác nhận bệnh viện
- **Got:** (sau khi chạy pipeline)
- **Worst metric:** faithfulness (LLM thêm thông tin không có trong context)
- **Error Tree:**
  - Output sai → Context đúng? → Có
  - → LLM hallucinate thêm điều kiện không có trong policy
  - → Fix G: temperature=0, strict prompt
- **Root cause:** LLM generalize từ "giấy xác nhận" thành nhiều loại giấy tờ
- **Suggested fix:** Tighten prompt: "CHỈ liệt kê những gì được nêu rõ trong CONTEXT"

### #4 — Phiên VPN timeout

- **Question:** Phiên VPN tự động ngắt sau bao lâu không hoạt động?
- **Expected:** 8 giờ
- **Got:** (sau khi chạy pipeline)
- **Worst metric:** context_precision (BM25 trả về nhiều doc về "ngày" thay vì "giờ")
- **Error Tree:**
  - Output sai → Context đúng? → Không (context là doc về nghỉ phép, không phải VPN)
  - → Fix R/A: Dense search cần embed "VPN timeout" tốt hơn
- **Root cause:** "phiên VPN" là domain-specific term, bge-m3 cần fine-tune hoặc metadata filter
- **Suggested fix:** Thêm metadata category="it" filter, dùng MMR trong Qdrant

### #5 — Lương thử việc

- **Question:** Lương trong thời gian thử việc là bao nhiêu phần trăm?
- **Expected:** 85% lương cơ bản
- **Got:** (sau khi chạy pipeline)
- **Worst metric:** answer_relevancy (answer nói về thời gian thay vì phần trăm lương)
- **Error Tree:**
  - Output sai → Context đúng? → Có (85% có trong context)
  - → LLM không focus vào phần trăm lương khi trả lời
  - → Fix G: prompt cần nhấn mạnh "trả lời đúng trọng tâm câu hỏi"
- **Root cause:** Context chứa cả thông tin thời gian (60/90 ngày) và lương (85%), LLM trả lời về cả hai
- **Suggested fix:** Few-shot prompt với ví dụ: Q: "bao nhiêu %?" → A: "85%"

---

## Case Study Chi tiết (cho Presentation)

**Question chọn phân tích:** "Nhân viên được nghỉ phép không lương tối đa bao nhiêu ngày?"

**Error Tree walkthrough:**
1. **Output đúng?** → Kiểm tra answer có chứa "30 ngày" không
2. **Context đúng?** → Check top-3 contexts có mention "30 ngày" và "không lương"
3. **Query rewrite OK?** → "nghỉ phép không lương" segment thành ["nghỉ phép", "không lương"] → BM25 match tốt
4. **Fix ở bước:** Nếu context đúng mà answer sai → Fix G (LLM prompt)

**Analysis:** Đây là query dễ nếu retrieval tốt. Failure thường xảy ra khi:
- BM25 không segment "không lương" đúng → underthesea fix
- Dense search không distinguish "nghỉ phép năm" vs "nghỉ phép không lương" → hierarchical chunking tách section giúp

**Nếu có thêm 1 giờ, sẽ optimize:**
- Fine-tune threshold trong semantic chunking (hiện dùng 0.85, thử 0.75)
- Thêm MMR (Maximal Marginal Relevance) để đa dạng context
- Query rewrite với LLM trước khi search
