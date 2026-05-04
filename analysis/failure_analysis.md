# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Làm một mình — 2A202600398  
**Thành viên:** Nguyễn Lê Minh Luân (M1 + M2 + M3 + M4 + M5 + Pipeline)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.9867 | 0.9923 | +0.0056 |
| Context Precision | 0.9167 | 0.8667 | -0.0500 |
| Context Recall | 1.0000 | 1.0000 | +0.0000 |

Nguồn số liệu:
- `reports/naive_baseline_report.json`
- `reports/ragas_report.json`

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

### #1 — Mật khẩu cần thay đổi sau bao nhiêu ngày?

- **Expected:** 90 ngày
- **Worst metric:** `context_precision = 0.3333`
- **Diagnosis:** Too many irrelevant chunks
- **Error Tree:** Output có thể đúng nhưng context lẫn nhiều đoạn khác ngoài section mật khẩu.
- **Root cause:** Retrieval trả về thêm chunks từ các vùng IT policy không trực tiếp nói về chu kỳ đổi mật khẩu.
- **Suggested fix:** Thêm metadata filter theo `category="it"` hoặc rerank mạnh hơn cho các query bảo mật.

### #2 — Yêu cầu độ dài mật khẩu là bao nhiêu ký tự?

- **Expected:** Ít nhất 12 ký tự
- **Worst metric:** `context_precision = 0.3333`
- **Diagnosis:** Too many irrelevant chunks
- **Error Tree:** Context có câu trả lời nhưng nhiễu bởi các chunk khác cùng document về VPN và thiết bị.
- **Root cause:** Chunk structure-aware ở file IT vẫn hơi rộng đối với các câu hỏi ngắn về một policy cụ thể.
- **Suggested fix:** Chia nhỏ section IT policy hơn nữa hoặc áp dụng MMR để giảm chunk gần trùng nhau.

### #3 — Thời gian thử việc với vị trí quản lý là bao nhiêu ngày?

- **Expected:** 90 ngày
- **Worst metric:** `context_precision = 0.6667`
- **Diagnosis:** Too many irrelevant chunks
- **Error Tree:** Output sai chỉ xảy ra khi context chứa đồng thời cả “60 ngày nhân viên” và “90 ngày quản lý”.
- **Root cause:** Retrieval đúng tài liệu nhưng chưa đủ sắc nét ở mức đoạn, nên reranker vẫn giữ lại chunk có thông tin của cả hai đối tượng.
- **Suggested fix:** Thêm metadata hoặc query expansion để phân biệt rõ `nhân viên` và `quản lý`.

### #4 — Khi nghỉ ốm cần nộp giấy tờ gì và trong bao lâu?

- **Expected:** Giấy xác nhận y tế trong vòng 3 ngày làm việc
- **Worst metric:** `context_precision = 0.6667`
- **Diagnosis:** Too many irrelevant chunks
- **Error Tree:** Context đúng nhưng có thể kèm thêm các đoạn khác của HR policy như nghỉ phép năm hoặc thai sản.
- **Root cause:** Chunking theo section vẫn để section HR khá dài; câu hỏi cần 1 policy con rất cụ thể.
- **Suggested fix:** Tách nhỏ section theo tiểu mục `## Nghỉ ốm`, `## Nghỉ thai sản`, `## Nghỉ phép năm` sớm hơn ở giai đoạn ingest.

### #5 — Lương trong thời gian thử việc là bao nhiêu phần trăm?

- **Expected:** 85% lương cơ bản
- **Worst metric:** `context_precision = 0.6667`
- **Diagnosis:** Too many irrelevant chunks
- **Error Tree:** Context đúng nhưng đi cùng thông tin “60 ngày/90 ngày” khiến answer dễ lan sang thời lượng thử việc.
- **Root cause:** Query hỏi về tỷ lệ phần trăm nhưng retrieval vẫn ưu tiên đoạn tổng quát về thử việc thay vì chỉ phần lương.
- **Suggested fix:** Query rewrite bổ sung từ khóa `phần trăm`, `lương cơ bản`, hoặc metadata-aware reranking.

---

## Case Study Chi tiết

**Question chọn phân tích:** `Thời gian thử việc với vị trí quản lý là bao nhiêu ngày?`

**Walkthrough theo Error Tree:**
1. Output đúng không?  
   Không ổn định nếu context chứa đồng thời “60 ngày” và “90 ngày”.
2. Context đúng không?  
   Có. Các chunk top đầu đều đến từ IT policy đúng nguồn.
3. Vì sao vẫn fail ở `context_precision`?  
   Vì ngoài câu chứa “90 ngày đối với vị trí quản lý”, retrieval còn giữ thêm phần “60 ngày đối với vị trí nhân viên”.
4. Root cause cuối cùng:  
   Retrieval và rerank chưa tách đủ rõ hai biến thể cùng chủ đề “thử việc”.

**Hướng tối ưu tiếp theo:**
- Tách section thử việc thành chunks nhỏ hơn.
- Thêm rerank features theo entity/role (`nhân viên`, `quản lý`).
- Dùng metadata filter khi query có role-specific term.
