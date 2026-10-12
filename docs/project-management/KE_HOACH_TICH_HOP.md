# Kế hoạch Tích hợp — 8 Quyết định + 4 Giai đoạn


---

## PHẦN A: 8 QUYẾT ĐỊNH KIẾN TRÚC

### QĐ-1: CHUNK_OVERLAP → 50

| | Hoạt (hiện tại) | Hùng (hiện tại) | **Thống nhất** |
|--|-----------------|-----------------|----------------|
| CHUNK_SIZE | 512 | 512 | **512** |
| CHUNK_OVERLAP | 80 | 50 | **50** (đúng spec đề cương) |

**Lý do:** Overlap 50 đúng spec, giảm chunks trùng lắp, tiết kiệm token LLM.
**Ai sửa:** Hoạt — đổi trong `src/config.py`.

---

### QĐ-2: LLM wrapper → Gộp thành 1 file `src/llm.py`

| Tính năng | `llm.py` (Hoạt) | `llm_utils.py` (Hùng) | **Gộp** |
|-----------|-----------------|----------------------|---------|
| Singleton | Co | Khong | **Co** |
| Rate limiting | Khong | Co (thread-safe, 30 req/min) | **Co** |
| `parse_json()` | Khong | Co | **Co** |
| Retry + backoff | Co | Co | **Co** |

**Ai làm:** Hùng gộp, Hoạt review.

---

### QĐ-3: FAISS index → Dùng chung LangChain FAISS

| | Hoạt | Long | **Thống nhất** |
|--|------|------|----------------|
| Library | LangChain FAISS | Raw faiss | **LangChain FAISS** |
| Index type | IndexFlatL2 | IndexFlatIP | **IndexFlatL2** (equivalent with normalized vectors) |
| Save format | `.faiss` + `.pkl` | Pickle | **LangChain format** |

**Lý do:** Controlled experiment — cùng chunks, cùng vectors, chỉ khác retrieval technique.
**Ai làm:** Long viết adapter, Hoạt review.

---

### QĐ-4: Graph backend → Default InMemory

| | Neo4j | InMemoryGraphStore | **Chọn** |
|--|-------|---------------------|---------|
| Cài đặt | Cần Docker | Không cần | **InMemory** (default) |
| Demo | Khó (giảng viên phải cài) | Dễ (chạy ngay) | **InMemory** |
| Scale | Tốt | Nhỏ | Đủ cho demo |

**Lý do:** Giảng viên clone + chạy ngay, không cần cài Neo4j.
**Ai làm:** Hùng — đổi `GRAPH_BACKEND` default trong config.

---

### QĐ-5: Config → Gộp 4 file thành 1

Cấu trúc `src/config.py` thống nhất:

```
Paths → LLM → Embedding → Chunking → Retrieval → Graph → Evaluation
```

**Hai lỗi phải sửa:**
1. Hùng: `dotenv` import thiếu try/except → crash
2. Hùng: `NEO4J_PASSWORD="graphrag123"` hardcoded → chuyển env var

**Ai làm:** Hoạt (config gốc) + Hùng (bổ sung phần Graph).

---

### QĐ-6: Import reranker → Long tạo `src/rerank_rag.py`

Tuấn Anh's `pipelines.py` import `from src.rerank_rag import build_rerank_rag` — module chưa tồn tại.

**Giải pháp:** Long tạo `src/rerank_rag.py` chứa `RerankerAdapter` class, wrap `reranker_module/`, export `build_rerank_rag()` facade.

---

### QĐ-7: RAGAS judge → Truyền Groq LLM

`evaluator.py` gọi RAGAS `evaluate()` mà KHÔNG truyền `llm=` → default OpenAI GPT-4.

**Sửa:** Truyền `llm=get_llm()` và `embeddings=get_embeddings()` cho RAGAS.

**Ai làm:** Tuấn Anh.

---

### QĐ-8: Orphan branch → Tuấn Anh tạo nhánh mới

`feature-tuananh-eval-ui` là orphan branch — không merge bình thường được.

**Giải pháp:** Tạo nhánh mới từ `main`, copy files cần thiết sang.

---

## PHẦN B: ROADMAP 4 GIAI ĐOẠN

### Giai đoạn 1: FOUNDATION (Tuần 1)

> Mục tiêu: Code base thống nhất — 1 config, 1 LLM wrapper, cùng chunking params.

| Task | Người | Phụ thuộc | Chi tiết |
|------|-------|-----------|----------|
| 1.1 Gộp config.py | **Hoạt** | Không | Xem `TASK_HOAT.md` |
| 1.2 Gộp LLM wrapper | **Hùng** | Không | Xem `TASK_HUNG.md` |
| 1.3 CHUNK_OVERLAP=50 | **Hoạt** | 1.1 | Xem `TASK_HOAT.md` |
| 1.4 Tạo nhánh mới | **Tuấn Anh** | Không | Xem `TASK_TUANANH.md` |
| 1.5 Học LangChain FAISS API | **Long** | Không | Xem `TASK_LONG.md` |

**Song song:** 1.1 + 1.2 + 1.4 + 1.5 cùng lúc → 1.3 sau 1.1.

---

### Giai đoạn 2: INTEGRATION (Tuần 2)

> Mục tiêu: Ba pipeline chạy được với cùng data + cùng config + cùng FAISS index.

| Task | Người | Phụ thuộc | Chi tiết |
|------|-------|-----------|----------|
| 2.1 Naive RAG update | **Hoạt** | 1.1, 1.2, 1.3 | Xem `TASK_HOAT.md` |
| 2.2 GraphRAG update | **Hùng** | 1.1, 1.2 | Xem `TASK_HUNG.md` |
| 2.3 `src/rerank_rag.py` | **Long** | 1.1, 1.5 | Xem `TASK_LONG.md` |
| 2.4 `chunk_id_for()` NFC fix | **Hùng** | 1.2 | Xem `TASK_HUNG.md` |
| 2.5 Shared data pipeline | **Hoạt** | 1.1, 1.3 | Xem `TASK_HOAT.md` |

**Song song:** 2.1 + 2.2 + 2.3 cùng lúc → 2.4 + 2.5 sau.

---

### Giai đoạn 3: EVALUATION (Tuần 3)

> Mục tiêu: Chạy RAGAS evaluation công bằng trên cả 3 pipeline.

| Task | Người | Phụ thuộc | Chi tiết |
|------|-------|-----------|----------|
| 3.1 Fix RAGAS judge LLM | **Tuấn Anh** | 1.2 | Xem `TASK_TUANANH.md` |
| 3.2 Import 3 pipeline thật | **Tuấn Anh** | 2.1, 2.2, 2.3 | Xem `TASK_TUANANH.md` |
| 3.3 Bộ câu hỏi eval (≥20) | **Cả nhóm** | Không | Mỗi người 5+ câu |
| 3.4 Chạy eval | **Tuấn Anh** | 3.1, 3.2, 3.3 | Xem `TASK_TUANANH.md` |
| 3.5 Fix mock source bug | **Tuấn Anh** | 2.3 | Xem `TASK_TUANANH.md` |

---

### Giai đoạn 4: DEMO & POLISH (Tuần 4)

> Mục tiêu: UI hoàn chỉnh, báo cáo cuối kỳ.

| Task | Người | Phụ thuộc |
|------|-------|-----------|
| 4.1 Streamlit UI chạy thật | **Tuấn Anh** | 3.2 |
| 4.2 README.md | **Hoạt** | GĐ 2+3 |
| 4.3 Biểu đồ so sánh | **Tuấn Anh** | 3.4 |
| 4.4 Báo cáo cuối kỳ | **Cả nhóm** | GĐ 3+4 |

---

## PHẦN C: BIỂU ĐỒ PHỤ THUỘC

```
Tuần 1 (Foundation)          Tuần 2 (Integration)         Tuần 3 (Eval)        Tuần 4 (Demo)
─────────────────────────    ────────────────────────     ───────────────       ──────────────
1.1 Config (Hoạt) ─────────→ 2.1 Naive (Hoạt) ─────────→                      4.2 README
1.3 Overlap (Hoạt) ←── 1.1   2.5 Shared (Hoạt) ────────→

1.2 LLM (Hùng) ────────────→ 2.2 Graph (Hùng) ─────────→
                              2.4 NFC fix (Hùng) ───────→

1.5 FAISS learn (Long) ────→ 2.3 Rerank adapter (Long) →

1.4 Branch (T.Anh) ────────→                              3.1 RAGAS fix ──→ 4.1 UI
                                                           3.2 Import ────→ 4.3 Charts
                              ALL 2.x ──────────────────→ 3.3 Questions
                                                           3.4 Run eval ──→ 4.4 Report
                                                           3.5 Mock fix
```

---

## PHẦN D: CHECKLIST THEO DÕI TIẾN ĐỘ

### Tuần 1 — Foundation
- [ ] 1.1 Config gộp (Hoạt)
- [ ] 1.2 LLM wrapper gộp (Hùng)
- [ ] 1.3 CHUNK_OVERLAP = 50 (Hoạt)
- [ ] 1.4 Nhánh mới từ main (Tuấn Anh)
- [ ] 1.5 Hiểu LangChain FAISS API (Long)

### Tuần 2 — Integration
- [ ] 2.1 Naive RAG dùng config/llm mới (Hoạt)
- [ ] 2.2 GraphRAG dùng config/llm mới (Hùng)
- [ ] 2.3 `src/rerank_rag.py` adapter (Long)
- [ ] 2.4 `chunk_id_for()` NFC-safe (Hùng)
- [ ] 2.5 Shared data pipeline (Hoạt)

### Tuần 3 — Evaluation
- [ ] 3.1 RAGAS dùng Groq LLM (Tuấn Anh)
- [ ] 3.2 `pipelines.py` import thật (Tuấn Anh)
- [ ] 3.3 Bộ câu hỏi eval ≥20 câu (Cả nhóm)
- [ ] 3.4 Chạy eval + kết quả JSON (Tuấn Anh)
- [ ] 3.5 Fix mock source bug (Tuấn Anh)

### Tuần 4 — Demo & Polish
- [ ] 4.1 Streamlit UI chạy thật (Tuấn Anh)
- [ ] 4.2 README.md (Hoạt)
- [ ] 4.3 Biểu đồ so sánh (Tuấn Anh)
- [ ] 4.4 Báo cáo cuối kỳ (Cả nhóm)
