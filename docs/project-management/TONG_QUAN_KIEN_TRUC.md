# Tổng quan Kiến trúc Hệ thống

## 1. Bản đồ kiến trúc

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (app.py)                        │
│                   Tuấn Anh — feature/tuananh-eval-ui                │
│         ┌──────────┐    ┌──────────────┐    ┌──────────┐           │
│         │ Chat Demo│    │ RAGAS Results │    │ Charts   │           │
│         └────┬─────┘    └──────┬───────┘    └────┬─────┘           │
└──────────────┼─────────────────┼─────────────────┼─────────────────┘
               │                 │                 │
        ┌──────▼──────┐  ┌──────▼──────┐   ┌──────▼──────┐
        │ pipelines.py │  │evaluator.py │   │  charts.py  │
        │ (adapter)    │  │ (RAGAS)     │   │ (matplotlib)│
        └──┬───┬───┬───┘  └─────────────┘   └─────────────┘
           │   │   │
     ┌─────┘   │   └─────┐
     ▼         ▼         ▼
┌─────────┐┌─────────┐┌─────────┐
│Naive RAG││Rerank   ││GraphRAG │
│ (Hoạt)  ││ (Long)  ││ (Hùng)  │
└────┬────┘└────┬────┘└────┬────┘
     │         │         │
     ▼         ▼         ▼
┌───────────────────────────────┐
│     SHARED FOUNDATION         │
│  config.py | llm.py           │
│  chunker.py | embedder.py     │
│  vectorstore.py (FAISS)       │
│  pdf_loader.py                │
└───────────────────────────────┘
```

## 2. Data Flow — Từ câu hỏi đến câu trả lời

```
User Question
     │
     ▼
┌─────────────┐
│ Pipeline     │──→ Phân loại method (naive/rerank/graph)
│ Adapter      │
└──────┬──────┘
       │
       ├──── Naive RAG ─────→ FAISS search (top-50) → top-5 → LLM → Answer
       │
       ├──── Rerank RAG ────→ FAISS search (top-50) → Cross-encoder rerank → top-5 → LLM → Answer
       │
       └──── GraphRAG ──────→ ┌─ LOCAL:  Entity linking → Graph traversal → top chunks ─┐
                              │                                                          │
                              └─ GLOBAL: Community ranking → Map-reduce summaries ───────┘
                                         │
                                         ▼
                                  merge_chunks() (interleave + dedup by chunk_id)
                                         │
                                         ▼
                                      LLM → Answer
```

## 3. Document Processing Flow — Từ PDF đến index

```
PDF files (data/pdfs/)
     │
     ▼
┌─────────────────┐
│ pdf_loader.py   │  load_pdfs() — PyMuPDF
│ clean_viet_text │  NFC normalize + regex cleaning
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ chunker.py      │  char-based splitting
│ CHUNK_SIZE=512  │  separators: \n\n, \n, ". ", "? ", "! ", " "
│ OVERLAP=50      │  chunk_id = SHA1("source|page|NFC(text)")[:16]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ embedder.py     │  BGE-M3 (BAAI/bge-m3)
│ normalize=True  │  LangChain HuggingFaceEmbeddings
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────────┐
│ FAISS  │ │ Knowledge Graph  │ (chỉ GraphRAG)
│ Index  │ │ Entity extraction│
│(shared)│ │ Leiden community │
└────────┘ │ LLM summaries   │
           └──────────────────┘
```

## 4. So sánh 3 Pipeline

| Đặc điểm | Naive RAG | RAG + Re-ranking | GraphRAG |
|-----------|-----------|-------------------|----------|
| **Retrieval** | Vector search (FAISS) | Vector search → Cross-encoder rerank | Entity linking + Graph traversal + Vector search |
| **Ranking** | Cosine similarity | Cross-encoder score (BGE-reranker-v2-m3) | Graph relevance + vector similarity |
| **Ưu điểm** | Đơn giản, nhanh | Chính xác hơn naive | Hiểu quan hệ giữa entities |
| **Nhược điểm** | Không hiểu ngữ nghĩa sâu | Chậm hơn (2-stage) | Phức tạp, cần build graph trước |
| **Khi nào tốt** | Câu hỏi đơn giản, factoid | Câu hỏi cần chính xác | Câu hỏi multi-hop, tổng quan |

## 5. Shared Components — Gì dùng chung, gì riêng

| Component | Shared? | Ghi chú |
|-----------|---------|---------|
| `config.py` | **Shared** | 1 file config cho cả project |
| `llm.py` | **Shared** | Singleton + rate limit + parse_json |
| `pdf_loader.py` | **Shared** | Load + clean PDF |
| `chunker.py` | **Shared** | Char-based 512/50, chunk_id SHA1 |
| `embedder.py` | **Shared** | BGE-M3 embeddings |
| `vectorstore.py` | **Shared** | LangChain FAISS, dùng cho cả 3 |
| `naive_rag.py` | Riêng Hoạt | NaiveRAG class |
| `rerank_rag.py` | Riêng Long | RerankerAdapter + cross-encoder |
| `graph_rag.py` | Riêng Hùng | GraphRAG + LOCAL/GLOBAL search |
| `graph_builder.py` | Riêng Hùng | Entity extraction + community detection |
| `pipelines.py` | Riêng Tuấn Anh | Adapter gọi 3 pipeline |
| `evaluator.py` | Riêng Tuấn Anh | RAGAS scoring |
| `app.py` | Riêng Tuấn Anh | Streamlit UI |

## 6. Chunk ID — Tại sao quan trọng

Công thức: `SHA1("source|page|NFC(text)")[:16]`

Dùng **giống nhau** ở `chunker.py` (Hoạt) và `graph_builder.py` (Hùng).

Mục đích: khi GraphRAG gọi `merge_chunks()` để xen kẽ kết quả từ vector search và graph traversal, chunk_id cho phép **loại bỏ trùng lặp** — cùng một đoạn văn bản từ hai nguồn retrieval khác nhau chỉ xuất hiện 1 lần trong context gửi cho LLM.

## 7. GraphRAG: LOCAL vs GLOBAL Search

| | LOCAL Search | GLOBAL Search |
|--|-------------|---------------|
| **Khi nào** | Câu hỏi cụ thể ("X là gì?") | Câu hỏi tổng quan ("xu hướng...") |
| **Trigger** | Mặc định | Khi câu hỏi chứa GLOBAL_HINTS |
| **Cách làm** | Entity linking → traverse graph → lấy chunks liên quan | Ranking communities → map-reduce summaries |
| **GLOBAL_HINTS** | | "xu hướng", "tổng quan", "chủ đề chính", "nhìn chung", "toàn bộ tài liệu", "tổng kết", "hướng nghiên cứu chính", ... |
