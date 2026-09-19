# Naive RAG — nền tảng truy hồi (Nhóm 7 · phần của Hoạt, SV1)

Module **Pipeline 1 (Naive RAG)** cho đề tài *So sánh Naive RAG — RAG+Re-ranking — GraphRAG cho
hỏi-đáp tài liệu học thuật tiếng Việt*. Đây là **đường cơ sở** (baseline) để so sánh, đồng thời
là **nền tảng dùng chung** cung cấp chunking + embedding + FAISS index cho hai pipeline kia.

```
Câu hỏi → embed (bge-m3) → cosine search top-k (FAISS) → LLM (Groq) sinh câu trả lời + trích dẫn
```

## Chạy nhanh

```bash
pip install -r requirements.txt
cp .env.example .env            # điền GROQ_API_KEY (miễn phí: https://console.groq.com/)

python -m src.naive_rag build --sample                 # build index từ corpus mẫu (chạy ngay, không cần PDF)
python -m src.naive_rag ask "BERT được huấn luyện bằng phương pháp gì?"
python -m src.naive_rag ask "..." --retrieval-only     # xem chunk lấy được, không tốn quota LLM
python -m pytest                                       # 21 test, chạy offline
```

Kèm **corpus mẫu tiếng Việt** (`data/sample_corpus/`) nên chạy được ngay. Dùng tài liệu thật:
bỏ PDF vào `data/pdfs/` rồi `python -m src.naive_rag build`.

## Cấu trúc

```
src/
  config.py        # cấu hình chung (đọc .env)
  pdf_loader.py    # đọc PDF + làm sạch tiếng Việt (+ corpus mẫu fallback)
  chunker.py       # chia đoạn + chunk_id ổn định (khớp GraphRAG) + chunks.jsonl
  embedder.py      # bge-m3 (LangChain) + FakeEmbeddings (test)
  vectorstore.py   # FAISS (dùng chung) + NumpyVectorStore (dự phòng)
  naive_rag.py     # pipeline + giao diện cho thành viên khác + CLI
  llm.py           # gọi Groq (retry)
data/
  sample_corpus/   # 5 file .txt tiếng Việt (Transformer, RAG, FAISS, GraphRAG, RAGAS)
  eval/questions.json   # 10 câu: single/multi-hop/global
  pdfs/ processed/ faiss/   # PDF thật + chunks.jsonl + index (không commit)
tests/             # 21 pytest, không cần API/model
docs/NAIVE_RAG_MODULE.md    # tài liệu chi tiết + giao diện cho cả nhóm
```

Chi tiết cách các thành viên khác gọi lại module này (chunks cho Hùng, index cho Long, kết quả cho
Tuấn Anh): xem [docs/NAIVE_RAG_MODULE.md](docs/NAIVE_RAG_MODULE.md).

---
*Nhóm 7 — Học Máy Nâng Cao · Pipeline 1: Naive RAG (Hoạt, SV1)*
