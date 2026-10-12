# Module Naive RAG (phần của Hoạt — SV1)

Tài liệu dùng cho cả nhóm: vai trò từng file, cách chạy, cách các thành viên khác gọi lại,
và những chỗ code khác/hơn bản mẫu trong [SPRINT_DEMO_GRAPHRAG.md](../SPRINT_DEMO_GRAPHRAG.md).

Đây là **nền tảng dùng chung** của cả đề tài: Naive RAG là đường cơ sở (baseline) để so sánh,
đồng thời cung cấp *chunking*, *embedding* và *FAISS index* cho hai pipeline kia (re-ranking của
Long, GraphRAG của Hùng). Ba pipeline dùng chung một bộ chunk + một index thì bảng so sánh RAGAS
mới công bằng.

| File | Vai trò |
|---|---|
| [src/config.py](../src/config.py) | Cấu hình chung: đường dẫn, model, tham số chunk/retrieval (đọc từ `.env`) |
| [src/pdf_loader.py](../src/pdf_loader.py) | Đọc PDF (PyMuPDF) + làm sạch text tiếng Việt; tự dùng corpus mẫu khi chưa có PDF |
| [src/chunker.py](../src/chunker.py) | Chia đoạn (recursive), sinh `chunk_id` ổn định, xuất/nhập `chunks.jsonl` |
| [src/embedder.py](../src/embedder.py) | Nhúng vector bge-m3 (giao diện LangChain) + `FakeEmbeddings` cho test |
| [src/vectorstore.py](../src/vectorstore.py) | FAISS index (dùng chung) + `NumpyVectorStore` dự phòng, cùng giao diện `similarity_search` |
| [src/naive_rag.py](../src/naive_rag.py) | Pipeline Naive RAG + các hàm giao diện cho thành viên khác + dòng lệnh |
| [src/llm.py](../src/llm.py) | Gọi Groq: khởi tạo model, thử lại khi lỗi 429 |

## 1. Cài đặt

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env                               # rồi điền GROQ_API_KEY
```

Không có GPU cứ để `EMBEDDING_DEVICE=cpu`. Máy yếu (RAM < 8GB) đổi `EMBEDDING_MODEL` trong `.env`
sang `intfloat/multilingual-e5-small` (~470MB thay vì ~2GB của bge-m3).

## 2. Chạy

```bash
# Dựng index — chưa có PDF thì tự dùng corpus mẫu tiếng Việt trong data/sample_corpus/
python -m src.naive_rag build --sample            # build từ corpus mẫu (chạy được ngay)
python -m src.naive_rag build --pdf-dir data/pdfs # build từ PDF thật
python -m src.naive_rag build --limit 50          # chỉ 50 chunk đầu (chạy thử)

# Hỏi đáp
python -m src.naive_rag ask "BERT được huấn luyện bằng phương pháp gì?"
python -m src.naive_rag ask "BERT là gì?" --retrieval-only   # xem chunk lấy được, KHÔNG tốn quota LLM

# Đo trên cả bộ câu hỏi
python -m src.naive_rag eval --questions data/eval/questions.json
python -m src.naive_rag stats
```

`build` xuất đồng thời **hai** thứ: `data/faiss/` (index dùng chung) và
`data/processed/chunks.jsonl` (để Hùng dựng đồ thị trên cùng bộ chunk).

`--retrieval-only` bỏ bước gọi LLM: xem được đúng những chunk mà retriever lấy về mà không tốn
quota Groq. Đây là cách soi lỗi truy hồi nhanh nhất và cách chạy được khi chưa có API key.

Chạy test: `python -m pytest` → **21 test**, không cần API key, không tải model 2GB
(dùng `FakeEmbeddings` + `NumpyVectorStore`).

## 3. Giao diện cho các thành viên khác

**Chunks ra (cho Hùng — GraphRAG).** `load_and_chunk(pdf_dir)` trả list dict
`{chunk_id, text, source, page}`. `chunk_id` băm từ `source|page|NFC(text)` — **trùng đúng công
thức** `src.graph_builder.chunk_id_for` của Hùng, nên chunk lấy qua vector và chunk lấy qua đồ thị
khử trùng được với nhau. Xuất file trao đổi:

```python
from src.naive_rag import load_and_chunk
from src.chunker import save_chunks_jsonl
save_chunks_jsonl(load_and_chunk(), "data/processed/chunks.jsonl")
# Hùng nạp: python -m src.graph_builder build --chunks data/processed/chunks.jsonl
```

**Index + embedding dùng chung (cho Long — re-ranking; Hùng — vector search trong Local search).**

```python
from src.naive_rag import get_global_vectorstore        # nạp/dựng FAISS 1 lần, cache lại
from src.embedder import get_embeddings

vs = get_global_vectorstore()
candidates = vs.similarity_search(question, k=50)        # Long lấy top-50 rồi cross-encoder re-rank
# Hùng: FAISS.load_local("data/faiss", get_embeddings()) — dùng lại chính index này qua --faiss-dir
```

**Kết quả ra (cho Tuấn Anh — UI và RAGAS).** `NaiveRAG.query(question)` và `query_naive_rag(question)`
trả về cùng một "hình dạng" với hai pipeline kia, nên UI và evaluator xử lý thống nhất:

```python
{
  "answer":   str,
  "sources":  list[str],   # ["01.txt — trang 1", ...]
  "contexts": list[str],   # RAGAS đọc trường này (text từng chunk)
  "latency":  float,
  "method":   "Naive RAG",
}
```

`NaiveRAG` gọi được như hàm (`rag(question)`), cắm thẳng vào `run_evaluation(rag, questions, "Naive RAG")`.
Backward-compatible với `app.py` hiện tại của Tuấn Anh qua `query_naive_rag(question)`.

## 4. Khác gì so với code mẫu / bản trên nhánh chính

| Vấn đề ở bản mẫu / bản cũ | Cách xử lý ở đây |
|---|---|
| Không có PDF là không chạy được | Corpus mẫu tiếng Việt trong `data/sample_corpus/` → `build --sample` chạy ngay |
| Nối từ ngắt dòng làm dính cả dòng số trang ("- 12 -" + "doi") | `(\w)-\n(\w)` chỉ nối khi có ký tự chữ hai bên gạch nối |
| PDF tiếng Việt xuất NFD → tìm kiếm trượt | Chuẩn hóa NFC ngay khi đọc, và `chunk_id` băm trên text đã NFC |
| `chunk_id` không thống nhất với GraphRAG → chunk trùng không khử được | Cùng công thức `sha1(source\|page\|NFC(text))[:16]` với Hùng |
| Chỉ có hàm `query_*`, khó test và tái dùng | Có class `NaiveRAG` (inject được vectorstore/llm) + hàm wrapper giữ tương thích |
| Thiếu faiss/langchain là vỡ | Tự rơi về `NumpyVectorStore` cùng giao diện `similarity_search` |
| Không lưu index → mỗi lần chạy embed lại (chậm, tốn) | `save_local`/`load_local` FAISS ra `data/faiss/`; `get_global_vectorstore` nạp lại |
| Không có test | 21 pytest chạy offline (FakeEmbeddings + NumpyVectorStore), không tốn quota |
| Cắt cứng giữa câu làm mất ngữ cảnh | Recursive splitting (đoạn→câu→từ) + overlap 80 ký tự |

Vài lựa chọn đáng nói khi bảo vệ:

- **Chuẩn hóa L2 khi nhúng** (`normalize_embeddings=True`) để tích vô hướng chính là cosine →
  dùng được `IndexFlatIP` của FAISS, nhanh hơn cosine đầy đủ.
- **Recursive character splitting**: cắt ở ranh giới tự nhiên trước, chỉ cắt cứng khi một câu dài
  quá `CHUNK_SIZE × 1.5`. Overlap giữ ngữ cảnh ở ranh giới hai chunk.
- **So sánh công bằng**: Naive RAG cố tình *không* re-rank, *không* dùng đồ thị — đó là đường cơ
  sở. Mọi chênh lệch điểm RAGAS so với hai pipeline kia phản ánh đúng đóng góp của re-ranking và
  của đồ thị, vì cả ba dùng chung chunk + embedding + câu lệnh sinh.

## 5. Ngân sách & lưu ý

- Embedding chạy **local**, không tốn quota Groq. Chỉ bước sinh câu trả lời (`ask`/`eval` không có
  `--retrieval-only`) mới gọi LLM — mỗi câu 1 lượt.
- Lần đầu `build` tải model bge-m3 (~2GB) và cần internet; các lần sau dùng cache của
  `sentence-transformers`.
- `data/faiss/` và `data/processed/` bị `.gitignore` (chỉ giữ `.gitkeep`) để repo nhẹ — mỗi người
  clone xong tự chạy `python -m src.naive_rag build --sample` một lần.

## 6. Ghi chú khi merge vào repo chung

- `src/llm.py` ở đây và `src/llm_utils.py` của Hùng làm cùng việc (get_llm + call_llm). Khi merge
  giữ **một** file dùng chung, sửa import cho khớp.
- `src/config.py` nên gộp với config của Hùng thành một; các hằng số chunk/retrieval ở đây và của
  Hùng đã đặt cùng tên (`CHUNK_SIZE`, `TOP_K_RETRIEVE`, `TOP_K_FINAL`) nên gộp không xung đột giá trị.
- Sau khi gộp, chạy `python -m src.naive_rag build --sample` rồi
  `python -m src.graph_builder build --chunks data/processed/chunks.jsonl` để xác nhận hai module ăn khớp.
