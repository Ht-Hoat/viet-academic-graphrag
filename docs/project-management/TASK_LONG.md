# Task Chi Tiết — LONG (Re-ranking Adapter)

**Vai trò:** Re-ranking pipeline owner — cross-encoder, adapter integration
**Nhánh làm việc:** `long` (hoặc tạo nhánh mới `feature/long-rerank-adapter`)

---

## TUẦN 1

### Task 1.5: Đọc hiểu LangChain FAISS API

**Mục tiêu:** Hiểu cách LangChain FAISS hoạt động để viết adapter ở tuần 2.

**Đọc code:**
- File `src/vectorstore.py` trên nhánh `feature/hoat-naive-rag`:
  ```bash
  git show feature/hoat-naive-rag:src/vectorstore.py
  ```

**Những gì cần hiểu:**

1. **Tạo index:**
   ```python
   from langchain_community.vectorstores import FAISS
   vectorstore = FAISS.from_documents(docs, embeddings)
   ```

2. **Search:** (đây là hàm bạn sẽ gọi thay cho faiss.IndexFlatIP)
   ```python
   results = vectorstore.similarity_search_with_score(query, k=50)
   # Trả về: List[Tuple[Document, float]]
   # Document có: .page_content (str), .metadata (dict)
   # float là distance score (L2, nhỏ hơn = gần hơn)
   ```

3. **So sánh với code hiện tại của bạn:**

   | | Code hiện tại (FaissIndex) | Sẽ dùng (LangChain FAISS) |
   |--|---------------------------|---------------------------|
   | Build | `faiss.IndexFlatIP(dim)` + manual add | `FAISS.from_documents()` |
   | Search | `index.search(vector, k)` → distances, indices | `vectorstore.similarity_search_with_score(query_text, k)` |
   | Input | Vector (numpy array) | Text string (tự embed) |
   | Output | `(distances, indices)` → manual lookup | `List[Tuple[Document, float]]` |

4. **Điểm khác biệt quan trọng:**
   - LangChain FAISS tự embed query → bạn KHÔNG cần gọi bi_encoder.encode()
   - Output là Document objects → bạn lấy `.page_content` trực tiếp
   - Score là L2 distance (nhỏ = tốt), không phải inner product (lớn = tốt)
   - Nhưng với normalized vectors, thứ tự ranking GIỐNG NHAU

**Viết code demo** (file tạm, không commit):
```python
# demo_langchain_faiss.py
from src.vectorstore import load_faiss_index
from src.embedder import get_embeddings

# Load shared index
vs = load_faiss_index(get_embeddings())

# Search — thay thế bi_encoder.encode() + faiss.search()
results = vs.similarity_search_with_score("RAG la gi?", k=50)

# Chuyển sang format cho cross-encoder
query = "RAG la gi?"
pairs = [(query, doc.page_content) for doc, score in results]
doc_texts = [doc.page_content for doc, score in results]
metadatas = [doc.metadata for doc, score in results]

print(f"Top-5 trước rerank:")
for doc, score in results[:5]:
    print(f"  [{score:.4f}] {doc.page_content[:80]}...")
```

**Checklist:**
- [ ] Đọc xong `src/vectorstore.py` của Hoạt
- [ ] Hiểu `similarity_search_with_score()` trả gì
- [ ] Viết được demo code chạy thành công
- [ ] Biết cách chuyển output sang pairs cho cross-encoder

---

## TUẦN 2

### Task 2.3: Tạo `src/rerank_rag.py` adapter (Task chính)

**Mục tiêu:** Tạo file `src/rerank_rag.py` wrap `reranker_module/`, nhận LangChain FAISS index, export interface giống `NaiveRAG` và `GraphRAG`.

**File `src/rerank_rag.py`:**

```python
"""
Re-ranking RAG adapter.
Dùng chung FAISS index (LangChain) + cross-encoder reranking.
Interface thống nhất với NaiveRAG và GraphRAG.
"""

from src.llm import get_llm, call_llm
from src import config


class RerankRAG:
    """Two-stage retrieval: vector search → cross-encoder rerank → LLM."""

    def __init__(self, vectorstore, rerank_top_k=None):
        self.vectorstore = vectorstore  # LangChain FAISS (shared)
        self.rerank_top_k = rerank_top_k or config.TOP_K_FINAL
        self.retrieve_top_k = config.TOP_K_RETRIEVE
        self.llm = get_llm()

        # Lazy load cross-encoder (tốn RAM, chỉ load khi cần)
        self._reranker = None

    def _get_reranker(self):
        if self._reranker is None:
            from reranker_module.reranker.cross_encoder import CrossEncoderReranker
            from reranker_module.reranker.config import RerankConfig
            cfg = RerankConfig()
            self._reranker = CrossEncoderReranker(cfg)
        return self._reranker

    def query(self, question: str) -> dict:
        """
        Query pipeline:
        1. Vector search (FAISS, top-50)
        2. Cross-encoder rerank (top-5)
        3. LLM generate answer

        Returns:
            dict with keys: answer, sources, method, contexts
        """
        # === Bước 1: Retrieve từ shared FAISS ===
        raw_results = self.vectorstore.similarity_search_with_score(
            question, k=self.retrieve_top_k
        )
        # raw_results: List[Tuple[Document, float]]

        # === Bước 2: Cross-encoder rerank ===
        reranker = self._get_reranker()
        # Chuẩn bị pairs cho cross-encoder
        pairs = [(question, doc.page_content) for doc, _ in raw_results]
        scores = reranker.predict(pairs)  # hoặc reranker.rerank() tùy API

        # Sắp xếp theo score giảm dần
        scored_results = list(zip(raw_results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_results[:self.rerank_top_k]

        # === Bước 3: Build context ===
        contexts = []
        sources = []
        for i, ((doc, vec_score), rerank_score) in enumerate(top_results):
            contexts.append(f"[{i+1}] {doc.page_content}")
            sources.append({
                "content": doc.page_content[:200],
                "metadata": doc.metadata,
                "vector_score": float(vec_score),
                "rerank_score": float(rerank_score),
            })

        context_text = "\n\n".join(contexts)

        # === Bước 4: LLM generate ===
        prompt = (
            f"Dua vao ngu canh sau day, hay tra loi cau hoi bang tieng Viet.\n\n"
            f"Ngu canh:\n{context_text}\n\n"
            f"Cau hoi: {question}\n\n"
            f"Tra loi:"
        )
        answer = call_llm(self.llm, prompt)

        return {
            "answer": answer,
            "sources": sources,
            "method": "RAG + Re-ranking",
            "contexts": [doc.page_content for (doc, _), _ in top_results],
        }


def build_rerank_rag(vectorstore, **kwargs) -> RerankRAG:
    """
    Factory function.
    Tuan Anh gọi hàm này từ src/pipelines.py:
        from src.rerank_rag import build_rerank_rag
        rerank = build_rerank_rag(shared_vectorstore)
    """
    return RerankRAG(vectorstore, **kwargs)
```

**Lưu ý quan trọng:**
- `reranker.predict(pairs)` hay `reranker.rerank(docs)` — kiểm tra API của `CrossEncoderReranker` class mà bạn viết. Điều chỉnh cho phù hợp.
- Output format phải giống `NaiveRAG.query()` và `GraphRAG.query()`: dict có keys `answer`, `sources`, `method`, `contexts`.
- Cross-encoder lazy load — không tốn RAM khi chưa gọi `query()`.
- Prompt LLM bằng tiếng Việt.

**Test:**
```python
from src.rerank_rag import build_rerank_rag
from src.vectorstore import load_faiss_index
from src.embedder import get_embeddings

vs = load_faiss_index(get_embeddings())
rag = build_rerank_rag(vs)

result = rag.query("RAG la gi?")
assert "answer" in result
assert "sources" in result
assert result["method"] == "RAG + Re-ranking"

# Kiểm tra rerank thực sự thay đổi thứ tự
raw = vs.similarity_search_with_score("RAG la gi?", k=5)
raw_texts = [doc.page_content for doc, _ in raw]
rerank_texts = result["contexts"]
# Nếu rerank hoạt động, thứ tự phải khác
print("Raw top-1:", raw_texts[0][:50])
print("Rerank top-1:", rerank_texts[0][:50])
```

**Checklist:**
- [ ] File `src/rerank_rag.py` tồn tại
- [ ] `from src.rerank_rag import build_rerank_rag` — import OK
- [ ] `build_rerank_rag(vs).query("test")` trả dict đúng format
- [ ] Cross-encoder thay đổi thứ tự so với vector-only
- [ ] Output format giống Naive và Graph RAG
- [ ] Tuấn Anh confirm import thành công từ `pipelines.py`

---

## TUẦN 3

### Review tasks

- [ ] Review Tuấn Anh's `pipelines.py` import cho reranker
- [ ] Đóng góp 5+ câu hỏi eval (single-hop + multi-hop) cho task 3.3

---

## TUẦN 4

### Báo cáo phần Re-ranking

Viết phần báo cáo:
- Mô tả kiến trúc 2-stage retrieval (bi-encoder → cross-encoder)
- Giải thích tại sao cross-encoder chính xác hơn bi-encoder
- So sánh BGE-M3 (bi-encoder) vs BGE-reranker-v2-m3 (cross-encoder)
- Kết quả RAGAS so sánh

---

## LƯU Ý

**Bạn KHÔNG cần:**
- Viết chunker riêng (dùng chung `chunker.py` của Hoạt)
- Tạo FAISS index riêng (dùng chung LangChain FAISS)
- Viết LLM wrapper riêng (dùng chung `src/llm.py`)

**Bạn CHỈ CẦN giữ lại:**
- `reranker_module/reranker/cross_encoder.py` — logic cross-encoder scoring
- `reranker_module/reranker/config.py` — config cho cross-encoder (batch size, max length, etc.)
- `src/rerank_rag.py` — adapter mới wrap cross-encoder + LangChain FAISS

**Có thể bỏ:**
- `reranker_module/reranker/bi_encoder.py` — thay bằng LangChain FAISS
- `reranker_module/reranker/faiss_index.py` — thay bằng LangChain FAISS
- `reranker_module/reranker/utils.py` (phần chunk_text) — dùng chunker chung
