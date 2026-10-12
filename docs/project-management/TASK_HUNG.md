# Task Chi Tiết — HÙNG (GraphRAG + LLM Wrapper)

**Vai trò:** GraphRAG owner + LLM wrapper (rate limiter, parse_json)
**Nhánh làm việc:** `feature/hung-graphrag`

---

## TUẦN 1

### Task 1.2: Gộp LLM wrapper → `src/llm.py` (Ưu tiên cao nhất)

**Mục tiêu:** Merge `llm.py` (Hoạt) + `llm_utils.py` (bạn) thành 1 file `src/llm.py`.

**File mới `src/llm.py`:**

```python
"""Unified LLM wrapper — singleton + rate limit + parse_json."""

import json
import threading
import time
import unicodedata

_llm_instance = None
_rate_lock = threading.Lock()
_last_call = 0.0


def get_llm(model=None, temperature=None):
    """Singleton ChatGroq instance. Trả cùng object mỗi lần gọi."""
    global _llm_instance
    if _llm_instance is None:
        from langchain_groq import ChatGroq
        from src import config
        _llm_instance = ChatGroq(
            model=model or config.LLM_MODEL,
            temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
            api_key=config.GROQ_API_KEY,
        )
    return _llm_instance


def _wait_for_rate_limit():
    """Thread-safe rate limiter: max LLM_REQUESTS_PER_MINUTE calls/min."""
    from src import config
    global _last_call
    min_interval = 60.0 / config.LLM_REQUESTS_PER_MINUTE
    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _last_call = time.monotonic()


def call_llm(llm, prompt, retries=3, backoff=2.0, throttle=False):
    """
    Call LLM with retry + optional rate limiting.

    Args:
        throttle: True khi batch processing (graph extraction).
                  False (default) khi single query.
    """
    if throttle:
        _wait_for_rate_limit()
    for attempt in range(retries):
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise


def parse_json(text):
    """
    Extract first valid JSON object from LLM output.
    Handles ```json fences and extra text.
    """
    import re
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    decoder = json.JSONDecoder()
    for i in range(len(text)):
        if text[i] in "{[":
            try:
                obj, _ = decoder.raw_decode(text, i)
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No valid JSON found in: {text[:200]}")
```

**Sau khi tạo file mới:**
1. Sửa import trong tất cả file của bạn:
   - `graph_builder.py`: `from src.llm_utils import ...` → `from src.llm import ...`
   - `graph_rag.py`: tương tự
2. Xóa file `src/llm_utils.py`
3. Test singleton: `assert get_llm() is get_llm()`
4. Test rate limit: gọi `call_llm(..., throttle=True)` 5 lần, đo thời gian ≥ expected interval
5. Test parse_json: `parse_json('```json\n{"a": 1}\n```')` → `{"a": 1}`

**Checklist:**
- [ ] File `src/llm.py` mới, `src/llm_utils.py` đã xóa
- [ ] Singleton hoạt động: `get_llm() is get_llm()` → True
- [ ] Rate limit hoạt động: throttle=True không vượt 30 req/min
- [ ] `parse_json()` xử lý JSON trong code fence
- [ ] Tất cả import đã sửa
- [ ] Hoạt review + approve

---

## TUẦN 2

### Task 2.2: GraphRAG dùng config + LLM mới

**Làm gì:**
1. Sửa import trong `graph_builder.py`:
   ```python
   from src.llm import get_llm, call_llm, parse_json  # không còn llm_utils
   from src import config
   ```

2. Sửa import trong `graph_rag.py`:
   ```python
   from src.llm import get_llm, call_llm
   from src import config
   ```

3. Đổi `GRAPH_BACKEND` default:
   ```python
   # Trong config (đã ở task 1.1 của Hoạt):
   GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "memory")  # ← memory, không phải neo4j
   ```

4. Đảm bảo `dotenv` import có try/except (đã ở config gộp).

**Test:**
```python
from src.graph_builder import build_knowledge_graph, build_communities
from src.graph_rag import build_graph_rag

# Build graph (dùng InMemory store)
chunks = [...]  # từ shared load_and_chunk()
graph = build_knowledge_graph(chunks)
communities = build_communities(graph)

# Query
rag = build_graph_rag(vectorstore, graph)
result = rag.query("machine learning la gi?")  # LOCAL search
result = rag.query("tong quan xu huong NLP")    # GLOBAL search (trigger GLOBAL_HINTS)
```

**Checklist:**
- [ ] Không còn `from src.llm_utils import ...`
- [ ] InMemory store là default
- [ ] LOCAL search hoạt động
- [ ] GLOBAL search kích hoạt khi câu hỏi chứa GLOBAL_HINTS
- [ ] Graph build + community detection chạy xong

---

### Task 2.4: Fix `chunk_id_for()` NFC-safe

**Vấn đề:** Hàm `chunk_id_for()` trong `graph_builder.py` KHÔNG tự NFC-normalize text. Nó phụ thuộc vào `to_chunk()` đã NFC trước đó. Nếu ai gọi `chunk_id_for()` trực tiếp với text không NFC, chunk ID sẽ khác với `chunker.py` của Hoạt → **phá vỡ deduplication** trong `merge_chunks()`.

**Sửa:**

```python
import unicodedata
import hashlib

def chunk_id_for(source: str, page: int, text: str) -> str:
    text = unicodedata.normalize("NFC", text)  # ← THÊM DÒNG NÀY
    raw = f"{source}|{page}|{text}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]
```

**Test:**
```python
# "café" viết bằng 2 cách Unicode khác nhau phải cho cùng ID
id1 = chunk_id_for("test.pdf", 1, "café")           # NFC sẵn
id2 = chunk_id_for("test.pdf", 1, "café")      # NFD (chưa normalize)
assert id1 == id2, f"NFC mismatch: {id1} != {id2}"
```

**Checklist:**
- [ ] `chunk_id_for()` tự NFC-normalize
- [ ] Test NFD→NFC pass
- [ ] `merge_chunks()` dedup đúng (cùng text từ 2 sources → 1 chunk)

---

## TUẦN 3

### Review tasks

- [ ] Review Long's `src/rerank_rag.py` — kiểm tra output format giống `GraphRAG.query()`
- [ ] Đóng góp 5+ câu hỏi eval (multi-hop + global) cho task 3.3
- [ ] Review Tuấn Anh's RAGAS fix (task 3.1)

---

## TUẦN 4

### Báo cáo phần GraphRAG

Viết phần báo cáo:
- Mô tả kiến trúc GraphRAG (entity extraction, Leiden communities, LOCAL/GLOBAL search)
- Giải thích tại sao GraphRAG tốt hơn cho câu hỏi multi-hop và global
- Kết quả RAGAS so sánh
