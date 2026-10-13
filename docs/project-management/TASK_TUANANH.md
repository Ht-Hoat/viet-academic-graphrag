# Task Chi Tiết — TUẤN ANH (UI + Evaluation)

**Vai trò:** UI/UX + Evaluation pipeline owner
**Nhánh làm việc:** `feature/tuananh-eval-ui-v2` (nhánh MỚI, xem task 1.4)

---

## TUẦN 1

### Task 1.4: Tạo nhánh mới từ `main` (Ưu tiên cao nhất)

**Vấn đề:** Nhánh hiện tại `feature-tuananh-eval-ui` là **orphan branch** — không có merge base với `main`. Git sẽ từ chối merge khi tạo PR.

**Cách kiểm tra:**
```bash
git merge-base main feature-tuananh-eval-ui
# → fatal: No common ancestor → ĐÚNG, đây là orphan
```

**Cách sửa:**
```bash
# 1. Lấy main mới nhất
git fetch origin main
git checkout main
git pull origin main

# 2. Tạo nhánh mới TỪ main
git checkout -b feature/tuananh-eval-ui-v2

# 3. Copy files từ nhánh cũ
git checkout feature-tuananh-eval-ui -- app.py
git checkout feature-tuananh-eval-ui -- src/pipelines.py
git checkout feature-tuananh-eval-ui -- src/evaluator.py
git checkout feature-tuananh-eval-ui -- src/charts.py
git checkout feature-tuananh-eval-ui -- scripts/run_eval.py

# 4. KHÔNG copy src/config.py — sẽ dùng config chung (Hoạt tạo)

# 5. Commit + push
git add app.py src/pipelines.py src/evaluator.py src/charts.py scripts/run_eval.py
git commit -m "feat: add evaluation UI and RAGAS scoring"
git push -u origin feature/tuananh-eval-ui-v2
```

**Kiểm tra thành công:**
```bash
git merge-base main feature/tuananh-eval-ui-v2
# → hiện SHA commit → OK, có merge base
```

**Checklist:**
- [ ] Nhánh mới `feature/tuananh-eval-ui-v2` tạo từ main
- [ ] `git merge-base main feature/tuananh-eval-ui-v2` có kết quả
- [ ] Files `app.py`, `src/pipelines.py`, `src/evaluator.py`, `src/charts.py`, `scripts/run_eval.py` có mặt
- [ ] KHÔNG có `src/config.py` riêng (dùng chung)
- [ ] Push thành công lên remote

---

## TUẦN 3

### Task 3.1: Fix RAGAS judge LLM

**Vấn đề:** `src/evaluator.py` gọi RAGAS `evaluate()` mà KHÔNG truyền `llm=` và `embeddings=`. Mặc định RAGAS dùng **OpenAI GPT-4** — nhóm dùng **Groq/Llama-3.3**.

**Sửa trong `src/evaluator.py`:**

```python
# TRƯỚC (lỗi):
from ragas import evaluate

def score_with_ragas(dataset, metrics):
    result = evaluate(dataset, metrics=metrics)  # ← default OpenAI!
    return result

# SAU (đúng):
from ragas import evaluate
from src.llm import get_llm
from src.embedder import get_embeddings

def score_with_ragas(dataset, metrics):
    llm = get_llm()
    embeddings = get_embeddings()
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,              # ← Groq LLM
        embeddings=embeddings,  # ← BGE-M3
    )
    return result
```

**Lưu ý:** RAGAS có thể cần LangChain-compatible LLM. `ChatGroq` từ `langchain_groq` đã là LangChain-compatible → nên hoạt động. Nếu gặp lỗi, thử:
```python
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=config.GROQ_API_KEY)
```

**Test:**
```python
# Test nhỏ với 1 sample
from src.evaluator import score_with_ragas
from datasets import Dataset

sample = Dataset.from_dict({
    "question": ["RAG la gi?"],
    "answer": ["RAG la Retrieval-Augmented Generation."],
    "contexts": [["RAG (Retrieval-Augmented Generation) la ky thuat ket hop..."]],
    "ground_truth": ["Retrieval-Augmented Generation"]
})
result = score_with_ragas(sample, ["faithfulness"])
print(result)  # Phải trả score, KHÔNG báo lỗi OpenAI API key
```

**Checklist:**
- [ ] `evaluate()` nhận `llm=` và `embeddings=`
- [ ] Test chạy KHÔNG cần OpenAI API key
- [ ] Score trả về hợp lệ (0.0 - 1.0)

---

### Task 3.2: Fix `pipelines.py` import 3 pipeline thật

**Vấn đề hiện tại:** `src/pipelines.py` import `src.rerank_rag` nhưng module chưa tồn tại → fallback mock.

**Sửa imports** (sau khi Hoạt, Hùng, Long hoàn thành GĐ2):

```python
# src/pipelines.py

# Import 3 pipeline
try:
    from src.naive_rag import NaiveRAG
    _HAS_NAIVE = True
except ImportError:
    _HAS_NAIVE = False

try:
    from src.rerank_rag import build_rerank_rag  # Long tạo ở task 2.3
    _HAS_RERANK = True
except ImportError:
    _HAS_RERANK = False

try:
    from src.graph_rag import build_graph_rag
    _HAS_GRAPH = True
except ImportError:
    _HAS_GRAPH = False


class PipelineAdapter:
    """Adapter thống nhất cho 3 pipeline."""

    def __init__(self, method, vectorstore, graph_store=None):
        self.method = method
        if method == "Naive RAG" and _HAS_NAIVE:
            self._rag = NaiveRAG(vectorstore=vectorstore)
        elif method == "RAG + Re-ranking" and _HAS_RERANK:
            self._rag = build_rerank_rag(vectorstore)
        elif method == "GraphRAG" and _HAS_GRAPH:
            self._rag = build_graph_rag(vectorstore, graph_store)
        else:
            self._rag = None  # fallback mock

    def query(self, question: str) -> dict:
        if self._rag is None:
            return _mock_response(question, self.method)
        return self._rag.query(question)
```

**Sửa GLOBAL_HINTS** (nếu cần):
Hiện tại mock dùng 5 hints, GraphRAG thật dùng 10 hints. Xóa GLOBAL_HINTS duplicate trong pipelines.py — dùng trực tiếp từ `graph_rag.py`.

**Checklist:**
- [ ] Import `src.naive_rag` OK
- [ ] Import `src.rerank_rag` OK (Long đã tạo)
- [ ] Import `src.graph_rag` OK
- [ ] `PipelineAdapter("Naive RAG", vs).query("test")` → real answer
- [ ] `PipelineAdapter("RAG + Re-ranking", vs).query("test")` → real answer
- [ ] `PipelineAdapter("GraphRAG", vs, graph).query("test")` → real answer
- [ ] Không dùng mock khi module tồn tại

---

### Task 3.3: Đóng góp câu hỏi eval

Đóng góp ≥5 câu hỏi loại **global**:

```json
[
  {
    "question": "Tong quan cac xu huong nghien cuu NLP tieng Viet hien nay la gi?",
    "ground_truth": "Cac xu huong chinh bao gom: ...",
    "type": "global"
  },
  {
    "question": "Nhung thach thuc lon nhat khi xu ly ngon ngu tieng Viet?",
    "ground_truth": "...",
    "type": "global"
  }
]
```

---

### Task 3.4: Chạy evaluation

**Sau khi 3.1 + 3.2 + 3.3 xong:**

```bash
python scripts/run_eval.py
```

Hoặc viết script mới:
```python
from src.pipelines import PipelineAdapter, load_shared_data
from src.evaluator import score_with_ragas, aggregate
import json

# Load shared data
vectorstore, graph_store = load_shared_data()
questions = json.load(open("data/evaluation/questions.json"))

results = {}
for method in ["Naive RAG", "RAG + Re-ranking", "GraphRAG"]:
    adapter = PipelineAdapter(method, vectorstore, graph_store)
    answers = [adapter.query(q["question"]) for q in questions]
    scores = score_with_ragas(answers, questions)
    results[method] = aggregate(scores)
    json.dump(results[method], open(f"data/results/{method}.json", "w"))

print("Evaluation done!")
```

**Output:** 3 file JSON trong `data/results/`, mỗi file chứa 4 RAGAS metrics.

**Checklist:**
- [ ] 3 pipeline chạy thật (không mock)
- [ ] RAGAS scoring dùng Groq (không OpenAI)
- [ ] Kết quả JSON lưu cho mỗi pipeline
- [ ] Mỗi metric có giá trị hợp lệ (0.0 - 1.0)

---

### Task 3.5: Fix mock source:"real" bug

**Vấn đề:** `scripts/run_eval.py` khi chạy mock pipeline, output ghi `source: "real"` thay vì `source: "mock"`.

**Sửa:** Trong mock factory, đổi:
```python
# TRƯỚC:
{"source": "real", ...}
# SAU:
{"source": "mock", ...}
```

Và trong real pipeline output:
```python
{"source": "real", ...}
```

**Checklist:**
- [x] Mock output ghi `source: "mock"`
- [x] Real output ghi `source: "real"`

---

## TUẦN 4

### Task 4.1: Streamlit UI chạy 3 pipeline thật

**Sửa `app.py`:**
- Thay mock pipeline bằng real pipeline (task 3.2 đã fix imports)
- Test: `streamlit run app.py` → chọn từng pipeline → gửi câu hỏi → nhận answer
- Kiểm tra: Chat Demo + RAGAS Results hiển thị đúng

**Checklist:**
- [ ] `streamlit run app.py` khởi động OK
- [ ] Chat Demo: 3 pipeline đều trả lời
- [ ] RAGAS Results: hiển thị scores từ file JSON
- [ ] Không crash khi chuyển pipeline

---

### Task 4.3: Biểu đồ so sánh

Dùng `src/charts.py` hoặc plotly trong Streamlit:

```python
import plotly.express as px
import pandas as pd

# Data từ evaluation results
data = []
for method, scores in results.items():
    for metric, value in scores.items():
        data.append({"Method": method, "Metric": metric, "Score": value})

df = pd.DataFrame(data)
fig = px.bar(df, x="Metric", y="Score", color="Method", barmode="group",
             title="So sanh 3 Pipeline RAG - RAGAS Metrics")
fig.show()
```

**Checklist:**
- [ ] Bar chart 4 metrics × 3 methods
- [ ] Nhãn tiếng Việt
- [ ] Hiển thị trong Streamlit UI (tab RAGAS Results)
