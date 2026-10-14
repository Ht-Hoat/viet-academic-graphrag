# Task Chi Tiết — HOẠT (Naive RAG + Config + Data Pipeline)

**Vai trò:** Foundation owner — config, chunker, vectorstore, naive_rag
**Nhánh làm việc:** `feature/hoat-naive-rag`

---

## TUẦN 1

### Task 1.1: Gộp `src/config.py` (Ưu tiên cao nhất)

**Mục tiêu:** 1 file config duy nhất cho toàn project, chứa constants từ cả 4 nhánh.

**Cấu trúc file:**

```python
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
SAMPLE_DIR = DATA_DIR / "sample"
PROCESSED_DIR = DATA_DIR / "processed"
FAISS_DIR = DATA_DIR / "faiss_index"
EVAL_DIR = DATA_DIR / "evaluation"
RESULTS_DIR = DATA_DIR / "results"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
EVAL_QUESTIONS_PATH = EVAL_DIR / "questions.json"
# Graph-specific paths (Hùng bổ sung)
GRAPH_DATA_DIR = DATA_DIR / "graph"
EXTRACTION_CACHE = GRAPH_DATA_DIR / "extraction_cache.json"
COMMUNITIES_FILE = GRAPH_DATA_DIR / "communities.json"
GRAPH_FILE = GRAPH_DATA_DIR / "graph.json"
ENTITY_VECTORS_FILE = GRAPH_DATA_DIR / "entity_vectors.npz"

# === LLM ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", LLM_MODEL)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_REQUESTS_PER_MINUTE = int(os.getenv("LLM_REQUESTS_PER_MINUTE", "30"))

# === Embedding ===
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# === Chunking ===
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # spec = 50
MIN_PAGE_TEXT_LENGTH = int(os.getenv("MIN_PAGE_TEXT_LENGTH", "40"))

# === Retrieval ===
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "50"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))

# === Graph (Hùng bổ sung chi tiết) ===
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")  # KHÔNG hardcode
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "memory")  # default=memory

EXTRACT_UNIT_CHARS = 1500
ENTITY_TYPES = {
    "PERSON": "Tac gia, nha nghien cuu",
    "ORGANIZATION": "Truong DH, vien nghien cuu, cong ty",
    "CONCEPT": "Khai niem, ly thuyet, phuong phap",
    "TECHNOLOGY": "Cong nghe, framework, thu vien",
    "DATASET": "Bo du lieu",
    "METRIC": "Chi so danh gia",
    "TASK": "Nhiem vu NLP",
}
RELATION_TYPES = {
    "PROPOSED_BY": "Duoc de xuat boi",
    "USED_IN": "Duoc su dung trong",
    "EVALUATED_ON": "Duoc danh gia tren",
    "OUTPERFORMS": "Vuot troi hon",
    "PART_OF": "La thanh phan cua",
    "RELATED_TO": "Lien quan den",
    "BASED_ON": "Dua tren",
    "APPLIED_TO": "Ap dung cho",
    "TRAINED_ON": "Huan luyen tren",
    "DEVELOPED_BY": "Phat trien boi",
    "EXTENDS": "Mo rong tu",
}

LEIDEN_RESOLUTION = float(os.getenv("LEIDEN_RESOLUTION", "1.0"))
MIN_COMMUNITY_SIZE = int(os.getenv("MIN_COMMUNITY_SIZE", "3"))
MAX_SEED_ENTITIES = 5
MAX_NODE_DEGREE = 50
PATHS_PER_SEED = 20
MAX_GRAPH_PATHS = 15
MAX_GRAPH_CHUNKS = 5
MAX_CONTEXT_CHARS = 12000
ENTITY_LINK_MIN_SIM = 0.75
GLOBAL_TOP_COMMUNITIES = 5
GLOBAL_MIN_SCORE = 20

# === Evaluation (Tuan Anh bo sung) ===
METHODS = ["Naive RAG", "RAG + Re-ranking", "GraphRAG"]
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
QUESTION_TYPES = ["single-hop", "multi-hop", "global"]
METHOD_COLORS = {
    "Naive RAG": "#4285F4",
    "RAG + Re-ranking": "#EA4335",
    "GraphRAG": "#34A853",
}
METRIC_LABELS_VI = {
    "faithfulness": "Do trung thuc",
    "answer_relevancy": "Do lien quan cau tra loi",
    "context_precision": "Do chinh xac ngu canh",
    "context_recall": "Do bao phu ngu canh",
}

def result_filename(method: str) -> str:
    return method.lower().replace(" ", "_").replace("+", "plus") + ".json"
```

**Checklist:**
- [x] Tất cả constants từ 4 nhánh đều có mặt
- [x] `python -c "from src import config; print(config.CHUNK_OVERLAP)"` → 50
- [x] Không hardcode API key, password
- [x] dotenv có try/except
- [ ] Hùng review phần Graph constants
- [ ] Tuấn Anh review phần Evaluation constants

---

### Task 1.3: Đổi CHUNK_OVERLAP → 50

**Đã nằm trong Task 1.1** — config mới ghi `CHUNK_OVERLAP=50`.

Thêm:
- [x] Xóa file chunks cache cũ (nếu có) trong `data/processed/` — chunks cũ dùng overlap=80
- [x] Re-run `chunk_documents()` với overlap=50 để tạo chunks mới (verify: 5 doc mẫu → 28 chunks)

---

## TUẦN 2

### Task 2.1: Naive RAG dùng config + LLM mới

**Làm gì:**
- Sửa import trong `src/naive_rag.py`:
  - `from src.llm import get_llm, call_llm` (file mới do Hùng gộp)
  - `from src import config` (config gộp)
- Kiểm tra `NaiveRAG.query()` hoạt động đúng

**Test:**
```python
from src.naive_rag import NaiveRAG
rag = NaiveRAG()
rag.load_and_chunk()   # dùng CHUNK_OVERLAP=50
result = rag.query("RAG la gi?")
print(result)  # {"answer": "...", "sources": [...], "method": "Naive RAG"}
```

**Checklist:**
- [x] Import từ `src.llm` (không còn `src.llm_utils`)
- [x] Import từ `src.config` (config gộp)
- [x] `NaiveRAG.query()` trả kết quả đúng format (`{"answer","sources","contexts","latency","method"}`)

---

### Task 2.5: Shared data pipeline

**Mục tiêu:** Tạo functions dùng chung cho cả 3 pipeline.

**Làm gì:** Tạo hoặc update file chứa:

```python
# Có thể đặt trong src/naive_rag.py hoặc tách ra src/data_pipeline.py

def load_and_chunk():
    """Load PDFs → clean → chunk. Dùng chung cho 3 pipeline."""
    from src.pdf_loader import load_pdfs
    from src.chunker import chunk_documents
    docs = load_pdfs()
    chunks = chunk_documents(docs)
    return chunks

def get_shared_vectorstore():
    """Tạo hoặc load FAISS index dùng chung."""
    from src.vectorstore import build_faiss_index, load_faiss_index
    from src.embedder import get_embeddings
    embeddings = get_embeddings()
    try:
        return load_faiss_index(embeddings)
    except FileNotFoundError:
        chunks = load_and_chunk()
        return build_faiss_index(chunks, embeddings)
```

**Checklist:**
- [x] `load_and_chunk()` callable từ cả naive_rag, graph_rag, rerank_rag (API ổn định trong `src/naive_rag.py`)
- [x] `get_global_vectorstore()` trả LangChain FAISS index (cache lazy, auto-build khi chưa có)
- [ ] Long và Hùng gọi được các hàm này (chờ confirm từ Long & Hùng sau khi merge)

---

## TUẦN 4

### Task 4.2: README.md

**Viết hướng dẫn:**
```markdown
# Cài đặt
git clone ...
pip install -r requirements.txt

# Cấu hình
cp .env.example .env
# Sửa GROQ_API_KEY trong .env

# Chạy demo
streamlit run app.py

# Chạy evaluation
python scripts/run_eval.py
```

**Checklist:**
- [ ] Clone → pip install → chạy được
- [ ] Có `.env.example` (không chứa key thật)
- [ ] Hướng dẫn rõ từng bước
