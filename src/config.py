"""Cấu hình chung (unified) cho toàn bộ project viet-academic-graphrag.

File config DUY NHẤT cho 3 pipeline (Naive RAG — Hoạt, Re-ranking — Long, GraphRAG — Hùng)
và module Evaluation/UI của Tuấn Anh. Mọi hằng số nằm ở đây để 3 pipeline dùng chung bộ
tham số (chunking / embedding / retrieval / graph) — nhờ đó so sánh RAGAS mới công bằng.

Nguyên tắc:
  • Không hardcode API key, password — tất cả đọc từ os.environ (có default an toàn).
  • python-dotenv để trong try/except — thiếu dotenv vẫn chạy được (CI, Docker…).
  • CHUNK_OVERLAP = 50 (chốt chung của nhóm).
  • GRAPH_BACKEND mặc định "memory" (không phụ thuộc Neo4j để demo chạy ngay).
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    # Chưa cài python-dotenv cũng không sao — vẫn đọc được os.environ
    pass

# ===================================================================
# Paths — đường dẫn thư mục / file (khớp cấu trúc thực tế trong repo)
# ===================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
SAMPLE_DIR = DATA_DIR / "sample_corpus"            # corpus mẫu .txt dùng khi chưa có PDF
PROCESSED_DIR = DATA_DIR / "processed"             # chunks.jsonl trao đổi giữa 3 pipeline
FAISS_DIR = DATA_DIR / "faiss"                     # index FAISS lưu ra đĩa
EVAL_DIR = DATA_DIR / "eval"                       # bộ câu hỏi RAGAS
RESULTS_DIR = PROJECT_ROOT / "results"             # kết quả eval (json/plot)
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
EVAL_QUESTIONS_PATH = EVAL_DIR / "questions.json"

# Graph-specific paths (Hùng dùng)
GRAPH_DATA_DIR = DATA_DIR / "graph"
EXTRACTION_CACHE = GRAPH_DATA_DIR / "extractions.jsonl"
COMMUNITIES_FILE = GRAPH_DATA_DIR / "communities.json"
GRAPH_FILE = GRAPH_DATA_DIR / "graph.json"         # chỉ dùng với backend="memory"
ENTITY_VECTORS_FILE = GRAPH_DATA_DIR / "entity_vectors.npz"

# Tự tạo thư mục cần thiết (không fail nếu đã tồn tại)
for _dir in (PROCESSED_DIR, FAISS_DIR, RESULTS_DIR, GRAPH_DATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ===================================================================
# LLM — Groq (miễn phí)
# Lấy key tại https://console.groq.com/ → API Keys
# ===================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Giữ cùng model giữa 3 pipeline → so sánh công bằng.
# Tên model trên Groq có thể đổi theo thời gian — xem console.groq.com/docs/models.
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
# Trích xuất thực thể tốn quota nhất → có thể set model nhỏ hơn (vd llama-3.1-8b-instant)
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", LLM_MODEL)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_REQUESTS_PER_MINUTE = int(os.getenv("LLM_REQUESTS_PER_MINUTE", "30"))

# ===================================================================
# Embedding — chạy local, không cần API
# bge-m3 đa ngữ (hỗ trợ tiếng Việt tốt). Lần đầu chạy tải ~2GB.
# Máy yếu có thể đổi sang "intfloat/multilingual-e5-small" (~470MB) trong .env.
# ===================================================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")   # "cuda" nếu có GPU

# ===================================================================
# Chunking
# ===================================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))              # ký tự / chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))         # spec nhóm = 50
MIN_PAGE_TEXT_LENGTH = int(os.getenv("MIN_PAGE_TEXT_LENGTH", "40"))  # bỏ trang PDF gần trống

# ===================================================================
# Retrieval — dùng chung cho cả 3 pipeline
# ===================================================================
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "50"))   # top rộng cho Long re-rank
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))          # top đưa LLM
GRAPH_HOPS = int(os.getenv("GRAPH_HOPS", "2"))            # số bước nhảy trên đồ thị

# ===================================================================
# GraphRAG — Neo4j / backend
# NEO4J_PASSWORD KHÔNG hardcode — đọc từ .env (xem .env.example).
# Mặc định backend "memory" để demo chạy ngay, không cần Neo4j.
# ===================================================================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")          # KHÔNG hardcode
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or None
GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "memory")      # mặc định "memory"

# ===================================================================
# GraphRAG — Trích xuất đồ thị (offline)
# ===================================================================
EXTRACT_UNIT_CHARS = 1500   # gộp các chunk liền nhau tới ngưỡng này cho 1 lượt gọi LLM

# Loại thực thể/quan hệ cố định để đồ thị nhất quán.
# Chỉnh lại khi nhóm chốt lĩnh vực tài liệu.
ENTITY_TYPES = {
    "PERSON": "người, tác giả",
    "ORG": "tổ chức, trường, viện, công ty",
    "PAPER": "bài báo, sách, giáo trình, công trình",
    "METHOD": "phương pháp, mô hình, thuật toán, kiến trúc",
    "CONCEPT": "khái niệm, thuật ngữ, lý thuyết",
    "TASK": "bài toán, ứng dụng",
    "DATASET": "bộ dữ liệu, độ đo, benchmark",
}
RELATION_TYPES = {
    "là_một": "A là một loại / trường hợp của B",
    "thành_phần_của": "A là thành phần / bộ phận của B",
    "sử_dụng": "A sử dụng / dựa trên B",
    "đề_xuất": "A (người, bài báo, tổ chức) đề xuất / giới thiệu B",
    "tác_giả_của": "A là tác giả của B",
    "thuộc_về": "A thuộc / làm việc tại B",
    "cải_tiến": "A cải tiến / mở rộng B",
    "giải_quyết": "A được dùng để giải quyết bài toán B",
    "đánh_giá_trên": "A được đánh giá trên bộ dữ liệu / độ đo B",
    "so_sánh_với": "A được so sánh với B",
    "liên_quan_đến": "quan hệ khác (chỉ dùng khi không có nhãn phù hợp)",
}

# Community detection (Leiden)
LEIDEN_RESOLUTION = float(os.getenv("LEIDEN_RESOLUTION", "1.0"))
MIN_COMMUNITY_SIZE = int(os.getenv("MIN_COMMUNITY_SIZE", "3"))

# ===================================================================
# GraphRAG — Truy hồi (online)
# ===================================================================
MAX_SEED_ENTITIES = 5          # số thực thể trong câu hỏi dùng làm điểm xuất phát
MAX_NODE_DEGREE = 50           # không đi qua hub có quá nhiều cạnh khi mở rộng
PATHS_PER_SEED = 20            # số đường tối đa / độ dài / seed
MAX_GRAPH_PATHS = 15           # số đường quan hệ đưa vào ngữ cảnh
MAX_GRAPH_CHUNKS = 5           # số chunk lấy qua đồ thị (ngoài top-k vector)
MAX_CONTEXT_CHARS = 12000      # ngân sách ngữ cảnh gửi LLM
ENTITY_LINK_MIN_SIM = 0.75     # ngưỡng cosine khi nối thực thể bằng embedding
GLOBAL_TOP_COMMUNITIES = 5
GLOBAL_MIN_SCORE = 20          # bỏ ý trả lời có điểm hữu ích thấp hơn ngưỡng

# ===================================================================
# Evaluation / UI (Tuấn Anh bổ sung)
# ===================================================================
SAMPLE_RESULTS_DIR = RESULTS_DIR / "sample"   # kết quả mẫu để UI hiển thị khi chưa chạy eval thật
METHODS = ["Naive RAG", "RAG + Re-ranking", "GraphRAG"]
RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
QUESTION_TYPES = ["single-hop", "multi-hop", "global"]
METHOD_COLORS = {
    "Naive RAG": "#4285F4",
    "RAG + Re-ranking": "#EA4335",
    "GraphRAG": "#34A853",
}
METRIC_LABELS_VI = {
    "faithfulness": "Độ trung thực",
    "answer_relevancy": "Độ liên quan câu trả lời",
    "context_precision": "Độ chính xác ngữ cảnh",
    "context_recall": "Độ bao phủ ngữ cảnh",
}


def result_filename(method: str) -> str:
    """Chuẩn hoá tên file kết quả theo method: 'RAG + Re-ranking' → 'rag_plus_re-ranking.json'."""
    return method.lower().replace(" ", "_").replace("+", "plus") + ".json"
