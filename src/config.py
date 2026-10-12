"""Cấu hình chung cho module Naive RAG (phần của Hoạt — SV1).

Đọc .env và tập trung mọi hằng số vào một chỗ. Các thành viên khác (Long — re-ranking,
Hùng — GraphRAG, Tuấn Anh — UI/RAGAS) import hằng số từ đây để 3 pipeline dùng chung
cùng một bộ tham số chunking / embedding / retrieval, nhờ đó so sánh mới công bằng.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    # Chưa cài python-dotenv cũng không sao — vẫn đọc được os.environ
    pass

# ===== Đường dẫn =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
SAMPLE_DIR = DATA_DIR / "sample_corpus"
PROCESSED_DIR = DATA_DIR / "processed"          # chunks.jsonl để trao đổi với GraphRAG
FAISS_DIR = DATA_DIR / "faiss"                  # index FAISS lưu ra đĩa
EVAL_DIR = DATA_DIR / "eval"
RESULTS_DIR = PROJECT_ROOT / "results"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
EVAL_QUESTIONS_PATH = EVAL_DIR / "questions.json"

for _dir in (PROCESSED_DIR, FAISS_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ===== LLM (Groq — miễn phí) =====
# Lấy key tại https://console.groq.com/ → API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Giữ trùng model với bản GraphRAG của Hùng để 3 pipeline dùng một model, so sánh công bằng.
# Tên model trên Groq có thể đổi theo thời gian — kiểm tra tại console.groq.com/docs/models.
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_REQUESTS_PER_MINUTE = int(os.getenv("LLM_REQUESTS_PER_MINUTE", "30"))

# ===== Embedding (chạy local, không cần API) =====
# bge-m3 đa ngữ, hỗ trợ tiếng Việt tốt. Lần đầu chạy tải ~2GB.
# Máy yếu có thể đổi sang "intfloat/multilingual-e5-small" (~470MB) trong .env.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")   # "cuda" nếu có GPU

# ===== Chunking =====
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))          # số ký tự mỗi chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))     # chồng lấn giữa 2 chunk liền nhau

# ===== Retrieval =====
# TOP_K_RETRIEVE để Long dùng cho bước re-ranking (lấy rộng rồi lọc lại).
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "50"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))          # số chunk cuối đưa cho LLM

# ===== Corpus =====
MIN_PAGE_TEXT_LENGTH = 40   # bỏ trang PDF gần trống
"""Cấu hình chung cho 3 pipeline: đọc .env và các hằng số."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ===== LLM (Groq) =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
# Trích xuất thực thể tốn nhiều lượt gọi nhất → có thể đổi sang model nhỏ (VD: llama-3.1-8b-instant) để tiết kiệm quota
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", LLM_MODEL)
LLM_REQUESTS_PER_MINUTE = int(os.getenv("LLM_REQUESTS_PER_MINUTE", "30"))

# ===== Embedding =====
EMBEDDING_MODEL = "BAAI/bge-m3"  # Đa ngữ, hỗ trợ tiếng Việt
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# ===== Neo4j =====
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or None
# "neo4j" (mặc định) hoặc "memory" — backend chạy trong RAM khi không dùng được Neo4j
GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "neo4j")

# ===== Retrieval =====
TOP_K_RETRIEVE = 50  # Số chunks lấy từ FAISS (cho re-ranking)
TOP_K_FINAL = 5      # Số chunks cuối cùng đưa cho LLM
GRAPH_HOPS = 2       # Số bước nhảy trên đồ thị

# ===== GraphRAG — dựng đồ thị (offline) =====
GRAPH_DATA_DIR = PROJECT_ROOT / "data" / "graph"
EXTRACTION_CACHE = GRAPH_DATA_DIR / "extractions.jsonl"
COMMUNITIES_FILE = GRAPH_DATA_DIR / "communities.json"
GRAPH_FILE = GRAPH_DATA_DIR / "graph.json"  # chỉ dùng cho backend "memory"
ENTITY_VECTORS_FILE = GRAPH_DATA_DIR / "entity_vectors.npz"  # cache embedding tên thực thể (nối thực thể)
EXTRACT_UNIT_CHARS = 1500  # Gộp các chunk liền nhau tới ngưỡng này cho 1 lần gọi LLM

# Loại thực thể/quan hệ cố định để đồ thị nhất quán — chỉnh lại khi nhóm chốt lĩnh vực tài liệu
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

LEIDEN_RESOLUTION = 1.0
MIN_COMMUNITY_SIZE = 3

# ===== GraphRAG — truy hồi (online) =====
MAX_SEED_ENTITIES = 5      # Số thực thể trong câu hỏi dùng làm điểm xuất phát
MAX_NODE_DEGREE = 50       # Không đi xuyên qua node có quá nhiều cạnh (hub) khi mở rộng
PATHS_PER_SEED = 20        # Số đường tối đa lấy cho mỗi độ dài, mỗi thực thể xuất phát
MAX_GRAPH_PATHS = 15       # Số đường quan hệ đưa vào ngữ cảnh
MAX_GRAPH_CHUNKS = 5       # Số chunk lấy qua đồ thị (ngoài top-k vector)
MAX_CONTEXT_CHARS = 12000  # Ngân sách ngữ cảnh gửi LLM
ENTITY_LINK_MIN_SIM = 0.75 # Ngưỡng cosine khi nối thực thể bằng embedding
GLOBAL_TOP_COMMUNITIES = 5
GLOBAL_MIN_SCORE = 20      # Bỏ ý trả lời một phần có điểm hữu ích thấp hơn ngưỡng
