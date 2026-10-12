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
