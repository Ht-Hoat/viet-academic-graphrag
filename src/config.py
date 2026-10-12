"""Cấu hình cho module Demo & Đánh giá (Tuấn Anh — SV4).

SV4 là người tích hợp: nối Naive RAG (Hoạt), Re-ranking (Long) và GraphRAG (Hùng) vào một UI
và một khung đánh giá RAGAS chung. File này giữ các hằng số dùng chung của khâu đánh giá + UI.
"""
import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# ===== Đường dẫn =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = DATA_DIR / "eval"
EVAL_QUESTIONS_PATH = EVAL_DIR / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "results"
SAMPLE_RESULTS_DIR = RESULTS_DIR / "sample"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ===== Ba phương pháp so sánh =====
# Thứ tự cố định để mọi bảng/biểu đồ nhất quán. Tên là "khóa" chung của cả nhóm.
METHODS = ["Naive RAG", "RAG + Re-ranking", "GraphRAG"]

# Màu cố định cho từng phương pháp (dùng cả matplotlib lẫn UI)
METHOD_COLORS = {
    "Naive RAG": "#94a3b8",          # xám
    "RAG + Re-ranking": "#60a5fa",   # xanh dương
    "GraphRAG": "#22c55e",           # xanh lá
}

# ===== 4 chỉ số RAGAS =====
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_LABELS_VI = {
    "faithfulness": "Faithfulness (Độ trung thực)",
    "answer_relevancy": "Answer Relevancy (Đúng câu hỏi)",
    "context_precision": "Context Precision (Ngữ cảnh chính xác)",
    "context_recall": "Context Recall (Ngữ cảnh đầy đủ)",
}

# Loại câu hỏi — trục phân tích chính của đề tài (giả thuyết: GraphRAG thắng ở multi-hop/global)
QUESTION_TYPES = ["single-hop", "multi-hop", "global"]

# Tên file kết quả cho mỗi phương pháp (đọc/ghi ở results/)
def result_filename(method: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")
    return f"eval_{slug}.json"
    # "Naive RAG" -> eval_naive_rag.json
    # "RAG + Re-ranking" -> eval_rag_re_ranking.json
    # "GraphRAG" -> eval_graphrag.json


# ===== LLM (RAGAS cần LLM làm giám khảo) =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
