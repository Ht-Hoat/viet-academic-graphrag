"""Chia tài liệu thành các đoạn nhỏ (chunk) để nhúng vector.

Chiến lược "recursive splitting": cắt theo ranh giới tự nhiên (đoạn → câu → từ) trước,
cố gắng không cắt giữa câu, và giữ phần chồng lấn (overlap) để không mất ngữ cảnh ở ranh giới.

QUAN TRỌNG — chunk_id ổn định: id của mỗi chunk băm từ (source|page|text) đã chuẩn hóa NFC,
TRÙNG scheme của module GraphRAG (Hùng). Nhờ đó chunk lấy qua vector search và chunk lấy qua
đồ thị tri thức khử trùng được với nhau khi so sánh 3 pipeline.
"""
import hashlib
import json
import unicodedata
from pathlib import Path

from src import config

# Ranh giới cắt, từ "thô" đến "mịn"
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " "]


def chunk_id_for(text: str, source, page) -> str:
    """Id ổn định theo nội dung. Băm trên text đã NFC nên không đổi theo cách gõ dấu.

    Phải trùng công thức với src.graph_builder.chunk_id_for của Hùng để 2 pipeline
    tính ra CÙNG id cho CÙNG một chunk.
    """
    norm = unicodedata.normalize("NFC", text)
    return hashlib.sha1(f"{source}|{page}|{norm}".encode("utf-8")).hexdigest()[:16]


def _split_recursive(text: str, size: int, overlap: int) -> list[str]:
    """Cắt text theo thứ tự ưu tiên: đoạn → câu → từ, giữ overlap ký tự giữa các chunk."""
    if len(text) <= size:
        return [text]

    # Chọn separator thô nhất còn tách được
    sep = next((s for s in _SEPARATORS if s in text), " ")
    parts = text.split(sep)

    chunks, current = [], ""
    for part in parts:
        piece = part + sep
        if len(current) + len(piece) <= size:
            current += piece
        else:
            if current:
                chunks.append(current.strip())
            tail = current[-overlap:] if overlap and current else ""   # mang ngữ cảnh sang chunk mới
            current = tail + piece
    if current.strip():
        chunks.append(current.strip())

    # Câu siêu dài vượt size*1.5 → cắt cứng theo ký tự (an toàn cuối cùng)
    final = []
    for c in chunks:
        if len(c) <= size * 1.5:
            final.append(c)
        else:
            step = max(1, size - overlap)
            final.extend(c[i:i + size] for i in range(0, len(c), step))
    return [c for c in final if c.strip()]


def chunk_documents(documents: list[dict],
                    size: int = None, overlap: int = None) -> list[dict]:
    """Nhận list trang {text, source, page}, trả về list chunk:
    {chunk_id, text, source, page}.
    """
    size = size or config.CHUNK_SIZE
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP
    chunks = []
    for doc in documents:
        for piece in _split_recursive(doc["text"], size, overlap):
            chunks.append({
                "chunk_id": chunk_id_for(piece, doc["source"], doc["page"]),
                "text": piece,
                "source": doc["source"],
                "page": doc["page"],
            })
    print(f"Tạo {len(chunks)} chunks từ {len(documents)} tài liệu")
    return chunks


# ===== Trao đổi file với module GraphRAG (Hùng) =====
def save_chunks_jsonl(chunks: list[dict], path=config.CHUNKS_FILE):
    """Ghi chunks ra JSONL — mỗi dòng một chunk. Đây là định dạng Hùng nhận qua:
        python -m src.graph_builder build --chunks data/processed/chunks.jsonl
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Đã ghi {len(chunks)} chunks vào {path}")


def load_chunks_jsonl(path=config.CHUNKS_FILE) -> list[dict]:
    """Đọc lại chunks từ JSONL."""
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
