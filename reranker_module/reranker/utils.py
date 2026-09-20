"""
Tiện ích dùng chung: đo thời gian, sigmoid, đọc/ghi JSONL, chia chunk.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Cấu trúc dữ liệu kết quả
# ---------------------------------------------------------------------------
@dataclass
class Document:
    """Một đoạn (chunk) trong corpus."""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredDocument:
    """Document kèm điểm số từ retrieve hoặc rerank."""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Lưu vết điểm ở từng tầng để debug/phân tích.
    retrieve_score: Optional[float] = None
    rerank_score: Optional[float] = None

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"ScoredDocument(id={self.id!r}, score={self.score:.4f}, text={preview!r}...)"


# ---------------------------------------------------------------------------
# Toán học
# ---------------------------------------------------------------------------
def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Ánh xạ logit -> xác suất trong (0, 1). Ổn định số học."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


# ---------------------------------------------------------------------------
# Đo thời gian
# ---------------------------------------------------------------------------
@contextmanager
def timer(name: str = "block", verbose: bool = True) -> Iterator[Dict[str, float]]:
    """Context manager đo thời gian một khối lệnh.

    with timer("retrieve") as t:
        ...
    print(t["elapsed_ms"])
    """
    result: Dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = (time.perf_counter() - start) * 1000.0
        result["elapsed_ms"] = elapsed
        if verbose:
            print(f"[timer] {name}: {elapsed:.1f} ms")


# ---------------------------------------------------------------------------
# Đọc/ghi JSONL
# ---------------------------------------------------------------------------
def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_corpus(path: str) -> List[Document]:
    """Đọc corpus dạng JSONL. Mỗi dòng: {"id": ..., "text": ..., "metadata": {...}}.

    Nếu thiếu "id" sẽ tự sinh theo số thứ tự dòng.
    """
    docs: List[Document] = []
    for i, row in enumerate(read_jsonl(path)):
        docs.append(
            Document(
                id=str(row.get("id", i)),
                text=row["text"],
                metadata=row.get("metadata", {}),
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Chia chunk đơn giản (theo số từ, có overlap)
# ---------------------------------------------------------------------------
def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 40,
) -> List[str]:
    """Chia văn bản thành các đoạn ~chunk_size từ, chồng lấn `overlap` từ.

    Đây là bộ chia tối giản để demo. Trong thực tế nên dùng bộ chia theo
    câu/đoạn (ví dụ underthesea cho tiếng Việt) để giữ ranh giới ngữ nghĩa.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


def resolve_device(device: Optional[str]) -> str:
    """Chọn thiết bị: ưu tiên tham số truyền vào, ngược lại cuda nếu có."""
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
