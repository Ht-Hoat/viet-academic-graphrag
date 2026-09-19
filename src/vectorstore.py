"""Kho vector: lưu embedding của các chunk và tìm kiếm cosine.

Hai đường:
  1. FAISS (LangChain) — MẶC ĐỊNH. Index lưu ra data/faiss/ bằng save_local, đúng định dạng mà
     GraphRAG của Hùng nạp lại qua `python -m src.graph_rag ask ... --faiss-dir data/faiss`.
     Đây là "index dùng chung" của cả 3 pipeline.
  2. NumpyVectorStore — DỰ PHÒNG. Khi máy chưa cài faiss/langchain (hoặc chạy test nhanh),
     tự tính cosine bằng numpy. Cùng giao diện `.similarity_search(question, k)` nên phần trên
     không phải biết đang dùng đường nào.

Cả hai trả về object có `.page_content` và `.metadata` (giống LangChain Document) để NaiveRAG
xử lý thống nhất.
"""
import json
import pickle
from pathlib import Path

import numpy as np

from src import config


class Doc:
    """Thay thế nhẹ cho LangChain Document ở đường numpy."""

    __slots__ = ("page_content", "metadata")

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Doc({self.metadata}, {self.page_content[:40]!r}...)"


def chunks_to_documents(chunks: list[dict]):
    """Chuyển chunk dict → LangChain Document (giữ chunk_id/source/page trong metadata)."""
    from langchain_core.documents import Document

    return [
        Document(page_content=c["text"],
                 metadata={"chunk_id": c["chunk_id"], "source": c["source"], "page": c["page"]})
        for c in chunks
    ]


# ===== Đường 1: FAISS (LangChain) =====
def build_faiss_index(chunks: list[dict], embeddings):
    """Dựng FAISS index từ chunks + embeddings (dùng chung cho cả 3 pipeline)."""
    from langchain_community.vectorstores import FAISS

    docs = chunks_to_documents(chunks)
    vs = FAISS.from_documents(docs, embeddings)
    print(f"Đã lập chỉ mục FAISS với {len(docs)} vectors")
    return vs


def save_faiss_index(vs, path=config.FAISS_DIR):
    Path(path).mkdir(parents=True, exist_ok=True)
    vs.save_local(str(path))
    print(f"Đã lưu FAISS index: {path}")


def load_faiss_index(embeddings, path=config.FAISS_DIR):
    from langchain_community.vectorstores import FAISS

    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)


# ===== Đường 2: Numpy (dự phòng) =====
class NumpyVectorStore:
    """Cosine search thuần numpy. Đủ nhanh cho vài chục nghìn chunk — quy mô của đề tài."""

    def __init__(self, embeddings, chunks: list[dict], vectors: np.ndarray = None):
        self.embeddings = embeddings
        self.chunks = chunks
        if vectors is None:
            vectors = np.asarray(embeddings.embed_documents([c["text"] for c in chunks]), dtype="float32")
        self.vectors = self._normalize(vectors)

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype="float32")
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

    def similarity_search(self, question: str, k: int = 5) -> list[Doc]:
        q = np.asarray(self.embeddings.embed_query(question), dtype="float32")
        q /= np.linalg.norm(q) + 1e-9
        sims = self.vectors @ q                       # cosine vì đã chuẩn hóa
        idxs = np.argsort(-sims)[:k]
        return [Doc(self.chunks[i]["text"],
                    {"chunk_id": self.chunks[i]["chunk_id"], "source": self.chunks[i]["source"],
                     "page": self.chunks[i]["page"], "score": float(sims[i])})
                for i in idxs]

    def save(self, path=config.FAISS_DIR):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "numpy_vectors.npy", self.vectors)
        (path / "numpy_chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")
        print(f"Đã lưu numpy store: {path}")

    @classmethod
    def load(cls, embeddings, path=config.FAISS_DIR):
        path = Path(path)
        vectors = np.load(path / "numpy_vectors.npy")
        chunks = json.loads((path / "numpy_chunks.json").read_text(encoding="utf-8"))
        return cls(embeddings, chunks, vectors=vectors)


# ===== Factory: tự chọn đường phù hợp =====
def build_index(chunks: list[dict], embeddings, prefer_faiss: bool = True):
    """Thử FAISS trước; thiếu faiss/langchain thì rơi về numpy. Trả (store, backend_name)."""
    if prefer_faiss:
        try:
            return build_faiss_index(chunks, embeddings), "faiss"
        except ImportError:
            print("  [!] Chưa có faiss/langchain — dùng NumpyVectorStore dự phòng.")
    return NumpyVectorStore(embeddings, chunks), "numpy"
