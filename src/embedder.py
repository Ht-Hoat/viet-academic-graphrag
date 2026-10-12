"""Nhúng văn bản thành vector (chạy local, không cần API).

Trả về đối tượng embeddings theo giao diện LangChain (`embed_documents`, `embed_query`) để
FAISS index sinh ra ở đây dùng lại được nguyên vẹn cho re-ranking (Long) và GraphRAG (Hùng) —
cả ba pipeline chia sẻ CÙNG một embedding và CÙNG một index, nên so sánh mới công bằng.

Vector được chuẩn hóa L2 (độ dài = 1) để tích vô hướng chính là cosine similarity.
"""
import hashlib

import numpy as np

from src import config

_embeddings_instance = None


def get_embeddings(model_name: str = None, device: str = None):
    """Tải bge-m3 qua LangChain HuggingFaceEmbeddings, cache 1 lần (singleton).

    Lần đầu gọi sẽ tải model ~2GB. Máy yếu đổi EMBEDDING_MODEL sang model nhỏ hơn trong .env.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        name = model_name or config.EMBEDDING_MODEL
        print(f"Đang nạp embedding model: {name} ...")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs={"device": device or config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},   # để dot product = cosine
        )
    return _embeddings_instance


class FakeEmbeddings:
    """Embedding giả, xác định (deterministic) — KHÔNG tải model.

    Dùng cho test và smoke test: cùng một text luôn cho cùng một vector, text khác cho vector
    khác, và vector đã chuẩn hóa L2. Cài đúng giao diện LangChain Embeddings nên cắm được thẳng
    vào FAISS.from_documents / VectorStore mà không cần internet hay 2GB model.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        # Gieo RNG từ hash của text → vector cố định theo nội dung
        seed = int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim)
        v /= np.linalg.norm(v) + 1e-9
        return v.astype("float32").tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
