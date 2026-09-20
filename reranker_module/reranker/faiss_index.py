"""
FAISS index — lưu trữ và tìm kiếm vector cho tầng RETRIEVE.

- IndexFlatIP: tích vô hướng (inner product). Với vector đã chuẩn hoá L2,
  tích vô hướng == cosine similarity. Tìm kiếm CHÍNH XÁC (brute-force),
  hợp cho corpus vừa và nhỏ (đến vài trăm nghìn chunk).
- IndexHNSWFlat: đồ thị HNSW, tìm kiếm XẤP XỈ nhưng nhanh hơn nhiều trên
  corpus lớn.

Index được lưu kèm một "docstore" (id + text + metadata) để ánh xạ từ vị
trí vector trở lại nội dung chunk.
"""
from __future__ import annotations

import json
import os
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import RerankConfig
from .utils import Document, ScoredDocument


class FaissIndex:
    def __init__(self, dim: int, config: Optional[RerankConfig] = None):
        import faiss  # import trễ

        self._faiss = faiss
        self.config = config or RerankConfig()
        self.dim = dim
        self.documents: List[Document] = []

        if self.config.faiss_index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, self.config.hnsw_m,
                                        faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
            self.index = index
        else:  # "flat"
            self.index = faiss.IndexFlatIP(dim)

    # ------------------------------------------------------------------ #
    def add(self, embeddings: np.ndarray, documents: Sequence[Document]) -> None:
        """Thêm vector + document tương ứng vào index."""
        if embeddings.shape[0] != len(documents):
            raise ValueError(
                f"Số vector ({embeddings.shape[0]}) khác số document "
                f"({len(documents)})."
            )
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Chiều vector ({embeddings.shape[1]}) khác dim index ({self.dim})."
            )
        self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self.documents.extend(documents)

    # ------------------------------------------------------------------ #
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[ScoredDocument]:
        """Tìm top_k document gần nhất cho MỘT câu hỏi.

        query_embedding: mảng shape (dim,) hoặc (1, dim).
        """
        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        top_k = min(top_k, len(self.documents))
        if top_k == 0:
            return []
        scores, indices = self.index.search(np.ascontiguousarray(q), top_k)
        results: List[ScoredDocument] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS trả -1 khi không đủ kết quả
                continue
            doc = self.documents[idx]
            results.append(
                ScoredDocument(
                    id=doc.id,
                    text=doc.text,
                    score=float(score),
                    metadata=doc.metadata,
                    retrieve_score=float(score),
                )
            )
        return results

    def __len__(self) -> int:
        return len(self.documents)

    # ------------------------------------------------------------------ #
    # Lưu / nạp
    # ------------------------------------------------------------------ #
    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        self._faiss.write_index(self.index, os.path.join(dir_path, "index.faiss"))
        with open(os.path.join(dir_path, "docstore.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        with open(os.path.join(dir_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"dim": self.dim, "config": self.config.to_dict(),
                 "num_docs": len(self.documents)},
                f, ensure_ascii=False, indent=2,
            )

    @classmethod
    def load(cls, dir_path: str) -> "FaissIndex":
        import faiss

        with open(os.path.join(dir_path, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        config = RerankConfig.from_dict(meta.get("config", {}))
        obj = cls.__new__(cls)  # bỏ qua __init__ để nạp index sẵn có
        obj._faiss = faiss
        obj.config = config
        obj.dim = int(meta["dim"])
        obj.index = faiss.read_index(os.path.join(dir_path, "index.faiss"))
        with open(os.path.join(dir_path, "docstore.pkl"), "rb") as f:
            obj.documents = pickle.load(f)
        return obj
