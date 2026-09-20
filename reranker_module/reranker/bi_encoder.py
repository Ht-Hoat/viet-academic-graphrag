"""
Bi-Encoder — tầng RETRIEVE (truy xuất nhanh).

Ý tưởng (theo SBERT Retrieve & Re-Rank):
    - Bi-encoder mã hoá CÂU HỎI và CHUNK một cách ĐỘC LẬP thành 2 vector.
    - Độ liên quan = độ tương đồng cosine giữa 2 vector.
    - Vì chunk được mã hoá TRƯỚC (offline) và lưu vào FAISS, lúc truy vấn
      chỉ cần encode câu hỏi 1 lần rồi tìm hàng xóm gần nhất → RẤT NHANH,
      phù hợp để quét toàn bộ corpus và lấy về top-N ứng viên.

    So sánh: Cross-encoder (xem cross_encoder.py) mã hoá cặp (câu hỏi, chunk)
    CÙNG LÚC nên chính xác hơn nhưng chậm — không thể quét cả corpus, chỉ
    dùng để re-rank lại N ứng viên mà bi-encoder đã lọc.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from .config import RerankConfig
from .utils import Document, resolve_device


class BiEncoder:
    """Bao bọc SentenceTransformer để mã hoá câu hỏi và tài liệu."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        config: Optional[RerankConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or RerankConfig()
        if model_name:
            self.config.bi_encoder_name = model_name
        self.device = resolve_device(device or self.config.device)

        # Import trễ để chỉ cần cài khi thực sự dùng.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            self.config.bi_encoder_name,
            device=self.device,
        )

    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        """Số chiều vector nhúng — cần cho việc khởi tạo FAISS index."""
        d = self.model.get_sentence_embedding_dimension()
        if d is None:  # một số model trả None trước khi encode
            d = int(self.encode_queries(["x"]).shape[1])
        return int(d)

    # ------------------------------------------------------------------ #
    def _encode(
        self,
        texts: Sequence[str],
        prefix: str,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        if prefix:
            texts = [prefix + t for t in texts]
        emb = self.model.encode(
            list(texts),
            batch_size=self.config.embed_batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_queries(
        self,
        queries: Union[str, Sequence[str]],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Mã hoá 1 hoặc nhiều câu hỏi. Trả về mảng (n, dim) float32."""
        if isinstance(queries, str):
            queries = [queries]
        return self._encode(queries, self.config.query_prefix, show_progress_bar)

    def encode_documents(
        self,
        documents: Union[Sequence[str], Sequence[Document]],
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Mã hoá danh sách tài liệu (str hoặc Document). Trả về (n, dim)."""
        if len(documents) and isinstance(documents[0], Document):
            texts = [d.text for d in documents]  # type: ignore[union-attr]
        else:
            texts = list(documents)  # type: ignore[assignment]
        return self._encode(texts, self.config.passage_prefix, show_progress_bar)
