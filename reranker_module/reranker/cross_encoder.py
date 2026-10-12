"""
Cross-Encoder — tầng RE-RANK (chấm điểm lại, chính xác).

Khác biệt cốt lõi với Bi-Encoder:
    - Bi-Encoder: encode(câu hỏi) và encode(chunk) RIÊNG BIỆT -> 2 vector ->
      so cosine. Nhanh vì chunk đã encode sẵn, nhưng mô hình không "nhìn"
      hai văn bản cùng lúc nên kém tinh.
    - Cross-Encoder: đưa CẶP (câu hỏi, chunk) vào CÙNG một lần forward của
      Transformer. Cơ chế self-attention cho phép từng token của câu hỏi
      "nhìn thẳng" vào từng token của chunk -> điểm liên quan chính xác hơn
      nhiều. Cái giá: phải chạy 1 lần forward cho MỖI cặp -> chậm, không
      quét được cả corpus. Vì thế chỉ dùng để re-rank N ứng viên.

Module hỗ trợ 2 backend:
    1. "sentence-transformers" (mặc định): dùng class CrossEncoder.
       Hợp cho cả cross-encoder/ms-marco-* và BAAI/bge-reranker-v2-m3.
    2. "flag": dùng FlagEmbedding.FlagReranker (chỉ cho họ BAAI/bge-reranker-*).
       Có sẵn tham số normalize (sigmoid) và use_fp16.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import RerankConfig
from .utils import ScoredDocument, resolve_device, sigmoid


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: Optional[str] = None,
        config: Optional[RerankConfig] = None,
        device: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        self.config = config or RerankConfig()
        if model_name:
            self.config.cross_encoder_name = model_name
        if backend:
            self.config.cross_encoder_backend = backend
        self.device = resolve_device(device or self.config.device)
        self.backend = self.config.cross_encoder_backend

        if self.backend == "sentence-transformers":
            self._init_sbert()
        elif self.backend == "flag":
            self._init_flag()
        else:
            raise ValueError(f"Backend không hợp lệ: {self.backend!r}")

    # ------------------------------------------------------------------ #
    def _init_sbert(self) -> None:
        from sentence_transformers import CrossEncoder

        kwargs = {"device": self.device}
        # Tên tham số độ dài tối đa đổi giữa các phiên bản sentence-transformers
        # (max_length -> max_seq_length ở v5.4+). Thử lần lượt cho tương thích.
        model = None
        for key in ("max_length", "max_seq_length"):
            try:
                model = CrossEncoder(
                    self.config.cross_encoder_name,
                    **{key: self.config.cross_encoder_max_length},
                    **kwargs,
                )
                break
            except TypeError:
                continue
        if model is None:
            model = CrossEncoder(self.config.cross_encoder_name, **kwargs)
        self.model = model

    def _init_flag(self) -> None:
        from FlagEmbedding import FlagReranker

        self.model = FlagReranker(
            self.config.cross_encoder_name,
            use_fp16=self.config.use_fp16,
        )

    # ------------------------------------------------------------------ #
    def score_pairs(
        self,
        query: str,
        passages: Sequence[str],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Chấm điểm liên quan cho từng cặp (query, passage).

        Trả về mảng điểm float (n,). Nếu normalize_scores=True, điểm đã qua
        sigmoid nên nằm trong (0,1); ngược lại là logit thô (khoảng -10..10).
        """
        if not passages:
            return np.array([], dtype=np.float32)

        pairs = [[query, p] for p in passages]

        if self.backend == "sentence-transformers":
            scores = self.model.predict(
                pairs,
                batch_size=self.config.rerank_batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
            )
            scores = np.asarray(scores, dtype=np.float64).reshape(-1)
            if self.config.normalize_scores:
                # bge-reranker xuất logit 1 lớp -> sigmoid ra [0,1].
                # ms-marco-MiniLM cũng xuất logit -> sigmoid hợp lý.
                # (Nếu model đã có sigmoid ở default_activation thì giá trị
                #  vốn đã ~[0,1]; sigmoid lần nữa chỉ nén nhẹ, thứ hạng KHÔNG đổi.)
                scores = sigmoid(scores)
        else:  # flag
            scores = self.model.compute_score(
                pairs,
                normalize=self.config.normalize_scores,
            )
            scores = np.asarray(scores, dtype=np.float64).reshape(-1)

        return scores.astype(np.float32)

    # ------------------------------------------------------------------ #
    def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredDocument],
        top_k: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> List[ScoredDocument]:
        """Re-rank danh sách ứng viên (từ bi-encoder) và trả top_k tốt nhất.

        Điểm cross-encoder ghi vào `.rerank_score` và trở thành `.score`
        chính; điểm retrieve gốc vẫn giữ ở `.retrieve_score` để đối chiếu.
        """
        if not candidates:
            return []
        top_k = top_k or self.config.rerank_top_k

        passages = [c.text for c in candidates]
        scores = self.score_pairs(query, passages, show_progress_bar)

        reranked: List[ScoredDocument] = []
        for cand, s in zip(candidates, scores):
            reranked.append(
                ScoredDocument(
                    id=cand.id,
                    text=cand.text,
                    score=float(s),
                    metadata=cand.metadata,
                    retrieve_score=cand.retrieve_score,
                    rerank_score=float(s),
                )
            )
        reranked.sort(key=lambda d: d.score, reverse=True)
        return reranked[:top_k]
