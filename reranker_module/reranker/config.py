"""
Cấu hình cho re-ranking module.

Tất cả tham số của pipeline (bi-encoder, cross-encoder, FAISS, số lượng
top-k ở mỗi tầng) được gom về một dataclass duy nhất để dễ tái lập
(reproducibility) và dễ nạp từ file YAML/JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


# ---------------------------------------------------------------------------
# Các model gợi ý sẵn (có thể ghi đè trong config)
# ---------------------------------------------------------------------------

# Bi-encoder (dùng cho tầng RETRIEVE — nhanh):
#   - "BAAI/bge-m3"                       : đa ngữ, tiếng Việt tốt, ghép đôi
#                                           tự nhiên với bge-reranker-v2-m3.
#   - "intfloat/multilingual-e5-base"     : đa ngữ, cần prefix "query:"/"passage:".
#   - "keepitreal/vietnamese-sbert"       : chuyên tiếng Việt, nhẹ.
DEFAULT_BI_ENCODER = "BAAI/bge-m3"

# Cross-encoder (dùng cho tầng RE-RANK — chính xác):
#   - "cross-encoder/ms-marco-MiniLM-L-12-v2" : nhanh, tốt, nhưng chủ yếu tiếng Anh.
#   - "BAAI/bge-reranker-v2-m3"               : đa ngữ, tiếng Việt tốt hơn (khuyến nghị).
DEFAULT_CROSS_ENCODER = "BAAI/bge-reranker-v2-m3"


@dataclass
class RerankConfig:
    # ---- Bi-encoder / retrieve ----
    bi_encoder_name: str = DEFAULT_BI_ENCODER
    # Prefix cho model họ E5 (ví dụ intfloat/multilingual-e5-*).
    # Với bge-m3 / vietnamese-sbert để rỗng "".
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize_embeddings: bool = True     # bắt buộc True nếu dùng IndexFlatIP (cosine)
    embed_batch_size: int = 64

    # ---- Cross-encoder / re-rank ----
    cross_encoder_name: str = DEFAULT_CROSS_ENCODER
    cross_encoder_backend: str = "sentence-transformers"  # hoặc "flag"
    cross_encoder_max_length: int = 512
    rerank_batch_size: int = 32
    # Chuẩn hoá điểm cross-encoder về [0,1] bằng sigmoid (logit -> xác suất).
    normalize_scores: bool = True
    use_fp16: bool = False   # tăng tốc trên GPU (chút hao độ chính xác)

    # ---- Tham số 2 tầng của pipeline ----
    retrieve_top_k: int = 50   # tầng 1: bi-encoder lấy top-50
    rerank_top_k: int = 5      # tầng 2: cross-encoder chọn top-5

    # ---- FAISS ----
    faiss_index_type: str = "flat"   # "flat" (chính xác) hoặc "hnsw" (nhanh, xấp xỉ)
    hnsw_m: int = 32                 # số neighbor của HNSW (chỉ dùng khi type="hnsw")

    # ---- Khác ----
    device: Optional[str] = None     # None = tự chọn (cuda nếu có, ngược lại cpu)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "RerankConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> "RerankConfig":
        # Bỏ qua các khoá lạ để an toàn khi đọc config cũ.
        known = {f_.name for f_ in cls.__dataclass_fields__.values()}  # type: ignore
        return cls(**{k: v for k, v in data.items() if k in known})
