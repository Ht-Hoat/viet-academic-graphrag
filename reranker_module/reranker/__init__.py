"""
reranker — module re-ranking 2 tầng (Bi-Encoder + Cross-Encoder) cho RAG.

Ví dụ nhanh:
    from reranker import RerankPipeline, RerankConfig

    cfg = RerankConfig(retrieve_top_k=50, rerank_top_k=5)
    pipe = RerankPipeline(config=cfg)
    pipe.build_index(["đoạn 1...", "đoạn 2...", ...])
    result = pipe.run("câu hỏi của bạn")
    print(result.context)
"""
from .config import RerankConfig, DEFAULT_BI_ENCODER, DEFAULT_CROSS_ENCODER
from .bi_encoder import BiEncoder
from .cross_encoder import CrossEncoderReranker
from .faiss_index import FaissIndex
from .pipeline import RerankPipeline, RerankResult
from .utils import Document, ScoredDocument, load_corpus, chunk_text

__all__ = [
    "RerankConfig",
    "DEFAULT_BI_ENCODER",
    "DEFAULT_CROSS_ENCODER",
    "BiEncoder",
    "CrossEncoderReranker",
    "FaissIndex",
    "RerankPipeline",
    "RerankResult",
    "Document",
    "ScoredDocument",
    "load_corpus",
    "chunk_text",
]

__version__ = "1.0.0"
