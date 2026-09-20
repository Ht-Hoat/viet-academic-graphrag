"""
Pipeline 2 tầng: Bi-Encoder (retrieve) + Cross-Encoder (re-rank).

Luồng chuẩn (theo yêu cầu):
    Bước 1: FAISS retrieve top-50  (bi-encoder, nhanh)
    Bước 2: Cross-encoder chấm điểm 50 cặp -> sort -> top-5
    Bước 3: Top-5 sau re-rank -> ghép thành context -> đưa vào LLM generation

Cross-Encoder được dùng như TẦNG LỌC CUỐI, kết hợp với Bi-Encoder:
    1. Bi-Encoder: truy xuất nhanh top 30–50 documents
    2. Cross-Encoder: re-rank và chọn ra top 3–5 đoạn tốt nhất
    3. Ghép các đoạn này thành context cuối cùng.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Union

from .bi_encoder import BiEncoder
from .config import RerankConfig
from .cross_encoder import CrossEncoderReranker
from .faiss_index import FaissIndex
from .utils import Document, ScoredDocument, timer


@dataclass
class RerankResult:
    """Kết quả đầy đủ của một lượt truy vấn qua pipeline."""
    query: str
    retrieved: List[ScoredDocument]        # top-N từ bi-encoder
    reranked: List[ScoredDocument]         # top-k sau cross-encoder
    context: str                           # context đã ghép cho LLM
    timings_ms: dict = field(default_factory=dict)


class RerankPipeline:
    """Đóng gói toàn bộ: index hoá corpus, retrieve, rerank, dựng context."""

    def __init__(
        self,
        config: Optional[RerankConfig] = None,
        bi_encoder: Optional[BiEncoder] = None,
        cross_encoder: Optional[CrossEncoderReranker] = None,
        index: Optional[FaissIndex] = None,
    ):
        self.config = config or RerankConfig()
        self.bi_encoder = bi_encoder or BiEncoder(config=self.config)
        self.cross_encoder = cross_encoder or CrossEncoderReranker(config=self.config)
        self.index = index

    # ------------------------------------------------------------------ #
    # Lập chỉ mục (offline)
    # ------------------------------------------------------------------ #
    def build_index(
        self,
        documents: Sequence[Union[str, Document]],
        show_progress_bar: bool = True,
    ) -> FaissIndex:
        """Mã hoá toàn bộ corpus bằng bi-encoder rồi nạp vào FAISS."""
        docs: List[Document] = []
        for i, d in enumerate(documents):
            if isinstance(d, Document):
                docs.append(d)
            else:
                docs.append(Document(id=str(i), text=d))

        embeddings = self.bi_encoder.encode_documents(
            docs, show_progress_bar=show_progress_bar
        )
        self.index = FaissIndex(dim=self.bi_encoder.dim, config=self.config)
        self.index.add(embeddings, docs)
        return self.index

    def load_index(self, dir_path: str) -> FaissIndex:
        self.index = FaissIndex.load(dir_path)
        return self.index

    def save_index(self, dir_path: str) -> None:
        if self.index is None:
            raise RuntimeError("Chưa có index để lưu. Gọi build_index trước.")
        self.index.save(dir_path)

    # ------------------------------------------------------------------ #
    # Tầng 1: retrieve
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[ScoredDocument]:
        if self.index is None:
            raise RuntimeError("Chưa có index. Gọi build_index hoặc load_index.")
        top_k = top_k or self.config.retrieve_top_k
        q_emb = self.bi_encoder.encode_queries(query)[0]
        return self.index.search(q_emb, top_k=top_k)

    # ------------------------------------------------------------------ #
    # Tầng 2: re-rank
    # ------------------------------------------------------------------ #
    def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredDocument],
        top_k: Optional[int] = None,
    ) -> List[ScoredDocument]:
        return self.cross_encoder.rerank(query, candidates, top_k=top_k)

    # ------------------------------------------------------------------ #
    # Tầng 3: ghép context
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_context(
        docs: Sequence[ScoredDocument],
        template: str = "[{i}] {text}",
        separator: str = "\n\n",
        max_chars: Optional[int] = None,
    ) -> str:
        """Ghép các đoạn đã re-rank thành một context duy nhất cho LLM.

        max_chars: nếu đặt, cắt bớt để tôn trọng ngân sách ngữ cảnh
        (quan trọng khi so sánh công bằng giữa các hệ RAG).
        """
        parts: List[str] = []
        total = 0
        for i, d in enumerate(docs, start=1):
            piece = template.format(i=i, text=d.text, id=d.id, score=d.score)
            if max_chars is not None and total + len(piece) > max_chars:
                break
            parts.append(piece)
            total += len(piece) + len(separator)
        return separator.join(parts)

    # ------------------------------------------------------------------ #
    # Chạy full pipeline
    # ------------------------------------------------------------------ #
    def run(
        self,
        query: str,
        retrieve_top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        max_context_chars: Optional[int] = None,
        verbose: bool = False,
    ) -> RerankResult:
        """Chạy trọn: retrieve -> rerank -> build context. (Chưa gọi LLM.)"""
        timings: dict = {}

        with timer("retrieve", verbose=verbose) as t1:
            retrieved = self.retrieve(query, top_k=retrieve_top_k)
        timings["retrieve_ms"] = t1["elapsed_ms"]

        with timer("rerank", verbose=verbose) as t2:
            reranked = self.rerank(query, retrieved, top_k=rerank_top_k)
        timings["rerank_ms"] = t2["elapsed_ms"]

        context = self.build_context(reranked, max_chars=max_context_chars)

        return RerankResult(
            query=query,
            retrieved=retrieved,
            reranked=reranked,
            context=context,
            timings_ms=timings,
        )

    # ------------------------------------------------------------------ #
    # Chạy full pipeline + gọi LLM (tuỳ chọn)
    # ------------------------------------------------------------------ #
    def run_with_llm(
        self,
        query: str,
        llm_fn: Callable[[str, str], str],
        prompt_template: Optional[str] = None,
        retrieve_top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        max_context_chars: Optional[int] = None,
        verbose: bool = False,
    ) -> tuple[str, RerankResult]:
        """Như run() nhưng gọi tiếp LLM để sinh câu trả lời.

        llm_fn(prompt, context) -> str : hàm bạn tự cung cấp (OpenAI, vLLM,
        Ollama, transformers... tuỳ ý). Xem examples/04_rag_pipeline.py.
        """
        result = self.run(
            query,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k=rerank_top_k,
            max_context_chars=max_context_chars,
            verbose=verbose,
        )
        if prompt_template is None:
            prompt_template = (
                "Dựa vào NGỮ CẢNH dưới đây, hãy trả lời CÂU HỎI bằng tiếng Việt. "
                "Chỉ dùng thông tin trong ngữ cảnh; nếu không đủ thông tin, hãy nói rõ.\n\n"
                "### NGỮ CẢNH:\n{context}\n\n### CÂU HỎI:\n{query}\n\n### TRẢ LỜI:"
            )
        prompt = prompt_template.format(context=result.context, query=query)
        answer = llm_fn(prompt, result.context)
        return answer, result
