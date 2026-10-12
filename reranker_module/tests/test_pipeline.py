"""
Smoke test không cần tải model nặng: kiểm tra logic index/rerank bằng
các thành phần GIẢ (fake) để chạy nhanh trong CI.

    pytest tests/ -v
hoặc chạy trực tiếp:
    python tests/test_pipeline.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reranker import RerankConfig, FaissIndex
from reranker.utils import Document, ScoredDocument, sigmoid, chunk_text
from reranker.pipeline import RerankPipeline


# --------------------------------------------------------------------------- #
# Fake components (không tải model)
# --------------------------------------------------------------------------- #
class FakeBiEncoder:
    """Bi-encoder giả: nhúng = one-hot theo từ khoá để test xác định."""
    def __init__(self, vocab):
        self.vocab = vocab
        self._dim = len(vocab)

    @property
    def dim(self):
        return self._dim

    def _vec(self, text):
        v = np.zeros(self._dim, dtype=np.float32)
        for i, w in enumerate(self.vocab):
            if w in text.lower():
                v[i] = 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode_queries(self, queries, show_progress_bar=False):
        if isinstance(queries, str):
            queries = [queries]
        return np.stack([self._vec(q) for q in queries])

    def encode_documents(self, docs, show_progress_bar=False):
        texts = [d.text if isinstance(d, Document) else d for d in docs]
        return np.stack([self._vec(t) for t in texts])


class FakeCrossEncoder:
    """Cross-encoder giả: điểm = độ trùng từ giữa query và passage."""
    def __init__(self, config=None):
        self.config = config or RerankConfig()

    def rerank(self, query, candidates, top_k=None):
        top_k = top_k or self.config.rerank_top_k
        qs = set(query.lower().split())
        out = []
        for c in candidates:
            overlap = len(qs & set(c.text.lower().split()))
            out.append(ScoredDocument(id=c.id, text=c.text, score=float(overlap),
                                      metadata=c.metadata,
                                      retrieve_score=c.retrieve_score,
                                      rerank_score=float(overlap)))
        out.sort(key=lambda d: d.score, reverse=True)
        return out[:top_k]


# --------------------------------------------------------------------------- #
def test_sigmoid():
    assert abs(float(sigmoid(0.0)) - 0.5) < 1e-9
    assert float(sigmoid(100.0)) > 0.99
    assert float(sigmoid(-100.0)) < 0.01
    print("✓ test_sigmoid")


def test_chunk_text():
    text = " ".join(str(i) for i in range(500))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 100 for c in chunks)
    print("✓ test_chunk_text")


def test_faiss_index():
    cfg = RerankConfig(faiss_index_type="flat")
    idx = FaissIndex(dim=3, config=cfg)
    emb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    docs = [Document(id=str(i), text=f"doc {i}") for i in range(3)]
    idx.add(emb, docs)
    res = idx.search(np.array([1, 0, 0], dtype=np.float32), top_k=2)
    assert res[0].id == "0"
    assert len(res) == 2
    print("✓ test_faiss_index")


def test_faiss_save_load(tmp_path=None):
    import tempfile
    d = tmp_path or Path(tempfile.mkdtemp())
    cfg = RerankConfig()
    idx = FaissIndex(dim=2, config=cfg)
    idx.add(np.array([[1, 0], [0, 1]], dtype=np.float32),
            [Document(id="a", text="x"), Document(id="b", text="y")])
    idx.save(str(d))
    idx2 = FaissIndex.load(str(d))
    assert len(idx2) == 2
    res = idx2.search(np.array([1, 0], dtype=np.float32), top_k=1)
    assert res[0].id == "a"
    print("✓ test_faiss_save_load")


def test_pipeline_end_to_end():
    vocab = ["cross", "bi", "encoder", "faiss", "rerank", "llm", "graph", "vector"]
    cfg = RerankConfig(retrieve_top_k=5, rerank_top_k=2)
    pipe = RerankPipeline(
        config=cfg,
        bi_encoder=FakeBiEncoder(vocab),
        cross_encoder=FakeCrossEncoder(cfg),
    )
    docs = [
        Document(id="0", text="cross encoder rerank chính xác"),
        Document(id="1", text="bi encoder faiss vector nhanh"),
        Document(id="2", text="graph tri thức nhiều bước"),
        Document(id="3", text="llm sinh câu trả lời"),
    ]
    pipe.build_index(docs)
    result = pipe.run("cross encoder rerank", verbose=False)
    assert len(result.reranked) == 2
    assert result.reranked[0].id == "0"   # trùng nhiều từ nhất
    assert "[1]" in result.context
    print("✓ test_pipeline_end_to_end")


if __name__ == "__main__":
    test_sigmoid()
    test_chunk_text()
    test_faiss_index()
    test_faiss_save_load()
    test_pipeline_end_to_end()
    print("\nTất cả smoke test đã PASS.")
