"""Test NumpyVectorStore + FakeEmbeddings (không cần faiss, langchain hay model 2GB)."""
from src.embedder import FakeEmbeddings
from src.vectorstore import NumpyVectorStore


def _chunks():
    return [
        {"chunk_id": "a", "text": "BERT là mô hình ngôn ngữ hai chiều", "source": "01.txt", "page": 1},
        {"chunk_id": "b", "text": "FAISS là thư viện tìm kiếm vector của Meta", "source": "03.txt", "page": 1},
        {"chunk_id": "c", "text": "RAGAS đánh giá hệ thống RAG tự động", "source": "05.txt", "page": 1},
    ]


def test_search_returns_self_first():
    chunks = _chunks()
    emb = FakeEmbeddings(dim=32)
    vs = NumpyVectorStore(emb, chunks)
    # Truy vấn đúng bằng text của chunk 'b' → chunk 'b' phải đứng đầu (cosine với chính nó = 1)
    hits = vs.similarity_search(chunks[1]["text"], k=3)
    assert hits[0].metadata["chunk_id"] == "b"
    assert hits[0].metadata["score"] > 0.99


def test_search_respects_k():
    vs = NumpyVectorStore(FakeEmbeddings(), _chunks())
    assert len(vs.similarity_search("bất kỳ", k=2)) == 2


def test_doc_has_langchain_shape():
    vs = NumpyVectorStore(FakeEmbeddings(), _chunks())
    doc = vs.similarity_search("BERT", k=1)[0]
    assert hasattr(doc, "page_content") and hasattr(doc, "metadata")
    assert {"chunk_id", "source", "page"} <= set(doc.metadata)


def test_save_load_roundtrip(tmp_path):
    chunks = _chunks()
    emb = FakeEmbeddings(dim=32)
    NumpyVectorStore(emb, chunks).save(tmp_path)
    vs2 = NumpyVectorStore.load(emb, tmp_path)
    assert len(vs2.chunks) == 3
    assert vs2.similarity_search(chunks[0]["text"], k=1)[0].metadata["chunk_id"] == "a"
