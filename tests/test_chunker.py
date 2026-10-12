"""Test chia chunk + tính ổn định của chunk_id (điểm khớp với GraphRAG của Hùng)."""
import hashlib
import unicodedata

from src.chunker import (chunk_documents, chunk_id_for, load_chunks_jsonl,
                         save_chunks_jsonl, _split_recursive)


def test_short_text_stays_one_chunk():
    assert _split_recursive("ngắn", size=512, overlap=80) == ["ngắn"]


def test_long_text_splits_with_overlap():
    text = ". ".join(f"Câu số {i} khá dài để vượt ngưỡng cắt" for i in range(60))
    chunks = _split_recursive(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 * 1.5 for c in chunks)


def test_chunk_id_matches_hung_scheme():
    # Phải TRÙNG công thức của src.graph_builder.chunk_id_for bên Hùng:
    # sha1(f"{source}|{page}|{NFC(text)}")[:16]
    text, source, page = "BERT dùng pre-training", "01.txt", 1
    expected = hashlib.sha1(
        f"{source}|{page}|{unicodedata.normalize('NFC', text)}".encode("utf-8")
    ).hexdigest()[:16]
    assert chunk_id_for(text, source, page) == expected


def test_chunk_id_stable_across_dau_encoding():
    nfc = unicodedata.normalize("NFC", "ăn")
    nfd = unicodedata.normalize("NFD", "ăn")
    assert chunk_id_for(nfc, "s", 1) == chunk_id_for(nfd, "s", 1)


def test_chunk_documents_shape():
    docs = [{"text": "a" * 1500, "source": "s.txt", "page": 1}]
    chunks = chunk_documents(docs, size=300, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert set(c) == {"chunk_id", "text", "source", "page"}


def test_jsonl_roundtrip(tmp_path):
    chunks = chunk_documents([{"text": "nội dung mẫu tiếng Việt", "source": "s.txt", "page": 2}])
    path = tmp_path / "chunks.jsonl"
    save_chunks_jsonl(chunks, path)
    assert load_chunks_jsonl(path) == chunks
