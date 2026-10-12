"""Test lớp tích hợp: chuẩn hóa kết quả, adapter, mock, factory (không cần backend thật)."""
from src.pipelines import (PipelineAdapter, availability, load_pipelines,
                           normalize_result)


def test_normalize_coerces_dict_sources_to_str():
    raw = {"answer": "a", "sources": [{"source": "bert.pdf", "page": 4}, "x.pdf — trang 1"],
           "contexts": "một đoạn", "latency": 1.2}
    res = normalize_result(raw, "Naive RAG")
    assert res["sources"][0] == "bert.pdf — trang 4"
    assert res["sources"][1] == "x.pdf — trang 1"
    assert res["contexts"] == ["một đoạn"]        # str -> list[str]
    assert res["method"] == "Naive RAG"


def test_normalize_fills_defaults():
    res = normalize_result({}, "GraphRAG")
    assert res["answer"] == "" and res["sources"] == [] and res["contexts"] == []
    assert res["search_type"] is None and res["graph_paths"] == []


def test_adapter_sets_latency_and_mock_flag():
    pipe = PipelineAdapter("X", lambda q: {"answer": "ok"}, is_mock=True)
    res = pipe.query("hỏi gì đó")
    assert res["is_mock"] is True
    assert res["latency"] >= 0.0
    assert res["method"] == "X"


def test_adapter_callable():
    pipe = PipelineAdapter("X", lambda q: {"answer": q})
    assert pipe("abc")["answer"] == "abc"


def test_load_pipelines_falls_back_to_mock_when_no_backend():
    # Trong folder độc lập chưa có src.naive_rag... → mọi pipeline phải là mock
    pipes = load_pipelines(with_rerank=True, with_graph=True)
    assert set(pipes) == {"Naive RAG", "RAG + Re-ranking", "GraphRAG"}
    assert all(p.is_mock for p in pipes.values())
    assert availability(pipes) == {"Naive RAG": False, "RAG + Re-ranking": False, "GraphRAG": False}


def test_providers_injection_takes_priority():
    fake = lambda q: {"answer": "injected", "method": "Naive RAG"}
    pipes = load_pipelines(with_rerank=False, with_graph=False, providers={"Naive RAG": fake})
    assert set(pipes) == {"Naive RAG"}
    assert not pipes["Naive RAG"].is_mock
    assert pipes["Naive RAG"].query("x")["answer"] == "injected"


def test_graphrag_mock_detects_global():
    pipes = load_pipelines()
    g = pipes["GraphRAG"]
    assert g.query("Xu hướng nghiên cứu RAG hiện nay?")["search_type"] == "Global"
    assert g.query("BERT là gì?")["search_type"] == "Local"
