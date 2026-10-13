"""Lớp tích hợp — đưa 3 pipeline của nhóm về MỘT giao diện chung cho UI và evaluator.

Vấn đề: ba thành viên trả kết quả theo kiểu hơi khác nhau:
  • Hoạt (Naive RAG):    query_naive_rag(question) -> dict, sources là list[dict]
  • Long  (Re-ranking):  query_rerank_rag(question) -> dict
  • Hùng  (GraphRAG):    build_graph_rag(...).query(question) -> dict, có thêm search_type/graph_paths

SV4 gói tất cả vào `PipelineAdapter` với một hàm `.query(question)` trả về CÙNG một "hình dạng",
nhờ đó UI và evaluator không phải biết bên trong là pipeline nào.

Khi chạy độc lập (folder này tách khỏi repo chung), các import `src.naive_rag`... chưa có nên
adapter tự rơi về pipeline MOCK (dữ liệu giả, có nhãn) để UI/eval vẫn chạy được khi phát triển.
Sau khi merge vào repo chung, các import thật thành công và adapter dùng pipeline thật.
"""
from __future__ import annotations

import random
import time
from typing import Callable

from src import config

# ===== Hình dạng kết quả chuẩn =====
# {answer:str, sources:list[str], contexts:list[str], latency:float, method:str,
#  search_type:str|None, graph_paths:list[str]}


def _coerce_source(src) -> str:
    """sources có thể là str hoặc dict {source, page} (Naive RAG của Hoạt) — về chung dạng str."""
    if isinstance(src, dict):
        page = src.get("page")
        label = str(src.get("source", "?"))
        return f"{label} — trang {page}" if page not in (None, "", "N/A") else label
    return str(src)


def normalize_result(raw: dict, method: str) -> dict:
    """Chuẩn hóa dict thô của pipeline về hình dạng chung, điền mặc định cho trường thiếu."""
    raw = raw if isinstance(raw, dict) else {}
    contexts = raw.get("contexts") or []
    if isinstance(contexts, str):
        contexts = [contexts]
    return {
        "answer": str(raw.get("answer", "")),
        "sources": [_coerce_source(s) for s in (raw.get("sources") or [])],
        "contexts": [str(c) for c in contexts],
        "latency": float(raw.get("latency", 0.0) or 0.0),
        "method": raw.get("method", method),
        "search_type": raw.get("search_type"),          # chỉ GraphRAG có
        "graph_paths": raw.get("graph_paths") or [],     # chỉ GraphRAG có
    }


class PipelineAdapter:
    """Bọc một hàm query(question)->dict thành pipeline chuẩn, kèm cờ trạng thái."""

    def __init__(self, name: str, query_fn: Callable[[str], dict],
                 available: bool = True, is_mock: bool = False):
        self.name = name
        self._query_fn = query_fn
        self.available = available
        self.is_mock = is_mock

    def __call__(self, question: str) -> dict:
        return self.query(question)

    def query(self, question: str) -> dict:
        start = time.time()
        raw = self._query_fn(question)
        res = normalize_result(raw, self.name)
        if not res["latency"]:
            res["latency"] = round(time.time() - start, 2)
        res["is_mock"] = self.is_mock
        return res


# ===== Nạp pipeline THẬT (chỉ chạy được sau khi merge vào repo chung) =====
def _load_naive():
    from src.naive_rag import query_naive_rag  # noqa: PLC0415

    return query_naive_rag


def _load_rerank():
    from src.rerank_rag import query_rerank_rag  # noqa: PLC0415

    return query_rerank_rag


def _load_graph():
    """Ưu tiên interface class của Hùng (build_graph_rag().query); nếu không có thì thử query_graph_rag."""
    try:
        from src.graph_rag import build_graph_rag  # noqa: PLC0415
        from src.naive_rag import get_embeddings, get_global_vectorstore  # noqa: PLC0415

        rag = build_graph_rag(vectorstore=get_global_vectorstore(), embedder=get_embeddings())
        return rag.query
    except ImportError:
        from src.graph_rag import query_graph_rag  # noqa: PLC0415

        return query_graph_rag


_REAL_LOADERS = {
    "Naive RAG": _load_naive,
    "RAG + Re-ranking": _load_rerank,
    "GraphRAG": _load_graph,
}


# ===== Pipeline MOCK (khi chạy độc lập / thiếu backend) =====
_MOCK_SOURCES = [
    "Devlin_2018_BERT.pdf — trang 4", "Vaswani_2017_Attention.pdf — trang 2",
    "Edge_2024_GraphRAG.pdf — trang 7", "Lewis_2020_RAG.pdf — trang 3",
]
_GLOBAL_HINTS = ("xu hướng", "tổng quan", "nhìn chung", "toàn bộ", "hướng nghiên cứu")


def _mock_factory(method: str) -> Callable[[str], dict]:
    base_latency = {"Naive RAG": 0.6, "RAG + Re-ranking": 1.3, "GraphRAG": 1.9}[method]

    def _query(question: str) -> dict:
        rng = random.Random(hash((method, question)) & 0xFFFF)
        latency = round(base_latency + rng.uniform(-0.2, 0.4), 2)
        k = {"Naive RAG": 1, "RAG + Re-ranking": 2, "GraphRAG": 3}[method]
        res = {
            "answer": f"[MOCK · {method}] Câu trả lời giả lập cho: '{question}'.",
            "sources": rng.sample(_MOCK_SOURCES, k),
            "contexts": [f"[MOCK] đoạn ngữ cảnh {i + 1} cho {method}" for i in range(k)],
            "latency": latency,
            "method": method,
        }
        if method == "GraphRAG":
            is_global = any(h in question.lower() for h in _GLOBAL_HINTS)
            res["search_type"] = "Global" if is_global else "Local"
            res["graph_paths"] = ["[MOCK] BERT --(được đề xuất bởi)--> Devlin --(thuộc)--> Google"]
        return res

    return _query


# ===== Factory chính =====
def load_pipelines(with_rerank: bool = True, with_graph: bool = True,
                   providers: dict[str, Callable[[str], dict]] | None = None) -> dict[str, PipelineAdapter]:
    """Trả {tên: PipelineAdapter} cho các phương pháp cần dùng.

    Với mỗi phương pháp: dùng `providers` nếu được truyền (để test tiêm pipeline giả), rồi thử nạp
    bản thật, thất bại thì rơi về mock. UI đọc cờ `.available`/`.is_mock` để báo trạng thái backend.
    """
    wanted = ["Naive RAG"]
    if with_rerank:
        wanted.append("RAG + Re-ranking")
    if with_graph:
        wanted.append("GraphRAG")

    providers = providers or {}
    pipes: dict[str, PipelineAdapter] = {}
    for name in wanted:
        if name in providers:
            pipes[name] = PipelineAdapter(name, providers[name], available=True, is_mock=False)
            continue
        try:
            query_fn = _REAL_LOADERS[name]()
            pipes[name] = PipelineAdapter(name, query_fn, available=True, is_mock=False)
        except Exception:  # noqa: BLE001 — chưa merge / thiếu key / lỗi backend → mock
            pipes[name] = PipelineAdapter(name, _mock_factory(name), available=False, is_mock=True)
    return pipes


def availability(pipes: dict[str, PipelineAdapter]) -> dict[str, bool]:
    return {name: (p.available and not p.is_mock) for name, p in pipes.items()}
