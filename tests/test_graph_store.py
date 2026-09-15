"""Test hợp đồng của tầng lưu trữ đồ thị, chạy y hệt nhau trên cả hai backend.

- backend "memory" (NetworkX + JSON): luôn chạy.
- backend "neo4j" (Cypher thật): chỉ chạy khi đặt NEO4J_TEST_URI. CẢNH BÁO: test xoá toàn bộ
  Entity/Chunk/Community trong database được trỏ tới, nên đừng trỏ vào DB đang chứa đồ thị thật.

    NEO4J_TEST_URI=bolt://localhost:7687 python -m pytest tests/test_graph_store.py
"""
import json
import os

import pytest

from src import config
from src.graph_builder import aggregate, build_communities, load_communities
from src.graph_rag import GraphRAG, path_to_text
from src.graph_sample import sample_unit_results

TEST_URI = os.getenv("NEO4J_TEST_URI")


@pytest.fixture(scope="module", params=["memory", "neo4j"])
def store(request, tmp_path_factory):
    if request.param == "memory":
        from src.graph_memory_store import InMemoryGraphStore

        graph_store = InMemoryGraphStore(path=tmp_path_factory.mktemp("graph") / "graph.json")
    else:
        if not TEST_URI:
            pytest.skip("Đặt NEO4J_TEST_URI để chạy test trên Neo4j thật")
        from src.graph_store import Neo4jGraphStore

        graph_store = Neo4jGraphStore(uri=TEST_URI)

    graph = aggregate(sample_unit_results())
    graph_store.clear()
    graph_store.write_graph(graph["chunks"], graph["entities"], graph["relations"])
    yield graph_store
    graph_store.close()


def path_texts(paths):
    return [path_to_text(path) for path in paths]


def test_write_graph_ghi_du_node_va_canh(store):
    stats = store.stats()
    assert stats["entities"] == 11 and stats["chunks"] == 5
    assert stats["relations"] == 11 and stats["mentions"] > 0


def test_expand_paths_tra_ve_duong_hai_buoc_day_du(store):
    texts = path_texts(store.expand_paths(["bert"], hops=2, max_degree=50, limit=20))
    assert "Devlin et al. 2018 --(đề xuất)--> BERT" in texts
    # Đường 2 bước giữ node trung gian Devlin et al. 2018 (bản mẫu cũ làm mất node này)
    assert any("Devlin et al. 2018 --(thuộc về)--> Google AI Language" in text and "BERT" in text for text in texts)


def test_expand_paths_khong_di_xuyen_hub(store):
    # Chặn ở max_degree=1 thì không còn đường 2 bước đi xuyên qua Transformer (bậc 4) hay Devlin (bậc 2)
    texts = path_texts(store.expand_paths(["bert"], hops=2, max_degree=1, limit=20))
    assert all("Vaswani" not in text and "Google AI Language" not in text for text in texts)


def test_shortest_paths_noi_hai_thuc_the_trong_cau_hoi(store):
    texts = path_texts(store.shortest_paths([("bert", "gpt")], max_hops=4))
    assert texts and all("BERT" in text and "GPT" in text for text in texts)
    assert any("Transformer" in text or "tiền huấn luyện" in text for text in texts)


def test_search_entities(store):
    assert "google ai language" in [row["key"] for row in store.search_entities("Google", 5)]
    assert store.search_entities("(((", 5) == []  # ký tự đặc biệt không làm vỡ truy vấn


def test_chunks_for_entities_uu_tien_chunk_nhac_nhieu_thuc_the(store):
    rows = store.chunks_for_entities(["devlin et al. 2018", "google ai language"], limit=3)
    assert rows[0]["id"] == "bert-2" and rows[0]["hits"] == 2
    assert store.chunks_by_ids(["bert-1"])[0]["source"] == "bert.pdf"


def test_entities_va_relations_cho_tom_tat_cum(store):
    keys = ["bert", "devlin et al. 2018", "google ai language"]
    assert [row["key"] for row in store.entities_by_keys(keys)][0] == "bert"  # bậc cao nhất đứng trước
    relations = store.relations_among(keys, limit=10)
    assert ("devlin et al. 2018", "thuộc_về", "google ai language") in [
        (r["source"], r["type"], r["target"]) for r in relations]


def test_query_da_buoc_lay_duoc_bang_chung_o_hai_chunk_khac_nhau(store):
    """Ca multi-hop: đáp án nằm rải ở bert-1 (BERT → Devlin) và bert-2 (Devlin → Google AI Language)."""
    result = GraphRAG(store).query("Tác giả của BERT làm việc ở tổ chức nào?", generate=False)

    assert result["seed_entities"] == ["BERT"]
    context = " ".join(result["contexts"])
    assert "Google AI Language" in context and "Devlin et al. 2018" in context
    assert "bert.pdf — trang 2" in result["sources"]
    assert {n["id"] for n in result["subgraph"]["nodes"]} >= {"bert", "devlin et al. 2018", "google ai language"}


def test_query_diem_chung_giua_hai_thuc_the(store):
    result = GraphRAG(store).query("BERT và GPT có điểm gì chung?", generate=False)
    assert {"BERT", "GPT"} <= set(result["seed_entities"])
    assert any("Transformer" in fact or "tiền huấn luyện" in fact for fact in result["graph_paths"])


def test_build_communities_va_global_search(store, tmp_path):
    path = tmp_path / "communities.json"
    communities = build_communities(store, llm=None, path=path)  # llm=None → tóm tắt dự phòng, không tốn quota

    assert communities and all(c["size"] >= config.MIN_COMMUNITY_SIZE for c in communities)
    assert store.stats()["communities"] == len(communities)
    assert load_communities(path) == json.loads(path.read_text(encoding="utf-8"))

    result = GraphRAG(store, communities=communities).query("Các chủ đề chính trong kho tài liệu là gì?",
                                                            generate=False)
    assert result["search_type"] == "Global" and result["contexts"]
