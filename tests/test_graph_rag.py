"""Test luồng online: nối thực thể, xếp hạng đường đi, gộp ngữ cảnh, local/global search.

Dùng store giả (dữ liệu dựng sẵn) nên không cần Neo4j; phần Cypher được kiểm ở test_neo4j_integration.py.
"""
import pytest

from src.graph_builder import Chunk
from src.graph_rag import (GraphRAG, EntityLinker, evidence_hit, heuristic_type, merge_chunks, path_to_text,
                           paths_to_subgraph, question_type_hints, rank_paths)

ENTITIES = [
    {"key": "bert", "name": "BERT", "type": "METHOD", "aliases": ["BERT"], "degree": 4},
    {"key": "devlin et al. 2018", "name": "Devlin et al. 2018", "type": "PAPER",
     "aliases": ["Devlin et al. 2018"], "degree": 2},
    {"key": "google ai language", "name": "Google AI Language", "type": "ORG",
     "aliases": ["Google AI Language"], "degree": 1},
    {"key": "gpt", "name": "GPT", "type": "METHOD", "aliases": ["GPT"], "degree": 2},
    {"key": "transformer", "name": "Transformer", "type": "METHOD", "aliases": ["Transformer"], "degree": 3},
]
CHUNKS = {
    "bert-1": {"id": "bert-1", "text": "BERT do Devlin et al. 2018 đề xuất.", "source": "data/pdfs/bert.pdf", "page": 1},
    "bert-2": {"id": "bert-2", "text": "Devlin et al. 2018 làm việc tại Google AI Language.",
               "source": "data/pdfs/bert.pdf", "page": 2},
    "gpt-1": {"id": "gpt-1", "text": "GPT dựa trên Transformer.", "source": "data/pdfs/gpt.pdf", "page": 1},
}


def node(key):
    return {k: v for k, v in next(e for e in ENTITIES if e["key"] == key).items() if k in ("key", "name", "type")}


def rel(source, rel_type, target, chunk_ids=(), weight=1):
    return {"type": rel_type, "source": source, "target": target, "weight": weight, "chunk_ids": list(chunk_ids)}


PATH_BERT_DEVLIN = {"nodes": [node("bert"), node("devlin et al. 2018")],
                    "rels": [rel("devlin et al. 2018", "đề_xuất", "bert", ["bert-1"])]}
PATH_BERT_GOOGLE = {"nodes": [node("bert"), node("devlin et al. 2018"), node("google ai language")],
                    "rels": [rel("devlin et al. 2018", "đề_xuất", "bert", ["bert-1"]),
                             rel("devlin et al. 2018", "thuộc_về", "google ai language", ["bert-2"])]}
PATH_BERT_TRANSFORMER = {"nodes": [node("bert"), node("transformer")],
                         "rels": [rel("bert", "sử_dụng", "transformer", ["bert-1"], weight=3)]}
PATH_BERT_GPT = {"nodes": [node("bert"), node("transformer"), node("gpt")],
                 "rels": [rel("bert", "sử_dụng", "transformer", ["bert-1"]),
                          rel("gpt", "sử_dụng", "transformer", ["gpt-1"])]}


class FakeStore:
    def __init__(self, paths=(), shortest=(), fulltext=()):
        self.paths = list(paths)
        self.shortest = list(shortest)
        self.fulltext = list(fulltext)
        self.chunks_for_entities_calls = 0

    def entity_index(self):
        return [dict(entity) for entity in ENTITIES]

    def search_entities(self, text, limit=5):
        return self.fulltext[:limit]

    def expand_paths(self, keys, hops, max_degree, limit):
        return [p for p in self.paths if p["nodes"][0]["key"] in keys]

    def shortest_paths(self, pairs, max_hops, limit=3):
        return self.shortest

    def chunks_for_entities(self, keys, limit):
        self.chunks_for_entities_calls += 1
        return [dict(CHUNKS[cid], hits=1) for cid in list(CHUNKS)[:limit]]

    def chunks_by_ids(self, ids):
        return [dict(CHUNKS[cid]) for cid in ids if cid in CHUNKS]


class FakeLLM:
    """Trả output theo dấu hiệu trong prompt; đếm số lần gọi để kiểm tra số lượt LLM."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        for marker, response in self.responses.items():
            if marker in prompt:
                return type("Response", (), {"content": response})
        return type("Response", (), {"content": ""})


def test_path_to_text_giu_node_trung_gian_va_dung_chieu():
    text = path_to_text(PATH_BERT_GOOGLE)
    assert text == ("Devlin et al. 2018 --(đề xuất)--> BERT; "
                    "Devlin et al. 2018 --(thuộc về)--> Google AI Language")
    # Node trung gian không bị mất như code mẫu cũ (chỉ lấy quan hệ đầu tiên)
    assert "Devlin et al. 2018" in text and "Google AI Language" in text


def test_rank_paths_uu_tien_duong_noi_hai_thuc_the_cua_cau_hoi():
    ranked = rank_paths([PATH_BERT_TRANSFORMER, PATH_BERT_GPT], {"bert", "gpt"}, set())
    assert ranked[0] is PATH_BERT_GPT


def test_rank_paths_uu_tien_loai_thuc_the_cau_hoi_nham_toi():
    hints = question_type_hints("Tác giả của BERT làm việc ở tổ chức nào?")
    assert hints == {"PERSON", "ORG"}
    ranked = rank_paths([PATH_BERT_DEVLIN, PATH_BERT_TRANSFORMER, PATH_BERT_GOOGLE], {"bert"}, hints)
    assert ranked[0] is PATH_BERT_GOOGLE


def test_rank_paths_bo_duong_trung_nhau():
    assert len(rank_paths([PATH_BERT_GOOGLE, dict(PATH_BERT_GOOGLE)], {"bert"}, set())) == 1


def test_heuristic_type():
    assert heuristic_type("Xu hướng nghiên cứu chính là gì?") == "GLOBAL"
    assert heuristic_type("BERT là gì?") == "LOCAL"
    # "tổng quan" nằm trong một từ khác thì không tính
    assert heuristic_type("Mô tả kiến trúc BERT") == "LOCAL"


def test_merge_chunks_xen_ke_bo_trung_va_theo_ngan_sach():
    vector = [Chunk("v1", "a" * 10, "a.pdf", 1), Chunk("v2", "b" * 10, "a.pdf", 2)]
    graph = [Chunk("v1", "a" * 10, "a.pdf", 1), Chunk("g1", "c" * 10, "b.pdf", 1)]
    # Xen kẽ vector/graph, chunk trùng id chỉ lấy một lần
    assert [c.id for c in merge_chunks(vector, graph, max_chars=35)] == ["v1", "v2", "g1"]
    # Hết ngân sách thì bỏ chunk phía sau
    assert [c.id for c in merge_chunks(vector, graph, max_chars=25)] == ["v1", "v2"]
    # Chunk đầu luôn được giữ dù dài hơn ngân sách
    assert [c.id for c in merge_chunks(vector, graph, max_chars=5)] == ["v1"]


def test_entity_linker_khop_ten_va_quet_ten_trong_cau_hoi():
    linker = EntityLinker(FakeStore())
    # LLM chỉ nêu BERT, nhưng GPT xuất hiện nguyên văn trong câu hỏi nên vẫn được lấy làm điểm xuất phát
    seeds = linker.link("BERT và GPT khác nhau thế nào?", ["mô hình BERT"])
    assert [s["key"] for s in seeds] == ["bert", "gpt"]


def test_entity_linker_khong_khop_nham_ben_trong_tu_khac():
    linker = EntityLinker(FakeStore())
    assert linker.link("RoBERTa là gì?", []) == []


def test_entity_linker_dung_fulltext_khi_ten_khong_trung_khop():
    store = FakeStore(fulltext=[{"key": "google ai language", "name": "Google AI Language", "type": "ORG", "score": 2}])
    seeds = EntityLinker(store).link("Tổ chức Google AI có gì?", ["Google AI"])
    assert [s["key"] for s in seeds] == ["google ai language"]


def test_local_search_tra_ve_contexts_va_nguon_de_ragas_dung_duoc():
    store = FakeStore(paths=[PATH_BERT_DEVLIN, PATH_BERT_GOOGLE])
    llm = FakeLLM({"Phân tích câu hỏi": '{"type": "LOCAL", "entities": ["BERT"]}',
                   "Dựa vào ngữ cảnh": "Devlin et al. 2018 thuộc Google AI Language (bert.pdf — trang 2)."})
    result = GraphRAG(store, llm).query("Tác giả của BERT làm việc ở tổ chức nào?")

    assert result["search_type"] == "Local"
    assert result["seed_entities"] == ["BERT"]
    # contexts không được rỗng — evaluator RAGAS đọc trường này
    assert result["contexts"] and any("Google AI Language" in c for c in result["contexts"])
    assert result["graph_paths"][0].startswith("Devlin et al. 2018 --(đề xuất)--> BERT")
    assert "bert.pdf — trang 2" in result["sources"]
    assert result["sources"][-1].startswith("Đồ thị tri thức")
    assert result["method"] == "GraphRAG" and result["latency"] >= 0
    # 1 lần gọi LLM để phân tích câu hỏi (gộp phân loại + thực thể) + 1 lần sinh câu trả lời
    assert len(llm.calls) == 2


def test_local_search_khong_tim_thay_thuc_the_van_chay_duoc():
    result = GraphRAG(FakeStore(), FakeLLM({"Dựa vào ngữ cảnh": "Tôi không tìm thấy thông tin liên quan."})).query(
        "Một câu hỏi không liên quan tới đồ thị")
    assert result["seed_entities"] == [] and result["graph_paths"] == []
    assert result["answer"].startswith("Tôi không tìm thấy")


def test_analyze_llm_tra_rac_thi_dung_luat_tu_khoa():
    rag = GraphRAG(FakeStore(), FakeLLM({"Phân tích câu hỏi": "xin lỗi"}))
    assert rag.analyze("Xu hướng nghiên cứu chính là gì?") == {"type": "GLOBAL", "entities": []}


def test_retrieval_only_khong_goi_llm_sinh_cau_tra_loi():
    store = FakeStore(paths=[PATH_BERT_GOOGLE])
    rag = GraphRAG(store, llm=None)
    result = rag.query("Tác giả của BERT làm việc ở tổ chức nào?", generate=False)
    assert result["answer"] == "" and result["contexts"]


def test_paths_to_subgraph_danh_dau_thuc_the_xuat_phat():
    subgraph = paths_to_subgraph([PATH_BERT_GOOGLE], {"bert"})
    assert {n["id"] for n in subgraph["nodes"]} == {"bert", "devlin et al. 2018", "google ai language"}
    assert next(n for n in subgraph["nodes"] if n["id"] == "bert")["seed"] is True
    assert len(subgraph["edges"]) == 2


COMMUNITIES = [
    {"id": 0, "title": "Mô hình ngôn ngữ tiền huấn luyện", "summary": "BERT và GPT dựa trên Transformer.",
     "size": 5, "members": [{"key": "bert", "name": "BERT", "type": "METHOD"}],
     "relations": [], "sources": ["bert.pdf — trang 1"]},
    {"id": 1, "title": "Thị giác máy tính", "summary": "CNN cho phân loại ảnh.", "size": 3,
     "members": [{"key": "cnn", "name": "CNN", "type": "METHOD"}], "relations": [], "sources": []},
]


def test_global_search_map_reduce_chi_giu_cum_lien_quan():
    llm = FakeLLM({
        "Phân tích câu hỏi": '{"type": "GLOBAL", "entities": []}',
        "Mô hình ngôn ngữ tiền huấn luyện": '{"answer": "Chủ đề chính là mô hình tiền huấn luyện.", "score": 90}',
        "Thị giác máy tính": '{"answer": "Không liên quan.", "score": 5}',
        "Hãy tổng hợp": "Chủ đề chính là các mô hình ngôn ngữ tiền huấn luyện [C0].",
    })
    result = GraphRAG(FakeStore(), llm, communities=COMMUNITIES).query("Các chủ đề chính trong kho tài liệu là gì?")

    assert result["search_type"] == "Global"
    assert result["answer"].endswith("[C0].")
    # Cụm bị chấm điểm thấp bị loại khỏi nguồn và ngữ cảnh
    assert result["contexts"] == ["Mô hình ngôn ngữ tiền huấn luyện: BERT và GPT dựa trên Transformer."]
    assert result["sources"][0] == "Cụm tri thức C0: Mô hình ngôn ngữ tiền huấn luyện"


def test_global_search_khong_cum_nao_lien_quan_thi_quay_ve_local():
    llm = FakeLLM({"Phân tích câu hỏi": '{"type": "GLOBAL", "entities": []}',
                   "Chỉ dựa vào báo cáo": '{"answer": "", "score": 0}',
                   "Dựa vào ngữ cảnh": "Trả lời từ local search."})
    result = GraphRAG(FakeStore(paths=[PATH_BERT_DEVLIN]), llm, communities=COMMUNITIES).query(
        "Tổng quan về BERT trong kho tài liệu?")
    assert result["search_type"] == "Local"


def test_global_search_chua_co_community_thi_quay_ve_local():
    llm = FakeLLM({"Phân tích câu hỏi": '{"type": "GLOBAL", "entities": []}', "Dựa vào ngữ cảnh": "..."})
    assert GraphRAG(FakeStore(), llm).query("Xu hướng nghiên cứu chính?")["search_type"] == "Local"


def test_rank_communities_khong_co_embedder_thi_dem_tu_chung():
    rag = GraphRAG(FakeStore(), communities=COMMUNITIES)
    assert rag.rank_communities("BERT và Transformer", top_k=1)[0]["id"] == 0
    assert rag.rank_communities("phân loại ảnh bằng CNN", top_k=1)[0]["id"] == 1


def test_evidence_hit():
    result = {"contexts": ["Devlin et al. 2018 làm việc tại Google AI Language."]}
    assert evidence_hit(result, ["Devlin et al. 2018", "Google AI Language"]) == 1.0
    assert evidence_hit(result, ["OpenAI", "Google AI Language"]) == 0.5
    assert evidence_hit(result, None) is None
