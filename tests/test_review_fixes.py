"""Test hồi quy cho các lỗi được chỉ ra trong đợt review code."""
import unicodedata

import pytest

from src import config
from src.graph_builder import aggregate, build_communities, clean_name, to_chunk
from src.graph_memory_store import InMemoryGraphStore
from src.graph_rag import GraphRAG, evidence_hit
from src.graph_sample import sample_unit_results
from tests.test_graph_rag import PATH_BERT_DEVLIN, PATH_BERT_GOOGLE, FakeLLM, FakeStore, rel


def nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def test_chunk_text_duoc_chuan_hoa_nfc_nen_van_do_duoc_ten_thuc_the():
    """Text NFD từ PDF: trước đây không khớp tên thực thể (đã NFC) nên nguồn bị gán cho mọi chunk trong đơn vị."""
    unit = [to_chunk({"chunk_id": "c1", "text": nfd("Tiền huấn luyện là kỹ thuật của BERT."),
                      "source": "a.pdf", "page": 1}),
            to_chunk({"chunk_id": "c2", "text": "Một đoạn không liên quan.", "source": "a.pdf", "page": 2})]
    raw = {"entities": [{"name": "Tiền huấn luyện", "type": "CONCEPT"}, {"name": "BERT", "type": "METHOD"}],
           "relations": [{"source": "BERT", "relation": "sử_dụng", "target": "Tiền huấn luyện"}]}

    graph = aggregate([(unit, raw)])
    entities = {e["key"]: e for e in graph["entities"]}
    assert entities["tiền huấn luyện"]["chunk_ids"] == ["c1"]
    assert graph["relations"][0]["chunk_ids"] == ["c1"]


def test_chunk_id_khong_doi_theo_cach_go_dau():
    a = to_chunk({"text": nfd("Tiền huấn luyện"), "source": "a.pdf", "page": 1})
    b = to_chunk({"text": "Tiền huấn luyện", "source": "a.pdf", "page": 1})
    assert a.id == b.id and a.text == b.text


def test_clean_name_bo_het_tien_to_chong_nhau():
    assert clean_name("phương pháp Mô hình BERT") == "BERT"
    assert clean_name(clean_name("phương pháp Mô hình BERT")) == "BERT"  # đã ổn định


def test_evidence_hit_khop_du_text_go_dau_kieu_khac():
    result = {"contexts": [nfd("BERT dùng Tiền huấn luyện.")]}
    assert evidence_hit(result, ["Tiền huấn luyện"]) == 1.0


def test_contexts_tach_tung_duong_quan_he_cho_ragas():
    store = FakeStore(paths=[PATH_BERT_DEVLIN, PATH_BERT_GOOGLE])
    result = GraphRAG(store).query("Tác giả của BERT làm việc ở tổ chức nào?", generate=False)
    graph_contexts = [c for c in result["contexts"] if c.startswith("Quan hệ trong đồ thị tri thức:")]
    assert len(graph_contexts) == len(result["graph_paths"]) >= 2


def test_khong_truy_van_them_khi_da_du_chunk_bang_chung():
    path = {"nodes": PATH_BERT_DEVLIN["nodes"],
            "rels": [rel("devlin et al. 2018", "đề_xuất", "bert", [f"c{i}" for i in range(config.MAX_GRAPH_CHUNKS)])]}
    store = FakeStore(paths=[path])
    GraphRAG(store).query("BERT do ai đề xuất?", generate=False)
    assert store.chunks_for_entities_calls == 0

    store = FakeStore(paths=[PATH_BERT_DEVLIN])  # chỉ 1 chunk bằng chứng → vẫn cần truy vấn bổ sung
    GraphRAG(store).query("BERT do ai đề xuất?", generate=False)
    assert store.chunks_for_entities_calls == 1


def test_ten_thuc_the_chua_ai_khong_bi_stopword_nuot():
    from src.graph_rag import _tokens

    assert "ai" in _tokens("Google AI Language")


COMMUNITIES = [{"id": 0, "title": "Mô hình ngôn ngữ", "summary": "BERT và GPT.", "size": 4,
                "members": [{"key": "bert", "name": "BERT", "type": "METHOD"}], "relations": [], "sources": []}]


def test_global_search_moi_lan_goi_map_deu_loi_thi_bao_loi():
    """Trước đây lỗi bị nuốt và câu GLOBAL âm thầm thành Local — bảng so sánh sẽ ghi nhầm."""
    class BrokenLLM:
        def invoke(self, prompt):
            if "Phân tích câu hỏi" in prompt:
                return type("Response", (), {"content": '{"type": "GLOBAL", "entities": []}'})
            raise RuntimeError("429 rate limit")

    rag = GraphRAG(FakeStore(), BrokenLLM(), communities=COMMUNITIES)
    with pytest.raises(RuntimeError, match="bước map"):
        rag.query("Các chủ đề chính trong kho tài liệu là gì?")


def test_global_search_ghi_ro_da_qua_buoc_loc_map_hay_chua():
    llm = FakeLLM({"Phân tích câu hỏi": '{"type": "GLOBAL", "entities": []}',
                   "Chỉ dựa vào báo cáo": '{"answer": "Có.", "score": 80}',
                   "Hãy tổng hợp": "Tổng hợp [C0]."})
    rag = GraphRAG(FakeStore(), llm, communities=COMMUNITIES)
    assert rag.query("Các chủ đề chính?")["map_filtered"] is True
    assert rag.query("Các chủ đề chính?", generate=False)["map_filtered"] is False


def test_quay_ve_local_van_giu_thoi_gian_chon_cum():
    llm = FakeLLM({"Phân tích câu hỏi": '{"type": "GLOBAL", "entities": []}',
                   "Chỉ dựa vào báo cáo": '{"answer": "", "score": 0}',
                   "Dựa vào ngữ cảnh": "Trả lời từ local."})
    result = GraphRAG(FakeStore(paths=[PATH_BERT_DEVLIN]), llm, communities=COMMUNITIES).query("Tổng quan về BERT?")
    assert result["search_type"] == "Local"
    assert {"communities", "graph"} <= set(result["timings"])


def test_build_tiep_khong_lam_mat_nguon_cu(tmp_path):
    """Backend RAM phải giữ ngữ nghĩa MERGE của Neo4j khi chạy build --no-reset nhiều lần."""
    store = InMemoryGraphStore(path=tmp_path / "graph.json")
    store.write_graph([{"id": "c1", "text": "x", "source": "a.pdf", "page": 1}],
                      [{"key": "bert", "name": "BERT", "type": "METHOD", "aliases": ["BERT"],
                        "mentions": 1, "chunk_ids": ["c1"]}], [])
    store.write_graph([{"id": "c2", "text": "y", "source": "a.pdf", "page": 2}],
                      [{"key": "bert", "name": "BERT", "type": "METHOD", "aliases": ["BERT"],
                        "mentions": 1, "chunk_ids": ["c2"]}], [])
    assert store.entities["bert"]["chunk_ids"] == ["c1", "c2"]


def test_tom_tat_du_phong_khong_chan_lan_chay_lai_co_llm(tmp_path):
    """Chạy --no-llm trước rồi chạy lại có key: tóm tắt phải được tạo lại, không dùng bản dự phòng đã lưu."""
    store = InMemoryGraphStore(path=tmp_path / "graph.json")
    graph = aggregate(sample_unit_results())
    store.write_graph(graph["chunks"], graph["entities"], graph["relations"])
    path = tmp_path / "communities.json"

    build_communities(store, llm=None, path=path)
    llm = FakeLLM({"Bạn đang viết báo cáo": '{"title": "Tiêu đề thật", "summary": "Tóm tắt từ LLM."}'})
    communities = build_communities(store, llm=llm, path=path)

    assert len(llm.calls) == len(communities)
    assert all(c["title"] == "Tiêu đề thật" and c["generated_by"] == "llm" for c in communities)

    # Lần chạy sau, nội dung cụm không đổi → dùng lại tóm tắt của LLM, không gọi thêm
    build_communities(store, llm=llm, path=path)
    assert len(llm.calls) == len(communities)
