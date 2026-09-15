"""Test luồng offline: chuẩn hoá tên, làm sạch output LLM, gộp đồ thị, cache và phân cụm Leiden."""
import json

import pytest

from src.graph_builder import (ExtractionCache, aggregate, clean_extraction, clean_name, detect_communities,
                               entity_key, extract_units, group_chunks, normalize_relation, normalize_type, to_chunk)
from src.graph_sample import sample_unit_results
from src.llm_utils import parse_json


class FakeLLM:
    """Trả lần lượt các output đã dựng sẵn và đếm số lần bị gọi."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return type("Response", (), {"content": self.outputs.pop(0)})


def test_clean_name_bo_tien_to_truoc_ten_rieng():
    assert clean_name("  mô hình   BERT ") == "BERT"
    assert clean_name("kiến trúc Transformer") == "Transformer"
    assert clean_name('"Devlin et al. 2018".') == "Devlin et al. 2018"
    # Không cắt khi phần còn lại không phải tên riêng
    assert clean_name("mô hình ngôn ngữ") == "mô hình ngôn ngữ"
    assert clean_name(None) == ""


def test_entity_key_gop_khac_hoa_thuong_va_tien_to():
    assert entity_key("mô hình BERT") == entity_key("BERT") == "bert"
    assert entity_key("Tiền huấn luyện") == entity_key("tiền huấn luyện")


def test_normalize_type_va_relation():
    assert normalize_type("ORGANIZATION") == "ORG"
    assert normalize_type("mơ hồ") == "CONCEPT"
    assert normalize_relation("dựa_trên") == ("sử_dụng", False)
    assert normalize_relation("nhãn lạ") == ("liên_quan_đến", False)
    # Nhãn bị động → đảo chiều
    assert normalize_relation("được_đề_xuất_bởi") == ("đề_xuất", True)


def test_clean_extraction_bo_muc_hong_va_tu_them_thuc_the():
    raw = {
        "entities": [{"name": "BERT", "type": "MODEL"}, {"name": None}, "hỏng"],
        "relations": [
            {"source": "BERT", "relation": "được_đề_xuất_bởi", "target": "Devlin et al. 2018"},
            {"source": "BERT", "relation": "sử_dụng", "target": "BERT"},  # tự trỏ vào mình
            {"source": "", "relation": "sử_dụng", "target": "X"},
            "không phải dict",
        ],
    }
    result = clean_extraction(raw)
    names = {e["name"]: e for e in result["entities"]}
    assert names["BERT"]["type"] == "METHOD"
    # Thực thể chỉ xuất hiện trong quan hệ vẫn được thêm vào (nếu không sẽ mất cạnh)
    assert names["Devlin et al. 2018"]["inferred"] is True
    assert result["relations"] == [{"source": "Devlin et al. 2018", "relation": "đề_xuất", "target": "BERT"}]
    assert result["dropped_relations"] == 3


def test_aggregate_gop_thuc_the_va_gan_nguon_chunk():
    graph = aggregate(sample_unit_results())
    entities = {e["key"]: e for e in graph["entities"]}
    relations = {(r["source"], r["type"], r["target"]): r for r in graph["relations"]}

    # "kiến trúc Transformer" và "Transformer" là một thực thể
    assert "transformer" in entities
    assert set(entities["transformer"]["aliases"]) == {"Transformer"}
    assert entities["tiền huấn luyện"]["mentions"] == 2
    # Nhãn bị động đã được đảo về đúng chiều
    assert ("devlin et al. 2018", "đề_xuất", "bert") in relations
    # Cạnh mang theo chunk nguồn để trích dẫn được
    assert relations[("devlin et al. 2018", "thuộc_về", "google ai language")]["chunk_ids"] == ["bert-2"]
    assert entities["google ai language"]["chunk_ids"] == ["bert-2"]
    assert graph["stats"]["dropped_relations"] == 0


def test_group_chunks_gop_theo_file_va_ngan_sach_ky_tu():
    chunks = [to_chunk({"text": "a" * 100, "source": "1.pdf", "page": i}) for i in range(3)]
    chunks += [to_chunk({"text": "b" * 100, "source": "2.pdf", "page": 0})]
    units = group_chunks(chunks, max_chars=250)
    assert [len(u) for u in units] == [2, 1, 1]
    assert {c.source for c in units[0]} == {"1.pdf"}


def test_extract_units_dung_cache_va_khong_goi_lai_llm(tmp_path):
    units = [[to_chunk({"text": "BERT do Devlin đề xuất.", "source": "a.pdf", "page": 1})]]
    llm = FakeLLM(['Đây là kết quả: {"entities": [{"name": "BERT", "type": "METHOD"}], "relations": []} xong'])
    cache_path = tmp_path / "extractions.jsonl"

    first = extract_units(units, llm, ExtractionCache(cache_path), "test-model")
    assert first[0][1]["entities"][0]["name"] == "BERT"
    assert llm.calls == 1

    # Lần chạy sau đọc từ cache trên đĩa → không tốn thêm quota
    second = extract_units(units, llm, ExtractionCache(cache_path), "test-model")
    assert second[0][1] == first[0][1]
    assert llm.calls == 1
    assert len(cache_path.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_extract_units_output_khong_phai_json_thi_ghi_nhan_loi(tmp_path):
    units = [[to_chunk({"text": "abc", "source": "a.pdf", "page": 1})]]
    llm = FakeLLM(["xin lỗi, tôi không thể"])
    results = extract_units(units, llm, ExtractionCache(tmp_path / "c.jsonl"), "test-model")
    assert results[0][1] == {"error": "invalid_json"}
    assert aggregate(results)["stats"]["failed_units"] == 1


def test_parse_json_lay_duoc_json_trong_fence_va_van_thua():
    assert parse_json('```json\n{"a": 1}\n```')["a"] == 1
    assert parse_json('Kết quả [1] là: {"a": {"b": 2}} hết.')["a"]["b"] == 2
    assert parse_json("không có json") is None


def test_detect_communities_tach_hai_cum_va_bo_cum_nho():
    keys = [f"n{i}" for i in range(7)]
    edges = [("n0", "n1", 5), ("n1", "n2", 5), ("n2", "n0", 5),   # cụm 1
             ("n3", "n4", 5), ("n4", "n5", 5), ("n5", "n3", 5),   # cụm 2
             ("n2", "n3", 1)]                                      # cầu nối yếu
    assignment = detect_communities(keys, edges, min_size=3)
    assert assignment["n0"] == assignment["n1"] == assignment["n2"]
    assert assignment["n3"] == assignment["n4"] == assignment["n5"]
    assert assignment["n0"] != assignment["n3"]
    assert "n6" not in assignment  # node lẻ không thuộc cụm nào


def test_detect_communities_do_thi_rong():
    assert detect_communities([], []) == {}
    assert detect_communities(["a", "b"], []) == {}
