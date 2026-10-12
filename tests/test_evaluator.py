"""Test khung đánh giá: thu record, gộp tổng thể + theo loại câu, lưu/đọc JSON (không cần ragas)."""
import math

from src import config
from src.evaluator import (aggregate, collect_records, evaluate_pipeline,
                           load_all_results, load_eval_questions, save_results)


def _questions():
    return [
        {"question": "q1", "ground_truth": "g1", "type": "single-hop"},
        {"question": "q2", "ground_truth": "g2", "type": "multi-hop"},
        {"question": "q3", "ground_truth": "g3", "type": "global"},
    ]


def _pipeline(q):
    return {"answer": f"trả lời {q}", "contexts": [f"ctx {q}"], "latency": 1.0}


def _fixed_scorer(records):
    # điểm cố định để kiểm tra phép gộp
    return [{m: 0.8 for m in config.RAGAS_METRICS} for _ in records]


def test_collect_records_shape():
    recs = collect_records(_pipeline, _questions())
    assert len(recs) == 3
    assert recs[0]["ground_truth"] == "g1" and recs[0]["type"] == "single-hop"


def test_collect_records_survives_pipeline_error():
    def boom(q):
        raise ValueError("hỏng")
    recs = collect_records(boom, _questions())
    assert all("error" in r for r in recs)


def test_aggregate_overall_and_by_type():
    recs = collect_records(_pipeline, _questions())
    agg = aggregate(recs, _fixed_scorer(recs))
    assert agg["overall"]["faithfulness"] == 0.8
    assert agg["overall"]["latency"] == 1.0
    assert agg["overall"]["n"] == 3
    # 3 loại, mỗi loại 1 câu
    assert set(agg["by_type"]) == {"single-hop", "multi-hop", "global"}
    assert agg["by_type"]["multi-hop"]["n"] == 1


def test_aggregate_ignores_nan():
    recs = collect_records(_pipeline, _questions())
    scores = [{m: math.nan for m in config.RAGAS_METRICS} for _ in recs]
    scores[0] = {m: 0.6 for m in config.RAGAS_METRICS}
    agg = aggregate(recs, scores)
    assert agg["overall"]["faithfulness"] == 0.6   # chỉ tính câu không NaN


def test_evaluate_pipeline_payload_and_save(tmp_path):
    payload = evaluate_pipeline(_pipeline, _questions(), method="Naive RAG",
                               scorer=_fixed_scorer, save=False)
    assert payload["method"] == "Naive RAG"
    assert payload["num_questions"] == 3
    assert len(payload["records"]) == 3
    save_results("Naive RAG", payload, results_dir=tmp_path)
    loaded = load_all_results(results_dir=tmp_path, fallback_sample=False)
    assert "Naive RAG" in loaded


def test_load_eval_questions_from_repo():
    qs = load_eval_questions()
    assert len(qs) == 10
    assert {q["type"] for q in qs} <= {"single-hop", "multi-hop", "global"}


def test_sample_results_load_when_no_real(tmp_path):
    # Thư mục kết quả trống → phải rơi về results/sample/ (đánh dấu source='sample')
    res = load_all_results(results_dir=tmp_path, fallback_sample=True)
    assert set(res) == set(config.METHODS)
    assert all(r["source"] == "sample" for r in res.values())
