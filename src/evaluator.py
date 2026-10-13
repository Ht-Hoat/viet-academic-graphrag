"""Đánh giá 3 pipeline bằng RAGAS — và TÁCH KẾT QUẢ THEO LOẠI CÂU HỎI.

Điểm mấu chốt: giả thuyết của đề tài là "GraphRAG thắng ở câu multi-hop và global, tương đương
ở single-hop". Nếu chỉ lấy trung bình chung 4 chỉ số thì KHÔNG chứng minh được điều đó — phải tách
điểm theo từng loại câu hỏi. `aggregate()` ở đây làm đúng việc này (trục `by_type`).

RAGAS chỉ được gọi ở một chỗ (`score_with_ragas`) nên test có thể tiêm hàm chấm giả, không cần
API key hay cài ragas.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

from src import config

Scorer = Callable[[list[dict]], list[dict]]   # records -> điểm từng câu (list dict 4 metric)


# ===== Đọc câu hỏi =====
def load_eval_questions(path=config.EVAL_QUESTIONS_PATH) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("questions.json phải là một mảng list.")
    return data


# ===== Thu thập câu trả lời từ pipeline =====
def collect_records(pipeline, questions: list[dict]) -> list[dict]:
    """Chạy pipeline trên từng câu hỏi, thu {question, type, answer, contexts, ground_truth, latency}.

    `pipeline` là bất kỳ thứ gì gọi được pipeline(question) -> dict chuẩn (PipelineAdapter, hoặc
    một hàm/instance của thành viên). Câu nào lỗi thì bỏ qua và ghi nhận, không làm chết cả mẻ.
    """
    records = []
    for item in questions:
        q = item.get("question")
        if not q:
            continue
        try:
            res = pipeline(q)
        except Exception as exc:  # noqa: BLE001
            records.append({"question": q, "type": item.get("type", "?"), "error": f"{type(exc).__name__}: {exc}"})
            continue
        records.append({
            "question": q,
            "type": item.get("type", "?"),
            "answer": res.get("answer", ""),
            "contexts": res.get("contexts", []),
            "ground_truth": item.get("ground_truth", ""),
            "latency": float(res.get("latency", 0.0) or 0.0),
        })
    return records


# ===== Chấm điểm bằng RAGAS (điểm này được tiêm giả khi test) =====
def score_with_ragas(records: list[dict]) -> list[dict]:
    """Chấm 4 chỉ số RAGAS cho từng record. Trả list dict cùng thứ tự với records.

    Chỉ chấm những record không lỗi. Record lỗi nhận toàn NaN.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)

    ok = [r for r in records if "error" not in r]
    if not ok:
        return [{m: math.nan for m in config.RAGAS_METRICS} for _ in records]

    ds = Dataset.from_list([{
        "question": r["question"],
        "answer": r["answer"],
        "contexts": r["contexts"] if isinstance(r["contexts"], list) else [str(r["contexts"])],
        "ground_truth": r["ground_truth"],
    } for r in ok])
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    df = evaluate(ds, metrics=metrics).to_pandas()

    scored, it = [], iter(range(len(ok)))
    for r in records:
        if "error" in r:
            scored.append({m: math.nan for m in config.RAGAS_METRICS})
            continue
        i = next(it)
        scored.append({m: float(df[m].iloc[i]) if m in df.columns else math.nan
                       for m in config.RAGAS_METRICS})
    return scored


# ===== Gộp: tổng thể + theo loại câu hỏi =====
def _nanmean(values) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return round(sum(nums) / len(nums), 4) if nums else None


def aggregate(records: list[dict], per_question: list[dict]) -> dict:
    """Trả {"overall": {...}, "by_type": {loại: {...}}}. Mỗi ô gồm 4 metric + latency + n."""
    metrics = config.RAGAS_METRICS

    def _summary(rows: list[tuple[dict, dict]]) -> dict:
        out = {m: _nanmean([s[m] for _, s in rows]) for m in metrics}
        out["latency"] = _nanmean([r.get("latency") for r, _ in rows])
        out["n"] = len(rows)
        return out

    pairs = list(zip(records, per_question))
    by_type = {}
    for qtype in config.QUESTION_TYPES:
        rows = [(r, s) for r, s in pairs if r.get("type") == qtype]
        if rows:
            by_type[qtype] = _summary(rows)
    return {"overall": _summary(pairs), "by_type": by_type}


# ===== Chạy đầy đủ cho một pipeline =====
def evaluate_pipeline(pipeline, questions: list[dict], method: str = None,
                      scorer: Scorer = score_with_ragas, save: bool = True) -> dict:
    """Chạy end-to-end cho một pipeline: thu câu trả lời → chấm RAGAS → gộp → (lưu JSON)."""
    method = method or getattr(pipeline, "name", "Unknown")
    records = collect_records(pipeline, questions)
    per_question = scorer(records)
    agg = aggregate(records, per_question)

    detail = []
    for r, s in zip(records, per_question):
        row = {"question": r["question"], "type": r.get("type"), "latency": r.get("latency")}
        row.update(s)
        if "error" in r:
            row["error"] = r["error"]
        detail.append(row)

    payload = {
        "method": method,
        "num_questions": len(records),
        "source": "real",
        "overall": agg["overall"],
        "by_type": agg["by_type"],
        "records": detail,
    }
    if save:
        save_results(method, payload)
    return payload


# ===== Lưu / đọc kết quả =====
def save_results(method: str, payload: dict, results_dir=config.RESULTS_DIR) -> Path:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    path = Path(results_dir) / config.result_filename(method)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_all_results(results_dir=config.RESULTS_DIR, fallback_sample: bool = True) -> dict[str, dict]:
    """Đọc mọi eval_*.json. Chưa có kết quả thật thì đọc results/sample/ (đánh dấu source='sample')."""
    results = {}
    for method in config.METHODS:
        path = Path(results_dir) / config.result_filename(method)
        if path.exists():
            results[method] = json.loads(path.read_text(encoding="utf-8"))
    if not results and fallback_sample and config.SAMPLE_RESULTS_DIR.exists():
        for method in config.METHODS:
            path = config.SAMPLE_RESULTS_DIR / config.result_filename(method)
            if path.exists():
                results[method] = json.loads(path.read_text(encoding="utf-8"))
    return results
