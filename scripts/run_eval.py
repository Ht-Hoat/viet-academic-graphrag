"""Chạy đánh giá RAGAS cho cả 3 pipeline (headless) → lưu JSON + vẽ biểu đồ + in bảng.

    python -m scripts.run_eval                 # dùng pipeline thật (cần GROQ_API_KEY + đã merge repo)
    python -m scripts.run_eval --mock          # chạy thử toàn bộ luồng bằng pipeline + điểm giả
    python -m scripts.run_eval --metric context_recall   # chọn chỉ số cho biểu đồ theo loại câu

Kết quả: results/eval_*.json, results/comparison.png, results/by_type_<metric>.png
"""
import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.evaluator import aggregate  # noqa: E402
from src.evaluator import (collect_records, evaluate_pipeline, load_all_results,
                           load_eval_questions, save_results, score_with_ragas)
from src.pipelines import load_pipelines


def _mock_scorer(records):
    """Chấm điểm GIẢ để thử luồng mà không cần ragas/API. GraphRAG được ưu ái ở multi/global."""
    out = []
    for r in records:
        rng = random.Random(hash((r["question"], r.get("type"))) & 0xFFFF)
        base = 0.6 + rng.uniform(0, 0.25)
        out.append({m: round(min(1.0, base + rng.uniform(-0.05, 0.05)), 3) for m in config.RAGAS_METRICS})
    return out


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Chạy đánh giá RAGAS 3 pipeline")
    ap.add_argument("--mock", action="store_true", help="Dùng pipeline + điểm giả (thử luồng, không cần API)")
    ap.add_argument("--metric", default="context_recall", help="Chỉ số cho biểu đồ theo loại câu hỏi")
    ap.add_argument("--questions", default=str(config.EVAL_QUESTIONS_PATH))
    args = ap.parse_args(argv)

    questions = load_eval_questions(args.questions)
    pipes = load_pipelines(with_rerank=True, with_graph=True)
    scorer = _mock_scorer if args.mock else score_with_ragas

    for name, pipe in pipes.items():
        tag = "MOCK" if (args.mock or pipe.is_mock) else "REAL"
        print(f"\n=== Đánh giá {name} [{tag}] ===")
        payload = evaluate_pipeline(pipe, questions, method=name, scorer=scorer, save=True)
        ov = payload["overall"]
        print("  " + " · ".join(f"{m}={ov.get(m)}" for m in config.RAGAS_METRICS) + f" · latency={ov.get('latency')}s")

    results = load_all_results(fallback_sample=False)
    _print_table(results)

    try:
        from src.charts import by_type_chart, comparison_chart

        print(f"\nĐã lưu: {comparison_chart(results)}")
        print(f"Đã lưu: {by_type_chart(results, metric=args.metric)}")
    except ImportError as exc:
        print(f"[!] Bỏ qua vẽ biểu đồ: {exc}")


def _print_table(results: dict):
    print("\n===== BẢNG TỔNG HỢP (overall) =====")
    header = ["Phương pháp", *config.RAGAS_METRICS, "latency"]
    print(" | ".join(f"{h:>18}" if i else f"{h:<18}" for i, h in enumerate(header)))
    for method in config.METHODS:
        if method not in results:
            continue
        ov = results[method].get("overall", {})
        cells = [method] + [_fmt(ov.get(m)) for m in config.RAGAS_METRICS] + [_fmt(ov.get("latency"))]
        print(" | ".join(f"{c:>18}" if i else f"{c:<18}" for i, c in enumerate(cells)))

    print("\n===== THEO LOẠI CÂU HỎI (context_recall) — bằng chứng giả thuyết =====")
    print(" | ".join([f"{'Phương pháp':<18}"] + [f"{t:>12}" for t in config.QUESTION_TYPES]))
    for method in config.METHODS:
        if method not in results:
            continue
        bt = results[method].get("by_type", {})
        cells = [f"{method:<18}"] + [f"{_fmt(bt.get(t, {}).get('context_recall')):>12}" for t in config.QUESTION_TYPES]
        print(" | ".join(cells))


def _fmt(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) and not math.isnan(v) else "—"


if __name__ == "__main__":
    main()
