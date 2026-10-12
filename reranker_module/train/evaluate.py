"""
Đánh giá chất lượng re-ranking: nDCG@k, MRR@k, Recall@k, Hit@k.

Dùng để:
    - Đo lợi ích của cross-encoder so với chỉ dùng bi-encoder.
    - So sánh model trước/sau khi fine-tune.

Định dạng dữ liệu đánh giá (JSONL), mỗi dòng:
    {
      "query": "câu hỏi",
      "candidates": ["đoạn A", "đoạn B", ...],   # các đoạn cần xếp hạng
      "relevant": [0, 3]                          # chỉ số các đoạn ĐÚNG (gold)
    }
Hoặc dạng nhị phân theo từng cặp:
    {"query": ..., "candidates": [...], "labels": [1,0,0,1,...]}
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

# Cho phép chạy trực tiếp: python train/evaluate.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reranker import CrossEncoderReranker, RerankConfig  # noqa: E402
from reranker.utils import read_jsonl  # noqa: E402


# ---------------------------------------------------------------------------
# Các hàm metric (nhận thứ hạng dưới dạng nhãn liên quan 0/1 theo đúng thứ tự)
# ---------------------------------------------------------------------------
def dcg_at_k(rels: Sequence[int], k: int) -> float:
    rels = rels[:k]
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_rels: Sequence[int], k: int) -> float:
    dcg = dcg_at_k(ranked_rels, k)
    ideal = dcg_at_k(sorted(ranked_rels, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def mrr_at_k(ranked_rels: Sequence[int], k: int) -> float:
    for i, r in enumerate(ranked_rels[:k]):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_rels: Sequence[int], k: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(1 for r in ranked_rels[:k] if r > 0) / total_relevant


def hit_at_k(ranked_rels: Sequence[int], k: int) -> float:
    return 1.0 if any(r > 0 for r in ranked_rels[:k]) else 0.0


# ---------------------------------------------------------------------------
def _labels_from_row(row: dict) -> List[int]:
    n = len(row["candidates"])
    if "labels" in row:
        return [int(x) for x in row["labels"]]
    labels = [0] * n
    for idx in row.get("relevant", []):
        if 0 <= idx < n:
            labels[idx] = 1
    return labels


def evaluate_reranker(
    reranker: CrossEncoderReranker,
    eval_rows: List[dict],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """Trả về dict metric trung bình trên toàn bộ tập đánh giá."""
    metrics: Dict[str, List[float]] = {}

    def _add(name: str, val: float) -> None:
        metrics.setdefault(name, []).append(val)

    for row in eval_rows:
        query = row["query"]
        candidates = row["candidates"]
        labels = _labels_from_row(row)
        total_rel = sum(labels)

        scores = reranker.score_pairs(query, candidates)
        order = np.argsort(-scores)              # giảm dần theo điểm
        ranked_rels = [labels[i] for i in order]

        for k in k_values:
            _add(f"nDCG@{k}", ndcg_at_k(ranked_rels, k))
            _add(f"MRR@{k}", mrr_at_k(ranked_rels, k))
            _add(f"Recall@{k}", recall_at_k(ranked_rels, k, total_rel))
            _add(f"Hit@{k}", hit_at_k(ranked_rels, k))

    return {name: float(np.mean(vals)) for name, vals in metrics.items()}


def evaluate_bi_encoder_baseline(
    eval_rows: List[dict],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """Baseline: giữ NGUYÊN thứ tự candidates (giả định đó là thứ tự bi-encoder).

    So sánh kết quả này với evaluate_reranker để thấy lợi ích của re-rank.
    """
    metrics: Dict[str, List[float]] = {}

    def _add(name: str, val: float) -> None:
        metrics.setdefault(name, []).append(val)

    for row in eval_rows:
        labels = _labels_from_row(row)
        total_rel = sum(labels)
        ranked_rels = labels  # giữ nguyên thứ tự
        for k in k_values:
            _add(f"nDCG@{k}", ndcg_at_k(ranked_rels, k))
            _add(f"MRR@{k}", mrr_at_k(ranked_rels, k))
            _add(f"Recall@{k}", recall_at_k(ranked_rels, k, total_rel))
            _add(f"Hit@{k}", hit_at_k(ranked_rels, k))

    return {name: float(np.mean(vals)) for name, vals in metrics.items()}


def print_comparison(baseline: Dict[str, float], reranked: Dict[str, float]) -> None:
    print(f"\n{'Metric':<12}{'Bi-encoder':>14}{'+ Cross-enc':>14}{'Δ':>10}")
    print("-" * 50)
    for name in sorted(reranked.keys()):
        b = baseline.get(name, 0.0)
        r = reranked[name]
        print(f"{name:<12}{b:>14.4f}{r:>14.4f}{(r - b):>+10.4f}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Đánh giá cross-encoder re-ranker.")
    ap.add_argument("--eval-file", required=True, help="JSONL đánh giá.")
    ap.add_argument("--model", default=None, help="Tên/đường dẫn cross-encoder.")
    ap.add_argument("--backend", default="sentence-transformers",
                    choices=["sentence-transformers", "flag"])
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    args = ap.parse_args()

    cfg = RerankConfig()
    if args.model:
        cfg.cross_encoder_name = args.model
    cfg.cross_encoder_backend = args.backend

    rows = read_jsonl(args.eval_file)
    print(f"Đã nạp {len(rows)} mẫu đánh giá.")

    reranker = CrossEncoderReranker(config=cfg)
    baseline = evaluate_bi_encoder_baseline(rows, args.k)
    reranked = evaluate_reranker(reranker, rows, args.k)
    print_comparison(baseline, reranked)


if __name__ == "__main__":
    main()
