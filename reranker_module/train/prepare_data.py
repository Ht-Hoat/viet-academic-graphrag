"""
Chuẩn bị dữ liệu huấn luyện cho cross-encoder.

Cross-encoder học từ các cặp (query, passage) kèm nhãn:
    - Positive (1): passage TRẢ LỜI ĐÚNG cho query.
    - Negative (0): passage KHÔNG liên quan.

Chất lượng negative quyết định phần lớn chất lượng model. "Hard negative"
(đoạn TRÔNG có vẻ liên quan nhưng thực ra sai) dạy model phân biệt tinh vi
hơn nhiều so với "random negative". Ta khai thác hard negative bằng chính
bi-encoder: với mỗi query, lấy các đoạn có điểm cosine cao NHƯNG không phải
positive.

Đầu vào (JSONL), mỗi dòng một trong hai dạng:
    A) {"query": ..., "positive": "đoạn đúng"}                 (+ corpus để đào negative)
    B) {"query": ..., "positive": [...], "negative": [...]}    (đã có sẵn negative)

Đầu ra: JSONL các cặp {"query", "passage", "label"} sẵn sàng cho
BinaryCrossEntropyLoss.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reranker import RerankConfig  # noqa: E402
from reranker.bi_encoder import BiEncoder  # noqa: E402
from reranker.utils import read_jsonl, write_jsonl  # noqa: E402


# ---------------------------------------------------------------------------
def mine_hard_negatives_simple(
    rows: List[dict],
    corpus: List[str],
    bi_encoder: BiEncoder,
    num_negatives: int = 5,
    range_min: int = 5,     # bỏ qua top-range_min đoạn giống nhất (dễ là positive trùng)
    range_max: int = 60,    # chỉ xét trong top-range_max
) -> List[dict]:
    """Đào hard negative bằng bi-encoder, tự cài đặt (không phụ thuộc FAISS).

    Với mỗi query: encode query + toàn corpus, lấy các đoạn điểm cao trong
    khoảng [range_min, range_max) và loại các đoạn trùng positive.
    """
    import numpy as np

    print(f"Mã hoá corpus ({len(corpus)} đoạn) để đào hard negative...")
    corpus_emb = bi_encoder.encode_documents(corpus, show_progress_bar=True)

    queries = [r["query"] for r in rows]
    q_emb = bi_encoder.encode_queries(queries, show_progress_bar=True)

    # cosine vì vector đã chuẩn hoá
    sims = q_emb @ corpus_emb.T  # (num_query, num_corpus)

    out: List[dict] = []
    for r, sim in zip(rows, sims):
        positives = r["positive"]
        if isinstance(positives, str):
            positives = [positives]
        pos_set = set(positives)

        order = np.argsort(-sim)
        negs: List[str] = []
        for rank, idx in enumerate(order):
            if rank < range_min:
                continue
            if rank >= range_max:
                break
            passage = corpus[idx]
            if passage in pos_set:
                continue
            negs.append(passage)
            if len(negs) >= num_negatives:
                break

        for p in positives:
            out.append({"query": r["query"], "passage": p, "label": 1})
        for n in negs:
            out.append({"query": r["query"], "passage": n, "label": 0})
    return out


def expand_existing_negatives(rows: List[dict]) -> List[dict]:
    """Trải các dòng dạng B (đã có positive/negative) thành cặp có nhãn."""
    out: List[dict] = []
    for r in rows:
        positives = r.get("positive", [])
        if isinstance(positives, str):
            positives = [positives]
        negatives = r.get("negative", [])
        if isinstance(negatives, str):
            negatives = [negatives]
        for p in positives:
            out.append({"query": r["query"], "passage": p, "label": 1})
        for n in negatives:
            out.append({"query": r["query"], "passage": n, "label": 0})
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Tạo dữ liệu train cho cross-encoder.")
    ap.add_argument("--input", required=True, help="JSONL đầu vào (dạng A hoặc B).")
    ap.add_argument("--output", required=True, help="JSONL cặp có nhãn đầu ra.")
    ap.add_argument("--corpus", default=None,
                    help="JSONL/TXT corpus để đào hard negative (dạng A).")
    ap.add_argument("--num-negatives", type=int, default=5)
    ap.add_argument("--bi-encoder", default=None, help="Ghi đè tên bi-encoder.")
    ap.add_argument("--val-ratio", type=float, default=0.1,
                    help="Tỷ lệ tách tập validation.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    rows = read_jsonl(args.input)
    print(f"Đã nạp {len(rows)} dòng từ {args.input}.")

    has_negatives = any("negative" in r for r in rows)
    if has_negatives:
        print("Phát hiện negative sẵn có -> trải trực tiếp thành cặp.")
        pairs = expand_existing_negatives(rows)
    else:
        if not args.corpus:
            raise SystemExit("Dữ liệu dạng A cần --corpus để đào hard negative.")
        corpus_rows = read_jsonl(args.corpus) if args.corpus.endswith(".jsonl") else None
        if corpus_rows is not None:
            corpus = [c["text"] for c in corpus_rows]
        else:
            with open(args.corpus, "r", encoding="utf-8") as f:
                corpus = [ln.strip() for ln in f if ln.strip()]
        cfg = RerankConfig()
        if args.bi_encoder:
            cfg.bi_encoder_name = args.bi_encoder
        bi = BiEncoder(config=cfg)
        pairs = mine_hard_negatives_simple(
            rows, corpus, bi, num_negatives=args.num_negatives
        )

    random.shuffle(pairs)
    n_val = int(len(pairs) * args.val_ratio)
    val, train = pairs[:n_val], pairs[n_val:]

    out_path = Path(args.output)
    write_jsonl(train, str(out_path))
    if n_val > 0:
        val_path = out_path.with_name(out_path.stem + "_val" + out_path.suffix)
        write_jsonl(val, str(val_path))
        print(f"Đã ghi {len(train)} cặp train -> {out_path}")
        print(f"Đã ghi {len(val)} cặp val   -> {val_path}")
    else:
        print(f"Đã ghi {len(train)} cặp train -> {out_path}")

    n_pos = sum(1 for p in pairs if p["label"] == 1)
    print(f"Tổng cặp: {len(pairs)} | positive: {n_pos} | negative: {len(pairs) - n_pos}")


if __name__ == "__main__":
    main()
