"""
Huấn luyện (fine-tune) cross-encoder re-ranker trên dữ liệu của bạn.

Dùng API mới của sentence-transformers v4/v5: CrossEncoderTrainer +
CrossEncoderTrainingArguments + BinaryCrossEntropyLoss. Nếu môi trường
đang ở bản < v4, script tự chuyển sang API cũ model.fit().

Đầu vào: JSONL các cặp có nhãn {"query", "passage", "label"} — tạo bằng
train/prepare_data.py.

Chạy ví dụ:
    python train/train_cross_encoder.py \
        --train-file data/pairs_train.jsonl \
        --val-file   data/pairs_train_val.jsonl \
        --base-model cross-encoder/ms-marco-MiniLM-L-12-v2 \
        --output-dir models/my-reranker \
        --epochs 2 --batch-size 16 --lr 2e-5

Fine-tune từ một reranker đa ngữ mạnh sẵn (khuyến nghị cho tiếng Việt):
    --base-model BAAI/bge-reranker-v2-m3
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reranker.utils import read_jsonl  # noqa: E402

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _sbert_major_version() -> int:
    import sentence_transformers as st
    try:
        return int(st.__version__.split(".")[0])
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Đường train chính: API mới (sentence-transformers >= 4.0)
# ---------------------------------------------------------------------------
def train_new_api(args) -> None:
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import (
        CrossEncoderTrainingArguments,
    )

    logger.info("Dùng API mới CrossEncoderTrainer (sentence-transformers >= 4.0).")

    # 1. Nạp model. num_labels=1 -> bài toán hồi quy điểm liên quan (reranker chuẩn).
    model = CrossEncoder(
        args.base_model,
        num_labels=1,
        max_length=args.max_length,
    )

    # 2. Nạp dữ liệu thành HuggingFace Dataset với đúng 3 cột.
    def to_dataset(path: str) -> "Dataset":
        rows = read_jsonl(path)
        return Dataset.from_dict({
            "query": [r["query"] for r in rows],
            "passage": [r["passage"] for r in rows],
            "label": [float(r["label"]) for r in rows],  # BCE cần float
        })

    train_ds = to_dataset(args.train_file)
    eval_ds = to_dataset(args.val_file) if args.val_file else None
    logger.info("Train: %d cặp | Val: %s cặp", len(train_ds),
                len(eval_ds) if eval_ds else "—")

    # 3. Loss. pos_weight cân bằng khi số negative > số positive.
    #    (Đặt xấp xỉ tỉ lệ negative/positive để tránh model thiên về lớp 0.)
    n_pos = sum(1 for x in train_ds["label"] if x > 0.5)
    n_neg = len(train_ds) - n_pos
    pos_weight = torch.tensor(max(1.0, n_neg / max(1, n_pos)))
    logger.info("pos_weight = %.2f (neg=%d / pos=%d)", pos_weight.item(), n_neg, n_pos)
    loss = BinaryCrossEntropyLoss(model=model, pos_weight=pos_weight)

    # 4. Tham số huấn luyện.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    train_args = CrossEncoderTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available() and not use_bf16,
        bf16=use_bf16,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=max(1, args.eval_steps // 5),
        load_best_model_at_end=bool(eval_ds),
        seed=args.seed,
        run_name=Path(args.output_dir).name,
    )

    # 5. Trainer + train.
    trainer = CrossEncoderTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=loss,
    )
    trainer.train()

    # 6. Lưu model cuối.
    final_dir = str(Path(args.output_dir) / "final")
    model.save_pretrained(final_dir)
    logger.info("Đã lưu model -> %s", final_dir)


# ---------------------------------------------------------------------------
# Đường train dự phòng: API cũ (sentence-transformers < 4.0)
# ---------------------------------------------------------------------------
def train_old_api(args) -> None:
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    logger.info("Dùng API cũ model.fit() (sentence-transformers < 4.0).")

    model = CrossEncoder(args.base_model, num_labels=1, max_length=args.max_length)

    train_rows = read_jsonl(args.train_file)
    train_examples = [
        InputExample(texts=[r["query"], r["passage"]], label=float(r["label"]))
        for r in train_rows
    ]
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=args.batch_size
    )

    evaluator = None
    if args.val_file:
        from sentence_transformers.cross_encoder.evaluation import (
            CrossEncoderClassificationEvaluator,
        )
        val_rows = read_jsonl(args.val_file)
        evaluator = CrossEncoderClassificationEvaluator(
            sentence_pairs=[[r["query"], r["passage"]] for r in val_rows],
            labels=[int(r["label"]) for r in val_rows],
            name="val",
        )

    warmup = int(len(train_dataloader) * args.epochs * 0.1)
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        output_path=str(Path(args.output_dir) / "final"),
    )
    logger.info("Đã lưu model -> %s", Path(args.output_dir) / "final")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune cross-encoder re-ranker.")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--val-file", default=None)
    ap.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    ap.add_argument("--output-dir", default="models/my-reranker")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if _sbert_major_version() >= 4:
        train_new_api(args)
    else:
        train_old_api(args)


if __name__ == "__main__":
    main()
