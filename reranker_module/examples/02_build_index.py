"""
Ví dụ 2 — Lập chỉ mục offline rồi lưu ra đĩa, sau đó nạp lại để truy vấn.

Trong thực tế, việc mã hoá corpus (bi-encoder) là tốn kém và làm MỘT LẦN;
ta lưu FAISS index + docstore ra đĩa rồi nạp lại tức thì ở các lần chạy sau.

    python examples/02_build_index.py --build
    python examples/02_build_index.py --query "câu hỏi của bạn"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reranker import RerankPipeline, RerankConfig, load_corpus

INDEX_DIR = str(Path(__file__).parent.parent / "artifacts" / "faiss_index")
CORPUS = str(Path(__file__).parent.parent / "data" / "sample_corpus.jsonl")


def build():
    cfg = RerankConfig()
    pipe = RerankPipeline(config=cfg)
    corpus = load_corpus(CORPUS)
    print(f"Lập chỉ mục {len(corpus)} đoạn...")
    pipe.build_index(corpus)
    pipe.save_index(INDEX_DIR)
    # Lưu kèm config để nạp lại nhất quán.
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    cfg.save(str(Path(INDEX_DIR) / "rerank_config.json"))
    print(f"Đã lưu index -> {INDEX_DIR}")


def query(text: str):
    cfg_path = Path(INDEX_DIR) / "rerank_config.json"
    cfg = RerankConfig.load(str(cfg_path)) if cfg_path.exists() else RerankConfig()
    pipe = RerankPipeline(config=cfg)
    pipe.load_index(INDEX_DIR)
    print(f"Index có {len(pipe.index)} đoạn.")
    result = pipe.run(text, verbose=True)
    print(f"\nCâu hỏi: {text}")
    for i, d in enumerate(result.reranked, 1):
        print(f"[{i}] {d.score:.4f} | {d.text[:100]}...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query", type=str, default=None)
    args = ap.parse_args()
    if args.build:
        build()
    elif args.query:
        query(args.query)
    else:
        print("Dùng --build để lập chỉ mục, hoặc --query \"...\" để truy vấn.")
