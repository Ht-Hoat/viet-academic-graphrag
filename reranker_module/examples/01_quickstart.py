"""
Ví dụ 1 — Quickstart: chạy trọn pipeline 2 tầng trong ~15 dòng.

    python examples/01_quickstart.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reranker import RerankPipeline, RerankConfig, load_corpus


def main():
    # Cấu hình: retrieve top-50 bằng bi-encoder, rerank còn top-5 bằng cross-encoder.
    cfg = RerankConfig(
        # Dùng model nhẹ để chạy nhanh khi thử. Đổi sang bge-m3 / bge-reranker-v2-m3
        # cho chất lượng tiếng Việt tốt hơn (xem README).
        retrieve_top_k=50,
        rerank_top_k=5,
    )
    pipe = RerankPipeline(config=cfg)

    # Nạp corpus mẫu và lập chỉ mục.
    corpus = load_corpus(str(Path(__file__).parent.parent / "data" / "sample_corpus.jsonl"))
    print(f"Đang lập chỉ mục {len(corpus)} đoạn...")
    pipe.build_index(corpus)

    # Truy vấn.
    query = "Cross-encoder khác bi-encoder như thế nào và dùng khi nào?"
    result = pipe.run(query, verbose=True)

    print(f"\nCâu hỏi: {query}\n")
    print("=== TOP-5 SAU RE-RANK ===")
    for i, d in enumerate(result.reranked, 1):
        print(f"\n[{i}] rerank={d.rerank_score:.4f} | retrieve={d.retrieve_score:.4f} | id={d.id}")
        print(f"    {d.text[:120]}...")

    print("\n=== CONTEXT GHÉP CHO LLM ===")
    print(result.context)


if __name__ == "__main__":
    main()
