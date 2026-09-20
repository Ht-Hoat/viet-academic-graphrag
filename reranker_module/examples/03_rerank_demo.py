"""
Ví dụ 3 — So sánh trực quan: CHỈ bi-encoder vs bi-encoder + cross-encoder.

Cho thấy vì sao cần tầng re-rank: thứ hạng top-k thay đổi ra sao sau khi
cross-encoder chấm điểm lại các ứng viên mà bi-encoder trả về.

    python examples/03_rerank_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reranker import RerankPipeline, RerankConfig, load_corpus


def main():
    cfg = RerankConfig(retrieve_top_k=10, rerank_top_k=5)
    pipe = RerankPipeline(config=cfg)
    corpus = load_corpus(str(Path(__file__).parent.parent / "data" / "sample_corpus.jsonl"))
    pipe.build_index(corpus)

    query = "Làm sao chọn ra đoạn văn tốt nhất để đưa vào LLM?"

    # Tầng 1: bi-encoder retrieve
    retrieved = pipe.retrieve(query, top_k=10)
    # Tầng 2: cross-encoder rerank
    reranked = pipe.rerank(query, retrieved, top_k=10)

    print(f"Câu hỏi: {query}\n")
    print(f"{'#':<3}{'BI-ENCODER (cosine)':<45}{'CROSS-ENCODER (rerank)':<45}")
    print("-" * 93)
    for i in range(min(5, len(retrieved))):
        b = retrieved[i]
        r = reranked[i]
        b_txt = f"{b.score:.3f} | {b.text[:32]}..."
        r_txt = f"{r.score:.3f} | {r.text[:32]}..."
        print(f"{i+1:<3}{b_txt:<45}{r_txt:<45}")

    print("\nNhận xét: cột phải là thứ hạng SAU khi cross-encoder chấm lại.")
    print("Nếu thứ tự khác cột trái, đó chính là giá trị mà tầng re-rank mang lại.")


if __name__ == "__main__":
    main()
