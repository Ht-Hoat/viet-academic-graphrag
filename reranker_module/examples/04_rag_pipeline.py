"""
Ví dụ 4 — RAG hoàn chỉnh: retrieve -> rerank -> LLM generation.

Bước 3 (LLM) được tách khỏi module để bạn tự cắm mô hình sinh tuỳ ý:
OpenAI, vLLM, Ollama, hoặc transformers cục bộ. Ở đây minh hoạ bằng một
"LLM giả" (mock) để chạy được ngay không cần API key, kèm hướng dẫn thay
bằng LLM thật ở phần cuối file.

    python examples/04_rag_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reranker import RerankPipeline, RerankConfig, load_corpus


# --------------------------------------------------------------------------- #
# LLM giả: chỉ để minh hoạ luồng. Thay bằng hàm gọi LLM thật của bạn.
# --------------------------------------------------------------------------- #
def mock_llm(prompt: str, context: str) -> str:
    n = context.count("[")
    return (
        f"[LLM giả] Đã nhận prompt {len(prompt)} ký tự với {n} đoạn ngữ cảnh.\n"
        f"Câu trả lời thật sẽ được sinh dựa trên các đoạn đã re-rank ở trên."
    )


def main():
    cfg = RerankConfig(retrieve_top_k=50, rerank_top_k=5)
    pipe = RerankPipeline(config=cfg)
    corpus = load_corpus(str(Path(__file__).parent.parent / "data" / "sample_corpus.jsonl"))
    pipe.build_index(corpus)

    query = "Giải thích pipeline RAG hai tầng và vai trò của cross-encoder."

    answer, result = pipe.run_with_llm(
        query,
        llm_fn=mock_llm,
        retrieve_top_k=50,
        rerank_top_k=5,
        max_context_chars=2000,   # cố định ngân sách ngữ cảnh
        verbose=True,
    )

    print(f"\nCâu hỏi: {query}")
    print(f"\nNgữ cảnh (top-{len(result.reranked)} sau re-rank):")
    print(result.context)
    print(f"\nTrả lời:\n{answer}")
    print(f"\nThời gian: retrieve={result.timings_ms['retrieve_ms']:.1f}ms, "
          f"rerank={result.timings_ms['rerank_ms']:.1f}ms")


# --------------------------------------------------------------------------- #
# Cách cắm LLM THẬT — chọn 1 trong các mẫu dưới, thay cho mock_llm:
#
# 1) OpenAI:
#     from openai import OpenAI
#     client = OpenAI()
#     def openai_llm(prompt, context):
#         r = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#         )
#         return r.choices[0].message.content
#
# 2) Ollama (LLM cục bộ):
#     import requests
#     def ollama_llm(prompt, context):
#         r = requests.post("http://localhost:11434/api/generate",
#                           json={"model": "qwen2.5", "prompt": prompt, "stream": False})
#         return r.json()["response"]
#
# 3) transformers cục bộ:
#     from transformers import pipeline as hf_pipeline
#     gen = hf_pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
#     def hf_llm(prompt, context):
#         return gen(prompt, max_new_tokens=256)[0]["generated_text"]
#
# Rồi gọi: pipe.run_with_llm(query, llm_fn=openai_llm, ...)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
