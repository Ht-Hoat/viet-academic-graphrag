"""Pipeline 1 — Naive RAG (đường cơ sở của cả đề tài).

    Câu hỏi → embed → cosine search (top-k) → LLM sinh câu trả lời kèm trích dẫn nguồn.

Module này còn là NỀN TẢNG DÙNG CHUNG cho hai pipeline kia:
  • Long (re-ranking) gọi get_global_vectorstore() để lấy top-50 rồi mới re-rank.
  • Hùng (GraphRAG) gọi load_and_chunk() để dựng đồ thị, và nạp lại FAISS index ở data/faiss.

Dòng lệnh:
    python -m src.naive_rag build [--pdf-dir data/pdfs] [--sample] [--limit N]
    python -m src.naive_rag ask "BERT là gì?" [--retrieval-only]
    python -m src.naive_rag eval [--questions data/eval/questions.json]
    python -m src.naive_rag stats
"""
import argparse
import json
import sys
import time
from pathlib import Path

from src import config
from src.chunker import chunk_documents, save_chunks_jsonl
from src.embedder import get_embeddings
from src.pdf_loader import load_documents
from src.vectorstore import build_index, load_faiss_index

METHOD = "Naive RAG"

ANSWER_SYSTEM = (
    "Bạn là trợ lý học thuật, chỉ trả lời dựa trên ngữ cảnh được cung cấp. "
    "Trả lời bằng tiếng Việt, ngắn gọn, chính xác, và luôn trích dẫn nguồn (tên file, trang). "
    "Nếu ngữ cảnh không chứa thông tin, hãy nói rõ 'Tôi không tìm thấy thông tin liên quan.' "
    "Không bịa thêm thông tin ngoài ngữ cảnh."
)


def source_label(metadata: dict) -> str:
    return f"{metadata.get('source', '?')} — trang {metadata.get('page', '?')}"


def build_answer_prompt(question: str, contexts: list[str], sources: list[str]) -> str:
    blocks = "\n\n---\n\n".join(f"[{s}]\n{c}" for s, c in zip(sources, contexts))
    return (
        f"{ANSWER_SYSTEM}\n\n"
        f"=== NGỮ CẢNH ===\n{blocks}\n\n"
        f"=== CÂU HỎI ===\n{question}\n\n"
        f"=== TRẢ LỜI ==="
    )


class NaiveRAG:
    """Gọi được như hàm: rag(question) → dict, để dùng thẳng với evaluator RAGAS của Tuấn Anh."""

    def __init__(self, vectorstore, llm=None, top_k: int = None):
        self.vs = vectorstore
        self.llm = llm
        self.top_k = top_k or config.TOP_K_FINAL

    def __call__(self, question: str) -> dict:
        return self.query(question)

    def retrieve(self, question: str, k: int = None):
        return self.vs.similarity_search(question, k=k or self.top_k)

    def query(self, question: str, generate: bool = True) -> dict:
        start = time.time()
        docs = self.retrieve(question)
        contexts = [d.page_content for d in docs]
        sources = [source_label(d.metadata) for d in docs]

        answer = ""
        if generate:
            from src.llm import call_llm, get_llm

            llm = self.llm or get_llm()
            answer = call_llm(llm, build_answer_prompt(question, contexts, sources))

        return {
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "latency": round(time.time() - start, 2),
            "method": METHOD,
        }


# ===== Giao diện cho các thành viên khác =====
def load_and_chunk(pdf_dir=config.PDF_DIR, use_sample_if_empty: bool = True) -> list[dict]:
    """Đọc PDF/corpus mẫu → chunk. Đầu ra là list dict {chunk_id, text, source, page}.

    Hùng (GraphRAG) import chính hàm này: `from src.naive_rag import load_and_chunk`.
    """
    docs = load_documents(pdf_dir, use_sample_if_empty=use_sample_if_empty)
    return chunk_documents(docs)


_global_vectorstore = None
_global_backend = None


def get_global_vectorstore():
    """Nạp (hoặc dựng) FAISS index dùng chung, cache lại. Long (re-ranking) import hàm này.

    Ưu tiên nạp index đã lưu ở data/faiss/; chưa có thì tự build từ tài liệu.
    """
    global _global_vectorstore, _global_backend
    if _global_vectorstore is None:
        embeddings = get_embeddings()
        if (config.FAISS_DIR / "index.faiss").exists():
            _global_vectorstore, _global_backend = load_faiss_index(embeddings), "faiss"
            print(f"Đã nạp FAISS index từ {config.FAISS_DIR}")
        else:
            chunks = load_and_chunk()
            _global_vectorstore, _global_backend = build_index(chunks, embeddings)
    return _global_vectorstore


def query_naive_rag(question: str, vectorstore=None) -> dict:
    """Wrapper hàm — tương thích với app.py của Tuấn Anh (gọi query_naive_rag(question))."""
    vs = vectorstore or get_global_vectorstore()
    return NaiveRAG(vs).query(question)


# ===== Build pipeline (offline) =====
def build(pdf_dir=config.PDF_DIR, use_sample: bool = False, limit: int = None, save: bool = True):
    """Dựng index đầu-cuối: đọc → chunk → xuất chunks.jsonl → embed → FAISS index → lưu đĩa.

    Xuất luôn data/processed/chunks.jsonl để Hùng dựng đồ thị trên CÙNG bộ chunk.
    """
    if use_sample:
        from src.pdf_loader import load_sample_corpus

        docs = load_sample_corpus()
        print(f"Dùng corpus mẫu: {len(docs)} tài liệu")
    else:
        docs = load_documents(pdf_dir, use_sample_if_empty=True)
    chunks = chunk_documents(docs)
    if limit:
        chunks = chunks[:limit]
    save_chunks_jsonl(chunks)                      # cho Hùng

    embeddings = get_embeddings()
    vs, backend = build_index(chunks, embeddings)
    if save:
        if backend == "faiss":
            from src.vectorstore import save_faiss_index

            save_faiss_index(vs)
        else:
            vs.save()
    return vs, chunks, backend


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Naive RAG — nền tảng truy hồi")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Dựng FAISS index + xuất chunks.jsonl")
    b.add_argument("--pdf-dir", default=str(config.PDF_DIR))
    b.add_argument("--sample", action="store_true", help="Bỏ qua PDF, dùng corpus mẫu")
    b.add_argument("--limit", type=int, help="Chỉ xử lý N chunk đầu (chạy thử)")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--retrieval-only", action="store_true",
                        help="Chỉ truy hồi, không gọi LLM (xem chunk lấy được mà không tốn quota)")
    a = sub.add_parser("ask", parents=[common], help="Hỏi một câu")
    a.add_argument("question")
    e = sub.add_parser("eval", parents=[common], help="Chạy cả bộ câu hỏi, đo latency")
    e.add_argument("--questions", default=str(config.EVAL_QUESTIONS_PATH))
    e.add_argument("--out", default=str(config.RESULTS_DIR / "naive_rag_trace.json"))
    sub.add_parser("stats", help="Thống kê index đã build")

    args = parser.parse_args(argv)

    if args.command == "build":
        _, chunks, backend = build(pdf_dir=Path(args.pdf_dir), use_sample=args.sample, limit=args.limit)
        print(f"\n✅ Build xong: {len(chunks)} chunks, backend={backend}")
        return

    vs = get_global_vectorstore()
    rag = NaiveRAG(vs)

    if args.command == "stats":
        n = getattr(vs, "index", None)
        count = n.ntotal if n is not None else len(getattr(vs, "chunks", []))
        print(f"Backend: {_global_backend} · số vector: {count} · TOP_K_FINAL={config.TOP_K_FINAL}")
    elif args.command == "ask":
        res = rag.query(args.question, generate=not args.retrieval_only)
        print(f"\n[{res['method']}] {res['latency']}s")
        print("\n--- Nguồn ---")
        for s in res["sources"]:
            print(f"- {s}")
        if res["answer"]:
            print("\n--- Câu trả lời ---\n" + res["answer"])
        else:
            print("\n--- Ngữ cảnh truy hồi ---")
            for i, c in enumerate(res["contexts"], 1):
                print(f"[{i}] {c[:200].strip()}...")
    elif args.command == "eval":
        questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
        rows = []
        for item in questions:
            res = rag.query(item["question"], generate=not args.retrieval_only)
            rows.append({"question": item["question"], "type": item.get("type"), **res})
            print(f"  [{item.get('type','?'):10}] {res['latency']}s · {item['question'][:50]}")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        avg = sum(r["latency"] for r in rows) / len(rows) if rows else 0
        print(f"\nĐã ghi {out} · latency trung bình {avg:.2f}s")


if __name__ == "__main__":
    main()
