"""Test pipeline NaiveRAG end-to-end với embedder giả + LLM giả (không gọi mạng)."""
from src.embedder import FakeEmbeddings
from src.naive_rag import NaiveRAG, build_answer_prompt, source_label
from src.vectorstore import NumpyVectorStore


class FakeLLM:
    """LLM giả: trả về prompt nhận được để test kiểm tra ngữ cảnh có được ghép đúng không."""

    def __init__(self):
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt

        class _Resp:
            content = "Câu trả lời mẫu."
        return _Resp()


def _vs():
    chunks = [
        {"chunk_id": "a", "text": "BERT được đề xuất bởi Devlin và cộng sự năm 2018.",
         "source": "01.txt", "page": 1},
        {"chunk_id": "b", "text": "FAISS là thư viện tìm kiếm vector.", "source": "03.txt", "page": 2},
    ]
    return NumpyVectorStore(FakeEmbeddings(dim=32), chunks)


def test_source_label_format():
    assert source_label({"source": "bert.pdf", "page": 4}) == "bert.pdf — trang 4"


def test_build_prompt_contains_context_and_question():
    prompt = build_answer_prompt("BERT là gì?", ["đoạn văn A"], ["01.txt — trang 1"])
    assert "BERT là gì?" in prompt
    assert "đoạn văn A" in prompt
    assert "01.txt — trang 1" in prompt


def test_query_returns_standard_shape():
    llm = FakeLLM()
    rag = NaiveRAG(_vs(), llm=llm)
    res = rag.query("Ai đề xuất BERT?")
    assert set(res) == {"answer", "sources", "contexts", "latency", "method"}
    assert res["method"] == "Naive RAG"
    assert res["answer"] == "Câu trả lời mẫu."
    # Ngữ cảnh phải được nhồi vào prompt gửi cho LLM
    assert "Devlin" in llm.last_prompt


def test_retrieval_only_skips_llm():
    llm = FakeLLM()
    rag = NaiveRAG(_vs(), llm=llm)
    res = rag.query("bất kỳ", generate=False)
    assert res["answer"] == ""
    assert llm.last_prompt is None          # không hề gọi LLM
    assert len(res["contexts"]) >= 1


def test_callable_interface_for_ragas():
    # rag(question) phải chạy được như rag.query(question) — để cắm vào evaluator RAGAS
    rag = NaiveRAG(_vs(), llm=FakeLLM())
    assert rag("BERT?")["method"] == "Naive RAG"
