"""Đồ thị mẫu (ví dụ BERT/GPT trong tài liệu nhóm) — chạy thử Neo4j, truy vấn đa bước và test khi chưa có PDF/API key.

Kết quả trích xuất được viết theo đúng định dạng LLM trả về (có cả nhãn bị động, tiền tố "mô hình", sai hoa/thường)
để đi qua cùng bước làm sạch như dữ liệu thật.
"""
from src.graph_builder import to_chunk

SAMPLE_CHUNKS = [
    {"chunk_id": "bert-1", "source": "bert.pdf", "page": 1,
     "text": "BERT là mô hình ngôn ngữ được đề xuất trong bài báo Devlin et al. 2018. BERT sử dụng kiến trúc "
             "Transformer và phương pháp tiền huấn luyện với tác vụ Masked Language Model."},
    {"chunk_id": "bert-2", "source": "bert.pdf", "page": 2,
     "text": "Nhóm tác giả của bài báo Devlin et al. 2018 làm việc tại Google AI Language."},
    {"chunk_id": "gpt-1", "source": "gpt.pdf", "page": 1,
     "text": "GPT được giới thiệu trong bài báo Radford et al. 2018. GPT cũng dựa trên kiến trúc Transformer "
             "và tiền huấn luyện trên kho văn bản lớn."},
    {"chunk_id": "gpt-2", "source": "gpt.pdf", "page": 2,
     "text": "Radford et al. 2018 là công trình của OpenAI."},
    {"chunk_id": "transformer-1", "source": "transformer.pdf", "page": 1,
     "text": "Transformer được đề xuất bởi Vaswani et al. 2017, dựa trên cơ chế self-attention."},
]

SAMPLE_EXTRACTIONS = {
    "bert-1": {
        "entities": [
            {"name": "BERT", "type": "METHOD"}, {"name": "Devlin et al. 2018", "type": "PAPER"},
            {"name": "kiến trúc Transformer", "type": "METHOD"}, {"name": "tiền huấn luyện", "type": "CONCEPT"},
            {"name": "Masked Language Model", "type": "METHOD"},
        ],
        "relations": [
            {"source": "BERT", "relation": "được_đề_xuất_bởi", "target": "Devlin et al. 2018"},
            {"source": "BERT", "relation": "sử_dụng", "target": "kiến trúc Transformer"},
            {"source": "BERT", "relation": "sử_dụng", "target": "tiền huấn luyện"},
            {"source": "BERT", "relation": "sử_dụng", "target": "Masked Language Model"},
        ],
    },
    "bert-2": {
        "entities": [{"name": "Devlin et al. 2018", "type": "PAPER"}, {"name": "Google AI Language", "type": "ORGANIZATION"}],
        "relations": [{"source": "Devlin et al. 2018", "relation": "thuộc_về", "target": "Google AI Language"}],
    },
    "gpt-1": {
        "entities": [
            {"name": "GPT", "type": "METHOD"}, {"name": "Radford et al. 2018", "type": "PAPER"},
            {"name": "Transformer", "type": "METHOD"}, {"name": "Tiền huấn luyện", "type": "CONCEPT"},
        ],
        "relations": [
            {"source": "Radford et al. 2018", "relation": "đề_xuất", "target": "GPT"},
            {"source": "GPT", "relation": "dựa_trên", "target": "Transformer"},
            {"source": "GPT", "relation": "sử_dụng", "target": "Tiền huấn luyện"},
        ],
    },
    "gpt-2": {
        "entities": [{"name": "Radford et al. 2018", "type": "PAPER"}, {"name": "OpenAI", "type": "ORG"}],
        "relations": [{"source": "Radford et al. 2018", "relation": "thuộc_về", "target": "OpenAI"}],
    },
    "transformer-1": {
        "entities": [
            {"name": "Transformer", "type": "METHOD"}, {"name": "Vaswani et al. 2017", "type": "PAPER"},
            {"name": "self-attention", "type": "CONCEPT"},
        ],
        "relations": [
            {"source": "Transformer", "relation": "proposed_by", "target": "Vaswani et al. 2017"},
            {"source": "Transformer", "relation": "sử_dụng", "target": "self-attention"},
        ],
    },
}


def sample_unit_results():
    """Mỗi chunk là một đơn vị trích xuất, kèm kết quả trích xuất dựng sẵn."""
    return [([to_chunk(row)], SAMPLE_EXTRACTIONS[row["chunk_id"]]) for row in SAMPLE_CHUNKS]
