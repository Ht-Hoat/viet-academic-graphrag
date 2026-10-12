# Module Demo & Đánh giá (phần của Tuấn Anh — SV4)

Tài liệu dùng cho cả nhóm: SV4 là **người tích hợp** — nối ba pipeline (Naive RAG của Hoạt,
Re-ranking của Long, GraphRAG của Hùng) vào một UI Streamlit và một khung đánh giá RAGAS chung,
rồi kết luận về giả thuyết nghiên cứu bằng số liệu.

| File | Vai trò |
|---|---|
| [src/config.py](../src/config.py) | Hằng số chung: 3 phương pháp, 4 chỉ số RAGAS, màu, loại câu hỏi, tên file kết quả |
| [src/pipelines.py](../src/pipelines.py) | **Lớp tích hợp**: đưa 3 pipeline về một giao diện `.query()`; tự mock khi thiếu backend |
| [src/evaluator.py](../src/evaluator.py) | Chạy RAGAS + **tách điểm theo loại câu hỏi** (bằng chứng giả thuyết) + lưu/đọc JSON |
| [src/charts.py](../src/charts.py) | Biểu đồ PNG cho báo cáo: so sánh tổng thể + theo loại câu hỏi |
| [scripts/run_eval.py](../scripts/run_eval.py) | Chạy đánh giá cả 3 pipeline (headless) → JSON + biểu đồ + bảng |
| [app.py](../app.py) | UI Streamlit: màn hình Chat (so sánh 3) + màn hình Kết quả RAGAS |

## 1. Cài đặt

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env                               # điền GROQ_API_KEY (RAGAS cần LLM làm giám khảo)
```

## 2. Chạy

```bash
# UI — chạy được NGAY cả khi chưa merge với 3 pipeline (tự dùng mock + số liệu mẫu)
streamlit run app.py

# Đánh giá
python -m scripts.run_eval --mock          # thử toàn bộ luồng bằng pipeline + điểm giả (không cần API)
python -m scripts.run_eval                 # đánh giá thật (cần GROQ_API_KEY + đã merge repo)
python -m scripts.run_eval --metric faithfulness   # đổi chỉ số cho biểu đồ theo loại câu

# Test — 16 test chạy offline, không cần API/ragas/streamlit
python -m pytest
```

Kết quả ghi vào `results/`: `eval_*.json` (mỗi phương pháp), `comparison.png`,
`by_type_<metric>.png`. Trước khi có số liệu thật, UI hiển thị **số liệu MẪU** trong
`results/sample/` (có nhãn cảnh báo) để không bị trống.

## 3. Giao diện với 3 pipeline (điều cả nhóm cần chốt)

SV4 gọi mỗi pipeline như một **hàm** `pipeline(question) -> dict`. Mỗi thành viên chỉ cần đảm bảo
module của mình export một trong các điểm vào sau, `src/pipelines.py` tự nhận diện:

| Pipeline | Điểm vào mà SV4 nạp | Ghi chú |
|---|---|---|
| Naive RAG (Hoạt) | `from src.naive_rag import query_naive_rag` | hoặc `NaiveRAG(...).query` |
| Re-ranking (Long) | `from src.rerank_rag import query_rerank_rag` | dùng chung vectorstore của Hoạt |
| GraphRAG (Hùng) | `from src.graph_rag import build_graph_rag` → `.query` | tự thử `query_graph_rag` nếu không có |

**Hình dạng dict chuẩn** mà mọi pipeline phải trả (SV4 tự điền mặc định cho trường thiếu):

```python
{
  "answer":   str,
  "sources":  list[str] | list[dict],   # dict {source,page} cũng được — adapter tự đổi thành str
  "contexts": list[str],                # RAGAS đọc trường này
  "latency":  float,
  "method":   str,
  # chỉ GraphRAG:
  "search_type": "Local" | "Global",
  "graph_paths": list[str],
}
```

Nếu một pipeline chưa sẵn sàng (chưa merge, thiếu API key, lỗi khởi tạo), adapter tự thay bằng
**pipeline mock** có nhãn, nên UI và eval không bao giờ chết. UI đọc cờ `.is_mock` để báo trạng thái.

Test tiêm pipeline giả không cần backend:

```python
from src.pipelines import load_pipelines
pipes = load_pipelines(providers={"Naive RAG": my_fake_fn})
```

## 4. Đánh giá — vì sao TÁCH THEO LOẠI CÂU HỎI

Giả thuyết của đề tài: *single-hop ba phương pháp tương đương; multi-hop và global thì GraphRAG
vượt lên*. Trung bình chung 4 chỉ số **không** chứng minh được điều này — một GraphRAG mạnh ở
multi-hop nhưng ngang ở single-hop có thể cho trung bình chung gần bằng Naive RAG.

Vì vậy `evaluator.aggregate()` trả cả `overall` lẫn `by_type` (single-hop / multi-hop / global).
Đây là điểm khác chính so với bản đánh giá cũ (chỉ có trung bình chung). Bảng và biểu đồ
`by_type_chart` là thứ đưa thẳng vào phần Kết quả của báo cáo.

RAGAS được cô lập trong đúng một hàm `score_with_ragas(records)`, nên:
- Test tiêm hàm chấm giả → chạy offline, không tốn quota.
- Đổi khung đánh giá khác (hoặc thêm chỉ số) chỉ sửa một chỗ.

## 5. Khác gì so với bản UI/eval cũ trên nhánh chính

| Vấn đề ở bản cũ | Cách xử lý ở đây |
|---|---|
| 3 pipeline có giao diện khác nhau, UI phải biết từng loại | `PipelineAdapter` đưa về một `.query()`; adapter chuẩn hóa cả `sources` dạng dict |
| Chỉ có trung bình chung 4 chỉ số | Thêm trục `by_type` (single/multi/global) — bằng chứng trực tiếp cho giả thuyết |
| RAGAS gọi rải rác, khó test | Cô lập trong `score_with_ragas`; test tiêm scorer giả, 16 test chạy offline |
| Thiếu backend/API là UI vỡ | Mock fallback có nhãn + số liệu mẫu `results/sample/` để UI luôn có gì hiển thị |
| GraphRAG có `search_type`/`graph_paths` nhưng UI bỏ qua | UI hiện loại truy vấn (Local/Global) và liệt kê đường quan hệ đồ thị |
| Một câu lỗi làm hỏng cả mẻ đánh giá | `collect_records` bắt lỗi từng câu, ghi nhận `error`, vẫn chấm các câu còn lại |

## 6. Ghi chú khi merge vào repo chung

- Bộ câu hỏi `data/eval/questions.json` ở đây nên là bộ **dùng chung** của cả nhóm (Naive RAG và
  GraphRAG cùng đánh giá trên nó). Nếu Hùng có thêm trường `graph_evidence` cho phân tích đồ thị,
  giữ nguyên — evaluator này chỉ đọc `question`, `ground_truth`, `type` nên không xung đột.
- `src/config.py` ở đây (config đánh giá/UI) tách biệt với config của Naive RAG/GraphRAG; khi gộp
  có thể để riêng namespace hoặc gộp cẩn thận các key trùng tên (`LLM_MODEL`, `EMBEDDING_MODEL`).
- Sau khi merge: chạy `python -m scripts.run_eval` (không có `--mock`) để ra số liệu thật, rồi
  `streamlit run app.py` mở màn hình Kết quả RAGAS xem bảng + biểu đồ.
- Nhớ thu thập PDF học thuật thật (về NLP/RAG/GraphRAG) bỏ vào `data/pdfs/` của module Naive RAG
  trước khi chạy đánh giá thật — nếu không, câu hỏi eval sẽ không có ground truth trong ngữ cảnh.
