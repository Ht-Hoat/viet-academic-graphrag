# Demo & Đánh giá — UI + RAGAS (Nhóm 7 · phần của Tuấn Anh, SV4)

Module **tích hợp** của đề tài *So sánh Naive RAG — RAG+Re-ranking — GraphRAG cho hỏi-đáp tài liệu
học thuật tiếng Việt*: nối 3 pipeline vào một UI Streamlit, đánh giá bằng RAGAS, và kết luận về
giả thuyết nghiên cứu bằng số liệu.

```
3 pipeline (Naive / Re-rank / GraphRAG) → adapter một giao diện → UI so sánh + đánh giá RAGAS
                                                                    → tách điểm theo loại câu hỏi
```

## Chạy nhanh

```bash
pip install -r requirements.txt
cp .env.example .env                       # điền GROQ_API_KEY (RAGAS cần LLM giám khảo)

streamlit run app.py                       # UI — chạy ngay cả khi chưa merge (mock + số liệu mẫu)
python -m scripts.run_eval --mock          # thử luồng đánh giá không cần API
python -m scripts.run_eval                 # đánh giá thật (sau khi merge 3 pipeline)
python -m pytest                           # 16 test, chạy offline
```

## Hai màn hình UI

- **💬 Chat Demo** — hỏi 1 câu, so sánh song song 3 pipeline: câu trả lời, độ trễ, nguồn trích dẫn;
  riêng GraphRAG hiện loại truy vấn (Local/Global) và đường quan hệ đồ thị.
- **📊 Kết quả RAGAS** — bảng + biểu đồ 4 chỉ số, **tách theo loại câu hỏi** (single/multi/global) —
  đây là bằng chứng cho giả thuyết "GraphRAG vượt ở multi-hop và global".

## Cấu trúc

```
app.py                     # UI Streamlit (2 màn hình)
src/
  config.py                # 3 phương pháp, 4 chỉ số RAGAS, màu, loại câu hỏi
  pipelines.py             # adapter hợp nhất 3 pipeline + mock fallback
  evaluator.py             # RAGAS + tách điểm theo loại câu + lưu/đọc JSON
  charts.py                # biểu đồ PNG cho báo cáo
scripts/run_eval.py        # chạy đánh giá headless → JSON + biểu đồ + bảng
data/eval/questions.json   # bộ câu hỏi dùng chung (single/multi/global)
results/sample/            # số liệu MẪU để UI có gì hiển thị trước khi chạy thật
tests/                     # 16 pytest, không cần API/ragas/streamlit
docs/EVAL_UI_MODULE.md     # giao diện với 3 pipeline + hướng dẫn cho cả nhóm
```

Chi tiết interface mà mỗi thành viên cần export để SV4 nạp được: xem
[docs/EVAL_UI_MODULE.md](docs/EVAL_UI_MODULE.md).

---
*Nhóm 7 — Học Máy Nâng Cao · Demo & Đánh giá (Tuấn Anh, SV4)*
