# Re-Ranking Module cho RAG (Bi-Encoder + Cross-Encoder)

Module re-ranking hai tầng hoàn chỉnh: **Bi-Encoder truy xuất nhanh → Cross-Encoder
chấm điểm lại chính xác → ghép context cho LLM**. Kèm code sẵn chạy, script huấn
luyện (fine-tune) cross-encoder, bộ đánh giá nDCG/MRR/Recall, dữ liệu mẫu tiếng Việt
và test.

> Bối cảnh: xây cho nghiên cứu so sánh *Naive RAG / RAG + Re-ranking / GraphRAG*.
> Tầng re-rank ở đây **dùng chung y hệt chỉ mục với Naive RAG** — nó chỉ là một
> tầng lọc thêm ở lúc truy vấn, đúng như bản chất đã phân tích trong ghi chú dự án.

---

## Mục lục

1. [Kiến thức nền: Bi-encoder vs Cross-encoder](#1-kiến-thức-nền-bi-encoder-vs-cross-encoder)
2. [Chọn model Cross-encoder](#2-chọn-model-cross-encoder)
3. [Kiến trúc pipeline 3 bước](#3-kiến-trúc-pipeline-3-bước)
4. [Cài đặt](#4-cài-đặt)
5. [Bắt đầu nhanh](#5-bắt-đầu-nhanh)
6. [Cấu trúc thư mục](#6-cấu-trúc-thư-mục)
7. [API chính](#7-api-chính)
8. [Hướng dẫn huấn luyện (fine-tune)](#8-hướng-dẫn-huấn-luyện-fine-tune)
9. [Đánh giá chất lượng](#9-đánh-giá-chất-lượng)
10. [Mẹo cho tiếng Việt & tinh chỉnh](#10-mẹo-cho-tiếng-việt--tinh-chỉnh)
11. [Câu hỏi thường gặp](#11-câu-hỏi-thường-gặp)

---

## 1. Kiến thức nền: Bi-encoder vs Cross-encoder

Đây là hai cách khác nhau để tính độ liên quan giữa **câu hỏi** và **đoạn văn (chunk)**.

### Bi-Encoder (dùng cho RETRIEVE — nhanh)

```
câu hỏi  ──► [Encoder] ──► vector_q  ┐
                                     ├──► cosine(vector_q, vector_d) = điểm
chunk    ──► [Encoder] ──► vector_d  ┘
```

- Mã hoá câu hỏi và chunk **riêng biệt** thành hai vector.
- Điểm liên quan = độ tương đồng cosine giữa hai vector.
- **Toàn bộ chunk được mã hoá TRƯỚC (offline)** và lưu vào FAISS. Lúc truy vấn chỉ
  cần encode câu hỏi **một lần** rồi tìm hàng xóm gần nhất → **cực nhanh**, quét được
  cả corpus hàng triệu đoạn.
- Điểm yếu: mô hình không "nhìn" hai văn bản cùng lúc, nên bỏ sót các liên hệ tinh vi.
- → Dùng cho **giai đoạn retrieve**.

### Cross-Encoder (dùng cho RE-RANK — chính xác)

```
[câu hỏi [SEP] chunk] ──► [Transformer + self-attention chéo] ──► 1 điểm liên quan
```

- Đưa **cặp (câu hỏi, chunk) vào cùng một lần forward**. Self-attention cho phép
  từng token câu hỏi "nhìn thẳng" vào từng token chunk → **chính xác hơn nhiều**.
- Cái giá: phải chạy một lần forward cho **mỗi cặp** → **chậm**, không thể quét cả
  corpus.
- → Dùng cho **giai đoạn re-rank** (chỉ vài chục ứng viên).

### Bảng so sánh

| | Bi-Encoder | Cross-Encoder |
|---|---|---|
| Cách encode | câu hỏi & chunk **riêng biệt** | cặp (câu hỏi, chunk) **cùng lúc** |
| Đầu ra | 2 vector → cosine | 1 điểm liên quan |
| Tốc độ | rất nhanh (chunk encode sẵn) | chậm (1 forward / cặp) |
| Độ chính xác | khá | cao |
| Quét cả corpus? | ✅ có | ❌ không |
| Vai trò | **retrieve** top 30–50 | **re-rank** còn top 3–5 |

**Kết luận:** hai loại **bổ sung cho nhau**. Bi-encoder lọc nhanh từ corpus khổng lồ
xuống vài chục ứng viên; cross-encoder — vốn quá chậm để quét cả corpus — chỉ phải
chấm lại vài chục ứng viên đó và chọn tinh ra top tốt nhất.

📄 Đọc thêm: [SBERT — Retrieve & Re-Rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html)

---

## 2. Chọn model Cross-encoder

Module hỗ trợ hai lựa chọn (đổi trong `RerankConfig` hoặc tham số dòng lệnh):

| Model | Ưu điểm | Nhược | Khi nào dùng |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | nhanh, nhẹ, chất lượng tốt | chủ yếu **tiếng Anh** | corpus tiếng Anh; cần tốc độ |
| `BAAI/bge-reranker-v2-m3` ⭐ | **đa ngữ, tiếng Việt tốt** | nặng hơn (~0.6B) | **khuyến nghị cho tiếng Việt** |

📄 Model card: [ms-marco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2) ·
[bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

**Về điểm số:** cross-encoder xuất **logit** (thường khoảng −10..10). Đặt
`normalize_scores=True` (mặc định) để đưa qua **sigmoid** về khoảng (0, 1) cho dễ đọc
và đặt ngưỡng. Việc này **không đổi thứ hạng**, chỉ đổi thang đo.

Bi-encoder mặc định là `BAAI/bge-m3` (đa ngữ, ghép đôi tự nhiên với bge-reranker-v2-m3).
Có thể đổi sang `intfloat/multilingual-e5-base` (nhớ đặt `query_prefix="query: "`,
`passage_prefix="passage: "`) hoặc `keepitreal/vietnamese-sbert` (nhẹ, thuần Việt).

---

## 3. Kiến trúc pipeline 3 bước

```
                    ┌─────────────────────────── OFFLINE (một lần) ───────────────────────────┐
   corpus (chunks) ─► Bi-Encoder.encode_documents ─► vectors ─► FAISS index (lưu ra đĩa)
                    └──────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────── ONLINE (mỗi truy vấn) ─────────────────────────────────┐
   │                                                                                              │
   │  câu hỏi ─► Bi-Encoder.encode_query ─► FAISS search                                          │
   │                                            │                                                 │
   │            BƯỚC 1: retrieve top-50 ◄───────┘   (nhanh, bi-encoder)                           │
   │                     │                                                                        │
   │                     ▼                                                                        │
   │            BƯỚC 2: Cross-Encoder chấm điểm 50 cặp ─► sort ─► top-5   (chính xác, chậm hơn)   │
   │                     │                                                                        │
   │                     ▼                                                                        │
   │            BƯỚC 3: ghép top-5 thành CONTEXT ─► LLM generation ─► câu trả lời                 │
   │                                                                                              │
   └──────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Bi-Encoder** truy xuất nhanh **top 30–50** documents từ FAISS.
2. **Cross-Encoder** re-rank và chọn ra **top 3–5** đoạn tốt nhất.
3. Ghép các đoạn này thành **context cuối** đưa vào LLM.

Con số top-k điều chỉnh trong `RerankConfig(retrieve_top_k=50, rerank_top_k=5)`.

---

## 4. Cài đặt

```bash
# 1) Tạo môi trường (khuyến nghị)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Cài phụ thuộc
pip install -r requirements.txt
#   Có GPU? Đổi faiss-cpu -> faiss-gpu trong requirements.txt.
#   Muốn dùng backend FlagEmbedding: pip install FlagEmbedding

# 3) (tuỳ chọn) cài dạng package để import từ mọi nơi
pip install -e .
```

Yêu cầu: Python ≥ 3.9, `sentence-transformers ≥ 4.0` (khuyến nghị ≥ 5.0 để có
`CrossEncoderTrainer` mới nhất).

---

## 5. Bắt đầu nhanh

```python
from reranker import RerankPipeline, RerankConfig, load_corpus

cfg = RerankConfig(
    bi_encoder_name="BAAI/bge-m3",
    cross_encoder_name="BAAI/bge-reranker-v2-m3",
    retrieve_top_k=50,   # bước 1
    rerank_top_k=5,      # bước 2
)
pipe = RerankPipeline(config=cfg)

# Lập chỉ mục corpus (list[str] hoặc list[Document])
corpus = load_corpus("data/sample_corpus.jsonl")
pipe.build_index(corpus)

# Chạy retrieve -> rerank -> ghép context
result = pipe.run("Cross-encoder khác bi-encoder như thế nào?", verbose=True)

for d in result.reranked:
    print(round(d.score, 4), d.text[:80])
print(result.context)   # <- đưa cái này vào prompt LLM
```

Chạy thử các ví dụ có sẵn:

```bash
python examples/01_quickstart.py     # end-to-end tối giản
python examples/02_build_index.py --build          # lập & lưu index
python examples/02_build_index.py --query "..."     # nạp index & truy vấn
python examples/03_rerank_demo.py    # so sánh bi-encoder vs +cross-encoder
python examples/04_rag_pipeline.py   # RAG đầy đủ + chỗ cắm LLM thật
```

---

## 6. Cấu trúc thư mục

```
reranker_module/
├── README.md                    # tài liệu này
├── requirements.txt
├── setup.py
├── reranker/                    # ── PACKAGE LÕI ──
│   ├── __init__.py
│   ├── config.py                # RerankConfig: mọi tham số + save/load
│   ├── utils.py                 # Document, ScoredDocument, sigmoid, jsonl, chunk
│   ├── bi_encoder.py            # BiEncoder — tầng retrieve (SentenceTransformer)
│   ├── cross_encoder.py         # CrossEncoderReranker — tầng re-rank (2 backend)
│   ├── faiss_index.py           # FaissIndex — build/add/search/save/load
│   └── pipeline.py              # RerankPipeline — nối 3 bước lại
├── train/                       # ── HUẤN LUYỆN & ĐÁNH GIÁ ──
│   ├── prepare_data.py          # tạo cặp (query,passage,label) + đào hard negative
│   ├── train_cross_encoder.py   # fine-tune (API v4/v5, tự fallback API cũ)
│   └── evaluate.py              # nDCG/MRR/Recall/Hit + so bi-encoder vs +rerank
├── examples/                    # ── VÍ DỤ CHẠY ĐƯỢC NGAY ──
│   ├── 01_quickstart.py
│   ├── 02_build_index.py
│   ├── 03_rerank_demo.py
│   └── 04_rag_pipeline.py
├── data/                        # ── DỮ LIỆU MẪU (tiếng Việt) ──
│   ├── sample_corpus.jsonl      # 15 đoạn kiến thức RAG/re-rank
│   ├── sample_train.jsonl       # 8 mẫu train (query/positive/negative)
│   └── sample_eval.jsonl        # 5 mẫu eval (query/candidates/relevant)
└── tests/
    └── test_pipeline.py         # smoke test (dùng encoder giả, chạy nhanh)
```

---

## 7. API chính

### `RerankConfig` (reranker/config.py)
Gom mọi tham số. Lưu/nạp bằng `cfg.save("cfg.json")` / `RerankConfig.load("cfg.json")`.
Các trường quan trọng: `bi_encoder_name`, `cross_encoder_name`, `cross_encoder_backend`
(`"sentence-transformers"` | `"flag"`), `retrieve_top_k`, `rerank_top_k`,
`normalize_scores`, `faiss_index_type` (`"flat"` | `"hnsw"`), `query_prefix`,
`passage_prefix`.

### `RerankPipeline` (reranker/pipeline.py)

| Phương thức | Vai trò |
|---|---|
| `build_index(docs)` | encode corpus + nạp vào FAISS |
| `save_index(dir)` / `load_index(dir)` | lưu/nạp index ra đĩa |
| `retrieve(query, top_k)` | **bước 1** — trả list `ScoredDocument` |
| `rerank(query, candidates, top_k)` | **bước 2** — chấm lại & cắt top-k |
| `build_context(docs, max_chars=...)` | **bước 3** — ghép thành context |
| `run(query)` | chạy trọn bước 1→3, trả `RerankResult` |
| `run_with_llm(query, llm_fn)` | như `run` + gọi LLM sinh câu trả lời |

`RerankResult` chứa: `.retrieved`, `.reranked`, `.context`, `.timings_ms`.
Mỗi `ScoredDocument` giữ cả `.retrieve_score` và `.rerank_score` để phân tích.

### Dùng riêng từng thành phần

```python
from reranker import BiEncoder, CrossEncoderReranker, FaissIndex, RerankConfig

cfg = RerankConfig()
bi = BiEncoder(config=cfg)
ce = CrossEncoderReranker(config=cfg)               # backend mặc định: sentence-transformers
# ce = CrossEncoderReranker(config=cfg, backend="flag")   # dùng FlagEmbedding

scores = ce.score_pairs("câu hỏi", ["đoạn A", "đoạn B"])   # mảng điểm [0,1]
```

---

## 8. Hướng dẫn huấn luyện (fine-tune)

Model cross-encoder có sẵn (ms-marco, bge-reranker) đã rất tốt. **Fine-tune** khi bạn
muốn nó hiểu sâu **domain riêng** (thuật ngữ, văn phong corpus của bạn) — thường cho
mức tăng nDCG/MRR đáng kể trên dữ liệu nhà.

### Bước 8.1 — Chuẩn bị dữ liệu

Cross-encoder học từ cặp **(query, passage, label)**: `label=1` nếu passage trả lời
đúng query, `label=0` nếu không. **Chất lượng negative quyết định phần lớn** — nên
dùng **hard negative** (đoạn *trông* liên quan nhưng thực chất sai) thay vì random.

Hai định dạng đầu vào (JSONL) được hỗ trợ:

```jsonc
// Dạng A — chỉ có positive, để script tự đào hard negative từ corpus
{"query": "...", "positive": "đoạn trả lời đúng"}

// Dạng B — đã có sẵn negative
{"query": "...", "positive": ["..."], "negative": ["...", "..."]}
```

Tạo dữ liệu huấn luyện:

```bash
# Dạng B (đã có negative) — trải thẳng thành cặp có nhãn:
python train/prepare_data.py \
    --input data/sample_train.jsonl \
    --output data/pairs_train.jsonl \
    --val-ratio 0.1

# Dạng A (chỉ positive) — đào hard negative bằng bi-encoder từ corpus:
python train/prepare_data.py \
    --input data/my_qa.jsonl \
    --corpus data/sample_corpus.jsonl \
    --output data/pairs_train.jsonl \
    --num-negatives 5
```

Kết quả: `data/pairs_train.jsonl` (+ `..._val.jsonl`) gồm các dòng
`{"query","passage","label"}`.

### Bước 8.2 — Huấn luyện

```bash
python train/train_cross_encoder.py \
    --train-file data/pairs_train.jsonl \
    --val-file   data/pairs_train_val.jsonl \
    --base-model cross-encoder/ms-marco-MiniLM-L-12-v2 \
    --output-dir models/my-reranker \
    --epochs 2 --batch-size 16 --lr 2e-5
```

- Muốn reranker **tiếng Việt** mạnh: đổi `--base-model BAAI/bge-reranker-v2-m3`.
- Script dùng **`CrossEncoderTrainer` + `BinaryCrossEntropyLoss`** (API
  sentence-transformers ≥ 4.0). Nếu máy đang ở bản < 4.0, script **tự chuyển** sang
  `model.fit()` cũ — không cần sửa gì.
- `pos_weight` được tính tự động theo tỉ lệ negative/positive để không thiên lệch.
- GPU: tự bật `bf16` (nếu hỗ trợ) hoặc `fp16`. CPU vẫn chạy được nhưng chậm.

Cơ chế bên trong (trích `train_cross_encoder.py`):

```python
model = CrossEncoder(base_model, num_labels=1, max_length=512)   # 1 điểm liên quan
loss  = BinaryCrossEntropyLoss(model=model, pos_weight=pos_weight)
args  = CrossEncoderTrainingArguments(output_dir=..., num_train_epochs=2,
                                      per_device_train_batch_size=16, learning_rate=2e-5,
                                      warmup_ratio=0.1, eval_strategy="steps", ...)
trainer = CrossEncoderTrainer(model=model, args=args,
                              train_dataset=train_ds, eval_dataset=eval_ds, loss=loss)
trainer.train()
model.save_pretrained("models/my-reranker/final")
```

### Bước 8.3 — Dùng model vừa train

```python
cfg = RerankConfig(cross_encoder_name="models/my-reranker/final")
pipe = RerankPipeline(config=cfg)
```

---

## 9. Đánh giá chất lượng

Đo **lợi ích thực sự của tầng re-rank** bằng nDCG@k, MRR@k, Recall@k, Hit@k. Định dạng
eval (JSONL): `{"query", "candidates": [...], "relevant": [chỉ số đoạn đúng]}`
(xem `data/sample_eval.jsonl`).

```bash
python train/evaluate.py \
    --eval-file data/sample_eval.jsonl \
    --model BAAI/bge-reranker-v2-m3 \
    --k 1 3 5 10
```

In ra bảng so sánh **Bi-encoder (giữ nguyên thứ tự) vs + Cross-encoder**:

```
Metric        Bi-encoder    + Cross-enc         Δ
--------------------------------------------------
nDCG@5            0.6431        0.8925    +0.2494
MRR@5             0.6000        0.9000    +0.3000
Recall@5          1.0000        1.0000    +0.0000
...
```

Δ dương ở nDCG/MRR = tầng re-rank đẩy đoạn đúng lên vị trí cao hơn. (Recall@k có thể
không đổi vì re-rank chỉ **sắp xếp lại** cùng tập ứng viên, không thêm đoạn mới.)

---

## 10. Mẹo cho tiếng Việt & tinh chỉnh

- **Tiếng Việt:** ưu tiên `BAAI/bge-reranker-v2-m3` (cross-encoder) + `BAAI/bge-m3`
  (bi-encoder). Cân nhắc tách từ (underthesea/pyvi) trước khi index nếu corpus nhiều
  từ ghép.
- **Chọn `retrieve_top_k`:** cao hơn (50→100) tăng recall nhưng cross-encoder phải chấm
  nhiều cặp hơn → chậm hơn. 30–50 là điểm cân bằng tốt.
- **Chọn `rerank_top_k`:** 3–5 đoạn thường đủ cho LLM; nhiều quá làm loãng context và
  tốn token.
- **Ngân sách ngữ cảnh:** dùng `build_context(..., max_chars=...)` để **cố định ngân
  sách** — quan trọng khi so sánh công bằng giữa các hệ RAG (so kiến trúc, không phải
  so lượng token nhồi vào prompt).
- **FAISS:** corpus nhỏ/vừa → `faiss_index_type="flat"` (chính xác). Corpus lớn (hàng
  trăm nghìn+) → `"hnsw"` (nhanh, xấp xỉ).
- **Tăng tốc cross-encoder:** đặt `use_fp16=True` (GPU) hoặc dùng backend `"flag"`.

---

## 11. Câu hỏi thường gặp

**Có bắt buộc phải fine-tune không?** Không. `bge-reranker-v2-m3` chạy tốt ngay
(zero-shot). Fine-tune chỉ để bơm thêm hiệu năng trên domain riêng.

**Điểm cross-encoder âm là sao?** Đó là logit thô (chưa sigmoid). Điểm âm = ít liên
quan, dương = liên quan. Đặt `normalize_scores=True` để có thang (0,1) dễ đọc.

**Chỉ dùng cross-encoder, bỏ bi-encoder được không?** Không nên: cross-encoder phải
chấm **mọi** đoạn trong corpus cho mỗi truy vấn → cực chậm với corpus lớn. Bi-encoder
lọc trước là bắt buộc để khả thi về tốc độ.

**Backend `"flag"` khác gì `"sentence-transformers"`?** `"flag"` dùng
`FlagEmbedding.FlagReranker` (chỉ cho họ `BAAI/bge-reranker-*`), có sẵn `normalize` và
`use_fp16`. `"sentence-transformers"` tổng quát hơn, chạy được cả ms-marco lẫn
bge-reranker. Kết quả xếp hạng tương đương.

**Tải model bị chặn mạng?** Tải trước ở máy có mạng rồi trỏ `cross_encoder_name`/
`bi_encoder_name` vào thư mục local, hoặc đặt biến môi trường `HF_HOME` tới cache đã có.

---

*Module này là tầng "RAG + Re-ranking" trong bộ so sánh ba phương pháp; nó dùng chung
chỉ mục với Naive RAG và chỉ thêm một bước lọc lúc truy vấn.*
