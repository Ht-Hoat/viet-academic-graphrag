# Module GraphRAG (phần của Hùng)

Tài liệu dùng cho cả nhóm: cách chạy, cách gọi từ module khác, và những chỗ code khác với bản mẫu trong
[SPRINT_DEMO_GRAPHRAG.md](../SPRINT_DEMO_GRAPHRAG.md).

| File | Vai trò |
|---|---|
| [src/graph_store.py](../src/graph_store.py) | Toàn bộ Cypher — ghi và đọc đồ thị (đổi sang backend khác chỉ cần sửa file này) |
| [src/graph_builder.py](../src/graph_builder.py) | Offline: trích xuất thực thể–quan hệ, chuẩn hoá, ghi Neo4j, Leiden, tóm tắt cộng đồng |
| [src/graph_rag.py](../src/graph_rag.py) | Online: local search (vector + đồ thị) và global search (map-reduce) |
| [src/graph_memory_store.py](../src/graph_memory_store.py) | Backend dự phòng chạy trong RAM (NetworkX + JSON), cùng giao diện với Neo4j |
| [src/graph_sample.py](../src/graph_sample.py) | Đồ thị mẫu BERT/GPT để chạy thử khi chưa có PDF hoặc API key |
| [src/llm_utils.py](../src/llm_utils.py) | Gọi LLM: thử lại khi lỗi, giãn nhịp theo rate limit, đọc JSON từ output |

## 1. Cài đặt

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env                               # rồi điền GROQ_API_KEY
docker compose up -d                                 # Neo4j: http://localhost:7474 (neo4j / graphrag123)
```

Neo4j cần phiên bản từ 5.23 trở lên (cú pháp `CALL (x) { ... }`). File `docker-compose.yml` đã ghim
`neo4j:5.26-community` và gắn volume nên dữ liệu không mất khi tạo lại container.

Không chạy được Docker/Neo4j thì đặt `GRAPH_BACKEND=memory` trong `.env`: đồ thị giữ trong RAM và lưu ra
`data/graph/graph.json`, mọi lệnh bên dưới chạy y hệt. Hai backend dùng chung một bộ test hợp đồng
(`tests/test_graph_store.py`) nên hành vi truy hồi giống nhau; Neo4j vẫn là backend chính để demo Cypher và
giao diện đồ thị.

## 2. Chạy

```bash
# Đồ thị mẫu — không cần LLM, dùng để kiểm tra Neo4j và xem truy vấn đa bước hoạt động
python -m src.graph_builder sample
python -m src.graph_rag ask "Tác giả của BERT làm việc ở tổ chức nào?" --retrieval-only

# Đồ thị thật
python -m src.graph_builder build --chunks data/processed/chunks.jsonl --limit 20   # chạy thử 20 chunk
python -m src.graph_builder build --chunks data/processed/chunks.jsonl              # chạy toàn bộ
python -m src.graph_builder communities            # Leiden + tóm tắt cộng đồng (thêm --no-llm để không gọi LLM)
python -m src.graph_builder stats

# Hỏi đáp và đo đạc
python -m src.graph_rag ask "BERT và GPT có điểm gì chung?" --faiss-dir data/faiss
python -m src.graph_rag eval --questions data/eval/questions.json --out results/graphrag_trace.json
```

`--retrieval-only` bỏ bước sinh câu trả lời: xem được thực thể xuất phát, đường quan hệ và ngữ cảnh mà không
tốn quota LLM. Đây là cách soi lỗi truy hồi nhanh nhất.

Chạy test: `python -m pytest` → 40 test chạy, 10 test Neo4j bị bỏ qua.
Thêm Cypher thật: `NEO4J_TEST_URI=bolt://localhost:7687 python -m pytest` → 50 test.
Test này **xoá sạch Entity/Chunk/Community** trong DB được trỏ tới, nên đừng trỏ vào DB đang chứa đồ thị thật.

## 3. Giao diện cho các thành viên khác

**Chunks vào (Hoạt).** GraphRAG nhận LangChain `Document` hoặc dict `{text, source, page[, chunk_id]}`.
Nếu không có `chunk_id`, id được băm từ `source|page|text` — Naive RAG và GraphRAG tính ra cùng id cho cùng một
chunk, nhờ đó chunk lấy qua đồ thị và chunk lấy qua vector khử trùng được. Xuất file trao đổi:

```python
from src.graph_builder import save_chunks_jsonl
save_chunks_jsonl(chunks, "data/processed/chunks.jsonl")
```

**Kết quả ra (Tuấn Anh — UI và RAGAS).** `rag.query(question)` và `rag(question)` trả về:

```python
{
  "answer": str,
  "sources": list[str],        # "bert.pdf — trang 2", cuối cùng là "Đồ thị tri thức: N đường quan hệ"
  "contexts": list[str],       # RAGAS đọc trường này: mỗi đường quan hệ 1 phần tử, rồi text từng chunk
  "latency": float,
  "timings": {"analyze": .., "communities": .., "graph": .., "vector": .., "generate": ..},
  "method": "GraphRAG",
  "search_type": "Local" | "Global",
  "seed_entities": list[str],
  "graph_paths": list[str],    # đường quan hệ đã viết thành câu
  "subgraph": {"nodes": [{"id", "label", "type", "seed"}], "edges": [{"source", "target", "label"}]},
  "map_filtered": bool,        # chỉ có ở Global search — xem lưu ý dưới
}
```

Lưu ý khi đo bằng ngữ cảnh: `--retrieval-only` bỏ bước map của Global search, mà bộ lọc điểm
(`GLOBAL_MIN_SCORE`) nằm ở chính bước đó. Câu GLOBAL chạy retrieval-only vì thế có nhiều cụm hơn lần chạy đầy
đủ, và kết quả mang `map_filtered: false`. Đừng đặt hai lần chạy cạnh nhau để so RAGAS context precision/recall.

```python
from src.graph_rag import build_graph_rag
rag = build_graph_rag(vectorstore=vectorstore, embedder=embeddings)   # dùng chung FAISS + embedding của Naive RAG
run_evaluation(rag, eval_questions, "GraphRAG")                       # rag gọi được như rag_func(question)
```

**Truy hồi thô (Long — hybrid search, Hoạt — context assembly).** `rag.retrieve(question)` chỉ truy hồi, không
gọi LLM sinh, trả `{"seeds", "paths", "graph_facts", "chunks", "timings"}`. Ranh giới: Hùng chịu trách nhiệm tới
đây; phần trộn kết quả với re-ranking nằm ở module của Long.

## 4. Khác gì so với code mẫu

| Vấn đề trong bản mẫu | Cách xử lý ở đây |
|---|---|
| Truy vấn 2 bước chỉ lấy quan hệ đầu tiên → mất node giữa | Trả về cả `nodes(p)` và `relationships(p)`, giữ đúng chiều bằng `startNode`/`endNode` |
| Không trả `contexts` → RAGAS chấm trên ngữ cảnh rỗng | `contexts` luôn có: khối quan hệ đồ thị + text từng chunk |
| Entity/quan hệ không có nguồn | `(:Entity)-[:MENTIONED_IN]->(:Chunk)`, cạnh `REL` mang `chunk_ids` |
| Lọc trùng theo chữ thường nhưng ghi theo tên gốc → rơi quan hệ | Mọi thứ khoá theo `entity_key` (NFC + bỏ tiền tố + chữ thường); quan hệ nhắc tới thực thể lạ thì tự thêm thực thể |
| `CONTAINS` khớp nhầm ("RAG" khớp "GraphRAG") | Khớp chính xác tên/alias → fulltext index → cosine embedding; quét tên trong câu hỏi theo ranh giới từ |
| `LIMIT 20` không xếp hạng, hub làm nổ số đường | Giới hạn theo từng độ dài, chặn đi xuyên node bậc cao (`MAX_NODE_DEGREE`), xếp hạng ở Python |
| Global search chỉ hỏi YES/NO từng cụm, không có reduce | Chọn cụm theo embedding → map song song (ý trả lời + điểm) → reduce; không cụm nào đạt điểm thì quay về local |
| Mỗi chunk 1 query, xoá sạch DB | Ghi theo lô `UNWIND`, chỉ xoá nhãn của GraphRAG |
| Gọi lại LLM là mất quota | Cache trích xuất theo đơn vị trong `data/graph/extractions.jsonl` |
| 3 lượt gọi LLM mỗi câu hỏi | Gộp phân loại + trích thực thể vào 1 lượt; map của global search chạy song song |

Vài lựa chọn đáng nói khi bảo vệ:

- **Đơn vị trích xuất.** Các chunk liền nhau cùng file được gộp tới `EXTRACT_UNIT_CHARS` (1500 ký tự) cho mỗi
  lượt gọi LLM, nhưng nguồn vẫn ghi về đúng chunk chứa tên thực thể, nên trích dẫn không bị thô đi.
- **Loại quan hệ cố định** (`config.RELATION_TYPES`). Nhãn tự do làm đồ thị vỡ vụn; nhãn lạ bị gom về
  `liên_quan_đến`, nhãn bị động ("được_đề_xuất_bởi") bị đảo chiều về dạng chủ động.
- **Chuẩn hoá NFC ngay khi nhận chunk.** PDF tiếng Việt xuất ra khi NFC khi NFD; nếu không chuẩn hoá thì tên
  thực thể không dò được trong text và nguồn trích dẫn bị gán sai chunk. `chunk_id` cũng băm trên text đã NFC nên
  không phụ thuộc cách gõ dấu.
- **Gợi ý loại thực thể theo câu hỏi.** "... tổ chức nào?" thì đường đi chạm node `ORG` được cộng điểm. Rẻ,
  không cần LLM, và giải thích được khi thầy hỏi vì sao chọn đường này.
- **So sánh công bằng.** GraphRAG dùng đúng chunk, embedding và câu lệnh sinh như Naive RAG; phần thêm chỉ là
  quan hệ đồ thị và tối đa `MAX_GRAPH_CHUNKS` chunk lấy qua đồ thị. Khi báo cáo cần nói rõ GraphRAG được nhiều
  ngữ cảnh hơn — đó là bản chất của phương pháp, nhưng phải ghi ra.

## 5. Ngân sách LLM

Trích xuất là chỗ tốn quota nhất: mỗi đơn vị ~1500 ký tự tốn 1 lượt gọi. 50 PDF × 20 trang ≈ 1000 đơn vị.
Gói miễn phí Groq giới hạn cả số request/ngày lẫn token/ngày (kiểm tra tại
console.groq.com/settings/limits trước khi chạy hàng loạt). Cách xử lý:

1. Chạy `--limit 20` trước, xem chất lượng triple rồi mới chạy rộng.
2. Đặt `EXTRACT_MODEL=llama-3.1-8b-instant` trong `.env` nếu hết quota model lớn (nhớ ghi lại trong báo cáo là
   hai model khác nhau, vì cache được khoá theo tên model nên đổi model là trích xuất lại).
3. Cache nằm ở `data/graph/extractions.jsonl` — nên commit để cả nhóm khỏi chạy lại.
4. Global search gọi LLM mỗi cụm một lượt ở bước map (có giãn nhịp theo `LLM_REQUESTS_PER_MINUTE`), nên
   `GLOBAL_TOP_COMMUNITIES` càng lớn thì càng tốn; mặc định 5.

Khi truyền `embedder`, embedding tên thực thể được cache ra `data/graph/entity_vectors.npz` (khoá theo danh sách
tên) để không phải tính lại mỗi lần chạy — file này sinh lại được nên không cần commit.

## 6. Phân tích lỗi (phần cần cho báo cáo)

Chạy `python -m src.graph_rag eval --retrieval-only` rồi đọc `results/graphrag_trace.json`
(mẫu chạy trên đồ thị demo: [results/graphrag_sample_trace.json](../results/graphrag_sample_trace.json)). Mỗi câu có
`seed_entities`, `graph_paths`, `contexts` và `evidence_hit` (tỉ lệ mốc thông tin kỳ vọng xuất hiện trong ngữ
cảnh — khai báo ở trường `graph_evidence` của bộ câu hỏi, xem
[data/eval/graph_sample_questions.json](../data/eval/graph_sample_questions.json)).

Bảng lỗi để điền khi có dữ liệu thật:

| Mã | Lỗi | Dấu hiệu trong trace | Hướng sửa |
|---|---|---|---|
| E1 | Nối sai thực thể | `seed_entities` trống hoặc lệch hẳn với câu hỏi | Thêm alias, hạ `ENTITY_LINK_MIN_SIM`, truyền `embedder` |
| E2 | Thiếu cạnh | Thực thể đúng nhưng không có đường nối tới đáp án | Xem lại chunk nguồn: LLM bỏ sót quan hệ hay chunking cắt mất |
| E3 | Nhiễu do node hub | `graph_paths` toàn quan hệ chung chung | Giảm `MAX_NODE_DEGREE`, siết danh sách loại quan hệ |
| E4 | Sai cấp độ local/global | Câu tổng quan lại chạy Local (hoặc ngược lại) | Bổ sung từ khoá `GLOBAL_HINTS`, sửa prompt phân tích |
| E5 | LLM bỏ qua ngữ cảnh đồ thị | Ngữ cảnh có đáp án nhưng câu trả lời nói không tìm thấy | Chỉnh `ANSWER_PROMPT`, giảm số đường đưa vào |
| E6 | Ngữ cảnh quá dài | `graph_paths` sát `MAX_GRAPH_PATHS`, câu trả lời loãng | Giảm `MAX_GRAPH_PATHS` / `MAX_CONTEXT_CHARS` |

E1–E6 khớp với các lỗi G1–G6 của bài khảo sát Agentic GraphRAG, tiện để đối chiếu trong báo cáo.
