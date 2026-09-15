# GRAPHRAG — BÁO CÁO BUỔI 2: THUẬT TOÁN & DEMO ĐỒ THỊ TRI THỨC


Nhóm sẽ xây một hệ thống cho phép người dùng đặt câu hỏi bằng tiếng Việt trên một kho tài liệu học thuật, và nhận câu trả lời kèm trích dẫn nguồn. Điểm mấu chốt của đề tài không phải chỉ là "làm một con chatbot", mà là so sánh ba cách lấy thông tin (truy hồi) để trả lời:

Naive RAG — tìm đoạn văn giống câu hỏi nhất rồi đưa cho mô hình trả lời.
RAG + Re-ranking — tìm nhiều đoạn rồi xếp hạng lại để lọc đoạn tốt nhất.
GraphRAG — dựng "bản đồ quan hệ" (Đồ thị tri thức) giữa các khái niệm trong tài liệu, rồi trả lời dựa trên quan hệ đó.
Câu hỏi nghiên cứu trung tâm: GraphRAG có trả lời tốt hơn hai cách kia không, đặc biệt với những câu hỏi phải nối thông tin từ nhiều chỗ (câu hỏi đa bước)? Đây chính là "hướng nghiên cứu" mà thầy muốn thấy: có giả thuyết rõ, có cái để so sánh, có cách đo.

| Thầy hỏi | Trả lời ngắn |
|---|---|
| Dự án dùng thuật toán gì? | 3 pipeline tăng dần: Naive RAG → RAG + Re-ranking → **GraphRAG** (dùng đồ thị tri thức) |
| Đồ thị tri thức là gì? | Graph G = (V, E): V = thực thể (tác giả, khái niệm, bài báo), E = quan hệ (viết bởi, trích dẫn, định nghĩa) |
| Xây đồ thị từ đâu? | PDF học thuật → trích xuất thực thể + quan hệ bằng LLM → nạp vào Neo4j |
| Sao cần đồ thị, vector search không đủ sao? | Vector search giỏi câu 1-hop, tệ câu multi-hop. Đồ thị giải quyết: "Tác giả A và B có cùng hướng nghiên cứu không?" |
| Đánh giá bằng gì? | RAGAS framework: 4 chỉ số tự động, không cần nhãn vàng |

---

## PHẦN 1 — THUẬT TOÁN 1: NAIVE RAG (Đường cơ sở)

### 1.1 Ý tưởng
> Chia tài liệu thành mảnh nhỏ, nhúng thành vector, khi hỏi thì tìm mảnh gần nhất, đưa cho LLM trả lời.

### 1.2 Pipeline chi tiết

```
PDF → [Chunk] → [Embed] → [Vector Index]
                                ↑
Câu hỏi → [Embed câu hỏi] → [Cosine Search] → Top-k chunks → [LLM] → Câu trả lời
```

**Bước 1 — Chunking (chia mảnh):**
- Chia văn bản thành đoạn ~256–512 token
- Chiến lược: fixed-size hoặc sentence-aware
- Vấn đề: chunk cắt đúng giữa câu → mất ngữ cảnh

**Bước 2 — Embedding (nhúng vector):**  
- Mô hình: `multilingual-e5-large` hoặc `bge-m3` (hỗ trợ tiếng Việt)
- Mỗi chunk c → vector **v**_c ∈ ℝ^768
- Công thức: **v** = Encoder(text) — output [CLS] token của transformer

**Bước 3 — Vector Index:**
- Lưu tất cả **v**_c vào FAISS hoặc Chroma
- FAISS dùng HNSW (Hierarchical Navigable Small World) để tìm kiếm gần đúng O(log n)

**Bước 4 — Retrieval (truy xuất):**
- Câu hỏi q → **v**_q = Encoder(q)
- Tính cosine similarity:

$$\text{sim}(\mathbf{v}_q, \mathbf{v}_c) = \frac{\mathbf{v}_q \cdot \mathbf{v}_c}{\|\mathbf{v}_q\| \cdot \|\mathbf{v}_c\|}$$

- Lấy top-k chunk có sim cao nhất

**Bước 5 — Generation:**
- Ghép top-k chunks vào prompt → LLM sinh câu trả lời

### 1.3 Giới hạn của Naive RAG
- ❌ Chỉ tốt với câu hỏi đơn (1 đoạn văn trả lời)
- ❌ Multi-hop: "Ai là cố vấn của tác giả viết bài X?" → cần kết nối nhiều đoạn văn khác nhau
- ❌ Global question: "Xu hướng nghiên cứu chính của lĩnh vực là gì?" → không có mảnh nào trả lời trực tiếp

---

## PHẦN 2 — THUẬT TOÁN 2: RAG + RE-RANKING (Cải tiến trung gian)

### 2.1 Ý tưởng
> Naive RAG dùng bi-encoder (nhúng độc lập) → nhanh nhưng kém chính xác. Re-ranking dùng cross-encoder (xét cặp query-chunk) → chậm hơn nhưng chính xác hơn.

### 2.2 Pipeline

```
Câu hỏi → [Bi-Encoder Retrieve] → 50 candidates
         → [Cross-Encoder Re-rank] → Top-5
         → [LLM] → Câu trả lời
```

**Cross-Encoder hoạt động thế nào:**
- Input: concat(câu_hỏi, chunk) → BERT → score thực sự
- Ký hiệu: score(q, c) = CrossEncoder(q ⊕ c) ∈ [0, 1]
- Vì phải xử lý từng cặp nên chậm hơn (O(k) lần forward pass)
- Chỉ áp dụng cho top-50 từ bước 1, không phải toàn bộ corpus

**Ví dụ cụ thể:**
- Bi-encoder lấy 50 chunks → Cross-encoder cho điểm lại từng cái → chỉ lấy top-5 → LLM
- Kết quả tốt hơn vì cross-encoder "đọc" cả câu hỏi + chunk cùng lúc

### 2.3 Vẫn còn giới hạn
- ❌ Vẫn không giải quyết được multi-hop reasoning qua nhiều tài liệu
- ❌ Không có tri thức cấu trúc (ai liên kết với ai)

---

## PHẦN 3 — THUẬT TOÁN 3: GRAPHRAG — CỐT LÕI

### 3.1 Ý tưởng gốc
> Thay vì chỉ tìm đoạn văn, ta **xây dựng đồ thị tri thức** từ toàn bộ tài liệu. Khi truy vấn, đi theo các cạnh trong đồ thị để thu thập thông tin phân tán ở nhiều nơi.

**Paper gốc:** "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" — Edge et al., Microsoft, 2024

### 3.2 Knowledge Graph là gì?

```
G = (V, E)

V = { "Nguyễn Văn A", "RAG", "BERT", "Bài báo X", "Đại học Y" }
     (thực thể: người, khái niệm, tổ chức, bài báo)

E = { ("Nguyễn Văn A", viết_bởi, "Bài báo X"),
      ("Bài báo X", đề_xuất, "RAG"),
      ("RAG", dựa_trên, "BERT"),
      ("Nguyễn Văn A", thuộc, "Đại học Y") }
     (quan hệ có nhãn)
```


### 3.3 Pipeline GraphRAG đầy đủ

#### PHASE 1 — OFFLINE INDEXING (xây đồ thị — chạy 1 lần)

```
PDF học thuật (50-100 bài)
    ↓
[1] PDF Parser (PyMuPDF)        → raw text
    ↓
[2] Text Chunking               → chunks 256-512 token
    ↓  
[3] Entity & Relation Extract   → LLM prompt → (entity, relation, entity) triplets
    ↓
[4] Graph Construction          → Neo4j: CREATE (:Entity)-[:RELATION]->(:Entity)
    ↓
[5] Community Detection         → Leiden algorithm → nhóm các thực thể liên quan
    ↓
[6] Community Summarization     → LLM tóm tắt từng cộng đồng → lưu vào DB
    ↓
[7] Parallel: Embed chunks      → FAISS vector index (vẫn giữ cho local search)
```

#### PHASE 2 — ONLINE QUERY (trả lời câu hỏi — chạy mỗi lần hỏi)

```
Câu hỏi người dùng
    ↓
[A] Query Classification        → local (thực thể cụ thể) hay global (xu hướng)?
    ↓
[B1] Local Search Path:
     Embed câu hỏi → FAISS → top chunks
     + Graph traversal: tìm thực thể liên quan → lấy neighbors trong Neo4j
     → Merge kết quả
                                         OR
[B2] Global Search Path:
     Lấy community summaries liên quan
     → Map-Reduce: LLM tóm tắt từng community → tổng hợp lại
    ↓
[C] Context Assembly            → ghép chunks + graph context + community summary
    ↓
[D] LLM Generation              → Qwen/Llama/GPT → câu trả lời có trích dẫn
```

### 3.4 Thuật toán trích xuất thực thể & quan hệ

**Prompt template (dùng cho LLM):**
```
Từ đoạn văn sau, hãy trích xuất các thực thể và quan hệ dưới dạng JSON.

Đoạn văn: {chunk_text}

Trả về JSON theo định dạng:
{
  "entities": [
    {"name": "tên thực thể", "type": "PERSON|CONCEPT|PAPER|ORGANIZATION|METHOD"},
    ...
  ],
  "relations": [
    {"source": "thực thể 1", "relation": "tên quan hệ", "target": "thực thể 2"},
    ...
  ]
}
```

**Ví dụ kết quả từ đoạn văn học thuật:**
```json
{
  "entities": [
    {"name": "BERT", "type": "METHOD"},
    {"name": "Devlin et al. 2018", "type": "PAPER"},
    {"name": "Google", "type": "ORGANIZATION"},
    {"name": "pre-training", "type": "CONCEPT"}
  ],
  "relations": [
    {"source": "BERT", "relation": "được_đề_xuất_bởi", "target": "Devlin et al. 2018"},
    {"source": "Devlin et al. 2018", "relation": "từ", "target": "Google"},
    {"source": "BERT", "relation": "sử_dụng", "target": "pre-training"}
  ]
}
```

### 3.5 Thuật toán Community Detection — Leiden Algorithm

Sau khi có đồ thị G = (V, E), dùng **Leiden algorithm** (cải tiến Louvain) để phân cộng đồng:

- **Input:** đồ thị G, độ phân giải γ
- **Output:** phân hoạch V thành các tập con C₁, C₂, ..., Cₖ
- **Mục tiêu:** tối đa hóa modularity Q:

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

Trong đó: A = ma trận kề, k_i = bậc đỉnh i, m = số cạnh, δ = 1 nếu cùng community

**Kết quả:** mỗi community = 1 nhóm thực thể liên quan chặt chẽ  
→ LLM tóm tắt từng community → dùng cho global search

---

## PHẦN 4 — SO SÁNH 3 THUẬT TOÁN

| Tiêu chí | Naive RAG | RAG + Re-rank | **GraphRAG** |
|---|---|---|---|
| Câu hỏi đơn (1-hop) | ✅ Tốt | ✅✅ Tốt hơn | ✅✅ Tốt |
| Câu hỏi multi-hop | ❌ Kém | ❌ Vẫn kém | ✅✅ **Vượt trội** |
| Câu hỏi toàn cục | ❌ Không được | ❌ Không được | ✅ Community summary |
| Tốc độ | ⚡ Nhanh | ⚡ Chậm hơn | 🐢 Chậm hơn nữa |
| Độ phức tạp xây dựng | 🟢 Thấp | 🟡 Trung bình | 🔴 Cao |
| Giải thích được không? | ❌ | ❌ | ✅ (trích dẫn qua graph) |

**Giả thuyết nghiên cứu:** GraphRAG vượt Naive RAG ≥15% F1 trên câu hỏi multi-hop, đổi lại latency cao hơn ~3-5x.

---

## PHẦN 5 — METRICS ĐÁNH GIÁ (RAGAS Framework)

| Chỉ số | Ý nghĩa | Công thức tóm tắt |
|---|---|---|
| **Faithfulness** | Câu trả lời có bịa đặt không? | % claims trong answer được support bởi context |
| **Answer Relevancy** | Có trả lời đúng câu hỏi không? | cosine(embed(câu_hỏi), embed(câu_trả_lời)) |
| **Context Precision** | Các chunk lấy về có liên quan không? | % chunks được dùng / tổng chunks lấy về |
| **Context Recall** | Có lấy đủ thông tin cần thiết không? | % thông tin cần có trong context |

**Điểm mạnh của RAGAS:** không cần nhãn vàng (gold labels) → LLM tự đánh giá → phù hợp nghiên cứu học thuật


**Q1: "Sao không dùng database quan hệ thay vì graph database?"**

Được, nhưng câu hỏi đa bước sẽ đắt. Trong SQL, đi từ BERT → tác giả → tổ chức là hai phép JOIN; mỗi bước nhảy
thêm là thêm một JOIN, và số bước lại phụ thuộc câu hỏi nên phải viết JOIN đệ quy hoặc sinh SQL động. Neo4j lưu
sẵn con trỏ từ node sang các cạnh của nó (index-free adjacency), nên chi phí đi một bước chỉ phụ thuộc số láng
giềng của node đó, không phụ thuộc kích thước bảng. Cypher cũng viết thẳng được mẫu cần tìm:
`MATCH p = allShortestPaths((a)-[:REL*..4]-(b))`. Với quy mô đồ thị của nhóm thì cả hai đều chạy được; chọn graph
database vì code truy vấn ngắn và đọc ra đúng ý đồ thuật toán.



**Q2: "Chi phí tính toán có quá cao không?"**

Tách làm hai phần. Phần dựng đồ thị chạy một lần: mỗi đơn vị văn bản (~1500 ký tự) tốn 1 lượt gọi LLM, 50 PDF
khoảng 1000 lượt; kết quả được cache ra `data/graph/extractions.jsonl` nên chạy lại không tốn thêm. Phần trả lời
mỗi câu hỏi tốn 2 lượt gọi LLM cho local search (1 phân tích câu hỏi + 1 sinh câu trả lời), so với 1 lượt của
Naive RAG; riêng global search tốn thêm mỗi cụm 1 lượt ở bước map nên nhóm chạy song song và chỉ lấy 5 cụm liên
quan nhất. Truy vấn trên đồ thị (mili-giây) không đáng kể so với thời gian gọi LLM. Nói cách khác, GraphRAG đắt ở
khâu chuẩn bị dữ liệu, không đắt ở khâu phục vụ — và đó chính là đánh đổi mà đề tài đo bằng cột latency.



**Q3: "Tại sao chọn tiếng Việt?"**

Vì đây là khoảng trống thật: các so sánh định lượng giữa naive RAG, RAG + re-ranking và GraphRAG hầu hết làm trên
tài liệu tiếng Anh, chưa có số liệu trên tài liệu học thuật tiếng Việt. Tiếng Việt cũng có mấy khó khăn riêng đáng
để khảo sát: dấu và chuẩn Unicode không đồng nhất giữa các PDF, thuật ngữ lẫn lộn Việt - Anh (`tiền huấn luyện`
và `pre-training` là một khái niệm), tên riêng viết theo nhiều cách khác nhau. Những chỗ này ảnh hưởng trực tiếp
tới chất lượng đồ thị, nên kết quả của nhóm nói được điều mà thí nghiệm trên tiếng Anh không nói.



