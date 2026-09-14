# NHÓM 7
> **Đề tài:** Nghiên cứu và so sánh các kỹ thuật RAG nâng cao (GraphRAG) cho hệ thống hỏi–đáp tài liệu học thuật tiếng Việt 
> **Thời gian:** 4 ngày (Tuần 3–4) 
> **Mục tiêu cuối:** Demo chạy được 3 pipeline, so sánh kết quả trên cùng bộ câu hỏi

---

## 1. TỔNG QUAN HỆ THỐNG

### 1.1 Bài toán nhóm giải quyết

Nhóm xây **hệ thống hỏi-đáp** trên tài liệu học thuật tiếng Việt:

```
Người dùng nhập câu hỏi → Hệ thống tìm thông tin trong PDF → Trả lời + trích dẫn nguồn
```

**Ví dụ:** Có 50 bài báo PDF về NLP. Người dùng hỏi "BERT được huấn luyện như thế nào?" → Hệ thống tìm đúng đoạn nói về BERT → Trả lời kèm dẫn nguồn "theo bài báo X, trang Y".

### 1.2 Ba phương pháp — từ đơn giản đến phức tạp

#### Phương pháp 1: Naive RAG (đường cơ sở)

```
Ý tưởng: Cắt tài liệu thành mảnh nhỏ → biến thành vector số → khi hỏi, tìm mảnh "gần" nhất
```

**Cách hoạt động từng bước:**

1. **Chunking (chia mảnh):** Mỗi PDF được chia thành đoạn 256–512 token (~150–300 từ). Giống cắt sách thành thẻ flashcard.

2. **Embedding (nhúng vector):** Mỗi đoạn văn được biến thành 1 dãy 768 số thực (vector). Hai đoạn nói về cùng chủ đề → vector của chúng "gần nhau" trong không gian 768 chiều. Mô hình embedding đọc đoạn văn → output vector.

3. **Indexing (lập chỉ mục):** Lưu tất cả vector vào FAISS — một thư viện tìm kiếm nhanh. FAISS dùng thuật toán HNSW để tìm kiếm gần đúng trong O(log n) thay vì O(n).

4. **Retrieval (truy xuất):** Khi có câu hỏi:
  - Biến câu hỏi thành vector (cùng mô hình embedding)
  - Tính **cosine similarity** giữa vector câu hỏi và mọi vector chunk:
   ```
   sim(q, c) = (q · c) / (|q| × |c|)
   ```
  - Lấy top-5 chunk có điểm cao nhất

5. **Generation (sinh câu trả lời):** Ghép 5 chunk vào prompt → gửi cho LLM → LLM đọc ngữ cảnh và sinh câu trả lời.

**Giới hạn:**
- Tốt với câu hỏi đơn (1 đoạn văn trả lời được)
- Fail với câu hỏi multi-hop: "Tác giả của BERT thuộc tổ chức nào?" → thông tin nằm ở 2 đoạn khác nhau
- Fail với câu hỏi tổng quát: "Xu hướng nghiên cứu chính là gì?" → không có mảnh nào trả lời trực tiếp

#### Phương pháp 2: RAG + Re-ranking (cải tiến trung gian)

```
Ý tưởng: Tìm nhiều ứng viên trước (nhanh, thô) → dùng model thứ 2 lọc lại (chậm, chính xác)
```

**Cách hoạt động:**

1. **Bước 1 — Bi-encoder retrieve:** Giống Naive RAG, dùng embedding tìm top-50 chunk (thay vì top-5)

2. **Bước 2 — Cross-encoder re-rank:**
  - Bi-encoder: nhúng câu hỏi và chunk **riêng biệt** → so sánh vector (nhanh nhưng mất ngữ cảnh)
  - Cross-encoder: đưa **cặp** (câu hỏi + chunk) vào cùng model → model đọc cả hai cùng lúc → cho điểm chính xác hơn
  ```
  score = CrossEncoder("câu hỏi [SEP] nội dung chunk") → [0, 1]
  ```
  - Chạy cross-encoder trên 50 chunk → sắp xếp theo điểm → lấy top-5

3. **Bước 3 — Generation:** Giống Naive RAG, đưa top-5 (đã re-rank) cho LLM

**So với Naive RAG:** Chính xác hơn vì cross-encoder "đọc hiểu" cặp câu hỏi-đoạn văn. Nhưng vẫn không giải quyết được multi-hop.

#### Phương pháp 3: GraphRAG (cốt lõi đề tài)

```
Ý tưởng: Ngoài vector, xây thêm "bản đồ quan hệ" (knowledge graph) giữa các khái niệm.
Khi truy vấn, đi theo các cạnh trong bản đồ để thu thập thông tin từ nhiều nơi.
```

**Knowledge Graph là gì?**

Đồ thị G = (V, E) trong đó:
- V = tập **thực thể** (đỉnh): tác giả, phương pháp, bài báo, tổ chức, khái niệm
- E = tập **quan hệ** (cạnh có nhãn): viết_bởi, đề_xuất, sử_dụng, thuộc_về

```
Ví dụ:

 [BERT] --đề_xuất_bởi--> [Devlin et al. 2018]
 [Devlin et al. 2018] --từ--> [Google]
 [BERT] --sử_dụng--> [Pre-training]
 [GPT] --đề_xuất_bởi--> [Radford et al. 2018]
 [Radford et al. 2018] --từ--> [OpenAI]

→ Hỏi "BERT và GPT có gì chung?" → đi theo cạnh: cả hai đều sử_dụng pre-training
→ Hỏi "Tác giả BERT thuộc tổ chức nào?" → đi 2 bước: BERT → Devlin → Google
```

**Pipeline GraphRAG đầy đủ:**

```
═══════════════ PHASE 1: OFFLINE (chạy 1 lần) ═══════════════

PDF (50-100 bài)
 │
 ├──→ [Text Extraction] → raw text
 │   │
 │   ├──→ [Chunking 512 token] → chunks
 │   │   │
 │   │   └──→ [Embedding bge-m3] → FAISS vector index ←── dùng cho Local Search
 │   │
 │   └──→ [LLM Entity Extraction] → (thực thể, quan hệ, thực thể) triplets
 │      │
 │      └──→ [Neo4j Graph] → Knowledge Graph
 │         │
 │         └──→ [Leiden Algorithm] → phân cụm communities
 │            │
 │            └──→ [LLM Summarize] → community summaries ←── dùng cho Global Search

═══════════════ PHASE 2: ONLINE (mỗi câu hỏi) ═══════════════

Câu hỏi người dùng
 │
 └──→ [Phân loại câu hỏi]
    │
    ├──→ LOCAL (câu hỏi cụ thể, VD: "BERT là gì?")
    │   │
    │   ├── Vector search (FAISS) → top chunks
    │   └── Graph traversal (Neo4j) → thực thể + quan hệ liên quan
    │   │
    │   └──→ Merge kết quả
    │
    └──→ GLOBAL (câu hỏi tổng quát, VD: "Xu hướng nghiên cứu chính?")
       │
       └── Lấy community summaries liên quan
         → Map: LLM tóm tắt từng community
         → Reduce: tổng hợp lại
    │
    └──→ [Context Assembly] → ghép chunks + graph context + summaries
       │
       └──→ [LLM Generation] → Câu trả lời + Trích dẫn nguồn
```

**Thuật toán Leiden (phân cụm):**

Sau khi có đồ thị, Leiden algorithm nhóm các thực thể liên quan chặt chẽ thành "cộng đồng" (community). Ví dụ: tất cả thực thể liên quan đến "computer vision" thành 1 cộng đồng, tất cả liên quan đến "NLP" thành 1 cộng đồng.

Mục tiêu: tối đa hóa **modularity Q** — các đỉnh trong cùng community có nhiều cạnh nối với nhau hơn kỳ vọng ngẫu nhiên.

```
Q = (1/2m) × Σ [A_ij - (k_i × k_j)/(2m)] × δ(c_i, c_j)

Trong đó:
 A_ij = 1 nếu có cạnh giữa i và j
 k_i = bậc (số cạnh) của đỉnh i
 m = tổng số cạnh
 δ = 1 nếu i và j cùng community, 0 nếu khác
```

Sau khi phân cụm, LLM tóm tắt từng community → dùng cho Global Search.

### 1.3 So sánh 3 phương pháp

| Tiêu chí | Naive RAG | RAG + Re-rank | GraphRAG |
|---|---|---|---|
| Câu hỏi đơn (1-hop) | Tốt | Tốt hơn | Tốt |
| Câu hỏi multi-hop | Kém | Vẫn kém | **Vượt trội** |
| Câu hỏi tổng quát | Không được | Không được | Community summary |
| Tốc độ phản hồi | ~1-2s | ~2-4s | ~5-10s |
| Độ phức tạp cài đặt | Thấp | Trung bình | Cao |
| Giải thích được? | | | Trích dẫn qua graph |

### 1.4 Đánh giá bằng RAGAS Framework

RAGAS đo chất lượng hệ thống RAG mà **không cần nhãn vàng** (gold labels). LLM tự đánh giá:

| Chỉ số | Đo cái gì | Cách tính đơn giản |
|---|---|---|
| **Faithfulness** | Câu trả lời có bịa không? | % claims trong answer được support bởi context |
| **Answer Relevancy** | Trả lời có đúng câu hỏi không? | cosine(embed(câu_hỏi), embed(câu_trả_lời)) |
| **Context Precision** | Chunks lấy về có liên quan không? | % chunks thực sự được dùng / tổng chunks |
| **Context Recall** | Có lấy đủ thông tin chưa? | % thông tin cần thiết có trong context |

---

## 2. TÓM TẮT 2 BÀI BÁO

### 2.1 Bài A — "Agentic GraphRAG: A Comprehensive Survey"

**Tác giả:** Zihan Chen et al. 
**Nội dung chính:** Khảo sát toàn cảnh cách kết hợp LLM Agent với GraphRAG.

**Điểm quan trọng cho nhóm:**

**(a) Bảng phân loại 4×3:**
- 4 vai trò Agent: Planning, Retrieval, Reasoning, Reflection
- 3 thao tác trên đồ thị: Construction, Retrieval, Reasoning
- Mỗi ô = 1 cách Agent thao tác với Graph → giúp nhóm hiểu "có bao nhiêu cách làm GraphRAG"

**(b) 6 lỗi phổ biến của GraphRAG (G1-G6):**

| Mã | Lỗi | Giải thích | Ảnh hưởng đến nhóm |
|---|---|---|---|
| G1 | Truy hồi tĩnh | Luôn lấy top-k cố định, không biết lọc theo câu hỏi | Nhóm cần adaptive retrieval |
| G2 | Suy luận yếu | Không đi sâu qua nhiều bước trên đồ thị | Cần graph traversal > 1 hop |
| G3 | Không có phản hồi | Lấy sai context nhưng không biết sửa | Có thể bỏ qua cho demo |
| G4 | Sai cấp độ chi tiết | Hỏi tổng quát nhưng lấy chi tiết (hoặc ngược lại) | Cần phân loại local/global |
| G5 | Đồ thị đóng băng | Graph không cập nhật khi có tài liệu mới | Không ảnh hưởng demo |
| G6 | Ngữ cảnh quá lớn | Lấy quá nhiều thông tin → LLM bị loãng | Giới hạn context window |

**(c) Cây quyết định (Figure 4):** Giúp chọn khi nào dùng Naive RAG, khi nào cần GraphRAG:
- Câu hỏi chỉ cần 1 đoạn văn → Naive RAG đủ tốt
- Câu hỏi cần kết nối nhiều thông tin → GraphRAG
- Câu hỏi tổng quát → GraphRAG Global Search

### 2.2 Bài B — "GraphRAG for Domain-Specific LLMs"

**Tác giả:** Qinggang Zhang et al. (arXiv 2501.13958) 
**Nội dung chính:** Hệ thống hóa GraphRAG thành 3 giai đoạn rõ ràng.

**Điểm quan trọng cho nhóm:**

**(a) 3 giai đoạn GraphRAG:**

```
Giai đoạn 1: Knowledge Organization (Tổ chức tri thức)
 → Xây Knowledge Graph từ tài liệu
 → Kỹ thuật: entity extraction, relation extraction, graph construction

Giai đoạn 2: Knowledge Retrieval (Truy hồi tri thức)
 → Tìm thông tin liên quan từ Graph
 → 6 kỹ thuật: subgraph matching, path finding, community detection,
         embedding similarity, hybrid search, iterative retrieval

Giai đoạn 3: Knowledge Integration (Tích hợp tri thức)
 → Đưa thông tin từ Graph vào LLM để sinh câu trả lời
 → Kỹ thuật: prompt engineering, context assembly, graph-to-text
```

**(b) Bảng so sánh RAG vs GraphRAG (Table I):**
- RAG truyền thống: flat text chunks, semantic similarity
- GraphRAG: structured knowledge, relation-aware retrieval
- Nhóm dùng bảng này trực tiếp cho phần "so sánh" trong báo cáo

**(c) 6 kỹ thuật truy hồi trên đồ thị:**

| # | Kỹ thuật | Mô tả | Nhóm dùng? |
|---|---|---|---|
| 1 | Subgraph matching | Tìm subgraph khớp pattern câu hỏi | Nên thử |
| 2 | Path finding | Tìm đường đi giữa 2 entity | Core cho multi-hop |
| 3 | Community-based | Dùng community summary | Core cho global search |
| 4 | Embedding similarity | Embed node → cosine search | Hybrid approach |
| 5 | Hybrid search | Kết hợp vector + graph | Local search |
| 6 | Iterative retrieval | Lặp nhiều vòng truy hồi | Nâng cao, sau demo |

### 2.3 Nhóm rút ra gì?

1. **Từ Bài A:** Biết lỗi hay gặp → phòng tránh. Cây quyết định → phân loại câu hỏi.
2. **Từ Bài B:** 3 giai đoạn rõ ràng → dễ chia module. Bảng so sánh → dùng cho báo cáo.
3. **Cho demo:** Focus vào Local Search (vector + graph traversal) và Global Search (community summary) — hai path chính của GraphRAG.

---

## 3. PHÂN CÔNG 4 NGÀY

### Vai trò từng thành viên

| Thành viên | Vai trò chính | Module phụ trách |
|---|---|---|
| Hoàng Thị Hoạt | Naive RAG Engineer | Chunking, embedding, FAISS, retrieval pipeline |
| Ngọc Long | Re-ranking Engineer | Cross-encoder, re-ranking module, hybrid search |
| Hùng | GraphRAG Engineer | Entity extraction, Neo4j/NetworkX, graph retrieval |
| Tuấn Anh | Demo + Evaluation | Streamlit UI, RAGAS evaluation, data preparation |

### Lịch trình chi tiết

Setup + Học nền tảng + Dữ liệu


| Hoàng Thị Hoạt | Học: Embedding là gì, cosine similarity, chunking strategies. Đọc LangChain RAG Tutorial | Cài đặt env, viết script load PDF + chunking + embedding | Script chạy được: PDF → chunks → vectors |
| Ngọc Long | Học: Bi-encoder vs Cross-encoder, cách re-ranking hoạt động. Đọc SBERT docs | Cài đặt env, test cross-encoder model trên ví dụ đơn giản | Hiểu re-ranking, chạy được cross-encoder |
| Hùng | Học: Knowledge Graph là gì, Neo4j cơ bản (Cypher query). Xem Neo4j Getting Started | Cài Neo4j Docker, tạo graph mẫu, test CRUD query | Neo4j chạy, biết tạo node/edge/query |
| Tuấn Anh | Thu thập 10-20 PDF tiếng Việt (giáo trình, bài báo mở). Clean text | Cài Streamlit, tạo skeleton UI (input box, output area, sidebar chọn method) | Có bộ PDF sạch + UI skeleton |


Naive RAG hoàn chỉnh + Bắt đầu module khác


| Hoàng Thị Hoạt | Hoàn thiện Naive RAG: FAISS index + retrieval + LLM generation | Test với 5 câu hỏi mẫu, fix bugs, tách thành module import được | `naive_rag.py` chạy end-to-end |
| Ngọc Long | Viết re-ranking module: nhận top-50 từ FAISS → cross-encoder → top-5 | Tích hợp với Hoàng Thị Hoạt's retrieval, test so sánh kết quả | `rerank_rag.py` chạy, thấy khác biệt với naive |
| Hùng | Viết prompt entity extraction + script parse JSON output | Chạy extraction trên 20 chunks đầu tiên, load triplets vào Neo4j | Có graph ~50-100 entities trong Neo4j |
| Tuấn Anh | Kết nối UI với Hoàng Thị Hoạt's naive_rag module, hiển thị answer + sources | Viết 20 câu hỏi đánh giá (10 single-hop, 10 multi-hop) | UI demo được Naive RAG + có bộ eval |


 — GraphRAG core + Tích hợp


| Hoàng Thị Hoạt | Chạy entity extraction cho toàn bộ PDF (song song với Hùng) | Hỗ trợ Hùng tích hợp graph retrieval, viết context assembly | Context assembly module |
| Ngọc Long | Tích hợp re-ranking vào UI | Viết hybrid search: kết hợp vector results + graph results | Hybrid search module cho local search |
| Hùng | Implement graph traversal: query → tìm entity → lấy neighbors 1-2 hop | Implement community detection (Leiden) + LLM summarize communities | `graph_rag.py` với local + global search |
| Tuấn Anh | Thêm tab so sánh 3 phương pháp vào UI | Chạy RAGAS evaluation trên Naive RAG (bộ 20 câu hỏi) | UI 3 tabs + kết quả eval Naive RAG |

 — Tích hợp + Demo + Đánh giá


| Hoàng Thị Hoạt | Fix bugs tích hợp, tối ưu retrieval | Chạy RAGAS cho cả 3 phương pháp (cùng Tuấn Anh) | Bảng so sánh 3 phương pháp |
| Ngọc Long | Fix bugs re-ranking, test edge cases | Chuẩn bị slide demo: kiến trúc + kết quả | Slide demo |
| Hùng | Fix bugs GraphRAG, test multi-hop questions | Demo GraphRAG trả lời câu multi-hop | GraphRAG chạy được multi-hop |
| Tuấn Anh | Polish UI: thêm graph visualization, timing display | Record demo video / chụp screenshots | Demo hoàn chỉnh |



---

NỘI DUNG HỌC CHI TIẾT — TỪNG THÀNH VIÊN

### Hoàng Thị Hoạt — Naive RAG Pipeline

**Kiến thức cần nắm:**

1. **Text Embedding là gì?** Biến đoạn văn thành vector số sao cho đoạn nghĩa giống nhau → vector gần nhau.
  - 📄 Đọc: [Sentence-BERT Paper giải thích](https://www.sbert.net/) — mục "How it Works"
  - 📄 Đọc: Bài B, Section III.A (Knowledge Organization) — cách tổ chức text cho RAG
  - 🔗 Model nhóm dùng: [BGE-M3](https://huggingface.co/BAAI/bge-m3) — hỗ trợ tiếng Việt

2. **Chunking strategies:** Cách cắt văn bản để không mất ngữ cảnh.
  - 📄 Đọc: [LangChain Text Splitters docs](https://python.langchain.com/docs/how_to/#text-splitters)
  - Dùng `RecursiveCharacterTextSplitter` — cắt theo câu trước, cố gắng không cắt giữa câu
  - Tham số: `chunk_size=512, chunk_overlap=50`

3. **FAISS vector search:** Thư viện tìm kiếm vector nhanh của Meta.
  - 📄 Đọc: [FAISS Wiki — Getting Started](https://github.com/facebookresearch/faiss/wiki/Getting-started)
  - Dùng `IndexFlatIP` (inner product) cho cosine similarity
  - LangChain wrapper: `FAISS.from_documents()`

4. **RAG Pipeline với LangChain:**
  - 📄 Đọc: [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
  - Đây là tutorial chính, làm theo từng bước là xong Naive RAG

### Ngọc Long — Re-ranking Module

**Kiến thức cần nắm:**

1. **Bi-encoder vs Cross-encoder:**
  - Bi-encoder: encode câu hỏi và chunk **riêng biệt** → nhanh, dùng cho giai đoạn retrieve
  - Cross-encoder: encode **cặp** (câu hỏi, chunk) cùng lúc → chậm hơn nhưng chính xác hơn, dùng cho re-rank
  - 📄 Đọc: [SBERT — Retrieve & Re-Rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html)
  - 📄 Đọc: Bài B, Section III.B (Knowledge Retrieval) — các kỹ thuật retrieval

2. **Cross-encoder models:**
  - Model đề xuất: `cross-encoder/ms-marco-MiniLM-L-12-v2` (nhanh, tốt)
  - Hoặc: `BAAI/bge-reranker-v2-m3` (đa ngữ, hỗ trợ tiếng Việt tốt hơn)
  - 📄 Đọc: [HuggingFace model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2)

3. **Tích hợp vào pipeline:**
  - Bước 1: FAISS retrieve top-50 (bi-encoder, nhanh)
  - Bước 2: Cross-encoder score 50 cặp → sort → top-5
  - Bước 3: Top-5 sau re-rank → LLM generation

### Hùng — Knowledge Graph + GraphRAG

**Kiến thức cần nắm:**

1. **Knowledge Graph cơ bản:**
  - Graph G = (V, E): V = thực thể (nodes), E = quan hệ (edges)
  - Triple: (subject, predicate, object) — VD: (BERT, đề_xuất_bởi, Google)
  - 📄 Đọc: [Neo4j Graph Database Concepts](https://neo4j.com/docs/getting-started/graph-database/)

2. **Neo4j — Cơ sở dữ liệu đồ thị:**
  - Cài đặt: `docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:latest`
  - Cypher query language (SQL cho graph):
   ```cypher
   // Tạo node
   CREATE (b:Entity {name: "BERT", type: "METHOD"})
   
   // Tạo quan hệ
   MATCH (a:Entity {name: "BERT"}), (b:Entity {name: "Google"})
   CREATE (a)-[:PROPOSED_BY]->(b)
   
   // Truy vấn 2-hop
   MATCH (a:Entity {name: "BERT"})-[*1..2]-(b)
   RETURN b
   ```
  - 📄 Đọc: [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/introduction/)
  - Hoặc dùng Neo4j Aura (cloud free tier): https://neo4j.com/cloud/aura-free/

3. **Entity & Relation Extraction:** Dùng LLM prompt để trích xuất.
  - 📄 Đọc: Bài A, Section về Graph Construction — cách các hệ thống SOTA trích xuất
  - 📄 Đọc: Bài B, Section III.A — knowledge organization techniques
  - Prompt template: xem phần Code Mẫu bên dưới

4. **Leiden Algorithm (phân cụm):**
  - 📄 Đọc: [python-igraph Leiden](https://python-igraph.org/) — `igraph` có implement sẵn Leiden
  - Hoặc: `pip install leidenalg` + `igraph`
  - 📄 Đọc: Bài A — phần Community Detection, giải thích modularity Q

5. **GraphRAG paper gốc (Microsoft):**
  - 📄 Đọc: [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) — Edge et al., 2024
  - Focus: Section 3 (Method) và Section 4 (Experiments)
  - Đây là paper nhóm implement theo

### Tuấn Anh — Demo UI + Đánh giá

**Kiến thức cần nắm:**

1. **Streamlit — Framework UI cho Python:**
  - 📄 Đọc: [Streamlit Get Started](https://docs.streamlit.io/get-started)
  - Đọc mục: Chat elements, Layouts, Session State
  - Chạy: `streamlit run app.py`

2. **RAGAS — Framework đánh giá RAG:**
  - 📄 Đọc: [RAGAS Documentation](https://docs.ragas.io/en/latest/)
  - Focus: Metrics (faithfulness, answer_relevancy, context_precision, context_recall)
  - Tutorial: [RAGAS Quick Start](https://docs.ragas.io/en/latest/getstarted/index.html)
  - 📄 Đọc: Bài A — Section Evaluation, cách các hệ thống SOTA đánh giá

3. **Data Preparation:**
  - Thu thập PDF tiếng Việt: giáo trình, bài báo mở từ tạp chí Việt Nam
  - Clean text: chuẩn hóa Unicode, bỏ header/footer PDF, fix lỗi OCR
  - Viết bộ eval: 10 câu single-hop + 10 câu multi-hop (xem mẫu bên dưới)

---

## 5. CÀI ĐẶT MÔI TRƯỜNG

### 5.1 Requirements

```
# requirements.txt
# === Core ===
langchain>=0.3.0
langchain-community>=0.3.0
langchain-groq>=0.2.0     # LLM qua Groq API (miễn phí)
# langchain-openai>=0.2.0   # Hoặc dùng OpenAI API (trả phí)

# === Embedding & Search ===
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
chromadb>=0.5.0        # Backup cho FAISS

# === Re-ranking ===
# (sentence-transformers đã cài ở trên, dùng CrossEncoder)

# === Knowledge Graph ===
neo4j>=5.20.0         # Neo4j Python driver
igraph>=0.11.0         # Leiden algorithm
leidenalg>=0.10.0       # Leiden implementation

# === PDF Processing ===
pymupdf>=1.24.0        # PyMuPDF để đọc PDF
# hoặc: pypdf>=4.0.0

# === Evaluation ===
ragas>=0.2.0
datasets>=2.20.0

# === Demo UI ===
streamlit>=1.37.0

# === Utilities ===
python-dotenv>=1.0.0
tqdm>=4.66.0
pandas>=2.2.0
matplotlib>=3.9.0       # Vẽ biểu đồ so sánh
```

### 5.2 Cài đặt nhanh

```bash
# 1. Clone repo
git clone https://github.com/nhom7/graphrag-demo.git
cd graphrag-demo

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Cài Neo4j (chọn 1 trong 2)
# Cách A: Docker (khuyên dùng)
docker run -d \
 --name neo4j \
 -p 7474:7474 -p 7687:7687 \
 -e NEO4J_AUTH=neo4j/graphrag123 \
 neo4j:latest

# Cách B: Neo4j Aura (cloud miễn phí)
# Đăng ký tại: https://neo4j.com/cloud/aura-free/

# 5. Tạo file .env
echo 'GROQ_API_KEY=your_groq_key_here' > .env
echo 'NEO4J_URI=bolt://localhost:7687' >> .env
echo 'NEO4J_USER=neo4j' >> .env
echo 'NEO4J_PASSWORD=graphrag123' >> .env

# 6. Lấy Groq API key (miễn phí)
# Đăng ký tại: https://console.groq.com/
# Vào API Keys → Create API Key

# 7. Chạy demo
streamlit run app.py
```

---

## 6. CẤU TRÚC THƯ MỤC DỰ ÁN

```
graphrag-demo/
├── README.md         # File này
├── requirements.txt
├── .env            # API keys (KHÔNG push lên GitHub)
├── .gitignore
│
├── data/
│  ├── pdfs/         # PDF gốc tiếng Việt
│  ├── processed/       # Text đã clean
│  └── eval/
│    └── questions.json   # Bộ câu hỏi đánh giá
│
├── src/
│  ├── __init__.py
│  ├── config.py       # Load .env, constants
│  ├── pdf_loader.py     # Đọc + clean PDF
│  ├── chunker.py       # Text splitting
│  ├── embedder.py      # Embedding model wrapper
│  ├── naive_rag.py      # Pipeline 1: Naive RAG
│  ├── rerank_rag.py     # Pipeline 2: RAG + Re-ranking
│  ├── graph_builder.py    # Entity extraction + graph construction
│  ├── graph_rag.py      # Pipeline 3: GraphRAG (local + global)
│  └── evaluator.py      # RAGAS evaluation
│
├── app.py           # Streamlit demo UI
│
├── notebooks/         # Jupyter notebooks thử nghiệm
│  ├── 01_explore_data.ipynb
│  ├── 02_naive_rag.ipynb
│  ├── 03_reranking.ipynb
│  └── 04_graphrag.ipynb
│
└── results/
  ├── eval_naive.json    # Kết quả RAGAS Naive RAG
  ├── eval_rerank.json    # Kết quả RAGAS Re-ranking
  ├── eval_graphrag.json   # Kết quả RAGAS GraphRAG
  └── comparison.png     # Biểu đồ so sánh
```

### .gitignore

```
.env
venv/
__pycache__/
*.pyc
data/pdfs/
*.faiss
.ipynb_checkpoints/
```

---

## 7. CODE MẪU KHỞI ĐẦU

### 7.1 Config chung (`src/config.py`)

```python
import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.1-70b-versatile" # Miễn phí trên Groq

# Embedding
EMBEDDING_MODEL = "BAAI/bge-m3" # Đa ngữ, hỗ trợ tiếng Việt
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag123")

# Retrieval
TOP_K_RETRIEVE = 50  # Số chunks lấy từ FAISS (cho re-ranking)
TOP_K_FINAL = 5    # Số chunks cuối cùng đưa cho LLM
GRAPH_HOPS = 2     # Số bước nhảy trên đồ thị
```

### 7.2 PDF Loader (`src/pdf_loader.py`)

```python
"""Đọc và clean PDF tiếng Việt."""
import os
import re
import fitz # PyMuPDF

def load_pdfs(pdf_dir: str) -> list[dict]:
  """Đọc tất cả PDF trong thư mục, trả về list {text, source, page}."""
  documents = []
  for filename in os.listdir(pdf_dir):
    if not filename.endswith(".pdf"):
      continue
    filepath = os.path.join(pdf_dir, filename)
    doc = fitz.open(filepath)
    for page_num, page in enumerate(doc):
      text = page.get_text()
      text = clean_vietnamese_text(text)
      if len(text.strip()) > 50: # Bỏ trang gần trống
        documents.append({
          "text": text,
          "source": filename,
          "page": page_num + 1
        })
    doc.close()
  print(f"Đã đọc {len(documents)} trang từ {pdf_dir}")
  return documents


def clean_vietnamese_text(text: str) -> str:
  """Chuẩn hóa text tiếng Việt."""
  import unicodedata
  text = unicodedata.normalize("NFC", text)   # Chuẩn hóa Unicode
  text = re.sub(r'\n{3,}', '\n\n', text)    # Bỏ nhiều dòng trống
  text = re.sub(r'[ \t]{2,}', ' ', text)    # Bỏ nhiều space
  text = re.sub(r'(\d+)\s*\n', '', text)    # Bỏ số trang
  return text.strip()
```

### 7.3 Naive RAG (`src/naive_rag.py`)

```python
"""Pipeline 1: Naive RAG — baseline."""
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.config import *


# ===== BƯỚC 1: Load & Chunk =====
def load_and_chunk(pdf_dir: str):
  """Đọc PDF và chia thành chunks."""
  docs = []
  for f in os.listdir(pdf_dir):
    if f.endswith(".pdf"):
      loader = PyMuPDFLoader(os.path.join(pdf_dir, f))
      docs.extend(loader.load())

  splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""] # Ưu tiên cắt theo đoạn/câu
  )
  chunks = splitter.split_documents(docs)
  print(f"Tạo {len(chunks)} chunks từ {len(docs)} trang PDF")
  return chunks


# ===== BƯỚC 2: Embedding & Index =====
def build_index(chunks):
  """Tạo FAISS vector index từ chunks."""
  embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}, # Dùng "cuda" nếu có GPU
  )
  vectorstore = FAISS.from_documents(chunks, embeddings)
  print(f"Đã tạo FAISS index với {len(chunks)} vectors")
  return vectorstore


# ===== BƯỚC 3: RAG Chain =====
def build_naive_rag(vectorstore):
  """Tạo Naive RAG pipeline."""
  llm = ChatGroq(
    model_name=LLM_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.1
  )

  prompt_template = """Dựa vào ngữ cảnh sau đây, hãy trả lời câu hỏi bằng tiếng Việt.
Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói "Tôi không tìm thấy thông tin liên quan."
Luôn trích dẫn nguồn (tên file, trang) khi trả lời.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""

  prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
  )

  qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
      search_kwargs={"k": TOP_K_FINAL}
    ),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
  )
  return qa_chain


# ===== CHẠY =====
def query_naive_rag(qa_chain, question: str) -> dict:
  """Hỏi và trả lời."""
  import time
  start = time.time()
  result = qa_chain.invoke({"query": question})
  latency = time.time() - start

  return {
    "answer": result["result"],
    "sources": [
      f"{doc.metadata.get('source', '?')} — trang {doc.metadata.get('page', '?')}"
      for doc in result["source_documents"]
    ],
    "latency": round(latency, 2),
    "method": "Naive RAG"
  }


# ===== MAIN =====
if __name__ == "__main__":
  chunks = load_and_chunk("data/pdfs")
  vectorstore = build_index(chunks)
  qa = build_naive_rag(vectorstore)

  # Test
  result = query_naive_rag(qa, "RAG là gì?")
  print(f"\nCâu trả lời: {result['answer']}")
  print(f"Nguồn: {result['sources']}")
  print(f"Thời gian: {result['latency']}s")
```

### 7.4 Re-ranking (`src/rerank_rag.py`)

```python
"""Pipeline 2: RAG + Re-ranking."""
import time
from sentence_transformers import CrossEncoder
from src.naive_rag import load_and_chunk, build_index
from src.config import *
from langchain_groq import ChatGroq


# Cross-encoder model (load 1 lần)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")


def rerank_retrieve(vectorstore, question: str, top_k_initial=50, top_k_final=5):
  """Retrieve + Re-rank: lấy nhiều rồi lọc lại."""
  # Bước 1: Bi-encoder retrieve (nhanh, thô)
  initial_docs = vectorstore.similarity_search(question, k=top_k_initial)

  # Bước 2: Cross-encoder re-rank (chậm, chính xác)
  pairs = [(question, doc.page_content) for doc in initial_docs]
  scores = reranker.predict(pairs)

  # Sắp xếp theo điểm cross-encoder
  scored_docs = sorted(
    zip(initial_docs, scores),
    key=lambda x: x[1],
    reverse=True
  )

  return [doc for doc, score in scored_docs[:top_k_final]]


def query_rerank_rag(vectorstore, question: str) -> dict:
  """Pipeline Re-ranking hoàn chỉnh."""
  start = time.time()

  # Retrieve + Re-rank
  top_docs = rerank_retrieve(vectorstore, question)

  # Generation
  context = "\n\n---\n\n".join([doc.page_content for doc in top_docs])

  llm = ChatGroq(model_name=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.1)

  prompt = f"""Dựa vào ngữ cảnh sau đây, hãy trả lời câu hỏi bằng tiếng Việt.
Nếu không tìm thấy thông tin, hãy nói rõ. Trích dẫn nguồn khi trả lời.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""

  response = llm.invoke(prompt)
  latency = time.time() - start

  return {
    "answer": response.content,
    "sources": [
      f"{doc.metadata.get('source', '?')} — trang {doc.metadata.get('page', '?')}"
      for doc in top_docs
    ],
    "latency": round(latency, 2),
    "method": "RAG + Re-ranking"
  }
```

### 7.5 Entity Extraction (`src/graph_builder.py`)

```python
"""Trích xuất thực thể & quan hệ → xây Knowledge Graph."""
import json
import re
from tqdm import tqdm
from langchain_groq import ChatGroq
from neo4j import GraphDatabase
from src.config import *


llm = ChatGroq(model_name=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)

EXTRACT_PROMPT = """Bạn là chuyên gia trích xuất thông tin. Từ đoạn văn học thuật sau, hãy trích xuất:
1. Các thực thể (entity): người, phương pháp, bài báo, tổ chức, khái niệm
2. Các quan hệ (relation) giữa chúng

Đoạn văn:
\"\"\"
{text}
\"\"\"

Trả về JSON duy nhất (không giải thích thêm):
{{
 "entities": [
  {{"name": "tên thực thể", "type": "PERSON|METHOD|PAPER|ORG|CONCEPT"}}
 ],
 "relations": [
  {{"source": "thực thể 1", "relation": "tên quan hệ", "target": "thực thể 2"}}
 ]
}}"""


def extract_from_chunk(chunk_text: str) -> dict:
  """Trích xuất entities và relations từ 1 chunk."""
  try:
    response = llm.invoke(EXTRACT_PROMPT.format(text=chunk_text))
    # Parse JSON từ response
    content = response.content
    # Tìm JSON trong response (phòng trường hợp model trả thêm text)
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
      return json.loads(json_match.group())
  except Exception as e:
    print(f"Lỗi extraction: {e}")
  return {"entities": [], "relations": []}


def build_knowledge_graph(chunks: list, neo4j_uri=NEO4J_URI,
             neo4j_user=NEO4J_USER, neo4j_pass=NEO4J_PASSWORD):
  """Trích xuất từ tất cả chunks và load vào Neo4j."""
  driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

  all_entities = []
  all_relations = []

  print("Đang trích xuất entities & relations...")
  for chunk in tqdm(chunks):
    text = chunk.page_content if hasattr(chunk, 'page_content') else chunk
    result = extract_from_chunk(text)
    all_entities.extend(result.get("entities", []))
    all_relations.extend(result.get("relations", []))

  # Deduplicate entities theo tên (lowercase)
  seen = set()
  unique_entities = []
  for e in all_entities:
    key = e["name"].lower().strip()
    if key not in seen:
      seen.add(key)
      unique_entities.append(e)

  print(f"Tìm được {len(unique_entities)} entities, {len(all_relations)} relations")

  # Load vào Neo4j
  with driver.session() as session:
    # Clear graph cũ
    session.run("MATCH (n) DETACH DELETE n")

    # Tạo entities
    for entity in tqdm(unique_entities, desc="Loading entities"):
      session.run(
        "MERGE (e:Entity {name: $name}) SET e.type = $type",
        name=entity["name"].strip(),
        type=entity.get("type", "CONCEPT")
      )

    # Tạo relations
    for rel in tqdm(all_relations, desc="Loading relations"):
      session.run(
        """MATCH (a:Entity {name: $source})
          MATCH (b:Entity {name: $target})
          MERGE (a)-[r:RELATES {type: $rel_type}]->(b)""",
        source=rel["source"].strip(),
        target=rel["target"].strip(),
        rel_type=rel["relation"].strip()
      )

  driver.close()
  print(" Knowledge Graph đã được load vào Neo4j!")
  return unique_entities, all_relations
```

### 7.6 GraphRAG Retrieval (`src/graph_rag.py`)

```python
"""Pipeline 3: GraphRAG — Local Search + Global Search."""
import time
from neo4j import GraphDatabase
from src.config import *
from src.naive_rag import build_index
from langchain_groq import ChatGroq


llm = ChatGroq(model_name=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.1)


def graph_local_search(question: str, vectorstore, neo4j_driver, top_k=5, hops=2):
  """Local Search: kết hợp vector search + graph traversal."""
  # Bước 1: Vector search lấy top chunks (giống Naive RAG)
  vector_docs = vectorstore.similarity_search(question, k=top_k)

  # Bước 2: Trích xuất entities từ câu hỏi
  entity_prompt = f"""Từ câu hỏi sau, liệt kê các thực thể chính (tên người, phương pháp, tổ chức).
Câu hỏi: {question}
Trả về JSON: {{"entities": ["entity1", "entity2"]}}"""

  response = llm.invoke(entity_prompt)
  import json, re
  match = re.search(r'\{.*\}', response.content, re.DOTALL)
  query_entities = json.loads(match.group())["entities"] if match else []

  # Bước 3: Graph traversal — lấy neighbors trong N hops
  graph_context = []
  with neo4j_driver.session() as session:
    for entity in query_entities:
      result = session.run(
        f"""MATCH (a:Entity)-[r*1..{hops}]-(b:Entity)
          WHERE toLower(a.name) CONTAINS toLower($name)
          RETURN a.name AS source, 
              [rel in r | type(rel) + ': ' + rel.type][0] AS relation,
              b.name AS target, b.type AS target_type
          LIMIT 20""",
        name=entity
      )
      for record in result:
        graph_context.append(
          f"{record['source']} → {record['relation']} → {record['target']} ({record['target_type']})"
        )

  return vector_docs, graph_context


def graph_global_search(question: str, community_summaries: list):
  """Global Search: dùng community summaries (map-reduce)."""
  # Map: LLM đánh giá từng community summary
  relevant_summaries = []
  for summary in community_summaries:
    check = llm.invoke(
      f"Đoạn tóm tắt sau có liên quan đến câu hỏi '{question}' không? "
      f"Trả lời YES hoặc NO.\n\nTóm tắt: {summary}"
    )
    if "YES" in check.content.upper():
      relevant_summaries.append(summary)

  # Reduce: tổng hợp
  return relevant_summaries


def query_graph_rag(question: str, vectorstore, neo4j_driver,
          community_summaries=None) -> dict:
  """Pipeline GraphRAG hoàn chỉnh."""
  start = time.time()

  # Phân loại câu hỏi: local hay global?
  classify = llm.invoke(
    f"""Câu hỏi sau là LOCAL (hỏi về thực thể/khái niệm cụ thể) hay GLOBAL (hỏi về xu hướng/tổng quan)?
Câu hỏi: {question}
Trả lời 1 từ: LOCAL hoặc GLOBAL"""
  )
  is_global = "GLOBAL" in classify.content.upper()

  if is_global and community_summaries:
    # Global Search
    summaries = graph_global_search(question, community_summaries)
    context = "Tóm tắt từ các cộng đồng tri thức:\n" + "\n\n".join(summaries)
    sources = [f"Community summary #{i+1}" for i in range(len(summaries))]
  else:
    # Local Search
    vector_docs, graph_context = graph_local_search(
      question, vectorstore, neo4j_driver
    )
    # Ghép context từ cả vector lẫn graph
    text_context = "\n\n".join([doc.page_content for doc in vector_docs])
    graph_text = "\n".join(graph_context) if graph_context else "Không tìm thấy quan hệ trong đồ thị."
    context = f"=== Từ tài liệu ===\n{text_context}\n\n=== Từ đồ thị tri thức ===\n{graph_text}"
    sources = [
      f"{doc.metadata.get('source', '?')} — trang {doc.metadata.get('page', '?')}"
      for doc in vector_docs
    ]
    if graph_context:
      sources.append(f"+ {len(graph_context)} quan hệ từ Knowledge Graph")

  # Generation
  prompt = f"""Dựa vào ngữ cảnh sau (bao gồm cả thông tin từ tài liệu và đồ thị tri thức),
hãy trả lời câu hỏi bằng tiếng Việt. Trích dẫn nguồn khi trả lời.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""

  response = llm.invoke(prompt)
  latency = time.time() - start

  return {
    "answer": response.content,
    "sources": sources,
    "latency": round(latency, 2),
    "method": "GraphRAG",
    "search_type": "Global" if is_global else "Local"
  }
```

### 7.7 Streamlit Demo (`app.py`)

```python
"""Demo UI — So sánh 3 phương pháp RAG."""
import streamlit as st
import time

st.set_page_config(
  page_title="GraphRAG Demo — Nhóm 7",
  page_icon="🕸️",
  layout="wide"
)

st.title("🕸️ Hệ thống Hỏi-Đáp Tài liệu Học thuật tiếng Việt")
st.caption("Nhóm 7 — Môn Học Máy Nâng Cao")

# ===== Sidebar =====
with st.sidebar:
  st.header("⚙️ Cấu hình")
  method = st.radio(
    "Chọn phương pháp:",
    ["Naive RAG", "RAG + Re-ranking", "GraphRAG"],
    index=0
  )
  st.markdown("---")
  st.markdown("""
  **So sánh:**
  - Naive RAG: Nhanh, đơn giản
  - Re-ranking: Chính xác hơn
  - 🟣 GraphRAG: Tốt nhất cho multi-hop
  """)

  compare_mode = st.checkbox("So sánh tất cả 3 phương pháp")

# ===== Main =====
query = st.text_input("💬 Đặt câu hỏi về tài liệu:", placeholder="VD: BERT là gì?")

if query:
  if compare_mode:
    # So sánh cả 3
    col1, col2, col3 = st.columns(3)

    with col1:
      st.subheader(" Naive RAG")
      with st.spinner("Đang xử lý..."):
        # result1 = query_naive_rag(qa_chain, query)
        result1 = {"answer": "Demo: ...", "sources": ["file.pdf — trang 5"], "latency": 1.2}
      st.write(result1["answer"])
      st.caption(f"⏱️ {result1['latency']}s")
      with st.expander("📄 Nguồn"):
        for s in result1["sources"]:
          st.write(f"- {s}")

    with col2:
      st.subheader(" Re-ranking")
      with st.spinner("Đang xử lý..."):
        # result2 = query_rerank_rag(vectorstore, query)
        result2 = {"answer": "Demo: ...", "sources": ["file.pdf — trang 3"], "latency": 2.5}
      st.write(result2["answer"])
      st.caption(f"⏱️ {result2['latency']}s")
      with st.expander("📄 Nguồn"):
        for s in result2["sources"]:
          st.write(f"- {s}")

    with col3:
      st.subheader("🟣 GraphRAG")
      with st.spinner("Đang xử lý..."):
        # result3 = query_graph_rag(query, vectorstore, neo4j_driver)
        result3 = {"answer": "Demo: ...", "sources": ["file.pdf + KG"], "latency": 5.1}
      st.write(result3["answer"])
      st.caption(f"⏱️ {result3['latency']}s")
      with st.expander("📄 Nguồn"):
        for s in result3["sources"]:
          st.write(f"- {s}")
  else:
    # Chạy 1 phương pháp
    with st.spinner(f"Đang xử lý bằng {method}..."):
      # result = run_method(method, query)
      result = {"answer": "Demo placeholder...", "sources": ["source"], "latency": 1.0}

    st.markdown("### 📝 Câu trả lời:")
    st.write(result["answer"])
    st.caption(f"⏱️ Thời gian: {result['latency']}s | Phương pháp: {method}")

    with st.expander("📄 Nguồn trích dẫn"):
      for s in result["sources"]:
        st.write(f"- {s}")

# ===== Footer =====
st.markdown("---")
st.markdown("*Nhóm 7 — Học Máy Nâng Cao — 2024*")
```

### 7.8 RAGAS Evaluation (`src/evaluator.py`)

```python
"""Đánh giá hệ thống bằng RAGAS framework."""
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
  faithfulness,
  answer_relevancy,
  context_precision,
  context_recall
)


def load_eval_questions(filepath="data/eval/questions.json"):
  """Load bộ câu hỏi đánh giá."""
  with open(filepath, "r", encoding="utf-8") as f:
    return json.load(f)


def run_evaluation(rag_func, eval_questions: list, method_name: str):
  """Chạy evaluation trên 1 phương pháp RAG.
  
  Args:
    rag_func: Hàm nhận câu hỏi, trả về dict {answer, sources, contexts}
    eval_questions: List of {question, ground_truth, type}
    method_name: Tên phương pháp (để lưu kết quả)
  """
  questions = []
  answers = []
  contexts = []
  ground_truths = []

  for item in eval_questions:
    result = rag_func(item["question"])
    questions.append(item["question"])
    answers.append(result["answer"])
    contexts.append(result.get("contexts", [])) # List of context strings
    ground_truths.append(item["ground_truth"])

  # Tạo dataset cho RAGAS
  eval_dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
  })

  # Chạy RAGAS evaluation
  results = evaluate(
    eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
  )

  # Lưu kết quả
  output = {
    "method": method_name,
    "scores": {
      "faithfulness": float(results["faithfulness"]),
      "answer_relevancy": float(results["answer_relevancy"]),
      "context_precision": float(results["context_precision"]),
      "context_recall": float(results["context_recall"])
    },
    "details": results.to_pandas().to_dict()
  }

  output_path = f"results/eval_{method_name.lower().replace(' ', '_')}.json"
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

  print(f"\n{'='*50}")
  print(f"KẾT QUẢ ĐÁNH GIÁ: {method_name}")
  print(f"{'='*50}")
  for metric, score in output["scores"].items():
    print(f" {metric}: {score:.4f}")

  return output
```

---

## 8. BỘ CÂU HỎI ĐÁNH GIÁ MẪU

Lưu vào `data/eval/questions.json`:

```json
[
 {
  "question": "RAG là viết tắt của gì và hoạt động như thế nào?",
  "ground_truth": "RAG là viết tắt của Retrieval-Augmented Generation. Hoạt động bằng cách truy hồi tài liệu liên quan làm ngữ cảnh trước khi sinh câu trả lời.",
  "type": "single-hop"
 },
 {
  "question": "BERT được huấn luyện bằng phương pháp gì?",
  "ground_truth": "BERT được huấn luyện bằng phương pháp pre-training với hai tác vụ: Masked Language Model (MLM) và Next Sentence Prediction (NSP).",
  "type": "single-hop"
 },
 {
  "question": "Embedding là gì và tại sao dùng cho RAG?",
  "ground_truth": "Embedding là biểu diễn văn bản dưới dạng vector số trong không gian nhiều chiều. Dùng cho RAG vì cho phép tìm kiếm semantic similarity giữa câu hỏi và tài liệu.",
  "type": "single-hop"
 },
 {
  "question": "Cosine similarity được tính như thế nào?",
  "ground_truth": "Cosine similarity được tính bằng tích vô hướng của hai vector chia cho tích của độ dài (norm) của chúng: sim(a,b) = (a·b)/(|a|×|b|).",
  "type": "single-hop"
 },
 {
  "question": "FAISS là gì và tại sao dùng thay vì tìm kiếm brute-force?",
  "ground_truth": "FAISS là thư viện tìm kiếm vector của Meta. Dùng thuật toán HNSW để tìm kiếm gần đúng trong O(log n) thay vì O(n) của brute-force.",
  "type": "single-hop"
 },
 {
  "question": "Tác giả nào đề xuất BERT và họ thuộc tổ chức nào?",
  "ground_truth": "BERT được đề xuất bởi Devlin et al. (2018), thuộc Google AI Language.",
  "type": "multi-hop",
  "note": "Cần kết nối: BERT → tác giả → tổ chức (2 bước)"
 },
 {
  "question": "GraphRAG và Naive RAG khác nhau ở điểm nào khi trả lời câu hỏi đa bước?",
  "ground_truth": "Naive RAG chỉ tìm chunk theo semantic similarity nên fail với câu hỏi đa bước. GraphRAG xây knowledge graph và đi theo quan hệ giữa entities để thu thập thông tin phân tán ở nhiều nơi.",
  "type": "multi-hop"
 },
 {
  "question": "Các phương pháp nào đã được đề xuất để cải thiện khâu truy hồi trong RAG?",
  "ground_truth": "Các cải tiến gồm: re-ranking bằng cross-encoder, hybrid search (BM25 + vector), graph-based retrieval, community summarization, iterative retrieval.",
  "type": "multi-hop",
  "note": "Cần tổng hợp từ nhiều nguồn"
 },
 {
  "question": "So sánh bi-encoder và cross-encoder: ưu nhược điểm của mỗi loại?",
  "ground_truth": "Bi-encoder encode riêng biệt, nhanh (dùng cho retrieve). Cross-encoder encode cặp, chính xác hơn nhưng chậm (dùng cho re-rank). Thường kết hợp: bi-encoder retrieve → cross-encoder re-rank.",
  "type": "multi-hop"
 },
 {
  "question": "Xu hướng nghiên cứu chính trong lĩnh vực RAG hiện nay là gì?",
  "ground_truth": "Các xu hướng chính gồm: GraphRAG (dùng knowledge graph), Agentic RAG (dùng LLM agents), multimodal RAG, và tối ưu hóa retrieval bằng hybrid/adaptive methods.",
  "type": "global",
  "note": "Câu hỏi global — cần community summaries để trả lời tốt"
 }
]
```

---

## 9. CHECKLIST DEMO

 
- [ ] Tất cả cài đặt xong environment (Python, packages)
- [ ] Hoàng Thị Hoạt: Script load PDF + chunking chạy được
- [ ] Ngọc Long: Cross-encoder chạy trên ví dụ đơn giản
- [ ] Hùng: Neo4j khởi động, tạo/query được node mẫu
- [ ] Tuấn Anh: Có 10+ PDF sạch + Streamlit skeleton

### 
- [ ] Hoàng Thị Hoạt: `naive_rag.py` trả lời được câu hỏi
- [ ] Ngọc Long: `rerank_rag.py` chạy, kết quả khác naive
- [ ] Hùng: Entity extraction chạy, có 50+ entities trong Neo4j
- [ ] Tuấn Anh: UI kết nối được với Naive RAG + có 20 câu hỏi eval
- [ ] **Push code lên GitHub**

### 
- [ ] Hùng: Graph traversal trả về kết quả hợp lý
- [ ] Hoàng Thị Hoạt+Hùng: GraphRAG local search hoạt động
- [ ] Ngọc Long: Hybrid search (vector + graph) chạy
- [ ] Tuấn Anh: UI có 3 tabs so sánh + eval Naive RAG xong

###  
- [ ] Cả 3 pipeline chạy end-to-end
- [ ] RAGAS evaluation cho cả 3 phương pháp
- [ ] Bảng so sánh kết quả
- [ ] Demo chạy mượt


---

## 10. LƯU Ý QUAN TRỌNG

1. **Dùng Groq API (miễn phí):** Đăng ký tại https://console.groq.com/, model `llama-3.1-70b-versatile` miễn phí và nhanh.

2. **Embedding model chạy local:** `BAAI/bge-m3` tải về máy, không cần API. Lần đầu chạy sẽ download ~2GB.

3. **Nếu Neo4j quá phức tạp:** Dùng NetworkX (Python library) thay thế. Không cần database, chạy trong RAM:
  ```python
  import networkx as nx
  G = nx.DiGraph()
  G.add_edge("BERT", "Google", relation="đề_xuất_bởi")
  # Tìm neighbors 2 hop
  neighbors = nx.single_source_shortest_path_length(G, "BERT", cutoff=2)
  ```

4. **Rate limit Groq:** Free tier có giới hạn ~30 requests/phút. Thêm `time.sleep(2)` giữa các request khi chạy batch.

5. **Git workflow:**
  ```bash
  # Mỗi người làm trên branch riêng
  git checkout -b feature/sv1-naive-rag
  # Commit thường xuyên

  # Push và tạo PR
  git push origin feature/sv1-naive-rag
  ```

---

## TÀI LIỆU THAM KHẢO

### Bài báo chính
1. Edge et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." Microsoft. https://arxiv.org/abs/2404.16130
2. Chen et al. "Agentic GraphRAG: A Comprehensive Survey." *(Bài A — đã đọc)*
3. Zhang et al. (2025). "GraphRAG for Domain-Specific LLMs." https://arxiv.org/abs/2501.13958 *(Bài B — đã đọc)*

### Documentation
4. LangChain RAG Tutorial: https://python.langchain.com/docs/tutorials/rag/
5. Sentence-Transformers: https://www.sbert.net/
6. FAISS: https://github.com/facebookresearch/faiss/wiki/Getting-started
7. Neo4j Getting Started: https://neo4j.com/docs/getting-started/
8. RAGAS Documentation: https://docs.ragas.io/
9. Streamlit: https://docs.streamlit.io/
10. Groq API: https://console.groq.com/docs/quickstart
11. BGE-M3 Model: https://huggingface.co/BAAI/bge-m3


