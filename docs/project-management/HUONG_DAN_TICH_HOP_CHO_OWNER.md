# HƯỚNG DẪN TÍCH HỢP — Dành cho Owner (Ht-Hoat)

**Mục tiêu:** Merge 3 nhánh (Hoạt → Hùng → Long) vào `main` để tạo code base thống nhất. Sau đó mỗi thành viên tạo nhánh từ `main` mới và làm task.

**Thời gian:** ~15-20 phút
**Yêu cầu:** Quyền push vào `main`

---

## BƯỚC 1: Clone repo mới (nếu chưa có)

```bash
git clone https://github.com/Ht-Hoat/viet-academic-graphrag.git
cd viet-academic-graphrag
git fetch origin
```

---

## BƯỚC 2: Merge nhánh Hoạt vào main (ưu tiên cao nhất)

Nhánh Hoạt là **nền tảng** — chứa config, chunker, embedder, vectorstore, naive_rag.

```bash
git checkout main
git pull origin main
git merge origin/feature/hoat-naive-rag -m "merge: integrate Hoat naive-rag foundation (config, chunker, embedder, vectorstore, naive_rag)"
```

**Không có conflict** (đã kiểm tra — nhánh Hoạt chỉ thêm file mới, không sửa file trên main).

```bash
git push origin main
```

---

## BƯỚC 3: Merge nhánh Hùng vào main

Nhánh Hùng thêm graph modules. Chỉ **2 file trùng** với Hoạt: `src/__init__.py` và `src/config.py`.

```bash
git merge origin/feature/hung-graphrag -m "merge: integrate Hung graphrag modules (graph_builder, graph_rag, graph_store, llm_utils)"
```

**Nếu có conflict ở `src/config.py`:**
- Giữ **CẢ HAI** phần — config của Hoạt (LLM, Embedding, Chunking, Retrieval) + config của Hùng (Graph, Neo4j, Leiden, Entity types)
- Mở file, sửa conflict markers `<<<<<<<`, `=======`, `>>>>>>>`:
  - Giữ phần Hoạt ở trên (paths, LLM, embedding, chunking, retrieval)
  - Thêm phần Hùng ở dưới (Neo4j, graph params, entity/relation types)
- `CHUNK_OVERLAP`: giữ **50** (của Hùng, đúng spec)
- dotenv: giữ kiểu **try/except** (của Hoạt)

**Nếu có conflict ở `src/__init__.py`:**
- File này thường rỗng hoặc rất ngắn — giữ phiên bản nào cũng được

```bash
# Sau khi resolve conflicts (nếu có):
git add -A
git commit -m "merge: resolve config conflicts - keep both Hoat + Hung sections"
git push origin main
```

---

## BƯỚC 4: Merge nhánh Long vào main

Nhánh Long nằm hoàn toàn trong `reranker_module/` — **không conflict** với ai.

```bash
git merge origin/long -m "merge: integrate Long reranker module"
git push origin main
```

---

## BƯỚC 5: Thêm tài liệu quản lý dự án

Em (Hoạt, coordinator) sẽ gửi cho bạn 7 file tài liệu. Bạn giải nén rồi push:

```bash
# Giải nén file tar.gz (Hoạt sẽ gửi)
tar xzf project-management-docs.tar.gz

# Hoặc tạo thư mục và copy từng file vào
mkdir -p docs/project-management
# (copy 7 file .md vào đây)

git add docs/project-management/
git commit -m "docs: add project management - architecture, integration plan, task assignments"
git push origin main
```

---

## BƯỚC 6: Xác nhận thành công

```bash
# Kiểm tra main có đầy đủ code
ls src/
# Phải thấy: __init__.py  chunker.py  config.py  embedder.py  
#            graph_builder.py  graph_memory_store.py  graph_rag.py
#            graph_sample.py  graph_store.py  llm.py  llm_utils.py
#            naive_rag.py  pdf_loader.py  vectorstore.py

ls reranker_module/reranker/
# Phải thấy: __init__.py  bi_encoder.py  config.py  cross_encoder.py
#            faiss_index.py  pipeline.py  utils.py

ls docs/project-management/
# Phải thấy: README.md  TONG_QUAN_KIEN_TRUC.md  KE_HOACH_TICH_HOP.md
#            TASK_HOAT.md  TASK_HUNG.md  TASK_LONG.md  TASK_TUANANH.md
```

---

## SAU KHI TÍCH HỢP XONG

Thông báo cho cả nhóm:

> **Main đã tích hợp xong!** Mỗi người:
> 1. `git fetch origin main && git checkout main && git pull`
> 2. Đọc file task của mình: `docs/project-management/TASK_<TEN>.md`
> 3. Tạo nhánh mới từ main: `git checkout -b feature/<ten>-phase1`
> 4. Làm task theo checklist trong file
> 5. Xong thì tạo PR để review

**Lưu ý cho Tuấn Anh:** Nhánh cũ `feature-tuananh-eval-ui` là orphan branch. Tuấn Anh PHẢI tạo nhánh mới từ `main` (xem TASK_TUANANH.md bước 1.4), KHÔNG dùng nhánh cũ.

---

## TÓM TẮT LỆNH (copy-paste nhanh)

```bash
# === TOÀN BỘ QUÁ TRÌNH ===
git clone https://github.com/Ht-Hoat/viet-academic-graphrag.git
cd viet-academic-graphrag
git fetch origin

# Merge Hoạt (không conflict)
git checkout main
git merge origin/feature/hoat-naive-rag -m "merge: integrate Hoat naive-rag foundation"
git push origin main

# Merge Hùng (có thể conflict ở config.py — xem hướng dẫn trên)
git merge origin/feature/hung-graphrag -m "merge: integrate Hung graphrag modules"
# Nếu conflict: resolve → git add -A → git commit
git push origin main

# Merge Long (không conflict)
git merge origin/long -m "merge: integrate Long reranker module"
git push origin main

# Thêm docs (sau khi nhận file từ coordinator)
mkdir -p docs/project-management
# Copy files...
git add docs/project-management/
git commit -m "docs: add project management docs"
git push origin main

echo "DONE! Thong bao cho nhom bat dau lam viec."
```
