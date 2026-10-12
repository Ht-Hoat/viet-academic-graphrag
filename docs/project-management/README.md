# Tài liệu Quản lý Dự án — Nhóm 7

**Dự án:** Nghiên cứu và so sánh các kỹ thuật RAG nâng cao (GraphRAG) cho hệ thống hỏi–đáp tài liệu học thuật tiếng Việt

**Repo:** `ht-hoat/viet-academic-graphrag`

---

## Danh sách tài liệu

| File | Nội dung | Ai cần đọc |
|------|---------|------------|
| [TONG_QUAN_KIEN_TRUC.md](./TONG_QUAN_KIEN_TRUC.md) | Bản đồ kiến trúc hệ thống, data flow, so sánh 3 pipeline | **Cả nhóm** |
| [KE_HOACH_TICH_HOP.md](./KE_HOACH_TICH_HOP.md) | 8 quyết định kiến trúc + Roadmap 4 giai đoạn | **Cả nhóm** |
| [TASK_HOAT.md](./TASK_HOAT.md) | Task chi tiết cho Hoạt (Naive RAG + Config + Data Pipeline) | **Hoạt** |
| [TASK_HUNG.md](./TASK_HUNG.md) | Task chi tiết cho Hùng (GraphRAG + LLM wrapper) | **Hùng** |
| [TASK_LONG.md](./TASK_LONG.md) | Task chi tiết cho Long (Re-ranking adapter) | **Long** |
| [TASK_TUANANH.md](./TASK_TUANANH.md) | Task chi tiết cho Tuấn Anh (UI + Evaluation) | **Tuấn Anh** |

---

## Quy trình làm việc

1. Đọc `TONG_QUAN_KIEN_TRUC.md` để hiểu tổng thể hệ thống
2. Đọc `KE_HOACH_TICH_HOP.md` để hiểu roadmap và quyết định kiến trúc
3. Đọc file `TASK_<TEN>.md` của mình để biết chi tiết từng task
4. Làm task theo thứ tự ưu tiên trong file
5. Tick checkbox khi hoàn thành, tạo PR để review

## Thứ tự merge vào main

```
1. Hoạt  → config + chunker + naive_rag + data_pipeline
2. Hùng  → llm.py + graph modules
3. Long  → reranker_module + src/rerank_rag.py
4. Tuấn Anh → UI + evaluator (cuối cùng)
```

Mỗi lần merge tạo **Pull Request**, có ít nhất 1 người khác review trước khi merge.
