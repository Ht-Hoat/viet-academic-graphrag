"""Đọc tài liệu (PDF thật hoặc corpus mẫu .txt) và làm sạch text tiếng Việt.

Mỗi tài liệu/trang trả về dict {text, source, page}. Đây là bước đầu tiên của toàn hệ thống:
đầu ra của file này là đầu vào của chunker.py.

Chạy thử:
    python -m src.pdf_loader                 # đọc data/pdfs, thiếu thì tự dùng corpus mẫu
"""
import re
import sys
import unicodedata
from pathlib import Path

from src import config

# Regex làm sạch — biên dịch sẵn cho nhanh
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")       # từ bị ngắt cuối dòng: "trans-\nformer"
_RE_PAGE_NUMBER_LINE = re.compile(r"(?im)^[ \t]*(trang\s*)?[-–—]?\s*\d{1,4}\s*[-–—]?[ \t]*$")
_RE_HEADER_FOOTER = re.compile(
    r"(?im)^[ \t]*(tạp chí .*|journal of .*|proceedings of .*|©\s*\d{4}.*|"
    r"all rights reserved.*|issn[:\s].*|doi[:\s].*)[ \t]*$"
)


def clean_vietnamese_text(text: str) -> str:
    """Chuẩn hóa Unicode tiếng Việt + dọn rác PDF.

    Các bước, theo thứ tự:
      1. NFC — gộp dấu tiếng Việt về một dạng chuẩn (PDF hay xuất ra NFD, dấu tách rời).
      2. Nối từ bị ngắt cuối dòng bằng dấu gạch nối.
      3. Bỏ dòng chỉ chứa số trang, và header/footer phổ biến (tạp chí, DOI, ISSN...).
      4. Gộp khoảng trắng và dòng trống thừa.
    """
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _RE_HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _RE_PAGE_NUMBER_LINE.sub("", text)
    text = _RE_HEADER_FOOTER.sub("", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def load_pdfs(pdf_dir=config.PDF_DIR) -> list[dict]:
    """Đọc mọi file .pdf trong thư mục bằng PyMuPDF, trích text theo từng trang."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("  [!] Chưa cài pymupdf (pip install pymupdf) — bỏ qua PDF.", file=sys.stderr)
        return []

    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        return []

    docs = []
    for path in sorted(pdf_dir.glob("*.pdf")):
        try:
            with fitz.open(path) as pdf:
                for i, page in enumerate(pdf):
                    text = clean_vietnamese_text(page.get_text("text"))
                    if len(text) >= config.MIN_PAGE_TEXT_LENGTH:   # bỏ trang bìa / trang trống
                        docs.append({"text": text, "source": path.name, "page": i + 1})
        except Exception as exc:  # noqa: BLE001 — 1 file hỏng không được làm chết cả mẻ
            print(f"  [!] Không đọc được {path.name}: {exc}", file=sys.stderr)
    return docs


def load_sample_corpus(sample_dir=config.SAMPLE_DIR) -> list[dict]:
    """Đọc corpus mẫu tiếng Việt dạng .txt (mỗi file = 1 tài liệu, page=1)."""
    docs = []
    for path in sorted(Path(sample_dir).glob("*.txt")):
        text = clean_vietnamese_text(path.read_text(encoding="utf-8"))
        if text:
            docs.append({"text": text, "source": path.name, "page": 1})
    return docs


def load_documents(pdf_dir=config.PDF_DIR, use_sample_if_empty: bool = True) -> list[dict]:
    """Ưu tiên PDF thật; nếu chưa có PDF nào thì tự dùng corpus mẫu để demo chạy được ngay.

    Đây là điểm khác biệt quan trọng: cả nhóm (kể cả Hùng — GraphRAG) có thể chạy toàn bộ
    pipeline mà chưa cần thu thập PDF, nhờ corpus mẫu trong data/sample_corpus/.
    """
    docs = load_pdfs(pdf_dir)
    if docs:
        print(f"Đã đọc {len(docs)} trang từ PDF trong {pdf_dir}")
        return docs
    if use_sample_if_empty:
        docs = load_sample_corpus()
        print(f"Không có PDF — dùng corpus mẫu: {len(docs)} tài liệu từ {config.SAMPLE_DIR}")
        return docs
    print("Không tìm thấy tài liệu nào.")
    return []


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    documents = load_documents()
    for d in documents:
        print(f"- [{d['source']} tr.{d['page']}] {len(d['text'])} ký tự")
