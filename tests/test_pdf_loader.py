"""Test làm sạch text tiếng Việt + đọc corpus mẫu (không cần PDF hay model)."""
import unicodedata

from src.pdf_loader import clean_vietnamese_text, load_sample_corpus


def test_clean_normalizes_to_nfc():
    # "ă" viết dạng NFD (a + dấu breve) phải được gộp về NFC
    nfd = unicodedata.normalize("NFD", "ăn")
    out = clean_vietnamese_text(nfd)
    assert out == unicodedata.normalize("NFC", "ăn")


def test_clean_joins_hyphen_linebreak():
    assert clean_vietnamese_text("trans-\nformer") == "transformer"


def test_clean_collapses_spaces_and_blank_lines():
    assert clean_vietnamese_text("a    b") == "a b"
    assert "\n\n\n" not in clean_vietnamese_text("a\n\n\n\n\nb")


def test_clean_drops_page_number_and_doi_lines():
    text = "Nội dung thật.\n- 12 -\ndoi: 10.1234/abc\nNội dung nữa."
    out = clean_vietnamese_text(text)
    assert "12" not in out.split("\n")
    assert "doi" not in out.lower()


def test_clean_handles_empty():
    assert clean_vietnamese_text("") == ""
    assert clean_vietnamese_text(None) == ""


def test_sample_corpus_loads_five_docs():
    docs = load_sample_corpus()
    assert len(docs) == 5
    assert all(d["text"] and d["source"].endswith(".txt") and d["page"] == 1 for d in docs)
