from __future__ import annotations

from pathlib import Path

import pytest

from src.pdf_extractor import ExtractedDoc, ExtractedPage, extract_pdf, needs_ocr
from src.pdf_to_markdown import doc_to_markdown


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
BCTC_PATH = RAW_DIR / "BCTC.pdf"


def test_extractor_returns_pages() -> None:
    if not BCTC_PATH.exists():
        pytest.skip("BCTC.pdf is missing")

    doc = extract_pdf(str(BCTC_PATH))

    assert len(doc.pages) > 0
    assert any(page.text.strip() for page in doc.pages)


def test_needs_ocr_heuristic() -> None:
    vietnamese_text = (
        "Chủ thể dữ liệu có quyền được biết, đồng ý, truy cập, chỉnh sửa và yêu cầu "
        "xóa dữ liệu cá nhân theo quy định của pháp luật Việt Nam."
    )

    assert needs_ocr("") is True
    assert needs_ocr(vietnamese_text) is False


def test_markdown_has_structure() -> None:
    doc = ExtractedDoc(
        source_path="data/raw/nghi_dinh.pdf",
        sha256="abc123",
        pages=[
            ExtractedPage(
                page_no=1,
                text="Chương I\n\nĐiều 1. Phạm vi điều chỉnh\n\nNội dung điều khoản.",
                tables=[],
                used_ocr=False,
                confidence=None,
            )
        ],
        meta={},
    )

    markdown = doc_to_markdown(
        doc,
        "Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân",
    )

    assert "## Điều 1. Phạm vi điều chỉnh" in markdown


def test_idempotent_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    if not BCTC_PATH.exists():
        pytest.skip("BCTC.pdf is missing")

    import src.pdf_extractor as pdf_extractor

    counter = {"calls": 0}

    def fake_render(_: str, dpi: int = 300) -> list[str]:
        return ["page-1", "page-2"]

    def fake_ocr_page(image: object, lang: str = "vie+eng", psm: int = 6) -> tuple[str, float]:
        counter["calls"] += 1
        return (f"Văn bản OCR {image}", 95.0)

    monkeypatch.setattr(pdf_extractor, "OCR_CACHE_DIR", tmp_path)
    monkeypatch.setattr(pdf_extractor, "render_pages_for_ocr", fake_render)
    monkeypatch.setattr(pdf_extractor, "ocr_page", fake_ocr_page)

    first = extract_pdf(str(BCTC_PATH), force_ocr=True)
    second = extract_pdf(str(BCTC_PATH), force_ocr=True)

    assert first.sha256 == second.sha256
    assert counter["calls"] == len(first.pages)
