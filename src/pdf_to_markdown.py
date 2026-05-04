from __future__ import annotations

"""Chuyển tài liệu PDF đã trích xuất sang Markdown chuẩn cho RAG."""

from datetime import datetime, timezone
import logging
from pathlib import Path
import re
import unicodedata

import ftfy

from src.pdf_extractor import ExtractedDoc, ExtractedPage


LOGGER = logging.getLogger(__name__)
MAX_REPEATED_LINE_RATIO = 0.5
MAX_HEADER_FOOTER_LENGTH = 120
BLANK_LINE_CAP_PATTERN = re.compile(r"\n{3,}")
HYPHEN_BREAK_PATTERN = re.compile(r"(?<=\w)-\n(?=\w)", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"[ \t]+")
CHAPTER_PATTERN = re.compile(r"^(Chương|Mục)\s+[IVXLCDM\d]+.*$", re.MULTILINE | re.IGNORECASE)
ARTICLE_PATTERN = re.compile(r"^Điều\s+\d+\..*$", re.MULTILINE)
NUMERIC_LINE_PATTERN = re.compile(r"\d")


def normalize_text(s: str) -> str:
    """Chuẩn hóa text OCR và text layer để giảm mojibake."""

    text = ftfy.fix_text(s or "")
    text = unicodedata.normalize("NFC", text)
    text = HYPHEN_BREAK_PATTERN.sub("", text)

    normalized_lines = [WHITESPACE_PATTERN.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in normalized_lines if line or "\n")
    text = BLANK_LINE_CAP_PATTERN.sub("\n\n", text)
    return text.strip()


def _strip_repeated_page_lines(pages: list[ExtractedPage]) -> list[ExtractedPage]:
    """Bỏ header/footer lặp trên hơn 50% số trang."""

    if not pages:
        return []

    first_line_counts: dict[str, int] = {}
    last_line_counts: dict[str, int] = {}
    extracted_lines: list[list[str]] = []

    for page in pages:
        lines = [line.strip() for line in normalize_text(page.text).splitlines() if line.strip()]
        extracted_lines.append(lines)
        if lines:
            first_line_counts[lines[0]] = first_line_counts.get(lines[0], 0) + 1
            last_line_counts[lines[-1]] = last_line_counts.get(lines[-1], 0) + 1

    repeated_first = {
        line
        for line, count in first_line_counts.items()
        if len(line) <= MAX_HEADER_FOOTER_LENGTH and count / len(pages) > MAX_REPEATED_LINE_RATIO
    }
    repeated_last = {
        line
        for line, count in last_line_counts.items()
        if len(line) <= MAX_HEADER_FOOTER_LENGTH and count / len(pages) > MAX_REPEATED_LINE_RATIO
    }

    cleaned_pages: list[ExtractedPage] = []
    for page, lines in zip(pages, extracted_lines):
        working_lines = lines[:]
        if working_lines and working_lines[0] in repeated_first:
            working_lines = working_lines[1:]
        if working_lines and working_lines[-1] in repeated_last:
            working_lines = working_lines[:-1]
        cleaned_pages.append(
            ExtractedPage(
                page_no=page.page_no,
                text="\n".join(working_lines).strip(),
                tables=page.tables,
                used_ocr=page.used_ocr,
                confidence=page.confidence,
            )
        )
    return cleaned_pages


def _promote_legal_headings(text: str) -> str:
    """Nâng cấp heading pháp lý để chunk structure-aware bắt được."""

    text = CHAPTER_PATTERN.sub(lambda match: f"## {match.group(0).strip()}", text)
    text = ARTICLE_PATTERN.sub(lambda match: f"### {match.group(0).strip()}", text)
    return text


def page_text_to_md(page: ExtractedPage) -> str:
    """Render một trang thành section Markdown."""

    body = normalize_text(page.text)
    return f"\n\n## Page {page.page_no}\n\n{body}".rstrip()


def _escape_cell(cell: str) -> str:
    return (cell or "").replace("|", "\\|").replace("\n", "<br>")


def tables_to_md(tables: list[list[list[str]]]) -> str:
    """Chuyển bảng trích xuất thành Markdown table."""

    rendered_tables: list[str] = []
    table_counter = 0
    for table in tables:
        cleaned_rows = [[_escape_cell(cell) for cell in row if cell is not None] for row in table if row]
        if not cleaned_rows:
            continue
        column_count = max(len(row) for row in cleaned_rows)
        if column_count <= 1:
            continue

        normalized_rows = [row + [""] * (column_count - len(row)) for row in cleaned_rows]
        header = normalized_rows[0]
        separator = ["---"] * column_count
        body_rows = normalized_rows[1:] or [[""] * column_count]
        lines = [
            f"### Table {table_counter + 1}",
            "",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in body_rows)
        rendered_tables.append("\n".join(lines))
        table_counter += 1
    return "\n\n".join(rendered_tables)


def numeric_lines_to_md(text: str) -> str:
    """Giữ thêm các dòng số liệu khi PDF tài chính không trích xuất được bảng."""

    numeric_lines = [
        line.strip()
        for line in normalize_text(text).splitlines()
        if NUMERIC_LINE_PATTERN.search(line) and len(line.strip()) >= 8
    ]
    if not numeric_lines:
        return ""

    rendered_lines = ["### Numeric lines", ""]
    rendered_lines.extend(f"- {line}" for line in numeric_lines)
    return "\n".join(rendered_lines)


def doc_to_markdown(doc: ExtractedDoc, title: str) -> str:
    """Chuyển toàn bộ tài liệu sang Markdown output."""

    cleaned_pages = _strip_repeated_page_lines(doc.pages)
    used_ocr = any(page.used_ocr for page in cleaned_pages)
    extraction_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    parts = [
        f"# {title}",
        "",
        f"- Source: {doc.source_path}",
        f"- SHA256: {doc.sha256}",
        f"- Extraction date: {extraction_date}",
        f"- Page count: {len(cleaned_pages)}",
        f"- Used OCR: {'yes' if used_ocr else 'no'}",
    ]

    for page in cleaned_pages:
        page_text = page.text
        normalized_title = title.lower()
        if "nghị định 13/2023" in normalized_title:
            page_text = _promote_legal_headings(page_text)

        page_markdown = page_text_to_md(
            ExtractedPage(
                page_no=page.page_no,
                text=page_text,
                tables=page.tables,
                used_ocr=page.used_ocr,
                confidence=page.confidence,
            )
        )
        parts.append(page_markdown)

        table_markdown = tables_to_md(page.tables)
        if table_markdown:
            parts.append(table_markdown)
        elif "báo cáo tài chính" in normalized_title:
            numeric_lines_markdown = numeric_lines_to_md(page_text)
            if numeric_lines_markdown:
                parts.append(numeric_lines_markdown)

    return "\n".join(part for part in parts if part).strip() + "\n"


def write_markdown(md: str, out_path: str | Path) -> None:
    """Ghi Markdown ra file UTF-8 và log thống kê cơ bản."""

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    line_count = md.count("\n") + 1 if md else 0
    byte_count = len(md.encode("utf-8"))
    LOGGER.info("Wrote %s (%s lines, %s bytes)", output_path, line_count, byte_count)
