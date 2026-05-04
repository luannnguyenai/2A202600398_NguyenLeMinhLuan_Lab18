from __future__ import annotations

"""CLI ingest PDF tiếng Việt vào Markdown và Qdrant."""

import argparse
import glob
import logging
import os
from pathlib import Path
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COLLECTION_NAME
from src.m1_chunking import (
    chunk_basic,
    chunk_hierarchical,
    chunk_semantic,
    chunk_structure_aware,
    load_documents,
)
from src.m2_search import DenseSearch
from src.pdf_extractor import DEFAULT_OCR_DPI, OCR_CACHE_DIR, extract_pdf, sha256_of
from src.pdf_to_markdown import doc_to_markdown, write_markdown


LOGGER = logging.getLogger(__name__)
DEFAULT_OUT_DIR = Path("data")
DEFAULT_STRATEGY = "structure"
TITLE_OVERRIDES = {
    "BCTC.pdf": "Báo cáo tài chính (BCTC)",
    "Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf": (
        "Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân"
    ),
}
SLUG_OVERRIDES = {
    "BCTC.pdf": "bctc",
    "Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf": "nghi_dinh_13_2023",
}


def build_parser() -> argparse.ArgumentParser:
    """Tạo parser cho lệnh ingest PDF."""

    parser = argparse.ArgumentParser(description="Ingest PDF vào markdown và Qdrant.")
    parser.add_argument("--input", nargs="+", action="append", required=True, help="PDF path hoặc glob pattern.")
    parser.add_argument("--output", help="Output markdown path khi chỉ ingest một file.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Thư mục output markdown.")
    parser.add_argument("--force-ocr", action="store_true", help="OCR toàn bộ trang.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_OCR_DPI, help="DPI render cho OCR.")
    parser.add_argument("--no-cache", action="store_true", help="Bỏ qua cache OCR hiện có.")
    parser.add_argument("--lang", default="vie+eng", help="Ngôn ngữ OCR.")
    parser.add_argument("--index", action="store_true", help="Index markdown vào Qdrant sau khi viết file.")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Tên collection Qdrant.")
    parser.add_argument(
        "--strategy",
        choices=["basic", "semantic", "structure", "hierarchical"],
        default=DEFAULT_STRATEGY,
        help="Chunking strategy để index.",
    )
    return parser


def slugify_filename(name: str) -> str:
    """Chuyển tên file thành slug ASCII ổn định."""

    if name in SLUG_OVERRIDES:
        return SLUG_OVERRIDES[name]

    base_name = Path(name).stem
    normalized = unicodedata.normalize("NFKD", base_name).encode("ascii", "ignore").decode("ascii")
    slug_chars = [char.lower() if char.isalnum() else "_" for char in normalized]
    slug = "".join(slug_chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "document"


def title_for_path(path: Path) -> str:
    """Lấy title output ưu tiên theo mapping yêu cầu."""

    return TITLE_OVERRIDES.get(path.name, path.stem.replace("_", " ").strip())


def resolve_inputs(raw_inputs: list[list[str]]) -> list[Path]:
    """Resolve input path và glob thành danh sách PDF."""

    resolved: list[Path] = []
    for input_group in raw_inputs:
        for item in input_group:
            matches = [Path(match) for match in glob.glob(item)]
            if matches:
                resolved.extend(matches)
            else:
                resolved.append(Path(item))

    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in resolved:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def remove_cache_for_pdf(pdf_path: Path) -> None:
    """Xóa cache OCR của file đầu vào nếu có."""

    cache_path = OCR_CACHE_DIR / f"{sha256_of(pdf_path)}.json"
    if cache_path.exists():
        cache_path.unlink()


def output_path_for_pdf(pdf_path: Path, output: str | None, out_dir: Path, total_inputs: int) -> Path:
    """Xác định output markdown path cho từng PDF."""

    if output:
        if total_inputs != 1:
            raise ValueError("--output chỉ dùng khi ingest đúng một file.")
        return Path(output)
    return out_dir / f"{slugify_filename(pdf_path.name)}.md"


def markdown_to_chunk_dicts(strategy: str) -> list[dict]:
    """Load toàn bộ markdown trong data/ và chunk theo strategy."""

    documents = load_documents()
    indexed_chunks: list[dict] = []
    for doc in documents:
        metadata = doc["metadata"]
        text = doc["text"]

        if strategy == "basic":
            chunks = chunk_basic(text, metadata=metadata)
            indexed_chunks.extend({"text": chunk.text, "metadata": chunk.metadata} for chunk in chunks)
            continue

        if strategy == "semantic":
            chunks = chunk_semantic(text, metadata=metadata)
            indexed_chunks.extend({"text": chunk.text, "metadata": chunk.metadata} for chunk in chunks)
            continue

        if strategy == "structure":
            chunks = chunk_structure_aware(text, metadata=metadata)
            indexed_chunks.extend({"text": chunk.text, "metadata": chunk.metadata} for chunk in chunks)
            continue

        parents, children = chunk_hierarchical(text, metadata=metadata)
        parent_map = {parent.parent_id: parent.text for parent in parents}
        indexed_chunks.extend(
            {
                "text": child.text,
                "metadata": {
                    **child.metadata,
                    "parent_id": child.parent_id,
                    "parent_text": parent_map.get(child.parent_id, ""),
                },
            }
            for child in children
        )
    return indexed_chunks


def ingest_pdfs(args: argparse.Namespace) -> list[Path]:
    """Ingest PDF thành markdown output."""

    input_paths = resolve_inputs(args.input)
    if not input_paths:
        raise FileNotFoundError("Không tìm thấy file PDF đầu vào.")

    out_dir = Path(args.out_dir)
    output_paths: list[Path] = []
    for pdf_path in input_paths:
        if not pdf_path.exists():
            raise FileNotFoundError(f"Thiếu PDF: {pdf_path}")
        if args.no_cache:
            remove_cache_for_pdf(pdf_path)

        document = extract_pdf(
            str(pdf_path),
            force_ocr=args.force_ocr,
            dpi=args.dpi,
            ocr_lang=args.lang,
        )
        title = title_for_path(pdf_path)
        markdown = doc_to_markdown(document, title)
        output_path = output_path_for_pdf(pdf_path, args.output, out_dir, len(input_paths))
        write_markdown(markdown, output_path)
        output_paths.append(output_path)
    return output_paths


def index_markdown(collection: str, strategy: str) -> tuple[int, int, float]:
    """Chunk toàn bộ markdown trong data/ và upsert vào Qdrant."""

    started_at = time.perf_counter()
    chunks = markdown_to_chunk_dicts(strategy)
    dense_search = DenseSearch()
    dense_search.index(chunks, collection=collection)
    elapsed = time.perf_counter() - started_at
    return len(chunks), len(chunks), elapsed


def main() -> int:
    """Entry point cho CLI ingest."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    output_paths = ingest_pdfs(args)
    LOGGER.info("Generated markdown: %s", ", ".join(str(path) for path in output_paths))

    if args.index:
        chunk_count, points_upserted, elapsed = index_markdown(args.collection, args.strategy)
        LOGGER.info(
            "Indexed %s chunks into %s (%s points upserted, %.2fs).",
            chunk_count,
            args.collection,
            points_upserted,
            elapsed,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
