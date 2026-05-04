from __future__ import annotations

"""Trích xuất PDF với text layer, OCR và cache theo SHA-256."""

from dataclasses import asdict, dataclass
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import site
import subprocess
import tempfile
import unicodedata


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
OCR_CACHE_DIR = REPO_ROOT / "data" / ".cache" / "ocr"
TESSDATA_CACHE_DIR = REPO_ROOT / "data" / ".cache" / "tessdata"
SHA256_BLOCK_SIZE = 1024 * 1024
DEFAULT_OCR_LANG = "vie+eng"
DEFAULT_OCR_PSM = 6
DEFAULT_OCR_DPI = 300
CONTROL_WHITESPACE = {"\n", "\r", "\t"}


@dataclass
class ExtractedPage:
    """Nội dung đã trích xuất của một trang."""

    page_no: int
    text: str
    tables: list[list[list[str]]]
    used_ocr: bool
    confidence: float | None


@dataclass
class ExtractedDoc:
    """Tài liệu PDF đã được trích xuất."""

    source_path: str
    sha256: str
    pages: list[ExtractedPage]
    meta: dict


def sha256_of(path: str | Path) -> str:
    """Tính SHA-256 theo block 1MB."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        while True:
            chunk = file_obj.read(SHA256_BLOCK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def extract_text_layer(pdf_path: str | Path) -> list[str]:
    """Đọc text layer từng trang bằng PyMuPDF."""

    import fitz

    texts: list[str] = []
    with fitz.open(pdf_path) as document:
        for page in document:
            texts.append(page.get_text("text"))
    return texts


def extract_tables_pdfplumber(pdf_path: str | Path) -> list[list[list[list[str]]]]:
    """Đọc bảng từng trang bằng pdfplumber."""

    import pdfplumber

    per_page_tables: list[list[list[list[str]]]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            normalized_tables = [
                [[(cell or "").strip() for cell in row] for row in table if row]
                for table in tables
            ]
            per_page_tables.append(normalized_tables)
    return per_page_tables


def needs_ocr(text: str, min_chars: int = 40, alpha_ratio: float = 0.5) -> bool:
    """Heuristic quyết định một trang có cần OCR hay không."""

    stripped = text.strip()
    if len(stripped) < min_chars:
        return True

    non_space_chars = [char for char in text if not char.isspace()]
    if not non_space_chars:
        return True

    alpha_chars = [char for char in non_space_chars if unicodedata.category(char).startswith("L")]
    alpha_score = len(alpha_chars) / len(non_space_chars)
    if alpha_score < alpha_ratio:
        return True

    control_chars = [
        char for char in text
        if unicodedata.category(char).startswith("C") and char not in CONTROL_WHITESPACE
    ]
    return (len(control_chars) / max(len(text), 1)) > 0.05


def _configure_tesseract() -> None:
    """Thiết lập đường dẫn binary Tesseract nếu được cài trong user site."""

    import pytesseract

    candidates = [
        shutil.which("tesseract"),
        str(Path(site.USER_BASE) / "bin" / "tesseract"),
        str(Path(site.getusersitepackages()).parent / "bin" / "tesseract"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return

    raise RuntimeError(
        "Không tìm thấy binary 'tesseract'. Cài Tesseract hoặc package tesseract-bin trước khi OCR."
    )


def _ensure_tessdata(lang: str) -> Path:
    """Chuẩn bị thư mục tessdata cục bộ cho các ngôn ngữ OCR cần dùng."""

    requested_languages = [item.strip() for item in lang.split("+") if item.strip()]
    if not requested_languages:
        requested_languages = ["eng"]

    source_dirs = [
        Path(site.getusersitepackages()) / "tesseract_bin" / "data" / "share" / "tessdata",
        Path(site.USER_BASE) / "share" / "tessdata",
    ]

    TESSDATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for traineddata in source_dir.glob("*.traineddata"):
            target = TESSDATA_CACHE_DIR / traineddata.name
            if not target.exists():
                shutil.copy2(traineddata, target)

    missing_languages = [
        language for language in requested_languages
        if not (TESSDATA_CACHE_DIR / f"{language}.traineddata").exists()
    ]
    if missing_languages:
        raise RuntimeError(
            "Thiếu tessdata cho các ngôn ngữ: "
            + ", ".join(missing_languages)
            + ". Hãy cài hoặc tải traineddata tương ứng."
        )

    os.environ["TESSDATA_PREFIX"] = str(TESSDATA_CACHE_DIR)
    return TESSDATA_CACHE_DIR


def _deskew_image(image_array: object) -> object:
    """Hiệu chỉnh nghiêng nhẹ cho ảnh nhị phân."""

    import cv2
    import numpy as np

    coords = np.column_stack(np.where(image_array < 255))
    if coords.size == 0:
        return image_array

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    if abs(angle) < 0.25:
        return image_array

    height, width = image_array.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image_array,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _ocr_page_via_tesseract_cli(image: object, lang: str, psm: int) -> tuple[str, float]:
    """Fallback OCR qua CLI để tránh lỗi PNG của binary đóng gói."""

    from PIL import Image

    _configure_tesseract()
    tessdata_dir = _ensure_tessdata(lang)
    tesseract_cmd = shutil.which("tesseract") or str(Path(site.USER_BASE) / "bin" / "tesseract")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "page.tiff"
        output_base = temp_path / "ocr"
        Image.fromarray(image).save(input_path, format="TIFF")

        env = os.environ.copy()
        env["TESSDATA_PREFIX"] = str(tessdata_dir)
        command = [
            tesseract_cmd,
            str(input_path),
            str(output_base),
            "-l",
            lang,
            "-c",
            "tessedit_create_tsv=1",
            "--oem",
            "3",
            "--psm",
            str(psm),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True, env=env)

        output_path = output_base.with_suffix(".tsv")
        with output_path.open(encoding="utf-8") as file_obj:
            reader = csv.DictReader(file_obj, delimiter="\t")
            return _rows_to_text_and_conf(list(reader))


def _rows_to_text_and_conf(rows: list[dict[str, str]]) -> tuple[str, float]:
    """Ghép kết quả TSV thành văn bản theo từng dòng OCR."""

    grouped_lines: list[str] = []
    current_key: tuple[str, str, str, str] | None = None
    current_tokens: list[str] = []
    confidences: list[float] = []

    for row in rows:
        token = (row.get("text") or "").strip()
        key = (
            row.get("page_num", ""),
            row.get("block_num", ""),
            row.get("par_num", ""),
            row.get("line_num", ""),
        )

        if current_key is None:
            current_key = key
        elif key != current_key:
            if current_tokens:
                grouped_lines.append(" ".join(current_tokens).strip())
            current_tokens = []
            current_key = key

        if token:
            current_tokens.append(token)

        try:
            conf_value = float(row.get("conf") or -1)
        except ValueError:
            conf_value = -1.0
        if conf_value > 0:
            confidences.append(conf_value)

    if current_tokens:
        grouped_lines.append(" ".join(current_tokens).strip())

    text = "\n".join(line for line in grouped_lines if line).strip()
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return text, confidence


def ocr_page(image: object, lang: str = DEFAULT_OCR_LANG, psm: int = DEFAULT_OCR_PSM) -> tuple[str, float]:
    """OCR một trang bằng pytesseract và trả về text + average confidence."""

    from PIL import Image
    from PIL import ImageFilter
    import numpy as np
    import pytesseract

    _configure_tesseract()
    _ensure_tessdata(lang)

    image_array = np.array(image)
    try:
        import cv2

        if image_array.ndim == 3:
            grayscale = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            grayscale = image_array

        _, thresholded = cv2.threshold(
            grayscale,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        denoised = cv2.medianBlur(thresholded, 3)
        processed = _deskew_image(denoised)
    except Exception:
        grayscale_image = Image.fromarray(image_array).convert("L")
        thresholded_image = grayscale_image.point(lambda pixel: 255 if pixel > 180 else 0)
        processed = np.array(thresholded_image.filter(ImageFilter.MedianFilter(size=3)))

    config = f"--oem 3 --psm {psm}"
    try:
        data = pytesseract.image_to_data(
            Image.fromarray(processed),
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractError as error:
        LOGGER.warning("pytesseract image_to_data failed, fallback to CLI OCR: %s", error)
        return _ocr_page_via_tesseract_cli(processed, lang=lang, psm=psm)

    rows = [
        {key: str(value_list[index]) for key, value_list in data.items()}
        for index in range(len(data.get("text", [])))
    ]
    return _rows_to_text_and_conf(rows)


def render_pages_for_ocr(pdf_path: str | Path, dpi: int = DEFAULT_OCR_DPI) -> list[object]:
    """Render PDF thành ảnh để OCR.

    Ưu tiên pdf2image theo yêu cầu; fallback sang PyMuPDF nếu thiếu Poppler.
    """

    try:
        from pdf2image import convert_from_path

        return list(convert_from_path(pdf_path, dpi=dpi))
    except Exception as error:
        LOGGER.warning("pdf2image failed, fallback to PyMuPDF rendering: %s", error)

    import fitz
    from PIL import Image

    scale = dpi / 72.0
    rendered_images: list[object] = []
    with fitz.open(pdf_path) as document:
        for page in document:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            rendered_images.append(image)
    return rendered_images


def _looks_table_like(text: str, tables: list[list[list[str]]]) -> bool:
    """Nhận diện tài liệu nhiều bảng để giữ output bảng."""

    if tables:
        return True
    has_digits = bool(re.search(r"\d", text))
    has_columns = bool(re.search(r"[ ]{2,}", text) or "|" in text)
    return has_digits and has_columns


def _cache_path_for_sha(sha256: str) -> Path:
    return OCR_CACHE_DIR / f"{sha256}.json"


def _doc_from_dict(payload: dict) -> ExtractedDoc:
    pages = [ExtractedPage(**page) for page in payload["pages"]]
    return ExtractedDoc(
        source_path=payload["source_path"],
        sha256=payload["sha256"],
        pages=pages,
        meta=payload.get("meta", {}),
    )


def _save_cache(document: ExtractedDoc) -> None:
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path_for_sha(document.sha256)
    with cache_path.open("w", encoding="utf-8") as file_obj:
        json.dump(asdict(document), file_obj, ensure_ascii=False, indent=2)


def _load_cache(sha256: str) -> ExtractedDoc | None:
    cache_path = _cache_path_for_sha(sha256)
    if not cache_path.exists():
        return None
    with cache_path.open(encoding="utf-8") as file_obj:
        return _doc_from_dict(json.load(file_obj))


def extract_pdf(
    pdf_path: str | Path,
    force_ocr: bool = False,
    dpi: int = DEFAULT_OCR_DPI,
    ocr_lang: str = DEFAULT_OCR_LANG,
) -> ExtractedDoc:
    """Trích xuất PDF và cache kết quả OCR theo SHA-256."""

    resolved_path = Path(pdf_path)
    sha256 = sha256_of(resolved_path)
    cached = _load_cache(sha256)
    if cached is not None:
        LOGGER.info("Loaded OCR cache for %s", resolved_path.name)
        return cached

    text_layers = extract_text_layer(resolved_path)
    page_tables = extract_tables_pdfplumber(resolved_path)
    rendered_images: list[object] | None = None

    pages: list[ExtractedPage] = []
    used_ocr_pages = 0
    for index, raw_text in enumerate(text_layers, start=1):
        should_ocr = force_ocr or needs_ocr(raw_text)
        page_text = raw_text
        confidence: float | None = None

        if should_ocr:
            if rendered_images is None:
                rendered_images = render_pages_for_ocr(resolved_path, dpi=dpi)
            page_text, confidence = ocr_page(rendered_images[index - 1], lang=ocr_lang)
            used_ocr_pages += 1

        tables = page_tables[index - 1] if index - 1 < len(page_tables) else []
        if not _looks_table_like(page_text, tables):
            tables = []

        pages.append(
            ExtractedPage(
                page_no=index,
                text=page_text,
                tables=tables,
                used_ocr=should_ocr,
                confidence=confidence,
            )
        )

    document = ExtractedDoc(
        source_path=str(resolved_path),
        sha256=sha256,
        pages=pages,
        meta={
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "page_count": len(pages),
            "used_ocr_pages": used_ocr_pages,
            "force_ocr": force_ocr,
            "ocr_lang": ocr_lang,
            "dpi": dpi,
        },
    )
    _save_cache(document)
    return document
