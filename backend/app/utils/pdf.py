from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfPageImage:
    page_number: int  # 1-based
    mime_type: str
    image_bytes: bytes


def pdf_page_count(pdf_bytes: bytes) -> int:
    """Return page count without rasterizing (fast; used to emit OCR start early)."""
    import fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return doc.page_count
    finally:
        doc.close()


def pdf_page_to_image(pdf_bytes: bytes, page_number: int, dpi: int = 150) -> PdfPageImage:
    """Rasterize a single 1-based PDF page to JPEG bytes."""
    import fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_number < 1 or page_number > doc.page_count:
            raise ValueError(f"page_number {page_number} out of range 1..{doc.page_count}")
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        page = doc.load_page(page_number - 1)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        jpg = pix.tobytes("jpeg", jpg_quality=95)
        return PdfPageImage(page_number=page_number, mime_type="image/jpeg", image_bytes=jpg)
    finally:
        doc.close()


def pdf_bytes_to_page_images(pdf_bytes: bytes, dpi: int = 150) -> list[PdfPageImage]:
    """Rasterize all PDF pages to JPEG bytes using PyMuPDF."""
    import fitz  # PyMuPDF

    pdf_size = len(pdf_bytes)
    logger.info("Rasterizing PDF: size=%d bytes dpi=%d", pdf_size, dpi)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        logger.exception("fitz.open failed: size=%d bytes", pdf_size)
        raise

    page_count = doc.page_count
    logger.info("PDF opened: pages=%d dpi=%d", page_count, dpi)

    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    out: list[PdfPageImage] = []
    try:
        for i in range(page_count):
            try:
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                # Quality 95 preserves fine Devanāgarī strokes; still ~2× smaller than PNG.
                jpg = pix.tobytes("jpeg", jpg_quality=95)
            except Exception:
                logger.exception("Failed to rasterize page %d of %d", i + 1, page_count)
                raise
            logger.debug(
                "Rasterized page %d/%d: %d bytes (%dx%d px)",
                i + 1, page_count, len(jpg), pix.width, pix.height,
            )
            out.append(
                PdfPageImage(
                    page_number=i + 1,
                    mime_type="image/jpeg",
                    image_bytes=jpg,
                )
            )
    finally:
        doc.close()

    logger.info(
        "PDF rasterization complete: pages=%d total_bytes=%d",
        len(out), sum(len(p.image_bytes) for p in out),
    )
    return out


def iter_pdf_pages(pdf_bytes: bytes, dpi: int = 150) -> Iterator[PdfPageImage]:
    return iter(pdf_bytes_to_page_images(pdf_bytes, dpi=dpi))
