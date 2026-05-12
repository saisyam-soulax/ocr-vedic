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


def pdf_bytes_to_page_images(pdf_bytes: bytes, dpi: int = 200) -> list[PdfPageImage]:
    """Rasterize PDF pages to PNG bytes using PyMuPDF."""
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
                png = pix.tobytes("png")
            except Exception:
                logger.exception("Failed to rasterize page %d of %d", i + 1, page_count)
                raise
            logger.debug(
                "Rasterized page %d/%d: output_size=%d bytes (%dx%d px)",
                i + 1, page_count, len(png), pix.width, pix.height,
            )
            out.append(
                PdfPageImage(
                    page_number=i + 1,
                    mime_type="image/png",
                    image_bytes=png,
                )
            )
    finally:
        doc.close()

    logger.info(
        "PDF rasterization complete: pages=%d total_image_bytes=%d",
        len(out), sum(len(p.image_bytes) for p in out),
    )
    return out


def iter_pdf_pages(pdf_bytes: bytes, dpi: int = 200) -> Iterator[PdfPageImage]:
    return iter(pdf_bytes_to_page_images(pdf_bytes, dpi=dpi))
