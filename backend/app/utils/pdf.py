from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Some scanned PDFs declare an oversized page box (e.g. 39x59in instead of the
# actual ~8x11in scan), which multiplies dpi-based rendering into a
# 90+ megapixel image per page for no quality gain — the source content isn't
# actually that detailed, so we're just upsampling. Cap the longest rendered
# edge so a bad page box can't blow up the payload; normal-sized pages never
# hit this cap at any DPI we use.
MAX_RENDER_EDGE_PX = 4000


def _force_all_layers_visible(doc) -> None:
    """Force every optional content group (PDF 'layer') visible before
    rendering.

    Some scanned/processed PDFs put the actual page image on a layer marked
    hidden-by-default (e.g. from whatever tool split/prepared the file) —
    many PDF viewers ignore or override this and show the content anyway,
    but PyMuPDF's rendering respects the stored default, producing a
    perfectly valid, blank-white page image with zero errors. This is
    silent and easy to miss: the page "succeeds" with no exception, Gemini
    correctly reports nothing legible on a genuinely blank input, and
    nothing in the pipeline flags it as wrong. See INC-007 in
    docs/INCIDENT_LOG.md — confirmed via a synthetic PDF with a
    hidden-by-default OCG: rendering before this fix produced a blank page,
    identical after forcing the layer on.

    A no-op (and safe) for the vast majority of PDFs, which have no layers
    at all — ``layer_ui_configs()`` returns an empty list for them.
    """
    try:
        configs = doc.layer_ui_configs()
    except Exception:
        return
    for cfg in configs:
        if not cfg.get("on"):
            try:
                doc.set_layer_ui_config(cfg["number"], action=1)  # 1 = turn ON
            except Exception:
                logger.warning(
                    "Could not force PDF layer %r visible (number=%s) — "
                    "rendering may be missing content on this layer.",
                    cfg.get("text"), cfg.get("number"),
                )


def _capped_render_matrix(page, dpi: int):
    """Render matrix for `page` at `dpi`, capped so neither output edge exceeds
    MAX_RENDER_EDGE_PX. Returns a fitz.Matrix."""
    import fitz

    scale = dpi / 72.0
    rect = page.rect
    long_edge_px = max(rect.width, rect.height) * scale
    if long_edge_px > MAX_RENDER_EDGE_PX:
        scale *= MAX_RENDER_EDGE_PX / long_edge_px
        logger.warning(
            "Page box %.0fx%.0fpt at %d dpi would render to %.0fpx long edge; "
            "capping to %dpx (effective dpi=%.0f)",
            rect.width, rect.height, dpi, long_edge_px, MAX_RENDER_EDGE_PX, scale * 72.0,
        )
    return fitz.Matrix(scale, scale)


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
        _force_all_layers_visible(doc)
        if page_number < 1 or page_number > doc.page_count:
            raise ValueError(f"page_number {page_number} out of range 1..{doc.page_count}")
        page = doc.load_page(page_number - 1)
        matrix = _capped_render_matrix(page, dpi)
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

    _force_all_layers_visible(doc)
    page_count = doc.page_count
    logger.info("PDF opened: pages=%d dpi=%d", page_count, dpi)

    out: list[PdfPageImage] = []
    try:
        for i in range(page_count):
            try:
                page = doc.load_page(i)
                matrix = _capped_render_matrix(page, dpi)
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
