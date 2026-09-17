"""Regression tests for hidden-PDF-layer rendering (INC-007).

Some PDFs put real page content on an optional content group (OCG / layer)
marked hidden by default. PyMuPDF respects that default and silently renders
a blank page — no exception, no error, just an empty image handed to Gemini.
_force_all_layers_visible() must force every such layer on before rendering.
"""
import fitz

from app.utils.pdf import pdf_bytes_to_page_images, pdf_page_to_image


def _pdf_with_hidden_layer_text() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    ocg_xref = doc.add_ocg("HiddenScanLayer", on=False)
    page.insert_text((72, 100), "REAL PAGE CONTENT", fontsize=20, oc=ocg_xref)
    data = doc.tobytes()
    doc.close()
    return data


def _rendered_image_is_blank(jpg_bytes: bytes) -> bool:
    """True if every pixel is (near-)white — i.e. nothing was drawn."""
    import io

    from PIL import Image

    im = Image.open(io.BytesIO(jpg_bytes)).convert("L")
    extrema = im.getextrema()
    return extrema[0] > 250  # darkest pixel is still essentially white


def test_pdf_page_to_image_reveals_hidden_layer_content() -> None:
    pdf_bytes = _pdf_with_hidden_layer_text()
    result = pdf_page_to_image(pdf_bytes, page_number=1, dpi=150)
    assert not _rendered_image_is_blank(result.image_bytes), (
        "hidden-layer content was not forced visible — page rendered blank"
    )


def test_pdf_bytes_to_page_images_reveals_hidden_layer_content() -> None:
    pdf_bytes = _pdf_with_hidden_layer_text()
    images = pdf_bytes_to_page_images(pdf_bytes, dpi=150)
    assert len(images) == 1
    assert not _rendered_image_is_blank(images[0].image_bytes)


def test_normal_pdf_without_layers_still_renders_fine() -> None:
    """A PDF with no OCGs at all must render unaffected (the common case)."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "NORMAL PAGE, NO LAYERS", fontsize=20)
    pdf_bytes = doc.tobytes()
    doc.close()

    result = pdf_page_to_image(pdf_bytes, page_number=1, dpi=150)
    assert not _rendered_image_is_blank(result.image_bytes)
