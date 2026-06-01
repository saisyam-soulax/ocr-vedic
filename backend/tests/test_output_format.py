from app.utils.output_format import (
    format_consolidated_ocr_text,
    format_page_marker_block,
    resolve_page_number,
)


def test_resolve_page_number_prefers_pdf_page():
    assert resolve_page_number({"index": 5, "page_in_source": 12}) == 12
    assert resolve_page_number({"index": 5}) == 6


def test_page_marker_block_matches_reference_script():
    block = format_page_marker_block(42)
    assert block.startswith("\n" + "▬" * 70)
    assert "PAGE 42\n" in block
    assert block.endswith("\n\n")


def test_format_consolidated_includes_page_markers():
    text = format_consolidated_ocr_text(
        [
            {"index": 0, "source_file": "book.pdf", "page_in_source": 3, "text": "ॐ"},
            {"index": 1, "source_file": "book.pdf", "page_in_source": 4, "text": "नमः"},
        ],
        source_files=["book.pdf"],
    )
    assert text.index("PAGE 3") < text.index("ॐ")
    assert text.index("PAGE 4") < text.index("नमः")
    assert "▬" * 70 in text
    assert "END OF OCR OUTPUT" in text
