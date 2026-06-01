"""Consolidated OCR text layout (aligned with pdfToGeminiRangeSpecific.py)."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

_RULE = "=" * 70
_PAGE_RULE = "▬" * 70


def resolve_page_number(page: dict[str, Any]) -> int:
    """1-based page number: PDF page when known, else global sequence."""
    pin = page.get("page_in_source")
    if pin is not None:
        return int(pin)
    return int(page.get("index", 0)) + 1


def format_page_marker_block(page_num: int) -> str:
    """Exact marker block from pdfToGeminiRangeSpecific.py (lines 606–608)."""
    return f"\n{_PAGE_RULE}\nPAGE {page_num}\n{_PAGE_RULE}\n\n"


def format_page_section(page: dict[str, Any], *, multi_source: bool) -> str:
    """Marker block plus transcription for one page."""
    num = resolve_page_number(page)
    text = (page.get("text") or "").strip()
    if not text:
        text = "[ERROR: Page OCR produced no text]"
    label = f"PAGE {num}"
    if multi_source:
        src = page.get("source_file") or ""
        if src:
            label = f"{label} — {src}"
    # Same layout as reference script; filename suffix only when multiple sources.
    if multi_source and page.get("source_file"):
        return f"\n{_PAGE_RULE}\n{label}\n{_PAGE_RULE}\n\n{text}\n"
    return format_page_marker_block(num) + text + "\n"


def page_sort_key(page: dict[str, Any]) -> tuple:
    pin = page.get("page_in_source")
    return (
        page.get("source_file") or "",
        pin if pin is not None else int(page.get("index", 0)) + 1,
        int(page.get("index", 0)),
    )


def format_consolidated_ocr_text(
    pages: list[dict[str, Any]],
    *,
    source_files: list[str] | None = None,
    provider: str | None = None,
    submitted_at: str | None = None,
    completed_at: str | None = None,
    elapsed_seconds: float | None = None,
) -> str:
    """Build a single plain-text file with header, per-page blocks, and footer."""
    sorted_pages = sorted(pages, key=page_sort_key)
    files = source_files or []
    unique_sources = {p.get("source_file") for p in sorted_pages if p.get("source_file")}
    multi_source = len(unique_sources) > 1

    now = datetime.now(UTC)
    proc_date = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    if completed_at:
        proc_date = completed_at.replace("T", " ").replace("+00:00", " UTC")

    lines: list[str] = [
        _RULE,
        "VEDIC OCR OUTPUT",
        f"Source file(s): {', '.join(files) if files else '(unknown)'}",
        f"Processing date: {proc_date}",
    ]
    if provider:
        lines.append(f"Provider: {provider}")
    if submitted_at:
        lines.append(f"Submitted: {submitted_at}")
    if elapsed_seconds is not None:
        lines.append(f"Elapsed: {elapsed_seconds:.1f} s")
    lines.append(f"Pages: {len(sorted_pages)}")
    lines.extend([_RULE, ""])

    body = "".join(format_page_section(p, multi_source=multi_source) for p in sorted_pages)
    footer = (
        f"\n{_RULE}\n"
        f"END OF OCR OUTPUT\n"
        f"Total pages processed: {len(sorted_pages)}\n"
        f"Completion time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"{_RULE}\n"
    )
    return "\n".join(lines) + "\n" + body + footer
