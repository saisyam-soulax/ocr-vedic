"""Shared prompt fragments for OCR providers."""
from __future__ import annotations


def user_instructions_prefix(user_prompt: str | None) -> str:
    """Optional block prepended to the per-page user message."""
    extra = (user_prompt or "").strip()
    if not extra:
        return ""
    return f"Additional instructions from the user:\n{extra}\n\n"
