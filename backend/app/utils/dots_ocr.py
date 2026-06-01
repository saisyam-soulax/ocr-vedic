"""dots.ocr + vLLM message layout (aligned with ``ocr_pipeline/vllm_client.py``)."""
from __future__ import annotations

# Required by dots.ocr chat template — text after the image in the user turn.
DOTS_IMAGE_TEXT_PREFIX = "<|img|><|imgpad|><|endofimg|>"

DEFAULT_USER_TAIL = (
    "Transcribe this page image exactly according to the system protocol. "
    "Output only the transcribed text as specified."
)

OCR_TASK_TEXT = "Extract the text content from this image."


def dots_user_text(*, user_instructions: str = "", use_arsha_tail: bool = True) -> str:
    """Build the text segment that follows the image in a dots.ocr user message."""
    parts: list[str] = [DOTS_IMAGE_TEXT_PREFIX]
    if user_instructions.strip():
        parts.append(user_instructions.strip())
        parts.append("")
    parts.append(OCR_TASK_TEXT)
    if use_arsha_tail:
        parts.append("")
        parts.append(DEFAULT_USER_TAIL)
    else:
        parts.append("")
        parts.append(
            "Output plain text only (no JSON, no markdown fences, no commentary). "
            "Preserve every character, diacritic, and line break from the source."
        )
    return "\n".join(parts).strip()


def dots_user_content_parts(
    image_url_part: dict,
    *,
    user_instructions: str = "",
) -> list[dict]:
    """Image first, then prefixed text — matches ``ocr_pipeline.vllm_client``."""
    return [
        image_url_part,
        {"type": "text", "text": dots_user_text(user_instructions=user_instructions)},
    ]
