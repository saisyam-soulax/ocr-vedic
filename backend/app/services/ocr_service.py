from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.config import Settings, get_settings
from app.providers.base import transcribe_with_provider
from app.schemas import OcrPageResult, OcrProvider
from app.storage_uploads import sniff_is_pdf
from app.utils.pdf import pdf_bytes_to_page_images

logger = logging.getLogger(__name__)

_IMAGE_RE = re.compile(r"\.(png|jpe?g|webp|gif|tiff?)$", re.IGNORECASE)


def _default_system_prompt() -> str:
    return (
        "You are an expert paleographer for Vedic and Sanskrit manuscripts. "
        "Transcribe printed or handwritten Śruti/Smṛti text with maximal fidelity. "
        "Preserve Devanāgarī conjuncts, daṇḍas, numerals, piṇḍīs/paragraph markers, and any "
        "diacritic-rich Latin transliteration (IAST) if shown. Keep Udātta (acute), Anudātta "
        "(grave under), circumflex Svarita, and combined svara marks exactly as in the source. "
        "Do not normalize sandhi or 'fix' readings; do not add commentary."
    )


def parse_few_shots_json(raw: str | None) -> list[dict]:
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"few_shots is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise HTTPException(status_code=422, detail="few_shots must be a JSON array")
    out: list[dict] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail=f"few_shots[{i}] must be an object")
        if "expected_text" not in item:
            raise HTTPException(
                status_code=422, detail=f"few_shots[{i}] missing expected_text"
            )
        out.append(
            {
                "expected_text": str(item["expected_text"]),
                "image_base64": item.get("image_base64"),
                "mime_type": item.get("mime_type"),
            }
        )
    return out


async def build_few_shots_for_provider(
    items: list[dict],
    few_shot_files: list[UploadFile] | None,
) -> list[dict[str, str]]:
    if not items:
        return []

    files = few_shot_files or []
    need_files = sum(1 for x in items if not x.get("image_base64"))
    if need_files != len(files):
        raise HTTPException(
            status_code=422,
            detail=(
                "Each few-shot needs an image: provide image_base64 in JSON or upload "
                f"few_shot_files[] in the same order (expected {need_files} files, got {len(files)})."
            ),
        )

    file_iter = iter(files)
    result: list[dict[str, str]] = []
    for item in items:
        if item.get("image_base64"):
            b64 = str(item["image_base64"])
            mime = str(item.get("mime_type") or "image/png")
        else:
            uf = next(file_iter)
            body = await uf.read()
            b64 = base64.b64encode(body).decode("ascii")
            mime = uf.content_type or str(item.get("mime_type") or "image/png")
        result.append(
            {
                "image_base64": b64,
                "expected_text": str(item["expected_text"]),
                "mime_type": mime.split(";")[0].strip().lower(),
            }
        )
    return result


def _mime_for_upload(filename: str | None, content_type: str | None) -> str:
    if content_type and content_type != "application/octet-stream":
        return content_type.split(";")[0].strip().lower()
    fn = (filename or "").lower()
    if fn.endswith(".png"):
        return "image/png"
    if fn.endswith(".jpg") or fn.endswith(".jpeg"):
        return "image/jpeg"
    if fn.endswith(".webp"):
        return "image/webp"
    if fn.endswith(".gif"):
        return "image/gif"
    return "image/png"


async def run_ocr_batch(
    *,
    saved_files: list[tuple[Path, str, str | None]],
    provider: str,
    system_prompt: str | None,
    few_shots: list[dict[str, str]],
    settings: Settings | None = None,
    model_id: str | None = None,
) -> tuple[list[OcrPageResult], str]:
    """Run OCR using files persisted on disk (see ``persist_ocr_uploads``)."""
    if not saved_files:
        raise HTTPException(status_code=422, detail="At least one file is required in files[].")

    settings = settings or get_settings()
    try:
        OcrProvider(provider)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Unknown provider: {provider}") from exc

    syn = system_prompt.strip() if system_prompt else _default_system_prompt()
    pages_out: list[OcrPageResult] = []

    for path, name, content_type in saved_files:
        raw = path.read_bytes()
        file_size = len(raw)
        logger.info("Processing file: name=%r size=%d bytes content_type=%s", name, file_size, content_type)

        if sniff_is_pdf(raw, name, content_type):
            logger.info("  Detected as PDF, rasterizing: name=%r", name)
            try:
                rendered = pdf_bytes_to_page_images(raw)
            except Exception as exc:
                logger.exception("Failed to rasterize PDF: name=%r", name)
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to read PDF {name}: {exc}",
                ) from exc
            logger.info("  PDF rasterized: name=%r pages=%d", name, len(rendered))

            for p in rendered:
                logger.info(
                    "  Sending page to provider: file=%r page=%d image_size=%d bytes provider=%s",
                    name, p.page_number, len(p.image_bytes), provider,
                )
                text = transcribe_with_provider(
                    provider,
                    image_bytes=p.image_bytes,
                    mime_type=p.mime_type,
                    system_prompt=syn,
                    few_shots=few_shots,
                    settings=settings,
                    model_id=model_id,
                )
                logger.info(
                    "  Provider response: file=%r page=%d response_chars=%d",
                    name, p.page_number, len(text),
                )
                pages_out.append(
                    OcrPageResult(
                        index=len(pages_out),
                        source_file=name,
                        page_in_source=p.page_number,
                        text=text,
                        mime_type=p.mime_type,
                    )
                )

        elif _IMAGE_RE.search(name) or ((content_type or "").lower().startswith("image/")):
            mime = _mime_for_upload(name, content_type)
            logger.info("  Detected as image: name=%r mime=%s size=%d bytes", name, mime, file_size)
            logger.info(
                "  Sending image to provider: file=%r provider=%s",
                name, provider,
            )
            text = transcribe_with_provider(
                provider,
                image_bytes=raw,
                mime_type=mime,
                system_prompt=syn,
                few_shots=few_shots,
                settings=settings,
                model_id=model_id,
            )
            logger.info(
                "  Provider response: file=%r page=1 response_chars=%d",
                name, len(text),
            )
            pages_out.append(
                OcrPageResult(
                    index=len(pages_out),
                    source_file=name,
                    page_in_source=1,
                    text=text,
                    mime_type=mime,
                )
            )
        else:
            logger.warning("Unsupported file type: name=%r content_type=%s", name, content_type)
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unsupported file type for {name}. "
                    "Upload PDF or images (png, jpeg, webp, gif)."
                ),
            )

    sep = "\n\n---\n\n"
    combined = sep.join(
        f"## {p.source_file}"
        + (f" (page {p.page_in_source})" if p.page_in_source else "")
        + "\n\n"
        + p.text
        for p in pages_out
    )
    logger.info(
        "Batch complete: total_pages=%d combined_chars=%d",
        len(pages_out), len(combined),
    )
    return pages_out, combined
