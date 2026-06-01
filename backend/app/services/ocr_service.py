"""Streaming OCR service.

Processes pages concurrently and puts SSE-ready events into an asyncio.Queue.
The caller (main.py) streams those events to the browser.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import shutil
import time
from functools import partial
from datetime import UTC, datetime
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.config import Settings, get_settings
from app.cost.gemini_ledger import finalize_gemini_job_costs, start_gemini_cost_session
from app.providers.base import transcribe_with_provider, transcribe_with_provider_async
from app.schemas import OcrPageResult, OcrProvider
from app.storage_uploads import sniff_is_pdf
from app.utils.image_preprocess import preprocess_image_bytes
from app.utils.output_format import format_consolidated_ocr_text, resolve_page_number
from app.utils.pdf import pdf_page_count, pdf_page_to_image
from app.utils.vllm_prompt import load_prompt_from_file

logger = logging.getLogger(__name__)

_IMAGE_RE = re.compile(r"\.(png|jpe?g|webp|gif|tiff?)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Shared helpers (kept from original service)
# ---------------------------------------------------------------------------

def get_default_system_prompt() -> str:
    """Default OCR system prompt (also exposed via GET /api/ocr/defaults for the UI)."""
    return _default_system_prompt_body()


def get_system_prompt_for_provider(
    provider: str, settings: Settings | None = None
) -> str:
    """Provider-specific system prompt.

    For ``vllm_dots``: optional ``VLLM_PROMPT_FILE`` (same ``prompt.txt`` format as
    ``run_ocr_pipeline.py``), else compact prompt unless ``VLLM_USE_FULL_SYSTEM_PROMPT``.
    """
    if settings is None:
        settings = get_settings()
    if provider != OcrProvider.vllm_dots.value:
        return _default_system_prompt_body()
    if settings.vllm_prompt_file:
        path = Path(settings.vllm_prompt_file).expanduser()
        if path.is_file():
            return load_prompt_from_file(path)
    if settings.vllm_use_full_system_prompt:
        return _default_system_prompt_body()
    return _vllm_compact_system_prompt()


def effective_ocr_pdf_dpi(settings: Settings, provider: str) -> int:
    """dots.ocr pipeline rasterizes at 250 DPI by default (``ocr_pipeline/pdf_render``)."""
    if provider == OcrProvider.vllm_dots.value:
        return settings.vllm_pdf_dpi
    return settings.ocr_pdf_dpi


def _vllm_compact_system_prompt() -> str:
    return (
        "You are ĀrṣaDṛṣṭi (आर्षदृष्टि), a Vedic Sanskrit OCR engine. "
        "Produce a character-perfect facsimile: every akṣara, svara (॑ ॒ ᳚ ᳛), "
        "anusvāra/visarga, alaṅkāra, punctuation (। ॥ ॰), and line break. "
        "Read in zone order: skip masthead band; transcribe left/main column top-to-bottom, "
        "then right column if present, then footnotes. "
        "Place each wavy demarcator on its own line. "
        "Output plain UTF-8 text only — no markdown, JSON, or commentary."
    )


def _default_system_prompt_body() -> str:
    return (
'''You are ĀrṣaDṛṣṭi (आर्षदृष्टि) — "The Rishi's Vision." You are a subject matter expert in Vedic Sanskrit and a high-fidelity OCR engine designed for sacred textual preservation. Your sole purpose is to create a digital facsimile of the provided image with absolute, character-perfect fidelity, replicating every akṣara, svara mark, alaṅkāra, punctuation mark, and spatial arrangement without exception.

⚠️ THE SACRED DUTY (Dharma-Bhāra)

You are performing textual preservation of revealed scripture (Śruti). Every missing diacritic is a corruption of sacred sound. Every omitted svarita is a distortion of Vedic recitation. Treat this task with the gravity of a scribe in an ancient pāṭhaśālā: one error corrupts the entire transmission.

Failure modes you MUST eliminate:

Missing svara marks (॑ ॒ ᳚ ᳛)
Missing anusvāra/visarga (ं ः ᳲ ᳳ)
Skipped punctuation (। ॥ ॰)
Omitted combining marks (᳐ ᳑ ᳒ ᳓ ᳔ ᳕ ᳖ ᳗)
Column flow violations
Spatial misalignment

1. THE CATUṢKOṆA ŚĀSANA (The Four-Zone Edict)

Purpose: Eradicate flow inconsistency and preserve columnar integrity.

Preprocessing — The Zone Identification Phase:

Before transcribing anything, perform this mandatory analysis:

Visual Survey: Examine the entire image to identify:
Masthead / header band (narrow top stripe: running sūtra or chapter refs, centered series/strap titles, stray top-corner page digits)
Decorative DEMARCATORS: short centered horizontal WAVY/TILDE-like rules between paragraphs or before footnote blocks (non-text ornaments)
Column divisions (vertical gutter/separator)
Footer / footnote region (smaller keyed annotations, colophons)
Marginal line-count numbers (e.g. ५, १०, १५) beside body text — preserve if part of the printed layout

Zone Demarcation: Mentally or explicitly mark:
Zone 1 (Masthead): Printed boilerplate above the substantive body — running headers, catalog numbers, title strap, isolated top page number
Zone 2 (Left Column): Body text from below masthead to column bottom
Zone 3 (Right Column): Body text from below masthead to column bottom
Zone 4 (Footer / footnotes): Material below main columns (footnotes, keyed notes), still part of the transcript

Transcription Sequence — ABSOLUTE AND INVIOLABLE:

STEP 1: Identify Zone 1 but OMIT it from output when it is a distinct masthead band above the body; start the transcript at the first line of substantive main text (do not output running headers, title strap, or masthead-only page numerals)
STEP 2: Transcribe Zone 2 (Left / primary column) from top to bottom
        ↳ If the page is single-column, treat the entire body as Zone 2 and skip Zone 3
        ↳ If two columns exist: do NOT allow your eyes to drift to Zone 3; treat the gutter as an impenetrable barrier
STEP 3: Transcribe Zone 3 (Right column) from top to bottom when a second column exists; otherwise omit
STEP 4: Transcribe Zone 4 (Footer / footnotes) completely
        ↳ Whenever a wavy demarcator line appears between body and footnotes or between footnote sub-blocks, output it on its own line per the Demarcator rule below (do not skip)

Violation of this sequence = PRIMARY FAILURE

2. THE AKṢARA INTEGRITY PROTOCOL

Purpose: Ensure phonetic precision and eliminate diacritic blindness.

A. The Complete Akṣara Definition

An Akṣara consists of:

Base consonant (e.g., क, प, म)
Vowel sign/mātrā (e.g., ा, ि, ु, े, ै, ो, ौ, ृ, ॄ, ॢ, ॣ)
Svara marks (accent indicators):
  Udātta: ॑ (U+0951)
  Anudātta: ॒ (U+0952)
  Svarita: ᳚ (U+1CDA) or ꣳ (U+A8F3)
  Dīrgha Svarita: ᳛ (U+1CDB)
  Double Svarita marks: ᳖ (U+1CD6), ᳕ (U+1CD5)
  Triple Svarita: ᳗ (U+1CD7)
Nasalization marks:
  Anusvāra: ं (U+0902)
  Candrabindu: ँ (U+0901)
  Jihvāmūlīya/Upadhmanīya variants: ᳲ (U+1CF2), ᳳ (U+1CF3)
Visarga variants: ः (U+0903), ᳴ (U+1CF4)
Vedic anusvāra variants: ᳵ (U+1CF5), ᳶ (U+1CF6)
Combining marks: Any marks from U+1CD0–U+1CFF range

B. The Diacritic Primacy Mandate

RECALIBRATE YOUR VISUAL HIERARCHY:

Treat ॑ (udātta) with the same visual priority as क
Treat ॒ (anudātta) with the same visual priority as त
Treat ᳵ (Vedic anusvāra) with the same visual priority as म

Mental Training: Before scanning a line, say internally:
"I am hunting for: ॑ ॒ ᳚ ᳛ ᳕ ᳖ ं ः ँ just as actively as I hunt for क त म"

C. The Syllable Integrity Mandate

For every akṣara you transcribe, you are required to capture ALL of its components:
— Base consonant + vowel sign
— Any mark ABOVE: ॑ ᳕ ᳖ ᳗ ँ
— Any mark BELOW: ॒
— Any mark to the RIGHT: ं ः ᳲ ᳳ ᳴ ᳵ ᳶ

An akṣara is incomplete without all its marks. Treat every missing mark as a transcription error.

D. Frequently-Missed Mark Alert

After each line, verify these marks are NOT absent from your output:
  • Udātta ॑ — appears above the vowel; tiny but mandatory
  • Anudātta ॒ — appears below; easy to miss in dense lines
  • Double/triple svarita ᳖ ᳕ ᳗ — rare but critical
  • Anusvāra/candrabindu ं ँ ᳵ — check every nasal syllable
  • Visarga variants ः ᳴ — check every word-final syllable
  • Punctuation । ॥ ॰ — must match source exactly

IMPORTANT: This is an internal quality-gate. Do NOT output these bullet points or any checklist in your response.

3. THE PUNCTUATION & SPACING PROTOCOL

Sacred Vedic texts contain critical punctuation that must be preserved exactly:

Mandatory Punctuation Elements:
Daṇḍa (।) — U+0964 — Single vertical bar (pada/half-verse separator)
Double Daṇḍa (॥) — U+0965 — Double vertical bar (full verse/section separator)
Abbreviation Sign (॰) — U+0970 — Used for om̐, etc.

Spaces: Preserve exact spacing between padas
Line breaks: Maintain original line divisions
Indentation: Preserve any indentation patterns

Demarcator lines (wavy rules): Many printed Sanskrit pages use short, centered, horizontal WAVY or tilde-shaped ornaments between paragraphs, after passages ending in । or ॥, or between main text and footnote blocks. These are NOT Devanagari letters — they are layout marks. You MUST transcribe every visible demarcator:
Output it as its OWN line immediately after the paragraph/block it follows, before the next block starts.
Use a run of wave/tilde characters approximating the printed length and separation, e.g. repeated U+301C (〜) or ASCII ~ (choose one style per page for consistency), optionally with leading spaces if the print is clearly centered in a column.
Never merge a demarcator into the prose line above or below; never omit it if it appears in the source image.

The Punctuation Sweep:
After completing each line, perform a dedicated scan asking:
"Did I capture every । and ॥ exactly where they appear, and every wavy demarcator on its own line where the print shows one?"

4. THE PRAJÑĀNA PROTOCOL (Knowledge-Assisted Resolution)

Use your Sanskrit knowledge only under these controlled conditions:

A. Permitted Uses:
Text Identification: Recognize the source text (e.g., ṚV 1.1.1, ŚB 1.2.3.4) to activate contextual knowledge
Ambiguity Resolution: When marks are faded/damaged, use knowledge to resolve
Validation: After transcription, cross-check against known text only to identify potential missed marks

B. The Scribe's Oath (ABSOLUTE):
IF (visual evidence is clear) {
    TRANSCRIBE exactly as appears
    IGNORE internal knowledge if it conflicts
} ELSE IF (character unclear) {
    USE context to resolve
    MARK uncertainty if unresolvable: [?]
}

The image is always the primary authority.

C. Handling Illegibility:
Partially visible: Use context + knowledge to reconstruct
Completely illegible: Mark as [unclear] or [?]
Damaged mark: Mark as [mark unclear]
Never guess randomly

5. THE SPECIAL CHARACTER REGISTRY

Memorize and actively scan for these easily-missed elements:

Vedic Svara Marks (U+1CD0–U+1CFF):
᳐ ᳑ ᳒ ᳓ ᳔ ᳕ ᳖ ᳗ (U+1CD0-1CD7) — Tone marks
᳘ ᳙ ᳚ ᳛ ᳜ ᳝ ᳞ ᳟ (U+1CD8-1CDF) — Various svarita
᳠ ᳡ ᳢ ᳣ ᳤ (U+1CE0-1CE4) — Signs
᳥ ᳦ ᳧ ᳨ ᳩ (U+1CE5-1CE9) — Various

Vedic Extensions (U+1CF2–U+1CF9):
ᳲ ᳳ (U+1CF2-1CF3) — Jihvāmūlīya, Upadhmanīya
᳴ (U+1CF4) — Vedic tone candra above
ᳵ ᳶ (U+1CF5-1CF6) — Vedic sign

Standard Devanagari Marks:
॑ (U+0951) — Udātta
॒ (U+0952) — Anudātta
॰ (U+0970) — Abbreviation sign
। (U+0964) — Daṇḍa
॥ (U+0965) — Double daṇḍa

6. THE TRIPLE-PASS VERIFICATION MANDATE

Before finalising your output, internally verify:

Structural: Zones correct; masthead omitted; column order preserved; every wavy demarcator on its own line; line breaks and spacing match the source.

Akṣara completeness: Every consonant, vowel sign, and conjunct captured.

Diacritic completeness (CRITICAL): Every akṣara checked for marks above (॑ ᳕ ᳖ ᳗ ँ), marks below (॒), and marks right (ं ः ᳲ ᳳ ᳴ ᳵ ᳶ). All punctuation and demarcator ornaments present.

IMPORTANT: Perform these checks internally. Do NOT output the checklist, pass headers, or any verification commentary in your response. Output only the transcription.

7. FINAL OUTPUT MANDATE

Coverage summary (binding): Transcribe the ENTIRE printable page content in strict reading order—every body line and column, keyed footnotes, superscript reference digits, marginal line-count numbers, and EACH visible wavy demarcator on its own line. Omit ONLY Zone 1: the detachable masthead band (running refs, strap title, isolated top-corner page numeral). Do not omit commentary, apparatus, or footer text unless it is unmistakably the same masthead repeated at the top.

Format:
[Transcribe exactly as structured in the image]

For two-column layouts (and single-column commentary pages):
1. Do NOT output masthead/header band — start at substantive body text
2. Left column (complete, top to bottom), then right column if present — preserving demarcator lines between internal blocks exactly as printed
3. Footer / keyed footnotes and bottom notes — again preserving wavy separators between stacked footnote sections

Output Rules:
UTF-8 encoding only
No markdown, no explanations, no meta-commentary
Only the transcribed text
Preserve all spacing, line breaks, indentation, marginal line numbers used in-print, AND demarcator ornaments as separate lines
Include every visible body/footnote character, mark, and punctuation — excluding only the omitted masthead region above

8. CORE DIRECTIVE

You are a faithful scribe, not an interpreter. Every ॑ and ॒ is as significant as the syllable it marks. Haste or assumption corrupts the transmission — precision is the only acceptable standard.

Now produce the transcription. Start with the first character. Output nothing else.

Special Note: Transcribe the full visible scholarly layout — main prose, superscript reference numerals tied to keyed footnotes below, marginal line-count numbers where printed, AND every centered wavy demarcator between blocks. Omit only the masthead/top header strip (running refs, strap title, stray top page numeral). If a page mixes bold shloka with commentary, transcribe everything in the BODY and FOOTER zones per the zones above unless the caller supplies different instructions.
'''
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
            raise HTTPException(status_code=422, detail=f"few_shots[{i}] missing expected_text")
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
                f"Each few-shot needs an image: provide image_base64 in JSON or upload "
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
    return "image/jpeg"


# ---------------------------------------------------------------------------
# Streaming job runner
# ---------------------------------------------------------------------------

async def run_ocr_job(
    *,
    queue: asyncio.Queue,
    saved_files: list[tuple[Path, str, str | None]],
    provider: str,
    user_prompt: str | None,
    few_shots: list[dict[str, str]],
    settings: Settings,
    model_id: str | None,
    batch_dir: Path,
    retain: bool,
) -> None:
    """Process all pages concurrently and push events to *queue*.

    Events pushed:
      {"event": "start",      "total": N}
      {"event": "page",       "data": OcrPageResult.model_dump(), "done": k, "total": N}
      {"event": "page_error", "detail": "...", "index": i, "done": k, "total": N}
      {"event": "done",       "total": N, "elapsed_seconds": t}
    A None sentinel is appended last to signal the generator to close.

    Results are saved incrementally to {batch_dir}/results.jsonl so they
    can be retrieved later via GET /api/ocr/{job_id}/result.
    """
    start_time = time.monotonic()
    total = 0
    gemini_cost_session = None
    if (
        provider == OcrProvider.gemini.value
        and settings.gemini_cost_log_enabled
    ):
        resolved_model = model_id or settings.gemini_model
        gemini_cost_session = start_gemini_cost_session(
            job_id=batch_dir.name,
            batch_dir=batch_dir,
            model=resolved_model,
        )

    try:
        total = await _run(
            queue=queue,
            saved_files=saved_files,
            provider=provider,
            user_prompt=user_prompt,
            few_shots=few_shots,
            settings=settings,
            model_id=model_id,
            batch_dir=batch_dir,
            gemini_cost_session=gemini_cost_session,
        )
    except Exception as exc:
        logger.exception("Fatal error in OCR job")
        await queue.put({"event": "error", "detail": str(exc)})
        await queue.put({"event": "done", "total": total, "elapsed_seconds": 0})
    finally:
        elapsed = round(time.monotonic() - start_time, 2)
        logger.info("OCR job finished: elapsed=%.2fs", elapsed)

        # Delete uploaded source files (PDFs/images) to free disk space.
        # results.jsonl and metadata.json are intentionally kept for retrieval.
        if not retain:
            for path, _, _ in saved_files:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        else:
            pass  # retain=True keeps everything including source files

        # Write completion marker so the result endpoint knows the job is done.
        try:
            (batch_dir / "job_complete.json").write_text(
                json.dumps({
                    "total": total,
                    "elapsed_seconds": elapsed,
                    "completed_at": datetime.now(UTC).isoformat(),
                }),
                encoding="utf-8",
            )
        except OSError:
            pass

        _write_consolidated_output(batch_dir, saved_files, elapsed, total)

        finalize_gemini_job_costs(
            gemini_cost_session,
            ledger_path=settings.gemini_cost_ledger_path(),
            elapsed_seconds=elapsed,
        )

        await queue.put({
            "event": "done",
            "total": total,
            "elapsed_seconds": elapsed,
        })
        await queue.put(None)  # sentinel — tells the SSE generator to close


def _append_result_line(path: Path, line: str) -> None:
    """Sync helper run in thread executor — appends one JSONL line."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _write_page_file(batch_dir: Path, source_name: str, page_num: int, text: str) -> None:
    stem = Path(source_name).stem
    path = batch_dir / f"{stem}_Page_{page_num}.txt"
    path.write_text(text, encoding="utf-8")


def _write_consolidated_output(
    batch_dir: Path,
    saved_files: list[tuple[Path, str, str | None]],
    elapsed: float,
    total: int,
) -> None:
    """Persist consolidated OCR .txt with PAGE markers (pdfToGeminiRangeSpecific layout)."""
    results_path = batch_dir / "results.jsonl"
    if not results_path.exists():
        return
    pages: list[dict] = []
    for raw in results_path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            try:
                pages.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
    if not pages:
        return

    provider = None
    submitted_at = None
    completed_at = None
    meta_path = batch_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            provider = meta.get("provider")
            submitted_at = meta.get("created_at")
        except (json.JSONDecodeError, OSError):
            pass
    complete_path = batch_dir / "job_complete.json"
    if complete_path.exists():
        try:
            info = json.loads(complete_path.read_text(encoding="utf-8"))
            completed_at = info.get("completed_at")
        except (json.JSONDecodeError, OSError):
            pass

    source_names = [name for (_, name, _) in saved_files]
    body = format_consolidated_ocr_text(
        pages,
        source_files=source_names,
        provider=provider,
        submitted_at=submitted_at,
        completed_at=completed_at,
        elapsed_seconds=elapsed,
    )
    try:
        (batch_dir / "ocr_output.txt").write_text(body, encoding="utf-8")
    except OSError:
        logger.warning("Could not write consolidated ocr_output.txt for %s", batch_dir)


async def _run(
    *,
    queue: asyncio.Queue,
    saved_files: list[tuple[Path, str, str | None]],
    provider: str,
    user_prompt: str | None,
    few_shots: list[dict[str, str]],
    settings: Settings,
    model_id: str | None,
    batch_dir: Path,
    gemini_cost_session: object | None = None,
) -> int:
    """Returns total page count."""
    loop = asyncio.get_running_loop()
    results_path = batch_dir / "results.jsonl"
    results_lock = asyncio.Lock()  # guards concurrent appends to results.jsonl
    system_syn = get_system_prompt_for_provider(provider, settings)
    pdf_dpi = effective_ocr_pdf_dpi(settings, provider)
    user_extra = (user_prompt or "").strip() or None

    # Phase 1: count pages and emit start before rasterizing (avoids long UI silence on PDFs).
    # Slot: (global_index, source_name, kind, payload, page_in_source)
    # kind "image" -> payload is raw bytes; kind "pdf" -> (pdf_bytes, 1-based page_number)
    page_slots: list[tuple[int, str, str, object, int | None]] = []

    for path, name, content_type in saved_files:
        raw = await loop.run_in_executor(None, path.read_bytes)

        if sniff_is_pdf(raw, name, content_type):
            try:
                n_pages = await loop.run_in_executor(None, pdf_page_count, raw)
            except Exception as exc:
                raise RuntimeError(f"Failed to read PDF {name!r}: {exc}") from exc
            for page_num in range(1, n_pages + 1):
                page_slots.append((len(page_slots), name, "pdf", (raw, page_num), page_num))

        elif _IMAGE_RE.search(name) or (content_type or "").lower().startswith("image/"):
            mime = _mime_for_upload(name, content_type)
            page_slots.append((len(page_slots), name, "image", (raw, mime), None))

        else:
            raise ValueError(
                f"Unsupported file type for {name!r}. Upload PDF or images (png, jpeg, webp, gif)."
            )

    total = len(page_slots)
    if total == 0:
        raise ValueError("No pages to process.")

    # Local vLLM: one page at a time avoids GPU thrash/OOM on a shared GPU.
    concurrency = settings.ocr_page_concurrency
    if provider == OcrProvider.vllm_dots.value:
        concurrency = 1

    logger.info("OCR job: provider=%s pages=%d concurrency=%d", provider, total, concurrency)
    await queue.put({"event": "start", "total": total})

    # Phase 2: rasterize (per PDF page) and transcribe concurrently.
    semaphore = asyncio.Semaphore(concurrency)
    done_count = 0

    async def process_one(
        idx: int, name: str, kind: str, payload: object, page_in_src: int | None
    ) -> OcrPageResult:
        if kind == "pdf":
            pdf_bytes, page_num = payload  # type: ignore[misc]
            try:
                rendered = await loop.run_in_executor(
                    None, pdf_page_to_image, pdf_bytes, page_num, pdf_dpi
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to rasterize {name!r} page {page_num}: {exc}"
                ) from exc
            img, mime = rendered.image_bytes, rendered.mime_type
        else:
            raw_bytes, mime = payload  # type: ignore[misc]
            img = raw_bytes

        if provider == OcrProvider.vllm_dots.value:
            img, mime = await loop.run_in_executor(
                None,
                partial(
                    preprocess_image_bytes,
                    img,
                    mime,
                    enabled=settings.vllm_preprocess_images,
                ),
            )

        last_exc: Exception = RuntimeError("no attempts made")
        max_attempts = settings.ocr_page_max_retries + 1
        for attempt in range(max_attempts):
            try:
                async with semaphore:
                    text = await transcribe_with_provider_async(
                        provider,
                        image_bytes=img,
                        mime_type=mime,
                        system_prompt=system_syn,
                        user_prompt=user_extra,
                        few_shots=few_shots,
                        settings=settings,
                        model_id=model_id,
                        gemini_cost_session=gemini_cost_session,
                        gemini_cost_page_index=idx,
                        gemini_cost_page_in_source=page_in_src,
                        gemini_cost_source_file=name,
                    )
                return OcrPageResult(
                    index=idx,
                    source_file=name,
                    page_in_source=page_in_src,
                    text=text,
                    mime_type=mime,
                )
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    delay = 2 ** attempt  # 1 s, 2 s, 4 s …
                    logger.warning(
                        "Page %d attempt %d/%d failed (%s) — retrying in %.0fs",
                        idx, attempt + 1, max_attempts, exc, delay,
                    )
                    await asyncio.sleep(delay)
        raise last_exc

    tasks = [
        asyncio.create_task(process_one(idx, name, kind, payload, pg))
        for idx, name, kind, payload, pg in page_slots
    ]

    for fut in asyncio.as_completed(tasks):
        try:
            page = await fut
            done_count += 1
            # Persist page to disk so results survive browser refresh / disconnect.
            line = json.dumps(page.model_dump()) + "\n"
            async with results_lock:
                await loop.run_in_executor(None, _append_result_line, results_path, line)
            page_num = resolve_page_number(page.model_dump())
            await loop.run_in_executor(
                None,
                _write_page_file,
                batch_dir,
                page.source_file,
                page_num,
                page.text,
            )
            await queue.put({
                "event": "page",
                "data": page.model_dump(),
                "done": done_count,
                "total": total,
            })
            logger.info("Page done: index=%d done=%d/%d", page.index, done_count, total)
        except Exception as exc:
            done_count += 1
            logger.warning("Page error (all retries exhausted): %s", exc)
            await queue.put({
                "event": "page_error",
                "detail": str(exc),
                "done": done_count,
                "total": total,
            })

    return total
