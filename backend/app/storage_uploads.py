"""Persist OCR request uploads under a configurable root with per-request folders."""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from pathlib import Path

from fastapi import UploadFile

logger = logging.getLogger(__name__)

def _safe_component(name: str | None) -> str:
    base = (name or "upload.bin").replace("\\", "/").rsplit("/", 1)[-1].strip() or "upload.bin"
    return base.replace("\x00", "").replace("/", "_")


async def persist_ocr_uploads(upload_root: Path, files: list[UploadFile]) -> tuple[str, Path, list[tuple[Path, str, str | None]]]:
    """
    Write each multipart file under ``upload_root /<batch_id>/<filename>``.
    Returns (batch_id, batch_dir, [(path, original_filename, content_type), ...]).
    """
    batch_id = str(uuid.uuid4())
    batch_dir = (upload_root / batch_id).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created upload batch dir: batch_id=%s path=%s", batch_id, batch_dir)
    saved: list[tuple[Path, str, str | None]] = []
    used_names: dict[str, int] = {}

    for uf in files:
        raw = await uf.read()
        orig = _safe_component(uf.filename)
        stem = Path(orig).stem
        suf = Path(orig).suffix or ""
        cnt = used_names.get(orig, 0) + 1
        used_names[orig] = cnt
        on_disk = f"{stem}{suf}" if cnt == 1 else f"{stem}_{cnt}{suf}"
        path = batch_dir / on_disk
        path.write_bytes(raw)
        logger.info(
            "Saved upload file: batch_id=%s name=%r on_disk=%s size=%d bytes content_type=%s",
            batch_id, orig, on_disk, len(raw), uf.content_type,
        )
        saved.append((path, orig, uf.content_type))

    logger.info("Upload batch persisted: batch_id=%s files=%d", batch_id, len(saved))
    return batch_id, batch_dir, saved


def write_batch_metadata(batch_dir: Path, payload: dict) -> None:
    meta_path = batch_dir / "metadata.json"
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prune_old_batches(upload_root: Path, max_age_hours: float) -> None:
    """Remove UUID-named subdirectories older than max_age_hours (mtime)."""
    if max_age_hours <= 0:
        return
    if not upload_root.is_dir():
        return
    ttl = float(max_age_hours) * 3600.0
    now = time.time()
    for entry in upload_root.iterdir():
        if not entry.is_dir():
            continue
        try:
            mtime = entry.stat().st_mtime
            age_hours = (now - mtime) / 3600.0
            if now - mtime <= ttl:
                continue
            shutil.rmtree(entry, ignore_errors=False)
            logger.info("Pruned aged upload batch: name=%s age_hours=%.1f", entry.name, age_hours)
        except OSError:
            logger.exception("Could not prune batch dir: path=%s", entry)


def sniff_is_pdf(raw: bytes, filename: str, content_type: str | None) -> bool:
    fn = filename.lower()
    ct = (content_type or "").lower().split(";")[0].strip()
    if ct == "application/pdf":
        return True
    if fn.endswith(".pdf"):
        return True
    return raw[: min(1024, len(raw))].startswith(b"%PDF")
