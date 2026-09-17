"""One-off recovery for a Gemini GCS batch job that succeeded on Vertex but
was marked 'failed' locally because polling died on a transient network
error (see docs/INCIDENT_LOG.md INC-005).

Reconstructs pages_meta and input_hashes directly from the still-present
input.jsonl (byte-exact — no re-rasterization needed since it's a single
source file submitted in page order), downloads + hash-matches the Vertex
output exactly as poll_gcs_batch() would, and writes the same on-disk
artifacts _run_gemini_batch() normally writes: per-page .txt files,
results.jsonl, ocr_output.txt, and job_complete.json.

Usage: python -m scripts.recover_orphaned_batch <job_id> <vertex_job_name>
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/app")

from app.config import get_settings
from app.providers.gemini_batch import BatchPageResult, _request_image_hash
from app.schemas import OcrPageResult
from app.utils.output_format import format_consolidated_ocr_text, resolve_page_number
from app.utils.size_tags import clean_text as strip_size_tags
from app.utils.size_tags import parse_segments


def _write_page_file(batch_dir: Path, source_name: str, page_num: int, text: str) -> None:
    stem = Path(source_name).stem
    (batch_dir / f"{stem}_Page_{page_num}.txt").write_text(text, encoding="utf-8")


async def main(job_id: str, vertex_job_name: str) -> None:
    from google import genai
    from google.cloud import storage as gcs

    settings = get_settings()
    batch_dir = settings.upload_root_path() / job_id
    if not batch_dir.is_dir():
        raise SystemExit(f"job dir not found: {batch_dir}")

    meta = json.loads((batch_dir / "metadata.json").read_text(encoding="utf-8"))
    source_file = meta["files"][0]
    gcs_bucket = settings.gcs_batch_bucket

    client = genai.Client(vertexai=True, project=settings.google_cloud_project, location="global")
    job = client.batches.get(name=vertex_job_name)
    state_str = job.state.name if hasattr(job.state, "name") else str(job.state)
    print(f"Vertex job state: {state_str}")
    if state_str != "JOB_STATE_SUCCEEDED":
        raise SystemExit(f"refusing to recover — job is not SUCCEEDED (state={state_str})")

    dest = job.dest
    gcs_output_uri = dest.gcs_uri
    print(f"output prefix: {gcs_output_uri}")

    storage_client = gcs.Client()

    # Reconstruct pages_meta + input_hashes byte-exactly from the submitted input.jsonl.
    input_blob = storage_client.bucket(gcs_bucket).blob(f"batch-jobs/{job_id}/input.jsonl")
    input_lines = input_blob.download_as_text(encoding="utf-8").splitlines()
    pages_meta: list[tuple[int, str, int | None, str]] = []
    input_hashes: list[str | None] = []
    for i, line in enumerate(input_lines):
        if not line.strip():
            continue
        req = json.loads(line)["request"]
        input_hashes.append(_request_image_hash({"contents": req["contents"]}))
        pages_meta.append((i, source_file, i + 1, "image/jpeg"))
    print(f"reconstructed pages_meta for {len(pages_meta)} pages from input.jsonl")

    # Download + hash-match output records (identical logic to poll_gcs_batch).
    output_prefix = gcs_output_uri.removeprefix(f"gs://{gcs_bucket}/")
    blobs = list(storage_client.list_blobs(gcs_bucket, prefix=output_prefix))
    output_records: list[dict] = []
    for blob in blobs:
        if not blob.name.endswith(".jsonl"):
            continue
        for out_line in blob.download_as_text(encoding="utf-8").splitlines():
            out_line = out_line.strip()
            if out_line:
                output_records.append(json.loads(out_line))
    print(f"downloaded {len(output_records)} output records from {len(blobs)} blob(s)")

    pos_by_hash: dict[str, list[int]] = {}
    for pos, h in enumerate(input_hashes):
        if h is not None:
            pos_by_hash.setdefault(h, []).append(pos)
    record_by_pos: dict[int, dict] = {}
    unmatched = 0
    for record in output_records:
        h = _request_image_hash(record.get("request") or {})
        positions = pos_by_hash.get(h) if h else None
        if positions:
            record_by_pos[positions.pop(0)] = record
        else:
            unmatched += 1
    print(f"matched {len(record_by_pos)}/{len(pages_meta)} pages by image hash (unmatched output records: {unmatched})")
    if unmatched:
        raise SystemExit("aborting — unmatched output records found, needs manual inspection")

    results: list[BatchPageResult] = []
    for pos in range(len(pages_meta)):
        idx, src, page_in_source, mime_type = pages_meta[pos]
        record = record_by_pos.get(pos)
        if record is None:
            results.append(BatchPageResult(
                index=idx, source_file=src, page_in_source=page_in_source,
                mime_type=mime_type, text="", error="No batch output matched this page",
                input_tokens=0, output_tokens=0, cached_tokens=0, total_tokens=0,
            ))
            continue
        error_str = str(record["error"]) if "error" in record else None
        text = ""
        input_tokens = output_tokens = cached_tokens = total_tokens = 0
        response = record.get("response") or {}
        if error_str is None and response:
            candidates = response.get("candidates") or []
            if candidates:
                content = candidates[0].get("content") or {}
                parts = content.get("parts") or []
                text = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()
            usage_meta = response.get("usageMetadata") or {}
            input_tokens = int(usage_meta.get("promptTokenCount") or 0)
            visible_out = int(usage_meta.get("candidatesTokenCount") or 0)
            thoughts_out = int(usage_meta.get("thoughtsTokenCount") or 0)
            output_tokens = visible_out + thoughts_out
            cached_tokens = int(usage_meta.get("cachedContentTokenCount") or 0)
            total_tokens = int(usage_meta.get("totalTokenCount") or input_tokens + output_tokens)
        results.append(BatchPageResult(
            index=idx, source_file=src, page_in_source=page_in_source,
            mime_type=mime_type, text=text, error=error_str,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cached_tokens=cached_tokens, total_tokens=total_tokens,
        ))

    # Write the same on-disk artifacts _run_gemini_batch()'s Phase D writes.
    results_path = batch_dir / "results.jsonl"
    if results_path.exists() and results_path.stat().st_size > 0:
        raise SystemExit(f"refusing to overwrite existing non-empty {results_path}")

    failed_pages = []
    with results_path.open("w", encoding="utf-8") as fh:
        for result in sorted(results, key=lambda r: r.index):
            if result.error and not result.text:
                failed_pages.append({"index": result.index, "source_file": result.source_file, "error": result.error})
                continue
            raw_text = result.text or ""
            segments = parse_segments(raw_text)
            clean = strip_size_tags(raw_text)
            page_dict = OcrPageResult(
                index=result.index, source_file=result.source_file,
                page_in_source=result.page_in_source, text=clean,
                mime_type=result.mime_type, segments=segments,
            ).model_dump()
            fh.write(json.dumps(page_dict, ensure_ascii=False) + "\n")
            page_num = resolve_page_number(page_dict)
            _write_page_file(batch_dir, result.source_file, page_num, clean)

    pages_ok = len(pages_meta) - len(failed_pages)
    print(f"wrote results.jsonl + {pages_ok} per-page .txt files ({len(failed_pages)} failed pages)")

    (batch_dir / "failed_pages.json").write_text(
        json.dumps(failed_pages, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    elapsed = (job.end_time - job.start_time).total_seconds()
    status = "complete" if not failed_pages else "partial"
    (batch_dir / "job_complete.json").write_text(
        json.dumps({
            "total": len(pages_meta), "elapsed_seconds": elapsed,
            "completed_at": datetime.now(UTC).isoformat(),
            "status": status, "failed_count": len(failed_pages),
            "failed_pages": failed_pages,
        }),
        encoding="utf-8",
    )

    pages_for_consolidated = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    body = format_consolidated_ocr_text(
        pages_for_consolidated, source_files=[source_file],
        provider=meta.get("provider"), submitted_at=meta.get("created_at"),
        completed_at=datetime.now(UTC).isoformat(), elapsed_seconds=elapsed,
    )
    (batch_dir / "ocr_output.txt").write_text(body, encoding="utf-8")

    print(f"status={status} pages_ok={pages_ok} elapsed={elapsed:.1f}s — recovery complete for {job_id}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
