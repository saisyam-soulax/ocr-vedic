"""Gemini Batch API support for OCR jobs.

Two modes depending on the endpoint:
- Developer API (GEMINI_USE_VERTEXAI=false, AIzaSy keys):
    Inline batch — all pages submitted as a list of dicts in one call.
    50% cost discount. No GCS required.
- Vertex AI (GEMINI_USE_VERTEXAI=true, AQ. keys):
    GCS batch — pages serialised to JSONL, uploaded to GCS, submitted as
    a batch prediction job, output downloaded from GCS when complete.
    Same 50% discount. Requires GCS_BATCH_BUCKET + GOOGLE_APPLICATION_CREDENTIALS.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import AsyncGenerator, Union

from app.providers.prompts import user_instructions_prefix
from app.utils.size_tags import SIZE_TAG_INSTRUCTION

logger = logging.getLogger(__name__)


@dataclass
class BatchPageResult:
    index: int
    source_file: str
    page_in_source: int | None
    mime_type: str
    text: str
    error: str | None
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    total_tokens: int


@dataclass
class BatchProgress:
    state: str
    elapsed_sec: float
    job_name: str
    page_count: int


# ── shared helpers ────────────────────────────────────────────────────────────

def _request_image_hash(request: dict) -> str | None:
    """MD5 hex of the first inline image in a (possibly echoed) request.

    Vertex AI batch prediction does NOT preserve input ordering in its output —
    output records come back shuffled. Each output record echoes its ``request``,
    so we hash the request's page image to map every output record back to the
    exact input page it belongs to. Matching by list position corrupts the whole
    document (see poll_gcs_batch).
    """
    try:
        parts = request["contents"][0]["parts"]
    except (KeyError, IndexError, TypeError):
        return None
    for part in parts:
        data = part.get("inline_data") or part.get("inlineData")
        if data and data.get("data"):
            return hashlib.md5(data["data"].encode("utf-8")).hexdigest()
    return None

def _build_inline_requests(
    *,
    pages: list[tuple[int, str, int | None, str, bytes]],
    system_prompt: str,
    user_prompt: str | None,
    few_shots: list,
    thinking_budget: int = -1,
) -> list[dict]:
    """Build one request dict per page (shared by both batch modes).

    thinking_budget: reasoning-token cap per call. 0 disables thinking (billed at
    output rate), a positive value caps it, -1 leaves the model's dynamic default.
    """
    prefix = user_instructions_prefix(user_prompt)
    transcribe_prompt = (
        prefix
        + "Transcribe the following image. Preserve Devanāgarī, IAST diacritics, "
        "anusvāra/visarga, all svaras (Udātta, Anudātta, Svarita, kampas, etc.), "
        "and punctuation exactly as printed or implied by the scan. "
        "Output plain text only, no commentary."
        + SIZE_TAG_INSTRUCTION
    )

    few_shot_parts: list[dict] = []
    for shot in few_shots:
        if hasattr(shot, "image_bytes"):
            shot_img: bytes = shot.image_bytes
            shot_mime: str = shot.mime_type
            shot_text: str = shot.expected_text
        else:
            raw_b64 = shot.get("image_base64", "")
            shot_img = base64.b64decode(raw_b64) if raw_b64 else b""
            shot_mime = shot.get("mime_type", "image/png")
            shot_text = shot.get("expected_text", "")
        if not shot_img:
            continue
        few_shot_parts.extend([
            {"text": "Few-shot example — manuscript snippet and its exact transcription (match this style, spacing, and diacritics):"},
            {"inline_data": {"mime_type": shot_mime, "data": base64.b64encode(shot_img).decode("ascii")}},
            {"text": f"Expected transcription:\n{shot_text}"},
        ])

    requests: list[dict] = []
    for _idx, _source_file, _page_in_source, mime_type, image_bytes in pages:
        parts: list[dict] = list(few_shot_parts)
        parts.append({"text": transcribe_prompt})
        parts.append({"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode("ascii")}})
        request: dict = {"contents": [{"role": "user", "parts": parts}]}
        cfg: dict = {}
        if system_prompt:
            cfg["system_instruction"] = {"parts": [{"text": system_prompt}]}
        if thinking_budget != -1:
            cfg["thinking_config"] = {"thinking_budget": thinking_budget}
        if cfg:
            request["config"] = cfg
        requests.append(request)
    return requests


# ── Developer API inline batch ────────────────────────────────────────────────

async def submit_gemini_batch(
    *,
    client,
    model: str,
    pages: list[tuple[int, str, int | None, str, bytes]],
    system_prompt: str,
    user_prompt: str | None,
    few_shots: list,
    display_name: str,
    thinking_budget: int = -1,
) -> str:
    """Submit an inline batch job (Developer API only)."""
    loop = asyncio.get_running_loop()
    inline_requests = _build_inline_requests(
        pages=pages, system_prompt=system_prompt,
        user_prompt=user_prompt, few_shots=few_shots,
        thinking_budget=thinking_budget,
    )
    logger.info("Gemini inline batch: display_name=%s requests=%d model=%s", display_name, len(inline_requests), model)

    def _create():
        return client.batches.create(model=model, src=inline_requests, config={"display_name": display_name})

    job = await loop.run_in_executor(None, _create)
    logger.info("Gemini inline batch submitted: job_name=%s", job.name)
    return job.name


async def poll_gemini_batch(
    *,
    client,
    job_name: str,
    pages_meta: list[tuple[int, str, int | None, str]],
    poll_interval_sec: float = 10.0,
    max_poll_interval_sec: float = 60.0,
    timeout_sec: float = 7200.0,
) -> AsyncGenerator[Union[BatchProgress, list[BatchPageResult]], None]:
    """Poll an inline batch job (Developer API). Yields BatchProgress then list[BatchPageResult]."""
    from app.providers.gemini import _usage_from_response

    loop = asyncio.get_running_loop()
    _TERMINAL_SUCCESS = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
    _TERMINAL_FAILURE = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    start = time.monotonic()
    interval = poll_interval_sec

    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout_sec:
            raise asyncio.TimeoutError(f"Gemini batch job {job_name!r} timed out after {timeout_sec:.0f}s")

        job = await loop.run_in_executor(None, lambda: _poll_get_batch_job(client, job_name))
        state_val = job.state
        state_str = state_val.name if hasattr(state_val, "name") else str(state_val)
        logger.info("Gemini inline batch poll: job=%s state=%s elapsed=%.1fs", job_name, state_str, elapsed)

        if state_str in _TERMINAL_FAILURE:
            raise RuntimeError(f"Gemini batch job {job_name!r} ended with state {state_str}.")

        if state_str in _TERMINAL_SUCCESS:
            results: list[BatchPageResult] = []
            inlined_responses = None
            dest = getattr(job, "dest", None)
            if dest is not None:
                inlined_responses = getattr(dest, "inlined_responses", None)

            if inlined_responses:
                for i, resp in enumerate(inlined_responses):
                    if i >= len(pages_meta):
                        break
                    idx, source_file, page_in_source, mime_type = pages_meta[i]
                    error_str: str | None = None
                    error_obj = getattr(resp, "error", None)
                    if error_obj is not None:
                        error_str = str(error_obj)
                    text = ""
                    response = getattr(resp, "response", None)
                    if error_str is None and response is not None:
                        raw_text = getattr(response, "text", None)
                        if raw_text:
                            text = raw_text.strip()
                        else:
                            candidates = getattr(response, "candidates", None) or []
                            if candidates:
                                content = getattr(candidates[0], "content", None)
                                parts_list = getattr(content, "parts", None) if content else None
                                if parts_list:
                                    chunks = [p.text for p in parts_list if getattr(p, "text", None)]
                                    if chunks:
                                        text = "\n".join(chunks).strip()
                    usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "total_tokens": 0}
                    if error_str is None and response is not None:
                        usage = _usage_from_response(response)
                    results.append(BatchPageResult(
                        index=idx, source_file=source_file, page_in_source=page_in_source,
                        mime_type=mime_type, text=text, error=error_str,
                        input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
                        cached_tokens=usage["cached_tokens"], total_tokens=usage["total_tokens"],
                    ))
                    if error_str:
                        logger.warning("Inline batch page index=%d error: %s", idx, error_str)
                    else:
                        logger.info("Inline batch page index=%d chars=%d tokens_in=%d tokens_out=%d", idx, len(text), usage["input_tokens"], usage["output_tokens"])
            else:
                logger.warning("Inline batch job %s succeeded but inlined_responses is empty", job_name)

            logger.info("Inline batch complete: job=%s pages=%d elapsed=%.1fs", job_name, len(results), elapsed)
            yield results
            return

        yield BatchProgress(state=state_str, elapsed_sec=elapsed, job_name=job_name, page_count=len(pages_meta))
        await asyncio.sleep(interval)
        interval = min(interval * 1.5, max_poll_interval_sec)


# ── Vertex AI GCS batch ───────────────────────────────────────────────────────

async def submit_gcs_batch(
    *,
    client,
    model: str,
    pages: list[tuple[int, str, int | None, str, bytes]],
    system_prompt: str,
    user_prompt: str | None,
    few_shots: list,
    display_name: str,
    gcs_bucket: str,
    job_id: str,
    thinking_budget: int = -1,
) -> tuple[str, str, list[str]]:
    """Upload input JSONL to GCS and submit a Vertex AI batch prediction job.

    Returns:
        (job_name, gcs_output_prefix, input_hashes) — job_name for polling,
        output prefix for downloading results, and per-page image hashes in input
        order so poll_gcs_batch can re-order Vertex's shuffled output correctly.
    """
    from google.cloud import storage as gcs

    loop = asyncio.get_running_loop()
    requests = _build_inline_requests(
        pages=pages, system_prompt=system_prompt,
        user_prompt=user_prompt, few_shots=few_shots,
        thinking_budget=thinking_budget,
    )

    # Vertex AI GCS batch input format: each line is {"request": <GenerateContentRequest>}
    input_lines = []
    for req in requests:
        # Move config.system_instruction into the request body for GCS format
        gcs_req: dict = {"contents": req["contents"]}
        cfg = req.get("config") or {}
        if "system_instruction" in cfg:
            gcs_req["system_instruction"] = cfg["system_instruction"]
        # Vertex request format nests thinking under generationConfig.thinkingConfig
        if "thinking_config" in cfg:
            gcs_req["generationConfig"] = {
                "thinkingConfig": {"thinkingBudget": cfg["thinking_config"]["thinking_budget"]}
            }
        input_lines.append(json.dumps({"request": gcs_req}, ensure_ascii=False))

    input_jsonl = "\n".join(input_lines) + "\n"
    input_blob = f"batch-jobs/{job_id}/input.jsonl"
    output_prefix = f"batch-jobs/{job_id}/output/"
    gcs_input_uri = f"gs://{gcs_bucket}/{input_blob}"
    gcs_output_uri = f"gs://{gcs_bucket}/{output_prefix}"

    payload = input_jsonl.encode("utf-8")
    payload_mb = len(payload) / (1024 * 1024)
    logger.info(
        "GCS batch: uploading %d requests (%.1f MB) to %s",
        len(requests),
        payload_mb,
        gcs_input_uri,
    )

    # A single slow/aborted write to GCS must not fail the whole job. The default
    # google-cloud-storage upload timeout (~60s) trips on large payloads over a
    # congested link, and resumable-upload write timeouts aren't retried by
    # default — so we use a generous per-attempt timeout plus explicit backoff.
    # This runs entirely before the batch is submitted to Gemini, so a retry here
    # never risks duplicate submission or cost.
    _UPLOAD_TIMEOUT_SEC = 600.0
    _UPLOAD_MAX_ATTEMPTS = 4

    def _upload():
        storage_client = gcs.Client()
        bucket = storage_client.bucket(gcs_bucket)
        blob = bucket.blob(input_blob)
        last_exc: Exception | None = None
        for attempt in range(1, _UPLOAD_MAX_ATTEMPTS + 1):
            try:
                blob.upload_from_string(
                    payload,
                    content_type="application/jsonl",
                    timeout=_UPLOAD_TIMEOUT_SEC,
                )
                if attempt > 1:
                    logger.info("GCS batch: upload succeeded on attempt %d", attempt)
                return
            except Exception as exc:  # noqa: BLE001 — retry any transient network error
                last_exc = exc
                if attempt == _UPLOAD_MAX_ATTEMPTS:
                    break
                backoff = 2.0 * (2 ** (attempt - 1))  # 2s, 4s, 8s
                logger.warning(
                    "GCS batch: upload attempt %d/%d failed (%s) — retrying in %.0fs",
                    attempt,
                    _UPLOAD_MAX_ATTEMPTS,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    await loop.run_in_executor(None, _upload)
    logger.info("GCS batch: input uploaded, submitting job")

    def _submit():
        return _create_gcs_batch_job(client, model, gcs_input_uri, gcs_output_uri, display_name)

    job = await loop.run_in_executor(None, _submit)
    logger.info("GCS batch submitted: job_name=%s output=%s", job.name, gcs_output_uri)
    input_hashes = [_request_image_hash(r) for r in requests]
    return job.name, gcs_output_uri, input_hashes


def _is_queue_full_error(err: object) -> bool:
    """True if a Vertex batch error is the transient 'maximum number of queued
    jobs' limit (RESOURCE_EXHAUSTED / code 9 / HTTP 429). This clears on its own
    once other batch jobs in the project drain, so it's safe to wait and resubmit."""
    if err is None:
        return False
    code = getattr(err, "code", None)
    message = (getattr(err, "message", None) or str(err)).lower()
    return code in (9, 429) or "queued jobs" in message or "resource_exhausted" in message


def _is_transient_network_error(exc: BaseException) -> bool:
    """True for connectivity blips (DNS hiccup, dropped connection, unreachable
    network) as opposed to a real API-level error from Vertex. These must not
    be treated as a batch job failure — the job itself may be running fine on
    Vertex's side while our container briefly can't reach it."""
    import httpx

    if isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError, OSError)):
        return True
    # google-genai wraps httpx errors; walk the cause chain too.
    cause = exc.__cause__
    return cause is not None and cause is not exc and _is_transient_network_error(cause)


def _poll_get_batch_job(client, job_name: str, max_attempts: int = 6):
    """client.batches.get() with retry for transient network errors only.

    A dropped connection while polling Vertex must not fail the whole OCR
    job — the batch prediction job keeps running on Vertex regardless of
    whether *we* can currently reach the API, and giving up on the first
    blip abandons a job that may complete successfully with nobody to
    collect its output.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.batches.get(name=job_name)
        except Exception as exc:  # noqa: BLE001
            if not _is_transient_network_error(exc) or attempt == max_attempts:
                raise
            last_exc = exc
            backoff = min(5.0 * (2 ** (attempt - 1)), 60.0)  # 5,10,20,40,60s
            logger.warning(
                "GCS batch poll: transient network error on attempt %d/%d (%s) — "
                "retrying in %.0fs. Batch job on Vertex is unaffected.",
                attempt, max_attempts, exc, backoff,
            )
            time.sleep(backoff)
    raise last_exc  # pragma: no cover — loop always returns or raises above


def _create_gcs_batch_job(client, model: str, gcs_input_uri: str, gcs_output_uri: str, display_name: str):
    """Create a Vertex AI GCS batch prediction job from an already-uploaded input."""
    from google.genai import types as genai_types

    return client.batches.create(
        model=model,
        src=genai_types.BatchJobSource(format="jsonl", gcs_uri=[gcs_input_uri]),
        config=genai_types.CreateBatchJobConfig(
            display_name=display_name,
            dest=genai_types.BatchJobDestination(format="jsonl", gcs_uri=gcs_output_uri),
        ),
    )


async def poll_gcs_batch(
    *,
    client,
    job_name: str,
    pages_meta: list[tuple[int, str, int | None, str]],
    gcs_output_uri: str,
    gcs_bucket: str,
    input_hashes: list[str] | None = None,
    poll_interval_sec: float = 30.0,
    max_poll_interval_sec: float = 120.0,
    timeout_sec: float = 7200.0,
    model: str | None = None,
    display_name: str | None = None,
    max_queue_retries: int = 5,
) -> AsyncGenerator[Union[BatchProgress, list[BatchPageResult]], None]:
    """Poll a Vertex AI GCS batch job. Yields BatchProgress then list[BatchPageResult]."""
    from google.cloud import storage as gcs

    loop = asyncio.get_running_loop()
    _TERMINAL_SUCCESS = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
    _TERMINAL_FAILURE = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    start = time.monotonic()
    interval = poll_interval_sec

    queue_retries = 0
    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout_sec:
            raise asyncio.TimeoutError(f"GCS batch job {job_name!r} timed out after {timeout_sec:.0f}s")

        job = await loop.run_in_executor(None, lambda: _poll_get_batch_job(client, job_name))
        state_val = job.state
        state_str = state_val.name if hasattr(state_val, "name") else str(state_val)
        logger.info("GCS batch poll: job=%s state=%s elapsed=%.1fs", job_name, state_str, elapsed)

        if state_str in _TERMINAL_FAILURE:
            # Surface Vertex's actual failure reason (e.g. quota / queued-jobs
            # limit, invalid input) instead of a generic state string — otherwise
            # the real cause is invisible in logs and the UI.
            err = getattr(job, "error", None)
            detail = ""
            if err is not None:
                code = getattr(err, "code", None)
                message = getattr(err, "message", None) or str(err)
                detail = f" reason=[code={code}] {message}"
            # Transient 'maximum number of queued jobs' rejection: the input is
            # already in GCS, so wait for the project's batch queue to drain and
            # resubmit (no re-upload, no cost — a failed-in-queue job never ran).
            if (
                _is_queue_full_error(err)
                and model is not None
                and display_name is not None
                and queue_retries < max_queue_retries
            ):
                queue_retries += 1
                backoff = min(30.0 * (2 ** (queue_retries - 1)), 240.0)  # 30,60,120,240,240s
                logger.warning(
                    "GCS batch job %s hit the queued-jobs limit (retry %d/%d) — "
                    "waiting %.0fs then resubmitting.%s",
                    job_name, queue_retries, max_queue_retries, backoff, detail,
                )
                await asyncio.sleep(backoff)
                gcs_input_uri = gcs_output_uri.rstrip("/").removesuffix("/output") + "/input.jsonl"
                try:
                    new_job = await loop.run_in_executor(
                        None,
                        lambda: _create_gcs_batch_job(
                            client, model, gcs_input_uri, gcs_output_uri, display_name
                        ),
                    )
                    job_name = new_job.name
                    interval = poll_interval_sec  # reset cadence for the fresh job
                    logger.info("GCS batch resubmitted after queue-full: job_name=%s", job_name)
                except Exception as exc:  # noqa: BLE001
                    if _is_queue_full_error(exc):
                        # Still full — keep the old (failed) job_name; next loop
                        # iteration re-observes the failure and backs off again.
                        logger.warning("GCS batch resubmit still queue-full: %s", exc)
                    else:
                        raise
                continue

            logger.error(
                "GCS batch job %s ended with state %s.%s", job_name, state_str, detail
            )
            raise RuntimeError(
                f"GCS batch job {job_name!r} ended with state {state_str}.{detail}"
            )

        if state_str in _TERMINAL_SUCCESS:
            # Download all output JSONL files from GCS output prefix
            output_prefix = gcs_output_uri.removeprefix(f"gs://{gcs_bucket}/")

            def _download_outputs() -> list[dict]:
                storage_client = gcs.Client()
                bucket_obj = storage_client.bucket(gcs_bucket)
                blobs = list(storage_client.list_blobs(gcs_bucket, prefix=output_prefix))
                records: list[dict] = []
                for blob in blobs:
                    if not blob.name.endswith(".jsonl"):
                        continue
                    content = blob.download_as_text(encoding="utf-8")
                    for line in content.splitlines():
                        line = line.strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                return records

            output_records = await loop.run_in_executor(None, _download_outputs)
            logger.info("GCS batch: downloaded %d output records", len(output_records))

            # Vertex batch output is NOT in input order — map each output record
            # back to its input page by the echoed image hash. Fall back to
            # positional only if we have no hashes (older callers).
            record_by_pos: dict[int, dict] = {}
            if input_hashes:
                pos_by_hash: dict[str, list[int]] = {}
                for pos, h in enumerate(input_hashes):
                    if h is not None:
                        pos_by_hash.setdefault(h, []).append(pos)
                unmatched = 0
                for record in output_records:
                    h = _request_image_hash(record.get("request") or {})
                    positions = pos_by_hash.get(h) if h else None
                    if positions:
                        record_by_pos[positions.pop(0)] = record
                    else:
                        unmatched += 1
                if unmatched:
                    logger.error(
                        "GCS batch: %d/%d output records could not be matched by "
                        "image hash — those pages will be marked as errors",
                        unmatched, len(output_records),
                    )
                else:
                    logger.info("GCS batch: all %d records matched to input pages by hash", len(output_records))
            else:
                logger.warning("GCS batch: no input_hashes provided — falling back to positional matching (unreliable)")
                for i, record in enumerate(output_records):
                    if i < len(pages_meta):
                        record_by_pos[i] = record

            # Emit results in input page order
            results: list[BatchPageResult] = []
            for pos in range(len(pages_meta)):
                idx, source_file, page_in_source, mime_type = pages_meta[pos]
                record = record_by_pos.get(pos)
                if record is None:
                    results.append(BatchPageResult(
                        index=idx, source_file=source_file, page_in_source=page_in_source,
                        mime_type=mime_type, text="", error="No batch output matched this page",
                        input_tokens=0, output_tokens=0, cached_tokens=0, total_tokens=0,
                    ))
                    logger.warning("GCS batch page index=%d had no matching output record", idx)
                    continue

                error_str: str | None = None
                if "error" in record:
                    error_str = str(record["error"])

                text = ""
                input_tokens = output_tokens = cached_tokens = total_tokens = 0
                response = record.get("response") or {}
                if error_str is None and response:
                    candidates = response.get("candidates") or []
                    if candidates:
                        content = candidates[0].get("content") or {}
                        parts = content.get("parts") or []
                        chunks = [p.get("text", "") for p in parts if p.get("text")]
                        text = "\n".join(chunks).strip()
                    usage_meta = response.get("usageMetadata") or {}
                    input_tokens = int(usage_meta.get("promptTokenCount") or 0)
                    # Include thinking tokens (billed at output rate) — see
                    # _usage_from_response in gemini.py for the full rationale.
                    visible_out = int(usage_meta.get("candidatesTokenCount") or 0)
                    thoughts_out = int(usage_meta.get("thoughtsTokenCount") or 0)
                    output_tokens = visible_out + thoughts_out
                    cached_tokens = int(usage_meta.get("cachedContentTokenCount") or 0)
                    total_tokens = int(usage_meta.get("totalTokenCount") or input_tokens + output_tokens)

                results.append(BatchPageResult(
                    index=idx, source_file=source_file, page_in_source=page_in_source,
                    mime_type=mime_type, text=text, error=error_str,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    cached_tokens=cached_tokens, total_tokens=total_tokens,
                ))
                if error_str:
                    logger.warning("GCS batch page index=%d error: %s", idx, error_str)
                else:
                    logger.info("GCS batch page index=%d chars=%d tokens_in=%d tokens_out=%d", idx, len(text), input_tokens, output_tokens)

            logger.info("GCS batch complete: job=%s pages=%d elapsed=%.1fs", job_name, len(results), elapsed)
            yield results
            return

        yield BatchProgress(state=state_str, elapsed_sec=elapsed, job_name=job_name, page_count=len(pages_meta))
        await asyncio.sleep(interval)
        interval = min(interval * 1.5, max_poll_interval_sec)
