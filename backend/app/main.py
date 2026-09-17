from __future__ import annotations

import asyncio
import json
import logging
import os
import re as _re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import starlette.requests
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

if TYPE_CHECKING:
    from starlette.datastructures import FormData

from app import vllm_runtime
from app.config import Settings, get_settings
from app.cost.gemini_ledger import (
    aggregate_gemini_costs,
    read_global_ledger,
    read_job_cost_summary,
    read_vllm_job_cost_summary,
)
from app.job_registry import (
    active_job_count,
    drop_job,
    get_job,
    is_cancelled,
    is_job_active,
    prune_stale_jobs,
    register_job,
    request_cancel,
)
from app.logging_config import append_job_log, configure_logging, log_ctx, set_request_id
from app.middleware.request_logging import RequestLoggingMiddleware
from app.schemas import (
    ErrorResponse,
    GeminiCostsOverviewResponse,
    GeminiJobCostEntry,
    GeminiJobCostSummary,
    GeminiModelCostBreakdown,
    GeminiProviderCostBreakdown,
    HealthResponse,
    OcrDefaultsResponse,
    OcrJobResponse,
    OcrJobStatusResponse,
    OcrPreviewResponse,
    OcrProvider,
    OcrResumeResponse,
    OcrSavedResult,
    ProvidersResponse,
    ProviderInfo,
    RecentJobSummary,
    VllmJobCostSummary,
    VllmState,
    VllmStatusResponse,
)
from app.services.job_helpers import (
    list_source_files,
    load_few_shots,
    persist_few_shots,
    preview_upload_pages,
    read_results_jsonl,
)
from app.services.ocr_service import (
    build_few_shots_for_provider,
    get_default_system_prompt,
    parse_few_shots_json,
    run_ocr_job,
)
from app.storage_uploads import persist_ocr_uploads, prune_old_batches, write_batch_metadata
from app.utils.model_id import resolve_model_id_for_provider
from app.utils.output_format import format_consolidated_ocr_text, page_sort_key

_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
configure_logging(level=_log_level)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Raise Starlette's per-part size limit so large PDFs aren't silently rejected.
# ---------------------------------------------------------------------------
_MAX_PART_SIZE = 500 * 1024 * 1024  # 500 MiB
_original_get_form = starlette.requests.Request._get_form


async def _patched_get_form(
    self: starlette.requests.Request,
    *,
    max_files: int | float = 1000,
    max_fields: int | float = 1000,
    max_part_size: int = _MAX_PART_SIZE,
) -> "FormData":
    return await _original_get_form(
        self,
        max_files=max_files,
        max_fields=max_fields,
        max_part_size=max_part_size,
    )


starlette.requests.Request._get_form = _patched_get_form  # type: ignore[method-assign]

# ---------------------------------------------------------------------------
# In-memory job registry lives in app.job_registry
# ---------------------------------------------------------------------------
_UUID_RE = _re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _content_disposition(filename: str) -> str:
    """Build a Content-Disposition header value that is safe for non-ASCII
    (e.g. Devanāgarī) filenames. HTTP header values must be latin-1 encodable,
    so we emit an ASCII-only ``filename=`` fallback plus an RFC 5987
    ``filename*=UTF-8''`` parameter carrying the real name percent-encoded."""
    from urllib.parse import quote

    ascii_fallback = filename.encode("ascii", "replace").decode("ascii").replace('"', "")
    utf8_encoded = quote(filename, safe="")
    return f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{utf8_encoded}"


def _provider_configured(settings: Settings, p: OcrProvider) -> tuple[bool, str | None]:
    if p == OcrProvider.gemini:
        api_key, _ = settings.effective_google_api_key()
        if settings.gemini_use_vertexai:
            ok = bool(api_key) or bool(settings.google_cloud_project)
            return ok, None if ok else (
                "Vertex AI mode requires a Gemini API key "
                "(GOOGLE_API_KEY-Sampath / GOOGLE_API_KEY) or GOOGLE_CLOUD_PROJECT + ADC"
            )
        ok = bool(api_key)
        return ok, None if ok else (
            "Set GOOGLE_API_KEY-Sampath (or GOOGLE_API_KEY)"
        )
    if p == OcrProvider.bedrock_claude:
        ok = bool(settings.aws_region and settings.bedrock_claude_model_id)
        return ok, None if ok else "Set AWS_REGION and BEDROCK_CLAUDE_MODEL_ID"
    if p == OcrProvider.bedrock_ocr:
        ok = bool(settings.aws_region and settings.bedrock_ocr_model_id)
        return ok, None if ok else "Set AWS_REGION and BEDROCK_OCR_MODEL_ID"
    if p in (OcrProvider.vllm_dots, OcrProvider.vllm_gemma):
        if not settings.vllm_enabled:
            return False, "Set VLLM_ENABLED=true"
        base_url = (
            settings.vllm_gemma_base_url
            if p == OcrProvider.vllm_gemma
            else settings.vllm_base_url
        )
        if not base_url:
            return False, (
                "Set VLLM_GEMMA_BASE_URL"
                if p == OcrProvider.vllm_gemma
                else "Set VLLM_BASE_URL"
            )
        return True, None
    return False, f"Unknown provider: {p}"


def _normalize_vllm_provider(provider: str) -> str:
    p = provider.strip()
    if p not in (OcrProvider.vllm_dots.value, OcrProvider.vllm_gemma.value):
        raise ValueError(
            f"Invalid vLLM provider {p!r}. Use: vllm_dots, vllm_gemma."
        )
    return p


def _vllm_status_payload(
    s: Settings, prov: str, *, state: VllmState, reachable: bool, message: str | None = None
) -> VllmStatusResponse:
    base_url, container, model = s.vllm_runtime_for_provider(prov)
    return VllmStatusResponse(
        state=state,
        reachable=reachable,
        message=message or vllm_runtime.get_error(),
        provider=prov,
        model=model,
        container_name=container,
        elapsed_load_seconds=vllm_runtime.get_load_elapsed_seconds(),
    )


async def _ensure_vllm_ready_for_submit(s: Settings, prov: OcrProvider) -> None:
    if prov not in (OcrProvider.vllm_dots, OcrProvider.vllm_gemma):
        return
    if not s.vllm_enabled:
        return
    if s.ocr_auto_load_vllm and not await vllm_runtime.vllm_ready_for_submit(s, prov.value):
        await vllm_runtime.load(s, prov.value)
        return
    if s.ocr_block_submit_if_vllm_stopped and not await vllm_runtime.vllm_ready_for_submit(
        s, prov.value
    ):
        raise HTTPException(
            409,
            detail=(
                "Local model is not loaded. Use Load model in the UI, "
                "or set OCR_BLOCK_SUBMIT_IF_VLLM_STOPPED=false."
            ),
        )


def _start_ocr_task(
    *,
    s: Settings,
    batch_id: str,
    batch_dir: Path,
    saved: list[tuple[Path, str, str | None]],
    provider: str,
    effective_user_prompt: str | None,
    resolved_shots: list[dict[str, str]],
    model_id_clean: str | None,
    preloaded_pages: list[dict] | None,
    process_mode: str,
    request_id: str | None,
) -> tuple[str, asyncio.Queue]:
    job_id = batch_id
    if is_job_active(job_id):
        # A prior submit/resume for this exact job is still running. Starting
        # a second concurrent task here would let both write to the same
        # on-disk job files, and whichever finishes LAST would silently
        # overwrite the other's (possibly correct) result — see INC-006 in
        # docs/INCIDENT_LOG.md. Refuse instead of racing.
        raise HTTPException(
            409,
            detail=(
                "This job is already running. Wait for it to finish (or cancel it) "
                "before resubmitting — submitting again while it's in progress "
                "can corrupt the job's results."
            ),
        )
    log_ctx(job_id=batch_id, provider=provider)
    append_job_log(batch_dir, "job_submitted", provider=provider, request_id=request_id)
    q: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(
        run_ocr_job(
            queue=q,
            saved_files=saved,
            provider=provider,
            user_prompt=effective_user_prompt,
            few_shots=resolved_shots,
            settings=s,
            model_id=model_id_clean,
            batch_dir=batch_dir,
            retain=s.upload_retain,
            preloaded_pages=preloaded_pages or None,
            process_mode=process_mode,
        )
    )
    register_job(job_id, q, task)

    def _drop_job_when_done(t: asyncio.Task) -> None:
        # Only clear the registry entry if it's still THIS task — a stale task
        # from a rejected/earlier run must never evict a newer active entry.
        current = get_job(job_id)
        if current is None or current[1] is t:
            drop_job(job_id)
        if t.cancelled():
            logger.info("OCR job task cancelled: %s", job_id)
        elif t.exception():
            logger.error("OCR job task failed: %s", job_id, exc_info=t.exception())
        else:
            logger.info("OCR job task finished: %s", job_id)

    task.add_done_callback(_drop_job_when_done)
    return job_id, q


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "App startup: log_level=%s concurrency=%d dpi=%d",
        _log_level, settings.ocr_page_concurrency, settings.ocr_pdf_dpi,
    )
    yield
    logger.info("App shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Vedic OCR API",
        version="0.2.0",
        lifespan=lifespan,
        description="Multimodal OCR for Devanāgarī + IAST with streaming page output.",
    )
    settings = get_settings()
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(RequestLoggingMiddleware)
    # Never combine allow_credentials=True with allow_origins=["*"] (invalid per CORS).
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health(verbose: bool = False, s: Settings = Depends(get_settings)) -> HealthResponse:
        details: dict[str, Any] | None = None
        if verbose:
            upload_root = s.upload_root_path()
            details = {
                "upload_dir_writable": os.access(upload_root, os.W_OK),
                "active_jobs_in_memory": active_job_count(),
                "vllm_enabled": s.vllm_enabled,
            }
            if s.vllm_enabled:
                details["vllm_dots_reachable"] = await vllm_runtime.check_health(
                    s.vllm_base_url
                )
                details["vllm_gemma_reachable"] = await vllm_runtime.check_health(
                    s.vllm_gemma_base_url
                )
        return HealthResponse(status="ok", details=details)

    # ------------------------------------------------------------------
    # Providers
    # ------------------------------------------------------------------

    @app.get("/api/providers", response_model=ProvidersResponse)
    def providers(s: Settings = Depends(get_settings)) -> ProvidersResponse:
        labels = {
            OcrProvider.gemini: "Google Gemini",
            OcrProvider.bedrock_claude: "AWS Bedrock — Claude",
            OcrProvider.bedrock_ocr: "AWS Bedrock — Open multimodal",
            OcrProvider.vllm_dots: "Local — dots.ocr (vLLM)",
            OcrProvider.vllm_gemma: "Local — Gemma 4 (vLLM)",
        }
        rows: list[ProviderInfo] = []
        for p in OcrProvider:
            configured, detail = _provider_configured(s, p)
            if p == OcrProvider.gemini:
                default_mid, mids = s.gemini_models_for_providers()
            elif p == OcrProvider.bedrock_claude:
                default_mid, mids = s.bedrock_claude_models_for_providers()
            elif p == OcrProvider.bedrock_ocr:
                default_mid, mids = s.bedrock_open_models_for_providers()
            elif p == OcrProvider.vllm_gemma:
                default_mid, mids = s.vllm_gemma_models_for_providers()
            else:
                default_mid, mids = s.vllm_models_for_providers()
            rows.append(
                ProviderInfo(
                    id=p.value,
                    label=labels[p],
                    configured=configured,
                    detail=detail if not configured else None,
                    default_model_id=default_mid,
                    model_options=mids,
                )
            )
        return ProvidersResponse(providers=rows)

    @app.get("/api/ocr/defaults", response_model=OcrDefaultsResponse)
    def ocr_defaults() -> OcrDefaultsResponse:
        # Default ĀrṣaDṛṣṭi protocol is applied server-side; not exposed in the UI.
        return OcrDefaultsResponse(system_prompt="")

    # ------------------------------------------------------------------
    # vLLM lifecycle
    # ------------------------------------------------------------------

    @app.get("/api/vllm/status", response_model=VllmStatusResponse)
    async def vllm_status(
        provider: str = OcrProvider.vllm_dots.value,
        s: Settings = Depends(get_settings),
    ) -> VllmStatusResponse:
        if not s.vllm_enabled:
            return VllmStatusResponse(
                state=VllmState.stopped, reachable=False, message="VLLM_ENABLED=false"
            )

        try:
            prov = _normalize_vllm_provider(provider)
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

        base_url, _, _ = s.vllm_runtime_for_provider(prov)
        internal = vllm_runtime.get_state()
        active = vllm_runtime.get_active_provider()

        # Active lifecycle states always win — avoids the polling resetting the UI
        # back to "stopped" while a load/unload is in progress.
        if internal in (VllmState.starting, VllmState.ready, VllmState.stopping):
            if active and active != prov and internal == VllmState.ready:
                return VllmStatusResponse(state=VllmState.stopped, reachable=False)
            if active and active != prov and internal == VllmState.starting:
                return VllmStatusResponse(state=VllmState.stopped, reachable=False)
            if active == prov or internal in (VllmState.starting, VllmState.stopping):
                return _vllm_status_payload(
                    s,
                    prov,
                    state=internal,
                    reachable=internal == VllmState.ready and active == prov,
                )
        if internal == VllmState.error and active == prov:
            return _vllm_status_payload(
                s, prov, state=VllmState.error, reachable=False,
            )

        reachable = await vllm_runtime.check_health(base_url)
        if reachable:
            await vllm_runtime.load(s, prov)
            return _vllm_status_payload(s, prov, state=VllmState.ready, reachable=True)
        return _vllm_status_payload(s, prov, state=VllmState.stopped, reachable=False)

    @app.post("/api/vllm/load", response_model=VllmStatusResponse)
    async def vllm_load(
        provider: str = OcrProvider.vllm_dots.value,
        s: Settings = Depends(get_settings),
    ) -> VllmStatusResponse:
        if not s.vllm_enabled:
            raise HTTPException(400, "VLLM_ENABLED=false — enable vLLM in .env first")
        try:
            prov = _normalize_vllm_provider(provider)
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        await vllm_runtime.load(s, prov)
        state = vllm_runtime.get_state()
        active = vllm_runtime.get_active_provider()
        return _vllm_status_payload(
            s,
            prov,
            state=state,
            reachable=state == VllmState.ready and active == prov,
        )

    @app.post("/api/vllm/unload", response_model=VllmStatusResponse)
    async def vllm_unload(
        provider: str = OcrProvider.vllm_dots.value,
        s: Settings = Depends(get_settings),
    ) -> VllmStatusResponse:
        if not s.vllm_enabled:
            raise HTTPException(400, "VLLM_ENABLED=false")
        try:
            prov = _normalize_vllm_provider(provider)
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        await vllm_runtime.unload(s, prov)
        return _vllm_status_payload(s, prov, state=VllmState.stopped, reachable=False)

    # ------------------------------------------------------------------
    # OCR — preview page counts
    # ------------------------------------------------------------------

    @app.post("/api/ocr/preview", response_model=OcrPreviewResponse)
    async def ocr_preview(
        files: Annotated[list[UploadFile], File(description="PDF and/or image files")],
    ) -> OcrPreviewResponse:
        total, breakdown = await preview_upload_pages(files)
        return OcrPreviewResponse(total_pages=total, file_breakdown=breakdown)

    # ------------------------------------------------------------------
    # OCR — submit job (returns immediately)
    # ------------------------------------------------------------------

    @app.post("/api/ocr", response_model=OcrJobResponse)
    async def submit_ocr(
        request: starlette.requests.Request,
        files: Annotated[list[UploadFile], File(description="PDF and/or image files")],
        provider: str = Form(...),
        user_prompt: str | None = Form(None),
        system_prompt: str | None = Form(
            None,
            description="Deprecated alias for user_prompt",
        ),
        few_shots: str | None = Form(None),
        few_shot_files: Annotated[
            list[UploadFile] | None,
            File(description="Images for few-shots; same order as few_shots"),
        ] = None,
        model_id: str | None = Form(None),
        resume_job_id: str | None = Form(
            None,
            description="Job ID of a previous partial run to resume from.",
        ),
        process_mode: str = Form("auto"),
        s: Settings = Depends(get_settings),
    ) -> OcrJobResponse:
        prune_stale_jobs(logger)

        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        set_request_id(request_id)

        logger.info(
            "OCR submit: provider=%s files=%d model=%s process_mode=%s request_id=%s",
            provider, len(files), model_id or "(default)", process_mode, request_id,
        )

        # Validate provider
        try:
            prov = OcrProvider(provider)
        except ValueError:
            raise HTTPException(
                422,
                detail=(
                    f"Invalid provider '{provider}'. "
                    "Use: gemini, bedrock_claude, bedrock_ocr, vllm_dots, vllm_gemma."
                ),
            )
        ok, msg = _provider_configured(s, prov)
        if not ok:
            raise HTTPException(400, detail=msg or "Provider not configured")

        await _ensure_vllm_ready_for_submit(s, prov)

        model_id_clean: str | None = None
        if model_id is not None:
            stripped = model_id.strip()
            if not stripped:
                raise HTTPException(422, detail="model_id must be non-empty when supplied")
            model_id_clean = stripped

        model_id_clean = resolve_model_id_for_provider(s, prov, model_id_clean)
        if model_id_clean != (model_id.strip() if model_id else None):
            logger.info(
                "OCR submit: model resolved to %s for provider=%s",
                model_id_clean or "(default)",
                provider,
            )

        # Build few-shots
        parsed_shots = parse_few_shots_json(few_shots)
        try:
            resolved_shots = await build_few_shots_for_provider(parsed_shots, few_shot_files or None)
        except HTTPException:
            raise

        # Persist uploads to disk
        upload_root = s.upload_root_path()
        upload_root.mkdir(parents=True, exist_ok=True)
        if s.upload_retain_hours > 0:
            prune_old_batches(upload_root, s.upload_retain_hours)

        batch_id, batch_dir, saved = await persist_ocr_uploads(upload_root, files)

        effective_user_prompt = (user_prompt or system_prompt or "").strip() or None
        persist_few_shots(batch_dir, resolved_shots)
        write_batch_metadata(
            batch_dir,
            {
                "batch_id": batch_id,
                "created_at": datetime.now(UTC).isoformat(),
                "provider": provider,
                "model_id": model_id_clean,
                "files": [fn for (_, fn, _) in saved],
                "few_shot_count": len(resolved_shots),
                "user_prompt": effective_user_prompt,
                "process_mode": process_mode,
                "request_id": request_id,
            },
        )

        # ── Resume: load already-completed pages from a prior partial run ────
        preloaded_pages: list[dict] = []
        if resume_job_id and _UUID_RE.match(resume_job_id):
            prev_dir = s.upload_root_path() / resume_job_id
            prev_results = prev_dir / "results.jsonl"
            if prev_results.is_file():
                for raw_line in prev_results.read_text(encoding="utf-8").splitlines():
                    raw_line = raw_line.strip()
                    if raw_line:
                        try:
                            preloaded_pages.append(json.loads(raw_line))
                        except json.JSONDecodeError:
                            pass
            logger.info(
                "Resume from %s: loaded %d preloaded pages",
                resume_job_id, len(preloaded_pages),
            )

        job_id, _q = _start_ocr_task(
            s=s,
            batch_id=batch_id,
            batch_dir=batch_dir,
            saved=saved,
            provider=provider,
            effective_user_prompt=effective_user_prompt,
            resolved_shots=resolved_shots,
            model_id_clean=model_id_clean,
            preloaded_pages=preloaded_pages or None,
            process_mode=process_mode,
            request_id=request_id,
        )

        return OcrJobResponse(
            job_id=job_id,
            stream_url=f"/api/ocr/{job_id}/stream",
            total_files=len(saved),
            resumed_pages=len(preloaded_pages),
            remaining_pages=None,
        )

    @app.post("/api/ocr/{job_id}/resume", response_model=OcrResumeResponse)
    async def resume_ocr_job(
        job_id: str,
        request: starlette.requests.Request,
        s: Settings = Depends(get_settings),
    ) -> OcrResumeResponse:
        if not _UUID_RE.match(job_id):
            raise HTTPException(400, "Invalid job ID format")
        batch_dir = s.upload_root_path() / job_id
        if not batch_dir.is_dir():
            raise HTTPException(404, "Job not found or results have expired")

        meta_path = batch_dir / "metadata.json"
        if not meta_path.is_file():
            raise HTTPException(404, "Job metadata not found")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        provider = meta.get("provider") or OcrProvider.gemini.value
        try:
            prov = OcrProvider(provider)
        except ValueError:
            raise HTTPException(422, detail=f"Unknown provider in metadata: {provider}")

        complete_path = batch_dir / "job_complete.json"
        if complete_path.is_file():
            info = json.loads(complete_path.read_text(encoding="utf-8"))
            if info.get("status") == "complete":
                raise HTTPException(409, detail="Job already completed")

        preloaded_pages = read_results_jsonl(batch_dir)
        filenames = meta.get("files") or []
        saved = list_source_files(batch_dir, filenames)

        await _ensure_vllm_ready_for_submit(s, prov)

        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        set_request_id(request_id)

        resume_prompt = meta.get("user_prompt")
        if isinstance(resume_prompt, str):
            resume_prompt = resume_prompt.strip() or None
        else:
            resume_prompt = None
        resume_mode = meta.get("process_mode") or "streaming"
        if resume_mode not in ("auto", "streaming", "batch"):
            resume_mode = "streaming"
        resolved_shots = load_few_shots(batch_dir)

        _job_id, _q = _start_ocr_task(
            s=s,
            batch_id=job_id,
            batch_dir=batch_dir,
            saved=saved,
            provider=provider,
            effective_user_prompt=resume_prompt,
            resolved_shots=resolved_shots,
            model_id_clean=meta.get("model_id"),
            preloaded_pages=preloaded_pages,
            process_mode=resume_mode,
            request_id=request_id,
        )

        total_pages = meta.get("total_pages")
        resumed = len(preloaded_pages)
        remaining = max(0, (total_pages or resumed) - resumed) if total_pages else 0

        return OcrResumeResponse(
            job_id=job_id,
            stream_url=f"/api/ocr/{job_id}/stream",
            resumed_pages=resumed,
            remaining_pages=remaining,
        )

    @app.post("/api/ocr/{job_id}/cancel")
    async def cancel_ocr_job(job_id: str) -> dict:
        if not _UUID_RE.match(job_id):
            raise HTTPException(400, "Invalid job ID format")
        if not request_cancel(job_id):
            raise HTTPException(404, detail="Job not active in memory (may already be finished)")
        logger.info("Cancel requested for job_id=%s", job_id)
        return {"job_id": job_id, "cancelled": True}

    # ------------------------------------------------------------------
    # OCR — SSE stream (long-lived GET)
    # ------------------------------------------------------------------

    @app.get("/api/ocr/{job_id}/stream")
    async def stream_ocr_results(job_id: str) -> StreamingResponse:
        entry = get_job(job_id)
        if not entry:
            raise HTTPException(404, detail="Job not found. It may have expired or already been consumed.")
        q, task, _ = entry

        async def generate():
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(q.get(), timeout=25.0)
                    except asyncio.TimeoutError:
                        # Keep-alive comment so nginx/browser don't close the connection.
                        yield ": ping\n\n"
                        continue
                    if item is None:
                        # Job complete — close the stream.
                        return
                    event_name = item.get("event", "message")
                    yield f"event: {event_name}\ndata: {json.dumps(item)}\n\n"
            finally:
                # Client disconnected — keep the background OCR task running.
                # Partial/final results remain on disk (GET /api/ocr/{job_id}/result).
                logger.info("SSE client disconnected for job %s (task still running)", job_id)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                # Tells nginx (and any other buffering proxy) not to buffer this response.
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # ------------------------------------------------------------------
    # OCR — retrieve saved results (survives browser refresh / disconnect)
    # ------------------------------------------------------------------

    def _read_saved_result(batch_dir: Path, job_id: str) -> OcrSavedResult:
        results_path = batch_dir / "results.jsonl"
        pages = []
        if results_path.exists():
            for raw in results_path.read_text(encoding="utf-8").splitlines():
                if raw.strip():
                    try:
                        pages.append(json.loads(raw))
                    except json.JSONDecodeError:
                        pass
        pages.sort(key=page_sort_key)

        complete_path = batch_dir / "job_complete.json"
        done = complete_path.exists()
        total = len(pages)
        elapsed = None
        completed_at = None
        status = None
        failed_pages: list[dict] = []
        if done:
            try:
                info = json.loads(complete_path.read_text(encoding="utf-8"))
                total = info.get("total", len(pages))
                elapsed = info.get("elapsed_seconds")
                completed_at = info.get("completed_at")
                status = info.get("status")
                failed_pages = info.get("failed_pages") or []
            except (json.JSONDecodeError, OSError):
                pass
        elif pages:
            status = "in_progress"

        provider = None
        files: list[str] = []
        submitted_at = None
        meta_path = batch_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                provider = meta.get("provider")
                files = meta.get("files", [])
                submitted_at = meta.get("created_at")
            except (json.JSONDecodeError, OSError):
                pass

        combined_path = batch_dir / "ocr_output.txt"
        if combined_path.exists():
            combined_text = combined_path.read_text(encoding="utf-8")
        elif pages:
            combined_text = format_consolidated_ocr_text(
                pages,
                source_files=files,
                provider=provider,
                submitted_at=submitted_at,
                completed_at=completed_at,
                elapsed_seconds=elapsed,
            )
        else:
            combined_text = ""

        gemini_cost = None
        cost_raw = read_job_cost_summary(batch_dir)
        if cost_raw and (batch_dir / "gemini_cost_log.json").is_file():
            gemini_cost = GeminiJobCostSummary(**cost_raw)

        vllm_cost = None
        vllm_raw = read_vllm_job_cost_summary(batch_dir)
        if vllm_raw:
            vllm_cost = VllmJobCostSummary(
                model=vllm_raw.get("pricing_model_key"),
                total_api_calls=int(vllm_raw.get("total_api_calls") or 0),
                total_input_tokens=int(vllm_raw.get("total_input_tokens") or 0),
                total_output_tokens=int(vllm_raw.get("total_output_tokens") or 0),
                estimated_total_cost_usd=float(vllm_raw.get("estimated_total_cost_usd") or 0),
                gpu_seconds=float(vllm_raw.get("gpu_seconds") or 0) or None,
                pricing_source=vllm_raw.get("pricing_source"),
            )

        return OcrSavedResult(
            job_id=job_id,
            done=done,
            total=total,
            done_count=len(pages),
            provider=provider,
            files=files,
            submitted_at=submitted_at,
            completed_at=completed_at,
            elapsed_seconds=elapsed,
            pages=pages,
            combined_text=combined_text,
            gemini_cost=gemini_cost,
            vllm_cost=vllm_cost,
            status=status,
            failed_pages=failed_pages,
        )

    @app.get("/api/ocr/{job_id}/status", response_model=OcrJobStatusResponse)
    async def ocr_job_status(job_id: str, s: Settings = Depends(get_settings)) -> OcrJobStatusResponse:
        if not _UUID_RE.match(job_id):
            raise HTTPException(400, "Invalid job ID format")
        batch_dir = s.upload_root_path() / job_id
        if not batch_dir.is_dir():
            raise HTTPException(404, "Job not found")
        saved = _read_saved_result(batch_dir, job_id)
        mode = None
        meta_path = batch_dir / "metadata.json"
        if meta_path.is_file():
            try:
                mode = json.loads(meta_path.read_text(encoding="utf-8")).get("process_mode")
            except (json.JSONDecodeError, OSError):
                pass
        return OcrJobStatusResponse(
            job_id=job_id,
            done=saved.done,
            done_count=saved.done_count,
            total=saved.total,
            provider=saved.provider,
            mode=mode,
            status=saved.status or ("complete" if saved.done else "in_progress"),
            failed_pages=saved.failed_pages,
            active_in_memory=get_job(job_id) is not None,
        )

    @app.get("/api/ocr/jobs", response_model=dict)
    async def list_ocr_jobs(s: Settings = Depends(get_settings)) -> dict:
        """List recent OCR jobs (completed or in-progress) from disk."""
        upload_root = s.upload_root_path()
        jobs: list[RecentJobSummary] = []
        if upload_root.is_dir():
            entries = sorted(
                (e for e in upload_root.iterdir() if e.is_dir()),
                key=lambda e: e.stat().st_mtime,
                reverse=True,
            )
            for batch_dir in entries[:30]:
                meta_path = batch_dir / "metadata.json"
                if not meta_path.exists():
                    continue
                try:
                    saved = _read_saved_result(batch_dir, batch_dir.name)
                    jobs.append(RecentJobSummary(
                        job_id=saved.job_id,
                        provider=saved.provider,
                        files=saved.files,
                        submitted_at=saved.submitted_at,
                        done=saved.done,
                        total=saved.total,
                        done_count=saved.done_count,
                        completed_at=saved.completed_at,
                        gemini_cost=saved.gemini_cost,
                        vllm_cost=saved.vllm_cost,
                        status=saved.status,
                    ))
                except Exception as exc:
                    logger.warning("Skipping corrupt job dir %s: %s", batch_dir.name, exc)
        return {"jobs": [j.model_dump() for j in jobs]}

    @app.get("/api/gemini-costs/summary", response_model=GeminiCostsOverviewResponse)
    async def gemini_costs_summary(s: Settings = Depends(get_settings)) -> GeminiCostsOverviewResponse:
        """Aggregate Gemini token usage and USD estimates across all saved OCR jobs."""
        raw = aggregate_gemini_costs(s.upload_root_path(), ledger_path=s.gemini_cost_ledger_path())
        return GeminiCostsOverviewResponse(
            job_count=raw["job_count"],
            total_pages=raw["total_pages"],
            total_api_calls=raw["total_api_calls"],
            total_input_tokens=raw["total_input_tokens"],
            total_output_tokens=raw["total_output_tokens"],
            input_cost_usd=raw["input_cost_usd"],
            output_cost_usd=raw["output_cost_usd"],
            estimated_total_cost_usd=raw["estimated_total_cost_usd"],
            pricing_source=raw["pricing_source"],
            pricing_effective=raw["pricing_effective"],
            by_model=[GeminiModelCostBreakdown(**m) for m in raw["by_model"]],
            by_provider=[GeminiProviderCostBreakdown(**p) for p in raw.get("by_provider", [])],
            jobs=[GeminiJobCostEntry(**j) for j in raw["jobs"]],
        )

    @app.get("/api/ocr/{job_id}/result", response_model=OcrSavedResult)
    async def get_ocr_result(job_id: str, s: Settings = Depends(get_settings)) -> OcrSavedResult:
        """Return saved pages for a completed (or in-progress) job."""
        if not _UUID_RE.match(job_id):
            raise HTTPException(400, "Invalid job ID format")
        batch_dir = s.upload_root_path() / job_id
        if not batch_dir.is_dir():
            raise HTTPException(404, "Job not found or results have expired")
        return _read_saved_result(batch_dir, job_id)

    def _build_combined_text(saved: OcrSavedResult) -> str:
        return format_consolidated_ocr_text(
            [p.model_dump() for p in saved.pages],
            source_files=saved.files,
            provider=saved.provider,
            submitted_at=saved.submitted_at,
            completed_at=saved.completed_at,
            elapsed_seconds=saved.elapsed_seconds,
        )

    @app.get("/api/ocr/{job_id}/download.txt")
    async def download_txt(job_id: str, s: Settings = Depends(get_settings)) -> Response:
        """Download all saved pages as a plain-text file."""
        if not _UUID_RE.match(job_id):
            raise HTTPException(400, "Invalid job ID format")
        batch_dir = s.upload_root_path() / job_id
        if not batch_dir.is_dir():
            raise HTTPException(404, "Job not found or results have expired")
        saved = _read_saved_result(batch_dir, job_id)
        if not saved.pages:
            raise HTTPException(404, "No pages have been saved for this job yet")
        content = saved.combined_text or _build_combined_text(saved)
        slug = (saved.files[0].rsplit(".", 1)[0] if saved.files else job_id[:8])
        return Response(
            content=content.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": _content_disposition(f"vedic-ocr-{slug}.txt")},
        )

    @app.get("/api/ocr/{job_id}/download.docx")
    async def download_docx(job_id: str, s: Settings = Depends(get_settings)) -> Response:
        """Download all saved pages as a Word document."""
        import io
        from docx import Document
        from docx.oxml.ns import qn
        from docx.shared import Pt

        from app.utils.size_tags import TIER_TO_PT

        _FONT = "Noto Serif"

        def _styled_line(doc, text: str, pt: int) -> None:
            """Add a paragraph whose run carries the given point size for BOTH
            Latin and complex-script (Devanāgarī) glyphs — Word uses szCs/rFonts
            w:cs for complex scripts, so setting only w:sz would leave Devanāgarī
            at the default size."""
            para = doc.add_paragraph()
            run = para.add_run(text)
            run.font.name = _FONT
            rpr = run._element.get_or_add_rPr()
            rfonts = rpr.get_or_add_rFonts()
            for attr in ("w:ascii", "w:hAnsi", "w:cs"):
                rfonts.set(qn(attr), _FONT)
            half = str(int(pt * 2))  # OOXML sizes are in half-points
            for tag in ("w:sz", "w:szCs"):
                el = rpr.find(qn(tag))
                if el is None:
                    el = rpr.makeelement(qn(tag), {})
                    rpr.append(el)
                el.set(qn("w:val"), half)

        if not _UUID_RE.match(job_id):
            raise HTTPException(400, "Invalid job ID format")
        batch_dir = s.upload_root_path() / job_id
        if not batch_dir.is_dir():
            raise HTTPException(404, "Job not found or results have expired")
        saved = _read_saved_result(batch_dir, job_id)
        if not saved.pages:
            raise HTTPException(404, "No pages have been saved for this job yet")

        doc = Document()
        # Set default font to Noto Serif for Devanagari rendering
        style = doc.styles["Normal"]
        style.font.name = _FONT
        style.font.size = Pt(TIER_TO_PT["BODY"])

        multi_source = len({p.source_file for p in saved.pages}) > 1
        for p in sorted(saved.pages, key=lambda x: x.index):
            num = p.page_in_source if p.page_in_source is not None else p.index + 1
            heading = f"PAGE {num}"
            if multi_source:
                heading = f"{heading} — {p.source_file}"
            h = doc.add_heading(heading, level=2)
            h.style.font.name = _FONT
            # Preferred: per-segment font sizing from the model's size tags.
            # Fallback (older jobs / non-tagging providers): uniform body size.
            if p.segments:
                for seg in p.segments:
                    _styled_line(doc, seg.text, TIER_TO_PT.get(seg.tier, TIER_TO_PT["BODY"]))
            else:
                for line in p.text.split("\n"):
                    _styled_line(doc, line, TIER_TO_PT["BODY"])
            doc.add_paragraph("▬" * 40)

        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        slug = (saved.files[0].rsplit(".", 1)[0] if saved.files else job_id[:8])
        return Response(
            content=buf.read(),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": _content_disposition(f"vedic-ocr-{slug}.docx")},
        )

    # ------------------------------------------------------------------
    # Administrator — Gemini cost ledger (persistent across sessions)
    # ------------------------------------------------------------------

    def _require_admin(request: starlette.requests.Request, s: Settings) -> None:
        if not s.admin_api_key:
            raise HTTPException(
                503,
                detail=(
                    "ADMIN_API_KEY is not configured. "
                    "Admin endpoints are disabled until it is set in .env."
                ),
            )
        if request.headers.get("X-Admin-Key") != s.admin_api_key:
            raise HTTPException(
                401,
                detail="Missing or invalid X-Admin-Key header.",
            )

    @app.get("/api/admin/gemini-costs")
    async def list_gemini_costs(
        request: starlette.requests.Request,
        limit: int = 100,
        s: Settings = Depends(get_settings),
    ) -> dict:
        """List recent Gemini OCR job cost summaries from the global ledger."""
        _require_admin(request, s)
        limit = max(1, min(limit, 500))
        records = read_global_ledger(s.gemini_cost_ledger_path(), limit=limit)
        return {
            "ledger_path": str(s.gemini_cost_ledger_path()),
            "pricing_source": "https://ai.google.dev/gemini-api/docs/pricing",
            "jobs": records,
        }

    @app.get("/api/admin/gemini-costs/{job_id}")
    async def get_gemini_costs_for_job(
        job_id: str,
        request: starlette.requests.Request,
        s: Settings = Depends(get_settings),
    ) -> dict:
        """Per-job cost log: ledger lines + on-disk gemini_cost_log.json if present."""
        _require_admin(request, s)
        if not _UUID_RE.match(job_id):
            raise HTTPException(400, "Invalid job ID format")
        ledger_records = read_global_ledger(
            s.gemini_cost_ledger_path(), limit=10_000, job_id=job_id
        )
        batch_dir = s.upload_root_path() / job_id
        job_log = None
        cost_path = batch_dir / "gemini_cost_log.json"
        if cost_path.is_file():
            try:
                job_log = json.loads(cost_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                job_log = None
        if not ledger_records and job_log is None:
            raise HTTPException(404, "No Gemini cost records found for this job.")
        return {
            "job_id": job_id,
            "ledger_records": ledger_records,
            "job_cost_log": job_log,
        }

    @app.get("/api/admin/jobs/{job_id}/log")
    async def get_job_log(
        job_id: str,
        request: starlette.requests.Request,
        s: Settings = Depends(get_settings),
    ) -> PlainTextResponse:
        _require_admin(request, s)
        if not _UUID_RE.match(job_id):
            raise HTTPException(400, "Invalid job ID format")
        log_path = s.upload_root_path() / job_id / "job.log"
        if not log_path.is_file():
            raise HTTPException(404, "No job.log for this job")
        return PlainTextResponse(log_path.read_text(encoding="utf-8"))

    @app.get("/api/ocr/{job_id}/log")
    async def get_ocr_job_log(
        job_id: str,
        s: Settings = Depends(get_settings),
    ) -> PlainTextResponse:
        """Public job log for Studio (UUID acts as capability token, same as downloads)."""
        if not _UUID_RE.match(job_id):
            raise HTTPException(400, "Invalid job ID format")
        log_path = s.upload_root_path() / job_id / "job.log"
        if not log_path.is_file():
            raise HTTPException(404, "No job.log for this job")
        return PlainTextResponse(log_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------

    @app.exception_handler(HTTPException)
    async def http_exc_handler(request, exc: HTTPException):
        detail = exc.detail
        extra: dict[str, Any] | None = None
        if isinstance(detail, dict):
            msg = str(detail.get("message") or detail.get("detail") or "Request failed")
            extra = {k: v for k, v in detail.items() if k not in ("message", "detail")}
            if not extra:
                extra = dict(detail)
        else:
            msg = str(detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                detail=msg, code="http_error", extra=extra
            ).model_dump(),
        )

    return app


app = create_app()
