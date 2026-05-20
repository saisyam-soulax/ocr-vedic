from __future__ import annotations

import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated

import starlette.requests
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from starlette.datastructures import FormData

# ---------------------------------------------------------------------------
# Configure root logger from LOG_LEVEL env var before any other module uses it.
# ---------------------------------------------------------------------------
_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Starlette 1.0 defaults max_part_size to 1 MiB, which rejects real PDFs and
# large images with an opaque HTTP 400.  Raise the per-part limit to 50 MiB
# application-wide by patching the private helper that caches the parsed form.
# ---------------------------------------------------------------------------
_MAX_PART_SIZE = 50 * 1024 * 1024  # 50 MiB

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
logger.info("Starlette _get_form patched: max_part_size=%d bytes (%.0f MiB)", _MAX_PART_SIZE, _MAX_PART_SIZE / 1024 / 1024)

from app.config import Settings, get_settings
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    OcrProvider,
    OcrResponse,
    ProvidersResponse,
    ProviderInfo,
)
from app.services.ocr_service import (
    build_few_shots_for_provider,
    parse_few_shots_json,
    run_ocr_batch,
)
from app.storage_uploads import persist_ocr_uploads, prune_old_batches, write_batch_metadata


def _provider_configured(settings: Settings, p: OcrProvider) -> tuple[bool, str | None]:
    if p == OcrProvider.gemini:
        if settings.gemini_use_vertexai:
            ok = bool(settings.google_api_key) or bool(settings.google_cloud_project)
            return ok, None if ok else (
                "Vertex AI mode requires GOOGLE_API_KEY (Express Mode) or "
                "GOOGLE_CLOUD_PROJECT + ADC (gcloud auth application-default login)"
            )
        ok = bool(settings.google_api_key)
        return ok, None if ok else "Set GOOGLE_API_KEY"
    if p == OcrProvider.bedrock_claude:
        ok = bool(settings.aws_region and settings.bedrock_claude_model_id)
        return ok, None if ok else "Set AWS_REGION and BEDROCK_CLAUDE_MODEL_ID"
    if p == OcrProvider.bedrock_ocr:
        ok = bool(settings.aws_region and settings.bedrock_ocr_model_id)
        return ok, None if ok else "Set AWS_REGION and BEDROCK_OCR_MODEL_ID"
    if not settings.vllm_enabled:
        return False, "Set VLLM_ENABLED=true and start vLLM (docker compose --profile vllm)"
    if not settings.vllm_base_url:
        return False, "Set VLLM_BASE_URL"
    from app.providers.vllm_gemma import check_vllm_reachable

    if not check_vllm_reachable(settings):
        return (
            False,
            "vLLM server not reachable — run: docker compose --profile vllm up -d",
        )
    return True, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "App startup: log_level=%s upload_root=%s retain=%s retain_hours=%s",
        _log_level,
        settings.upload_root_path(),
        settings.upload_retain,
        settings.upload_retain_hours,
    )
    yield
    logger.info("App shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Vedic OCR API",
        version="0.1.0",
        lifespan=lifespan,
        description="Multimodal OCR for high-diacritics and Devanāgarī using Gemini or AWS Bedrock.",
    )
    settings = get_settings()
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/api/providers", response_model=ProvidersResponse)
    def providers(s: Settings = Depends(get_settings)) -> ProvidersResponse:
        rows: list[ProviderInfo] = []
        labels = {
            OcrProvider.gemini: "Google Gemini",
            OcrProvider.bedrock_claude: "AWS Bedrock — Claude",
            OcrProvider.bedrock_ocr: "AWS Bedrock — Open multimodal",
            OcrProvider.vllm_gemma: "Local — Gemma 4 (vLLM)",
        }
        for p in OcrProvider:
            configured, detail = _provider_configured(s, p)
            default_mid: str | None
            mids: list[str]
            if p == OcrProvider.gemini:
                default_mid, mids = s.gemini_models_for_providers()
            elif p == OcrProvider.bedrock_claude:
                default_mid, mids = s.bedrock_claude_models_for_providers()
            elif p == OcrProvider.bedrock_ocr:
                default_mid, mids = s.bedrock_open_models_for_providers()
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

    @app.post("/api/ocr", response_model=OcrResponse)
    async def ocr(
        files: Annotated[list[UploadFile], File(description="PDF and/or image files")],
        provider: str = Form(
            ..., description="gemini | bedrock_claude | bedrock_ocr | vllm_gemma"
        ),
        system_prompt: str | None = Form(None),
        few_shots: str | None = Form(
            None,
            description='JSON array of {"expected_text","image_base64"?,"mime_type"?}',
        ),
        few_shot_files: Annotated[
            list[UploadFile] | None,
            File(description="Images for few-shots missing image_base64; same order as few_shots"),
        ] = None,
        model_id: str | None = Form(
            None,
            description="Optional model id overriding server default for the selected provider",
        ),
        s: Settings = Depends(get_settings),
    ) -> OcrResponse:
        t_start = time.perf_counter()

        # Log incoming request details
        file_info = [
            {
                "filename": f.filename,
                "content_type": f.content_type,
                "size": f.size,
            }
            for f in files
        ]
        logger.info(
            "OCR request: provider=%s model_id=%s files=%d few_shots=%s",
            provider,
            model_id or "(default)",
            len(files),
            len(few_shots) if few_shots else 0,
        )
        for fi in file_info:
            logger.info(
                "  file: name=%r content_type=%s size=%s bytes",
                fi["filename"],
                fi["content_type"],
                fi["size"],
            )

        parsed_shots = parse_few_shots_json(few_shots)
        try:
            resolved = await build_few_shots_for_provider(parsed_shots, few_shot_files or None)
        except HTTPException as exc:
            logger.warning("Validation error building few-shots: %s", exc.detail)
            raise
        logger.info("Few-shot pairs resolved: %d", len(resolved))

        try:
            prov = OcrProvider(provider)
        except ValueError as exc:
            logger.warning("Invalid provider value: %r", provider)
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Invalid provider '{provider}'. Use one of: "
                    "gemini, bedrock_claude, bedrock_ocr, vllm_gemma."
                ),
            ) from exc

        ok, msg = _provider_configured(s, prov)
        if not ok:
            logger.warning("Provider %s not configured: %s", provider, msg)
            raise HTTPException(status_code=400, detail=msg or "Provider not configured")

        override_model_id: str | None = None
        if model_id is not None:
            stripped = model_id.strip()
            if not stripped:
                logger.warning("Received empty model_id string")
                raise HTTPException(
                    status_code=422,
                    detail="model_id must be a non-empty string when supplied",
                )
            override_model_id = stripped

        upload_root = s.upload_root_path()
        upload_root.mkdir(parents=True, exist_ok=True)
        if s.upload_retain_hours > 0:
            prune_old_batches(upload_root, s.upload_retain_hours)

        batch_id, batch_dir, saved = await persist_ocr_uploads(upload_root, files)
        write_batch_metadata(
            batch_dir,
            {
                "batch_id": batch_id,
                "created_at": datetime.now(UTC).isoformat(),
                "provider": provider,
                "model_id": override_model_id,
                "files": [fn for (_, fn, _) in saved],
                "few_shot_count": len(resolved),
            },
        )

        try:
            pages, combined = await run_ocr_batch(
                saved_files=saved,
                provider=provider,
                system_prompt=system_prompt,
                few_shots=resolved,
                settings=s,
                model_id=override_model_id,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            logger.warning("OCR validation error: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("OCR provider error")
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected OCR error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        elapsed = time.perf_counter() - t_start
        logger.info(
            "OCR complete: provider=%s pages=%d combined_chars=%d elapsed=%.2fs",
            provider,
            len(pages),
            len(combined),
            elapsed,
        )

        if not s.upload_retain:
            shutil.rmtree(batch_dir, ignore_errors=True)

        return OcrResponse(provider=prov, pages=pages, combined_text=combined)

    @app.exception_handler(HTTPException)
    async def http_exc_handler(request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(detail=str(exc.detail), code="http_error").model_dump(),
        )

    return app


app = create_app()
