from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_upload_storage_dir() -> str:
    backend_root = Path(__file__).resolve().parents[1]
    return str((backend_root / "data" / "uploads").resolve())


def _comma_separated_nonempty(raw: str | None) -> list[str]:
    if not raw or not raw.strip():
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _dedupe_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(
            str(_REPO_ROOT / ".env"),
            str(_REPO_ROOT / "backend" / ".env"),
            ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Google Gemini
    google_api_key: str | None = None
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        validation_alias=AliasChoices("GEMINI_MODEL", "gemini_model"),
    )
    gemini_model_options: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_MODEL_OPTIONS", "gemini_model_options"),
    )
    gemini_use_vertexai: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "GEMINI_USE_VERTEXAI", "GOOGLE_GENAI_USE_VERTEXAI", "gemini_use_vertexai"
        ),
    )
    google_cloud_project: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_CLOUD_PROJECT", "google_cloud_project"),
    )
    google_cloud_location: str = Field(
        default="us-central1",
        validation_alias=AliasChoices("GOOGLE_CLOUD_LOCATION", "google_cloud_location"),
    )

    # AWS Bedrock
    aws_region: str | None = None
    aws_profile: str | None = None
    bedrock_claude_model_id: str | None = None
    bedrock_claude_model_options: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "BEDROCK_CLAUDE_MODEL_OPTIONS", "bedrock_claude_model_options"
        ),
    )
    bedrock_ocr_model_id: str | None = None
    bedrock_ocr_model_options: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "BEDROCK_OCR_MODEL_OPTIONS", "bedrock_ocr_model_options"
        ),
    )

    # OCR performance
    ocr_request_timeout_seconds: int = 120
    ocr_page_concurrency: int = Field(
        default=8,
        validation_alias=AliasChoices("OCR_PAGE_CONCURRENCY", "ocr_page_concurrency"),
    )
    ocr_pdf_dpi: int = Field(
        default=150,
        validation_alias=AliasChoices("OCR_PDF_DPI", "ocr_pdf_dpi"),
    )
    # How many times to retry a failed page before giving up (0 = no retries).
    ocr_page_max_retries: int = Field(
        default=2,
        validation_alias=AliasChoices("OCR_PAGE_MAX_RETRIES", "ocr_page_max_retries"),
    )

    # Local vLLM
    vllm_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("VLLM_ENABLED", "vllm_enabled"),
    )
    vllm_base_url: str = Field(
        default="http://vllm:8000/v1",
        validation_alias=AliasChoices("VLLM_BASE_URL", "vllm_base_url"),
    )
    vllm_model: str = Field(
        default="nvidia/Gemma-4-31B-IT-NVFP4",
        validation_alias=AliasChoices("VLLM_MODEL", "vllm_model"),
    )
    vllm_model_options: str | None = Field(
        default=None,
        validation_alias=AliasChoices("VLLM_MODEL_OPTIONS", "vllm_model_options"),
    )
    vllm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("VLLM_API_KEY", "vllm_api_key"),
    )
    vllm_request_timeout_seconds: int = Field(
        default=600,
        validation_alias=AliasChoices(
            "VLLM_REQUEST_TIMEOUT_SECONDS", "vllm_request_timeout_seconds"
        ),
    )
    # Set true to let the backend start/stop vLLM via the Docker socket.
    vllm_on_demand: bool = Field(
        default=False,
        validation_alias=AliasChoices("VLLM_ON_DEMAND", "vllm_on_demand"),
    )
    # Docker container name for the vLLM service (docker compose naming: <project>_<service>_1).
    vllm_container_name: str = Field(
        default="ocr-vedic-vllm-1",
        validation_alias=AliasChoices("VLLM_CONTAINER_NAME", "vllm_container_name"),
    )
    # Send a tiny warm-up request after load so first real page is fast.
    vllm_warmup_on_load: bool = Field(
        default=False,
        validation_alias=AliasChoices("VLLM_WARMUP_ON_LOAD", "vllm_warmup_on_load"),
    )

    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        validation_alias=AliasChoices("CORS_ORIGINS", "cors_origins"),
    )

    upload_storage_dir: str = Field(
        default_factory=_default_upload_storage_dir,
        validation_alias=AliasChoices("UPLOAD_STORAGE_DIR", "upload_storage_dir"),
    )
    upload_retain: bool = Field(
        default=False,
        validation_alias=AliasChoices("UPLOAD_RETAIN", "upload_retain"),
    )
    upload_retain_hours: float = Field(
        default=168.0,
        validation_alias=AliasChoices("UPLOAD_RETAIN_HOURS", "upload_retain_hours"),
    )

    def upload_root_path(self) -> Path:
        return Path(self.upload_storage_dir).expanduser().resolve()

    def gemini_models_for_providers(self) -> tuple[str | None, list[str]]:
        default = self.gemini_model
        opts = _dedupe_order(
            _comma_separated_nonempty(self.gemini_model_options) + [default]
        )
        return default, opts

    def bedrock_claude_models_for_providers(self) -> tuple[str | None, list[str]]:
        d = self.bedrock_claude_model_id
        opts = _dedupe_order(
            _comma_separated_nonempty(self.bedrock_claude_model_options)
            + ([d] if d else [])
        )
        return d, opts

    def bedrock_open_models_for_providers(self) -> tuple[str | None, list[str]]:
        d = self.bedrock_ocr_model_id
        opts = _dedupe_order(
            _comma_separated_nonempty(self.bedrock_ocr_model_options)
            + ([d] if d else [])
        )
        return d, opts

    def vllm_models_for_providers(self) -> tuple[str | None, list[str]]:
        default = self.vllm_model
        opts = _dedupe_order(
            _comma_separated_nonempty(self.vllm_model_options) + [default]
        )
        return default, opts


@lru_cache
def get_settings() -> Settings:
    return Settings()
