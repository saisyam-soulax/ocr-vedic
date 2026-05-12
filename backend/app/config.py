from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

def _default_upload_storage_dir() -> str:
    """Default upload root; resolved at Settings init (avoids class-body NameError on partial saves)."""
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

    google_api_key: str | None = None
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        validation_alias=AliasChoices("GEMINI_MODEL", "gemini_model"),
    )
    gemini_model_options: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_MODEL_OPTIONS", "gemini_model_options"),
    )
    #: Route Gemini calls through Vertex AI (required for Agentic Platform keys / `AQ.*`
    #: tokens and for Vertex-only models like `gemini-3.1-pro`). Accepts either the
    #: project-local `GEMINI_USE_VERTEXAI` env var or the SDK-native
    #: `GOOGLE_GENAI_USE_VERTEXAI`.
    gemini_use_vertexai: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "GEMINI_USE_VERTEXAI",
            "GOOGLE_GENAI_USE_VERTEXAI",
            "gemini_use_vertexai",
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

    aws_region: str | None = None
    aws_profile: str | None = None
    bedrock_claude_model_id: str | None = None
    bedrock_claude_model_options: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "BEDROCK_CLAUDE_MODEL_OPTIONS",
            "bedrock_claude_model_options",
        ),
    )
    bedrock_ocr_model_id: str | None = None
    bedrock_ocr_model_options: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "BEDROCK_OCR_MODEL_OPTIONS",
            "bedrock_ocr_model_options",
        ),
    )

    ocr_request_timeout_seconds: int = 120
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        validation_alias=AliasChoices("CORS_ORIGINS", "cors_origins"),
    )

    #: Root directory for OCR upload batches (`<dir>/<uuid>/...`). Defaults to ``<backend-root>/data/uploads``.
    upload_storage_dir: str = Field(
        default_factory=_default_upload_storage_dir,
        validation_alias=AliasChoices("UPLOAD_STORAGE_DIR", "upload_storage_dir"),
    )
    #: When false, successfully processed batches under ``upload_storage_dir`` are removed.
    upload_retain: bool = Field(
        default=False,
        validation_alias=AliasChoices("UPLOAD_RETAIN", "upload_retain"),
    )
    #: Prune stale batch folders older than this many hours when > 0 (by directory mtime).
    upload_retain_hours: float = Field(
        default=168.0,
        validation_alias=AliasChoices("UPLOAD_RETAIN_HOURS", "upload_retain_hours"),
    )

    def upload_root_path(self) -> Path:
        return Path(self.upload_storage_dir).expanduser().resolve()

    def gemini_models_for_providers(self) -> tuple[str | None, list[str]]:
        default = self.gemini_model
        opts = _dedupe_order(_comma_separated_nonempty(self.gemini_model_options) + [default])
        return default, opts

    def bedrock_claude_models_for_providers(self) -> tuple[str | None, list[str]]:
        d = self.bedrock_claude_model_id
        opts = _dedupe_order(_comma_separated_nonempty(self.bedrock_claude_model_options) + ([d] if d else []))
        return d, opts

    def bedrock_open_models_for_providers(self) -> tuple[str | None, list[str]]:
        d = self.bedrock_ocr_model_id
        opts = _dedupe_order(_comma_separated_nonempty(self.bedrock_ocr_model_options) + ([d] if d else []))
        return d, opts


@lru_cache
def get_settings() -> Settings:
    return Settings()
