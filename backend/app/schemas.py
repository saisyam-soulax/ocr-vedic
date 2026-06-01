from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OcrProvider(str, Enum):
    gemini = "gemini"
    bedrock_claude = "bedrock_claude"
    bedrock_ocr = "bedrock_ocr"
    vllm_dots = "vllm_dots"


class VllmState(str, Enum):
    stopped = "stopped"
    starting = "starting"
    ready = "ready"
    stopping = "stopping"
    error = "error"


class FewShotExample(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image bytes.")
    expected_text: str = Field(..., description="Gold transcription for the few-shot image.")
    mime_type: str = Field(default="image/png")


class OcrPageResult(BaseModel):
    index: int
    source_file: str
    page_in_source: int | None = None
    text: str
    mime_type: str | None = None


class OcrResponse(BaseModel):
    provider: OcrProvider
    pages: list[OcrPageResult]
    combined_text: str


class OcrJobResponse(BaseModel):
    job_id: str
    stream_url: str
    total_files: int


class OcrSavedResult(BaseModel):
    job_id: str
    done: bool
    total: int
    done_count: int
    provider: str | None = None
    files: list[str] = Field(default_factory=list)
    submitted_at: str | None = None
    completed_at: str | None = None
    elapsed_seconds: float | None = None
    pages: list[OcrPageResult] = Field(default_factory=list)
    combined_text: str = ""


class RecentJobSummary(BaseModel):
    job_id: str
    provider: str | None = None
    files: list[str] = Field(default_factory=list)
    submitted_at: str | None = None
    done: bool = False
    total: int = 0
    done_count: int = 0
    completed_at: str | None = None


class ProviderInfo(BaseModel):
    id: str
    label: str
    configured: bool
    detail: str | None = None
    default_model_id: str | None = None
    model_options: list[str] = Field(default_factory=list)


class ProvidersResponse(BaseModel):
    providers: list[ProviderInfo]


class OcrDefaultsResponse(BaseModel):
    system_prompt: str


class VllmStatusResponse(BaseModel):
    state: VllmState
    reachable: bool
    message: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
    extra: dict[str, Any] | None = None
