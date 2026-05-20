from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OcrProvider(str, Enum):
    gemini = "gemini"
    bedrock_claude = "bedrock_claude"
    bedrock_ocr = "bedrock_ocr"
    vllm_gemma = "vllm_gemma"


class FewShotExample(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image bytes (no data URL prefix).")
    expected_text: str = Field(..., description="Gold transcription for the few-shot image.")
    mime_type: str = Field(default="image/png", description="MIME type of the few-shot image.")


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


class ProviderInfo(BaseModel):
    id: str
    label: str
    configured: bool
    detail: str | None = None
    default_model_id: str | None = None
    model_options: list[str] = Field(default_factory=list)


class ProvidersResponse(BaseModel):
    providers: list[ProviderInfo]


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
    extra: dict[str, Any] | None = None
