from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings
    from app.schemas import OcrProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FewShotPair:
    image_bytes: bytes
    mime_type: str
    expected_text: str


class OcrProviderBase(ABC):
    timeout_seconds: int

    @abstractmethod
    def transcribe(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        few_shots: list[FewShotPair],
    ) -> str:
        raise NotImplementedError

    @staticmethod
    def _decode_b64(data: str) -> bytes:
        raw = data.strip()
        if raw.startswith("data:") and "base64," in raw:
            raw = raw.split("base64,", 1)[1]
        return base64.b64decode(raw, validate=False)


def transcribe_with_provider(
    provider_name: str,
    *,
    image_bytes: bytes,
    mime_type: str,
    system_prompt: str,
    few_shots: list[dict[str, str]],
    settings: Settings,
    model_id: str | None = None,
) -> str:
    """Dispatch to the correct provider implementation."""
    from app.providers.bedrock_claude import BedrockClaudeProvider
    from app.providers.bedrock_open import BedrockOpenMultimodalProvider
    from app.providers.gemini import GeminiProvider
    from app.providers.vllm_gemma import VllmGemmaProvider
    from app.schemas import OcrProvider

    parsed = OcrProvider(provider_name)
    pairs = [
        FewShotPair(
            image_bytes=OcrProviderBase._decode_b64(x["image_base64"]),
            mime_type=x.get("mime_type", "image/png"),
            expected_text=x["expected_text"],
        )
        for x in few_shots
    ]
    if parsed == OcrProvider.vllm_gemma:
        timeout = settings.vllm_request_timeout_seconds
    else:
        timeout = settings.ocr_request_timeout_seconds

    if parsed == OcrProvider.gemini:
        impl: OcrProviderBase = GeminiProvider(
            settings=settings, timeout_seconds=timeout, model_id=model_id
        )
        effective_model = model_id or settings.gemini_model
    elif parsed == OcrProvider.bedrock_claude:
        impl = BedrockClaudeProvider(
            settings=settings, timeout_seconds=timeout, model_id=model_id
        )
        effective_model = model_id or settings.bedrock_claude_model_id
    elif parsed == OcrProvider.bedrock_ocr:
        impl = BedrockOpenMultimodalProvider(
            settings=settings, timeout_seconds=timeout, model_id=model_id
        )
        effective_model = model_id or settings.bedrock_ocr_model_id
    elif parsed == OcrProvider.vllm_gemma:
        impl = VllmGemmaProvider(
            settings=settings, timeout_seconds=timeout, model_id=model_id
        )
        effective_model = model_id or settings.vllm_model
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")

    logger.debug(
        "Dispatching transcribe: provider=%s model=%s image_size=%d bytes few_shots=%d",
        provider_name, effective_model, len(image_bytes), len(pairs),
    )

    return impl.transcribe(
        image_bytes=image_bytes,
        mime_type=mime_type,
        system_prompt=system_prompt,
        few_shots=pairs,
    )


def transcribe_image(
    image_bytes: bytes,
    mime_type: str,
    system_prompt: str,
    few_shot_examples: list[dict[str, str]],
    provider: str,
    *,
    settings: Settings | None = None,
    model_id: str | None = None,
) -> str:
    """Unified OCR entrypoint: transcribe a single image with optional few-shots."""
    from app.config import get_settings

    return transcribe_with_provider(
        provider,
        image_bytes=image_bytes,
        mime_type=mime_type,
        system_prompt=system_prompt,
        few_shots=few_shot_examples,
        settings=settings or get_settings(),
        model_id=model_id,
    )
