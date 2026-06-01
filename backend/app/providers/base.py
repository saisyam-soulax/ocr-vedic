from __future__ import annotations

import asyncio
import base64
import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings

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
        user_prompt: str | None = None,
        few_shots: list[FewShotPair],
    ) -> str:
        raise NotImplementedError

    @staticmethod
    def _decode_b64(data: str) -> bytes:
        raw = data.strip()
        if raw.startswith("data:") and "base64," in raw:
            raw = raw.split("base64,", 1)[1]
        return base64.b64decode(raw, validate=False)


def _build_pairs(few_shots: list[dict[str, str]]) -> list[FewShotPair]:
    return [
        FewShotPair(
            image_bytes=OcrProviderBase._decode_b64(x["image_base64"]),
            mime_type=x.get("mime_type", "image/png"),
            expected_text=x["expected_text"],
        )
        for x in few_shots
    ]


def transcribe_with_provider(
    provider_name: str,
    *,
    image_bytes: bytes,
    mime_type: str,
    system_prompt: str,
    user_prompt: str | None = None,
    few_shots: list[dict[str, str]],
    settings: Settings,
    model_id: str | None = None,
    gemini_cost_session: object | None = None,
    gemini_cost_page_index: int | None = None,
    gemini_cost_page_in_source: int | None = None,
    gemini_cost_source_file: str | None = None,
) -> str:
    """Synchronous dispatch — used directly or via run_in_executor."""
    from app.providers.bedrock_claude import BedrockClaudeProvider
    from app.providers.bedrock_open import BedrockOpenMultimodalProvider
    from app.providers.gemini import GeminiProvider
    from app.providers.vllm_dots import VllmDotsProvider
    from app.schemas import OcrProvider

    parsed = OcrProvider(provider_name)
    pairs = _build_pairs(few_shots)

    timeout = (
        settings.vllm_request_timeout_seconds
        if parsed == OcrProvider.vllm_dots
        else settings.ocr_request_timeout_seconds
    )

    if parsed == OcrProvider.gemini:
        impl: OcrProviderBase = GeminiProvider(
            settings=settings,
            timeout_seconds=timeout,
            model_id=model_id,
            cost_session=gemini_cost_session,
            cost_page_index=gemini_cost_page_index,
            cost_page_in_source=gemini_cost_page_in_source,
            cost_source_file=gemini_cost_source_file,
        )
    elif parsed == OcrProvider.bedrock_claude:
        impl = BedrockClaudeProvider(settings=settings, timeout_seconds=timeout, model_id=model_id)
    elif parsed == OcrProvider.bedrock_ocr:
        impl = BedrockOpenMultimodalProvider(settings=settings, timeout_seconds=timeout, model_id=model_id)
    elif parsed == OcrProvider.vllm_dots:
        impl = VllmDotsProvider(settings=settings, timeout_seconds=timeout, model_id=model_id)
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")

    return impl.transcribe(
        image_bytes=image_bytes,
        mime_type=mime_type,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        few_shots=pairs,
    )


async def transcribe_with_provider_async(
    provider_name: str,
    *,
    image_bytes: bytes,
    mime_type: str,
    system_prompt: str,
    user_prompt: str | None = None,
    few_shots: list[dict[str, str]],
    settings: Settings,
    model_id: str | None = None,
    gemini_cost_session: object | None = None,
    gemini_cost_page_index: int | None = None,
    gemini_cost_page_in_source: int | None = None,
    gemini_cost_source_file: str | None = None,
) -> str:
    """Async dispatch.

    vLLM uses a native AsyncClient for true concurrency.
    All other providers run their sync implementation in the default thread pool
    so they don't block the event loop.
    """
    from app.schemas import OcrProvider

    parsed = OcrProvider(provider_name)

    if parsed == OcrProvider.vllm_dots:
        from app.providers.vllm_dots import VllmDotsProvider

        pairs = _build_pairs(few_shots)
        impl = VllmDotsProvider(
            settings=settings,
            timeout_seconds=settings.vllm_request_timeout_seconds,
            model_id=model_id,
        )
        return await impl.async_transcribe(
            image_bytes=image_bytes,
            mime_type=mime_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            few_shots=pairs,
        )

    # All other providers: run sync in thread pool.
    loop = asyncio.get_running_loop()
    fn = functools.partial(
        transcribe_with_provider,
        provider_name,
        image_bytes=image_bytes,
        mime_type=mime_type,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        few_shots=few_shots,
        settings=settings,
        model_id=model_id,
        gemini_cost_session=gemini_cost_session,
        gemini_cost_page_index=gemini_cost_page_index,
        gemini_cost_page_in_source=gemini_cost_page_in_source,
        gemini_cost_source_file=gemini_cost_source_file,
    )
    return await loop.run_in_executor(None, fn)


def transcribe_image(
    image_bytes: bytes,
    mime_type: str,
    system_prompt: str,
    few_shot_examples: list[dict[str, str]],
    provider: str,
    *,
    user_prompt: str | None = None,
    settings: Settings | None = None,
    model_id: str | None = None,
) -> str:
    from app.config import get_settings

    return transcribe_with_provider(
        provider,
        image_bytes=image_bytes,
        mime_type=mime_type,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        few_shots=few_shot_examples,
        settings=settings or get_settings(),
        model_id=model_id,
    )
