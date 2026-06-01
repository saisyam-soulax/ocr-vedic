from __future__ import annotations

import logging

import boto3
from botocore.config import Config

from app.config import Settings
from app.providers.base import FewShotPair, OcrProviderBase
from app.providers.prompts import user_instructions_prefix

logger = logging.getLogger(__name__)


def _mime_to_bedrock_format(mime_type: str) -> str:
    m = (mime_type or "image/png").lower().split(";")[0].strip()
    mapping = {
        "image/png": "png",
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/gif": "gif",
        "image/webp": "webp",
    }
    return mapping.get(m, "png")


def _image_part(image_bytes: bytes, mime_type: str) -> dict:
    fmt = _mime_to_bedrock_format(mime_type)
    return {"image": {"format": fmt, "source": {"bytes": image_bytes}}}


def _text_part(text: str) -> dict:
    return {"text": text}


def _model_disallows_temperature(model_id: str) -> bool:
    """Claude Opus 4.7+ (with adaptive thinking) rejects the `temperature` parameter."""
    mid = model_id.lower()
    return (
        "claude-opus-4-7" in mid
        or "claude-opus-5" in mid
    )


class BedrockClaudeProvider(OcrProviderBase):
    def __init__(
        self,
        *,
        settings: Settings,
        timeout_seconds: int,
        model_id: str | None = None,
    ) -> None:
        resolved = model_id or settings.bedrock_claude_model_id
        if not resolved:
            raise ValueError("BEDROCK_CLAUDE_MODEL_ID is not set.")
        if not settings.aws_region:
            raise ValueError("AWS_REGION is not set for Bedrock.")
        self._model_id = resolved
        self.timeout_seconds = timeout_seconds
        cfg = Config(
            read_timeout=timeout_seconds,
            connect_timeout=min(30, timeout_seconds),
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        session_kw: dict = {"region_name": settings.aws_region}
        if settings.aws_profile:
            session_kw["profile_name"] = settings.aws_profile
        session = boto3.Session(**session_kw)
        self._client = session.client("bedrock-runtime", config=cfg)

    def transcribe(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        user_prompt: str | None = None,
        few_shots: list[FewShotPair],
    ) -> str:
        logger.info(
            "Bedrock Claude transcribe: model=%s image_size=%d bytes few_shots=%d",
            self._model_id, len(image_bytes), len(few_shots),
        )

        messages: list[dict] = []
        for shot in few_shots:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        _text_part(
                            "Few-shot example — transcribe in this exact orthographic style."
                        ),
                        _image_part(shot.image_bytes, shot.mime_type),
                        _text_part("Provide the transcription for the example image."),
                    ],
                }
            )
            messages.append(
                {"role": "assistant", "content": [_text_part(shot.expected_text)]}
            )

        prefix = user_instructions_prefix(user_prompt)
        messages.append(
            {
                "role": "user",
                "content": [
                    _text_part(
                        prefix
                        + "Transcribe this manuscript page. Preserve Devanāgarī, Latin diacritics, "
                        "and all Vedic accent/svara marks. Plain text only, no commentary."
                    ),
                    _image_part(image_bytes, mime_type),
                ],
            }
        )

        inference_config: dict = {"maxTokens": 8192}
        if not _model_disallows_temperature(self._model_id):
            inference_config["temperature"] = 0.1
        kw: dict = {
            "modelId": self._model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
        }
        if system_prompt:
            kw["system"] = [{"text": system_prompt}]

        try:
            resp = self._client.converse(**kw)
        except Exception:
            logger.exception("Bedrock Claude converse failed: model=%s", self._model_id)
            raise

        out = (resp.get("output") or {}).get("message") or {}
        parts = out.get("content") or []
        texts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
        if not texts:
            logger.error("Bedrock Claude returned no text: model=%s response=%s", self._model_id, resp)
            raise RuntimeError("Bedrock Claude returned an empty transcription.")

        result = "\n".join(texts).strip()
        logger.info(
            "Bedrock Claude response received: model=%s chars=%d preview=%r",
            self._model_id, len(result), result[:200],
        )
        return result
