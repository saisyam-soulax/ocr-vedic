from __future__ import annotations

import logging

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import Settings
from app.providers.base import FewShotPair, OcrProviderBase
from app.providers.bedrock_claude import _image_part, _text_part

logger = logging.getLogger(__name__)


class BedrockOpenMultimodalProvider(OcrProviderBase):
    """Open-weight / non-Claude multimodal model on Bedrock (e.g. Llama 3.2 Vision, Amazon Nova)."""

    def __init__(
        self,
        *,
        settings: Settings,
        timeout_seconds: int,
        model_id: str | None = None,
    ) -> None:
        resolved = model_id or settings.bedrock_ocr_model_id
        if not resolved:
            raise ValueError("BEDROCK_OCR_MODEL_ID is not set.")
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
        few_shots: list[FewShotPair],
    ) -> str:
        logger.info(
            "Bedrock OCR (open) transcribe: model=%s image_size=%d bytes few_shots=%d",
            self._model_id, len(image_bytes), len(few_shots),
        )

        messages: list[dict] = []
        for shot in few_shots:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        _text_part("Few-shot example page — match script and diacritics style."),
                        _image_part(shot.image_bytes, shot.mime_type),
                        _text_part("Transcription for the example:"),
                    ],
                }
            )
            messages.append(
                {"role": "assistant", "content": [_text_part(shot.expected_text)]}
            )

        messages.append(
            {
                "role": "user",
                "content": [
                    _text_part(
                        "Transcribe this page. Preserve Devanāgarī, diacritics, and svara notation. "
                        "Plain text only."
                    ),
                    _image_part(image_bytes, mime_type),
                ],
            }
        )

        kw: dict = {
            "modelId": self._model_id,
            "messages": messages,
            "inferenceConfig": {"maxTokens": 8192, "temperature": 0.1},
        }
        if system_prompt:
            kw["system"] = [{"text": system_prompt}]

        try:
            resp = self._client.converse(**kw)
        except ClientError:
            logger.exception(
                "Bedrock OCR model converse failed: model=%s — confirm model supports "
                "Converse API and BEDROCK_OCR_MODEL_ID is correct for region",
                self._model_id,
            )
            raise RuntimeError(
                "Bedrock multimodal model request failed. Confirm the model supports the "
                "Converse API and that BEDROCK_OCR_MODEL_ID is correct for your region."
            )
        except Exception:
            logger.exception("Bedrock OCR model unexpected error: model=%s", self._model_id)
            raise

        out = (resp.get("output") or {}).get("message") or {}
        parts = out.get("content") or []
        texts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
        if not texts:
            logger.error("Bedrock OCR model returned no text: model=%s response=%s", self._model_id, resp)
            raise RuntimeError("Bedrock OCR model returned an empty transcription.")

        result = "\n".join(texts).strip()
        logger.info(
            "Bedrock OCR (open) response received: model=%s chars=%d preview=%r",
            self._model_id, len(result), result[:200],
        )
        return result
