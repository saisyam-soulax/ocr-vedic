from __future__ import annotations

import base64
import logging
import re

import httpx

from app.config import Settings
from app.providers.base import FewShotPair, OcrProviderBase

logger = logging.getLogger(__name__)

_THOUGHT_MARKERS = re.compile(
    r"<\|channel\>thought[\s\S]*?(?=<\|channel\>|$)", re.IGNORECASE
)


def _vllm_root_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root.rstrip("/")


def check_vllm_reachable(settings: Settings) -> bool:
    """Return True if the vLLM OpenAI server responds to /health."""
    if not settings.vllm_enabled or not settings.vllm_base_url:
        return False
    root = _vllm_root_url(settings.vllm_base_url)
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{root}/health")
            return r.status_code == 200
    except httpx.HTTPError:
        return False


def _text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def _image_url_part(image_bytes: bytes, mime_type: str) -> dict:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
    }


def _strip_gemma_artifacts(text: str) -> str:
    cleaned = _THOUGHT_MARKERS.sub("", text)
    return cleaned.strip()


class VllmGemmaProvider(OcrProviderBase):
    """Local Gemma 4 vision OCR via vLLM OpenAI-compatible chat completions."""

    def __init__(
        self,
        *,
        settings: Settings,
        timeout_seconds: int,
        model_id: str | None = None,
    ) -> None:
        if not settings.vllm_enabled:
            raise ValueError("VLLM_ENABLED is not true.")
        if not settings.vllm_base_url:
            raise ValueError("VLLM_BASE_URL is not set.")
        resolved = model_id or settings.vllm_model
        if not resolved:
            raise ValueError("VLLM_MODEL is not set.")
        self._base_url = settings.vllm_base_url.rstrip("/")
        self._model = resolved
        self._api_key = settings.vllm_api_key or "EMPTY"
        self.timeout_seconds = timeout_seconds
        self._settings = settings

    def transcribe(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        few_shots: list[FewShotPair],
    ) -> str:
        logger.info(
            "vLLM Gemma transcribe: model=%s image_size=%d bytes few_shots=%d",
            self._model,
            len(image_bytes),
            len(few_shots),
        )

        messages: list[dict] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        for shot in few_shots:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        _text_part(
                            "Few-shot example — manuscript snippet and its exact transcription "
                            "(match this style, spacing, and diacritics):"
                        ),
                        _image_url_part(shot.image_bytes, shot.mime_type),
                        _text_part("Expected transcription:"),
                    ],
                }
            )
            messages.append({"role": "assistant", "content": shot.expected_text})

        messages.append(
            {
                "role": "user",
                "content": [
                    _text_part(
                        "Transcribe the following image. Preserve Devanāgarī, IAST diacritics, "
                        "anusvāra/visarga, all svaras (Udātta, Anudātta, Svarita, kampas, etc.), "
                        "and layout line breaks. Output plain transcription text only — no commentary."
                    ),
                    _image_url_part(image_bytes, mime_type),
                ],
            }
        )

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 8192,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        url = f"{self._base_url}/chat/completions"

        try:
            with httpx.Client(
                timeout=httpx.Timeout(self.timeout_seconds, connect=30.0)
            ) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500] if exc.response is not None else str(exc)
            raise RuntimeError(
                f"vLLM request failed ({exc.response.status_code if exc.response else '?'}): {body}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"vLLM request failed: {exc}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("vLLM returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            text = "\n".join(parts).strip()
        elif isinstance(content, str):
            text = content.strip()
        else:
            text = ""

        text = _strip_gemma_artifacts(text)
        if not text:
            raise RuntimeError("vLLM returned empty transcription")

        logger.info(
            "vLLM Gemma response received: model=%s chars=%d preview=%r",
            self._model,
            len(text),
            text[:120],
        )
        return text
