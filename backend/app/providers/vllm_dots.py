"""Local dots.ocr via vLLM OpenAI-compatible chat completions."""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import TYPE_CHECKING

import httpx
from PIL import Image

from app.providers.base import FewShotPair, OcrProviderBase
from app.providers.prompts import user_instructions_prefix
from app.schemas import OcrProvider
from app.utils.dots_ocr import (
    DOTS_IMAGE_TEXT_PREFIX,
    OCR_TASK_TEXT,
    dots_user_content_parts,
)
from app.utils.model_id import resolve_model_id_for_provider

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_CONTEXT_OVERFLOW_RE = re.compile(
    r"maximum context length is (\d+) tokens and your request has (\d+) input tokens",
    re.IGNORECASE,
)
_DECODER_PROMPT_RE = re.compile(
    r"decoder prompt(?: length)? (\d+).*?max(?:_model)?_len (\d+)",
    re.IGNORECASE,
)
_PREAMBLE_RE = re.compile(
    r"^(?:"
    r"(?:here\s+is\s+(?:the\s+)?(?:transcri(?:ption|bed)|the\s+text|the\s+extracted)[^\n]*\n)"
    r"|(?:the\s+extracted\s+text[^\n]*\n)"
    r")+",
    re.IGNORECASE,
)


def _vllm_root_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root.rstrip("/")


def check_vllm_reachable(settings: Settings) -> bool:
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


def _shrink_image(image_bytes: bytes, scale: float = 0.5) -> tuple[bytes, str]:
    """Return a JPEG-compressed image scaled down by ``scale`` (default 50%)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


def _extract_text_from_layout_json(raw: str) -> str | None:
    """If dots.ocr returns layout JSON, flatten Text/Title cells to plain lines."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        if isinstance(data, dict) and "elements" in data:
            data = data["elements"]
        else:
            return None
    lines: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            lines.append(text.strip())
    return "\n".join(lines) if lines else None


def _normalize_dots_output(text: str) -> str:
    text = _PREAMBLE_RE.sub("", text).strip()
    if not text:
        return text
    m = _JSON_BLOCK_RE.search(text)
    if m:
        extracted = _extract_text_from_layout_json(m.group(1).strip())
        if extracted:
            return extracted
    if text.startswith("{") or text.startswith("["):
        extracted = _extract_text_from_layout_json(text)
        if extracted:
            return extracted
    return text


def _few_shot_user_content(shot: FewShotPair) -> list[dict]:
    """Image first, then dots.ocr image tokens (``ocr_pipeline/vllm_client``)."""
    text = (
        f"{DOTS_IMAGE_TEXT_PREFIX}\n"
        "Few-shot example — match this orthographic style exactly.\n\n"
        f"{OCR_TASK_TEXT}"
    )
    return [
        _image_url_part(shot.image_bytes, shot.mime_type),
        _text_part(text),
    ]


def _build_messages(
    image_bytes: bytes,
    mime_type: str,
    system_prompt: str,
    few_shots: list[FewShotPair],
    user_prompt: str | None = None,
) -> list[dict]:
    messages: list[dict] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for shot in few_shots:
        messages.append({"role": "user", "content": _few_shot_user_content(shot)})
        messages.append({"role": "assistant", "content": shot.expected_text})
    extra = user_instructions_prefix(user_prompt).strip()
    messages.append(
        {
            "role": "user",
            "content": dots_user_content_parts(
                _image_url_part(image_bytes, mime_type),
                user_instructions=extra,
            ),
        }
    )
    return messages


class VllmDotsProvider(OcrProviderBase):
    """Local dots.ocr (rednote-hilab/dots.ocr) via vLLM."""

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
        resolved = resolve_model_id_for_provider(
            settings, OcrProvider.vllm_dots, model_id
        ) or settings.vllm_model
        if not resolved:
            raise ValueError("VLLM_MODEL is not set.")
        self._base_url = settings.vllm_base_url.rstrip("/")
        self._model = resolved
        self._api_key = settings.vllm_api_key or "EMPTY"
        self.timeout_seconds = timeout_seconds
        self._max_model_len = settings.vllm_max_model_len
        self._max_output_tokens = settings.vllm_effective_max_output_tokens()
        self._temperature = settings.vllm_temperature
        self._top_p = settings.vllm_top_p

    def _build_payload(
        self,
        image_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        few_shots: list[FewShotPair],
        user_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        messages = _build_messages(
            image_bytes, mime_type, system_prompt, few_shots, user_prompt=user_prompt
        )
        max_out = max_tokens if max_tokens is not None else self._max_output_tokens
        return {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": max_out,
            "max_completion_tokens": max_out,
        }

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _completions_url(self) -> str:
        return f"{self._base_url}/chat/completions"

    def _parse_context_overflow_max_tokens(self, error_body: str) -> int | None:
        m = _CONTEXT_OVERFLOW_RE.search(error_body)
        if not m:
            return None
        context_len = int(m.group(1))
        input_tokens = int(m.group(2))
        return max(128, context_len - input_tokens - 16)

    @staticmethod
    def _is_decoder_prompt_overflow(body: str) -> bool:
        return "decoder prompt" in body.lower()

    @staticmethod
    def _raise_input_overflow(max_model_len: int) -> None:
        raise RuntimeError(
            f"vLLM input overflow: the image + prompt exceeds the server context "
            f"({max_model_len} tokens). "
            "Lower VLLM_PDF_DPI (e.g. VLLM_PDF_DPI=150 in .env), "
            "or increase VLLM_MAX_MODEL_LEN to 8192+ if GPU memory allows."
        )

    def _single_request(
        self, client: httpx.Client, url: str, payload: dict, headers: dict
    ) -> tuple[dict | None, str]:
        """Post one request. Returns (json_data, error_body). error_body=='' on success."""
        response = client.post(url, json=payload, headers=headers)
        if response.status_code not in (200, 201):
            return None, response.text
        return response.json(), ""

    async def _single_request_async(
        self, client: httpx.AsyncClient, url: str, payload: dict, headers: dict
    ) -> tuple[dict | None, str]:
        response = await client.post(url, json=payload, headers=headers)
        if response.status_code not in (200, 201):
            return None, response.text
        return response.json(), ""

    def _post_with_retry(
        self,
        client: httpx.Client,
        url: str,
        image_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        few_shots: list[FewShotPair],
        user_prompt: str | None,
    ) -> dict:
        """Sync POST with output-token shrink and image downscale on overflow."""
        headers = self._auth_headers()
        img, mt = image_bytes, mime_type
        max_out = self._max_output_tokens
        shrink_attempt = 0

        for attempt in range(6):
            payload = self._build_payload(img, mt, system_prompt, few_shots, user_prompt, max_out)
            data, err_body = self._single_request(client, url, payload, headers)
            if data is not None:
                return data

            out_adjusted = self._parse_context_overflow_max_tokens(err_body)
            if out_adjusted is not None:
                logger.warning("vLLM max_tokens too large; reducing to %d", out_adjusted)
                max_out = out_adjusted
                continue

            if self._is_decoder_prompt_overflow(err_body):
                if shrink_attempt >= 2:
                    self._raise_input_overflow(self._max_model_len)
                scale = 0.5 ** (shrink_attempt + 1)
                shrink_attempt += 1
                logger.warning(
                    "vLLM decoder-prompt overflow (attempt %d); shrinking image to %.0f%%",
                    attempt + 1,
                    scale * 100,
                )
                img, mt = _shrink_image(image_bytes, scale)
                max_out = self._max_output_tokens
                continue

            raise RuntimeError(f"vLLM request failed: {err_body[:500]}")

        raise RuntimeError(f"vLLM request failed after retries: {err_body[:300]}")  # type: ignore[possibly-undefined]

    async def _post_with_retry_async(
        self,
        client: httpx.AsyncClient,
        url: str,
        image_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        few_shots: list[FewShotPair],
        user_prompt: str | None,
    ) -> dict:
        """Async POST with output-token shrink and image downscale on overflow."""
        headers = self._auth_headers()
        img, mt = image_bytes, mime_type
        max_out = self._max_output_tokens
        shrink_attempt = 0

        for attempt in range(6):
            payload = self._build_payload(img, mt, system_prompt, few_shots, user_prompt, max_out)
            data, err_body = await self._single_request_async(client, url, payload, headers)
            if data is not None:
                return data

            out_adjusted = self._parse_context_overflow_max_tokens(err_body)
            if out_adjusted is not None:
                logger.warning("vLLM max_tokens too large; reducing to %d", out_adjusted)
                max_out = out_adjusted
                continue

            if self._is_decoder_prompt_overflow(err_body):
                if shrink_attempt >= 2:
                    self._raise_input_overflow(self._max_model_len)
                scale = 0.5 ** (shrink_attempt + 1)
                shrink_attempt += 1
                logger.warning(
                    "vLLM decoder-prompt overflow (attempt %d); shrinking image to %.0f%%",
                    attempt + 1,
                    scale * 100,
                )
                img, mt = _shrink_image(image_bytes, scale)
                max_out = self._max_output_tokens
                continue

            raise RuntimeError(f"vLLM request failed: {err_body[:500]}")

        raise RuntimeError(f"vLLM request failed after retries: {err_body[:300]}")  # type: ignore[possibly-undefined]

    def _parse_response(self, data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("vLLM returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            text = "\n".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ).strip()
        elif isinstance(content, str):
            text = content.strip()
        else:
            text = ""
        text = _normalize_dots_output(text)
        if not text:
            raise RuntimeError("vLLM returned empty transcription")
        return text

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
            "dots.ocr vLLM transcribe (sync): model=%s image=%d bytes shots=%d",
            self._model,
            len(image_bytes),
            len(few_shots),
        )
        url = self._completions_url()
        try:
            with httpx.Client(timeout=httpx.Timeout(self.timeout_seconds, connect=30.0)) as client:
                data = self._post_with_retry(
                    client, url, image_bytes, mime_type, system_prompt, few_shots, user_prompt
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"vLLM request failed: {exc}") from exc
        text = self._parse_response(data)
        logger.info("dots.ocr vLLM response: chars=%d preview=%r", len(text), text[:80])
        return text

    async def async_transcribe(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        user_prompt: str | None = None,
        few_shots: list[FewShotPair],
    ) -> str:
        logger.info(
            "dots.ocr vLLM transcribe (async): model=%s image=%d bytes shots=%d",
            self._model,
            len(image_bytes),
            len(few_shots),
        )
        url = self._completions_url()
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds, connect=30.0)
            ) as client:
                data = await self._post_with_retry_async(
                    client, url, image_bytes, mime_type, system_prompt, few_shots, user_prompt
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"vLLM request failed: {exc}") from exc
        text = self._parse_response(data)
        logger.info("dots.ocr vLLM async response: chars=%d preview=%r", len(text), text[:80])
        return text
