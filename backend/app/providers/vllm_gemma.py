from __future__ import annotations

import base64
import logging
import re
from typing import TYPE_CHECKING

import httpx

from app.providers.base import FewShotPair, OcrProviderBase

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

_THOUGHT_RE = re.compile(
    r"<\|channel\>thought[\s\S]*?(?=<\|channel\>|$)", re.IGNORECASE
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


# Phrases the model sometimes outputs before starting the transcription.
# These are artefacts of the prompt's invocation / verification language leaking into output.
_PREAMBLE_RE = re.compile(
    r"^(?:"
    r"(?:here\s+is\s+(?:the\s+)?(?:transcri(?:ption|bed)|the\s+text)[^\n]*\n)"
    r"|(?:i\s+(?:will\s+now|am\s+(?:now\s+)?going\s+to)\s+transcri[^\n]*\n)"
    r"|(?:applying\s+(?:the\s+)?(?:full\s+)?(?:protocol|transcription)[^\n]*\n)"
    r"|(?:transcription\s+begin[s]?[^\n]*\n)"
    r"|(?:begin\s+transcription[^\n]*\n)"
    r"|(?:end\s+of\s+protocol[^\n]*\n)"
    r")+",
    re.IGNORECASE,
)


def _strip_gemma_artifacts(text: str) -> str:
    text = _THOUGHT_RE.sub("", text).strip()
    text = _PREAMBLE_RE.sub("", text).strip()
    return text


def _build_messages(
    image_bytes: bytes,
    mime_type: str,
    system_prompt: str,
    few_shots: list[FewShotPair],
) -> list[dict]:
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
                    "Execute the full transcription protocol on this page.\n"
                    "Follow the zone sequence (omit masthead; left column → right column → footer).\n"
                    "Capture every akṣara, svara mark (॑ ॒ ᳚ ᳛), anusvāra (ं), visarga (ः), "
                    "daṇḍas (। ॥), and wavy demarcator lines exactly as printed.\n"
                    "⚠ Output ONLY the transcription — no preamble, no checklist output, "
                    "no commentary, no 'Here is the transcription:' header.\n"
                    "Begin your response with the very first transcribed character."
                ),
                _image_url_part(image_bytes, mime_type),
            ],
        }
    )
    return messages


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

    def _payload(
        self,
        image_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        few_shots: list[FewShotPair],
    ) -> tuple[str, dict, dict]:
        messages = _build_messages(image_bytes, mime_type, system_prompt, few_shots)
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 8192,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        url = f"{self._base_url}/chat/completions"
        return url, payload, headers

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
        text = _strip_gemma_artifacts(text)
        if not text:
            raise RuntimeError("vLLM returned empty transcription")
        return text

    # ------------------------------------------------------------------
    # Sync path (used by non-async callers / thread-pool dispatch)
    # ------------------------------------------------------------------

    def transcribe(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        few_shots: list[FewShotPair],
    ) -> str:
        logger.info(
            "vLLM transcribe (sync): model=%s image=%d bytes shots=%d",
            self._model, len(image_bytes), len(few_shots),
        )
        url, payload, headers = self._payload(image_bytes, mime_type, system_prompt, few_shots)
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

        text = self._parse_response(data)
        logger.info("vLLM response: chars=%d preview=%r", len(text), text[:80])
        return text

    # ------------------------------------------------------------------
    # Async path (used by the streaming OCR pipeline — true concurrency)
    # ------------------------------------------------------------------

    async def async_transcribe(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        few_shots: list[FewShotPair],
    ) -> str:
        logger.info(
            "vLLM transcribe (async): model=%s image=%d bytes shots=%d",
            self._model, len(image_bytes), len(few_shots),
        )
        url, payload, headers = self._payload(image_bytes, mime_type, system_prompt, few_shots)
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds, connect=30.0)
            ) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500] if exc.response is not None else str(exc)
            raise RuntimeError(
                f"vLLM request failed ({exc.response.status_code if exc.response else '?'}): {body}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"vLLM request failed: {exc}") from exc

        text = self._parse_response(data)
        logger.info("vLLM async response: chars=%d preview=%r", len(text), text[:80])
        return text
