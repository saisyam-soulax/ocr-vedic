from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from app.config import Settings
from app.providers.base import FewShotPair, OcrProviderBase
from app.providers.prompts import user_instructions_prefix

if TYPE_CHECKING:
    from app.cost.gemini_ledger import GeminiCostSession

logger = logging.getLogger(__name__)


def _usage_from_response(response: object) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }
    prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
    output = int(getattr(usage, "candidates_token_count", 0) or 0)
    cached = int(getattr(usage, "cached_content_token_count", 0) or 0)
    total = getattr(usage, "total_token_count", None)
    if total is None:
        total = prompt + output
    return {
        "input_tokens": prompt,
        "output_tokens": output,
        "cached_tokens": cached,
        "total_tokens": int(total),
    }


class GeminiProvider(OcrProviderBase):
    def __init__(
        self,
        *,
        settings: Settings,
        timeout_seconds: int,
        model_id: str | None = None,
        cost_session: GeminiCostSession | None = None,
        cost_page_index: int | None = None,
        cost_page_in_source: int | None = None,
        cost_source_file: str | None = None,
        cost_step: str = "ocr",
    ) -> None:
        if settings.gemini_use_vertexai:
            # Vertex AI accepts EITHER an Agentic Platform / Express API key OR
            # project + ADC, but not both — the SDK rejects mixing them. Pick the
            # mode that matches what the user supplied.
            if settings.google_api_key:
                self._client = genai.Client(
                    vertexai=True,
                    api_key=settings.google_api_key,
                )
            elif settings.google_cloud_project and settings.google_cloud_location:
                self._client = genai.Client(
                    vertexai=True,
                    project=settings.google_cloud_project,
                    location=settings.google_cloud_location,
                )
            else:
                raise RuntimeError(
                    "Gemini is configured for Vertex AI (GEMINI_USE_VERTEXAI=true) "
                    "but no auth was provided. Either set GOOGLE_API_KEY to a "
                    "Vertex AI / Agentic Platform key, or set GOOGLE_CLOUD_PROJECT "
                    "and GOOGLE_CLOUD_LOCATION and authenticate with ADC "
                    "(`gcloud auth application-default login`)."
                )
        elif settings.google_api_key:
            self._client = genai.Client(api_key=settings.google_api_key)
        else:
            # Fall back to env vars (GOOGLE_API_KEY / GEMINI_API_KEY) or ADC.
            self._client = genai.Client()
        self._model_name = model_id or settings.gemini_model
        self.timeout_seconds = timeout_seconds
        self._settings = settings
        self._cost_session = cost_session
        self._cost_page_index = cost_page_index
        self._cost_page_in_source = cost_page_in_source
        self._cost_source_file = cost_source_file or ""
        self._cost_step = cost_step

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
            "Gemini transcribe: model=%s image_size=%d bytes few_shots=%d",
            self._model_name, len(image_bytes), len(few_shots),
        )

        contents: list[object] = []
        for shot in few_shots:
            contents.append(
                "Few-shot example — manuscript snippet and its exact transcription "
                "(match this style, spacing, and diacritics):"
            )
            contents.append(
                types.Part.from_bytes(data=shot.image_bytes, mime_type=shot.mime_type)
            )
            contents.append(f"Expected transcription:\n{shot.expected_text}")

        prefix = user_instructions_prefix(user_prompt)
        contents.append(
            prefix
            + "Transcribe the following image. Preserve Devanāgarī, IAST diacritics, "
            "anusvāra/visarga, all svaras (Udātta, Anudātta, Svarita, kampas, etc.), "
            "and punctuation exactly as printed or implied by the scan. "
            "Output plain text only, no commentary."
        )
        contents.append(
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        )

        config = types.GenerateContentConfig(
            system_instruction=system_prompt or None,
        )

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=config,
            )
        except Exception:
            logger.exception("Gemini API call failed: model=%s", self._model_name)
            raise

        usage = _usage_from_response(response)
        if self._cost_session is not None and self._cost_page_index is not None:
            self._cost_session.record_call(
                page_index=self._cost_page_index,
                page_in_source=self._cost_page_in_source,
                source_file=self._cost_source_file,
                step=self._cost_step,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cached_tokens=usage["cached_tokens"],
                total_tokens=usage["total_tokens"],
            )
            logger.info(
                "Gemini cost recorded: job=%s page=%s in=%d out=%d",
                self._cost_session.job_id,
                self._cost_page_index,
                usage["input_tokens"],
                usage["output_tokens"],
            )
        elif usage["input_tokens"] or usage["output_tokens"]:
            logger.debug(
                "Gemini tokens (no cost session): model=%s in=%d out=%d",
                self._model_name,
                usage["input_tokens"],
                usage["output_tokens"],
            )

        text = getattr(response, "text", None)
        if text:
            result = text.strip()
            logger.info(
                "Gemini response received: model=%s chars=%d preview=%r",
                self._model_name, len(result), result[:200],
            )
            return result

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            cand = candidates[0]
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                chunks = [p.text for p in parts if getattr(p, "text", None)]
                if chunks:
                    result = "\n".join(chunks).strip()
                    logger.info(
                        "Gemini response received (from candidates): model=%s chars=%d preview=%r",
                        self._model_name, len(result), result[:200],
                    )
                    return result

        logger.warning(
            "Gemini returned no text: model=%s finish_reason may be SAFETY",
            self._model_name,
        )
        raise RuntimeError(
            "Gemini returned an empty response. Check safety filters or image content."
        )
