from __future__ import annotations

import logging

from google import genai
from google.genai import types

from app.config import Settings
from app.providers.base import FewShotPair, OcrProviderBase

logger = logging.getLogger(__name__)


class GeminiProvider(OcrProviderBase):
    def __init__(
        self,
        *,
        settings: Settings,
        timeout_seconds: int,
        model_id: str | None = None,
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

    def transcribe(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        system_prompt: str,
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

        contents.append(
            "Transcribe the following image. Preserve Devanāgarī, IAST diacritics, "
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
