"""Resolve per-provider model IDs submitted from the UI."""
from __future__ import annotations

import logging

from app.config import Settings
from app.schemas import OcrProvider

logger = logging.getLogger(__name__)


def resolve_model_id_for_provider(
    settings: Settings,
    provider: OcrProvider,
    model_id: str | None,
) -> str | None:
    """Return a model id safe for the given provider.

    For vLLM, only served names from VLLM_MODEL / VLLM_MODEL_OPTIONS are accepted.
    A Gemini or Bedrock id left in the UI model field is ignored (logged) and replaced
    with the vLLM default — this matches ``--served-model-name`` in docker-compose.
    """
    if not model_id:
        return None

    if provider == OcrProvider.vllm_dots:
        default, options = settings.vllm_models_for_providers()
        allowed = set(options)
        if model_id in allowed:
            return model_id
        logger.warning(
            "Ignoring model_id=%r for vllm_dots (allowed: %s); using %r",
            model_id,
            ", ".join(sorted(allowed)) or "(none)",
            default,
        )
        return default

    return model_id
