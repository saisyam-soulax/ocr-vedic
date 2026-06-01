from app.config import Settings
from app.schemas import OcrProvider
from app.utils.model_id import resolve_model_id_for_provider


def test_vllm_rejects_gemini_model_id() -> None:
    s = Settings(
        vllm_model="model",
        vllm_model_options="model,rednote-hilab/dots.ocr",
    )
    assert (
        resolve_model_id_for_provider(
            s, OcrProvider.vllm_dots, "gemini-3.1-pro-preview"
        )
        == "model"
    )


def test_vllm_accepts_served_name() -> None:
    s = Settings(vllm_model="model", vllm_model_options="model")
    assert resolve_model_id_for_provider(s, OcrProvider.vllm_dots, "model") == "model"


def test_gemini_passes_through() -> None:
    s = Settings(gemini_model="gemini-2.5-flash")
    assert (
        resolve_model_id_for_provider(s, OcrProvider.gemini, "gemini-3.1-pro-preview")
        == "gemini-3.1-pro-preview"
    )
