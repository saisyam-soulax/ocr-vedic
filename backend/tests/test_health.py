from fastapi.testclient import TestClient

from app.main import create_app


def test_health_ok() -> None:
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_providers_list() -> None:
    client = TestClient(create_app())
    r = client.get("/api/providers")
    assert r.status_code == 200
    rows = {p["id"]: p for p in r.json()["providers"]}
    assert "gemini" in rows
    assert "vllm_dots" in rows
    assert "vllm_gemma" not in rows


def test_vllm_dots_provider_label() -> None:
    client = TestClient(create_app())
    rows = {p["id"]: p for p in client.get("/api/providers").json()["providers"]}
    assert rows["vllm_dots"]["label"] == "Local — dots.ocr (vLLM)"


def test_vllm_dots_message_builder() -> None:
    from app.providers.vllm_dots import DOTS_OCR_BASE_PROMPT, _image_url_part, _text_part

    assert "Extract the text" in DOTS_OCR_BASE_PROMPT
    assert _text_part("hi") == {"type": "text", "text": "hi"}
    part = _image_url_part(b"\xff\xd8\xff", "image/jpeg")
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")
