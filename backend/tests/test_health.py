from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_providers_shape() -> None:
    r = client.get("/api/providers")
    assert r.status_code == 200
    data = r.json()
    assert "providers" in data
    assert len(data["providers"]) >= 4
    for row in data["providers"]:
        assert "default_model_id" in row
        assert "model_options" in row
        assert isinstance(row["model_options"], list)


def test_vllm_gemma_provider_present() -> None:
    r = client.get("/api/providers")
    rows = {p["id"]: p for p in r.json()["providers"]}
    assert "vllm_gemma" in rows
    assert rows["vllm_gemma"]["label"] == "Local — Gemma 4 (vLLM)"
    # Default in tests: VLLM_ENABLED=false → not configured
    assert rows["vllm_gemma"]["configured"] is False


def test_vllm_gemma_message_builder() -> None:
    from app.providers.vllm_gemma import _image_url_part, _text_part

    img = _image_url_part(b"\x89PNG", "image/png")
    assert img["type"] == "image_url"
    assert img["image_url"]["url"].startswith("data:image/png;base64,")
    assert _text_part("hi")["text"] == "hi"
