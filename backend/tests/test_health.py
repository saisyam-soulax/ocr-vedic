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
    assert len(data["providers"]) >= 3
    for row in data["providers"]:
        assert "default_model_id" in row
        assert "model_options" in row
        assert isinstance(row["model_options"], list)
