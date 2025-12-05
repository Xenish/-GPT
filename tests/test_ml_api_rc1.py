from __future__ import annotations

from fastapi.testclient import TestClient

from finantradealgo.api.server import create_app


def test_ml_targets_and_models_api_rc1(monkeypatch, tmp_path):
    app = create_app()
    client = TestClient(app)

    # Targets endpoint should always respond (based on config)
    resp = client.get("/api/ml/targets")
    assert resp.status_code == 200
    targets = resp.json()
    assert isinstance(targets, list)

    # Models endpoint: allow empty registry, but should 200 with list payload
    resp_models = client.get("/api/ml/models")
    assert resp_models.status_code == 200
    models = resp_models.json()
    assert isinstance(models, list)
