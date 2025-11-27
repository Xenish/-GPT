from __future__ import annotations

from fastapi.testclient import TestClient

from finantradealgo.api.server import create_app


client = TestClient(create_app())


def test_meta_endpoint():
    resp = client.get("/api/meta")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["symbols"], list)
    assert isinstance(data["timeframes"], list)
    assert isinstance(data["strategies"], list)
    assert isinstance(data.get("scenario_presets", []), list)
    assert len(data["symbols"]) > 0
    assert len(data["timeframes"]) > 0
    assert len(data["strategies"]) > 0
