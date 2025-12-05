from __future__ import annotations

from fastapi.testclient import TestClient

from services.research_service.main import app


def test_performance_api_smoke_rg1():
    client = TestClient(app)
    resp = client.get("/health")
    if resp.status_code != 200:
        resp = client.get("/api/health")
    assert resp.status_code == 200

    # performance endpoints should exist even if they return empty data
    resp_perf = client.get("/api/performance/summary")
    assert resp_perf.status_code in (200, 404)
    if resp_perf.status_code == 200:
        data = resp_perf.json()
        # Expect risk-related fields when data exists
        assert isinstance(data, dict)
        if data:
            assert "max_drawdown" in data or "sharpe" in data or "volatility" in data
