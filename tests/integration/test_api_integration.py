import pytest
from fastapi.testclient import TestClient

from finantradealgo.api.server import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    return TestClient(app)


@pytest.mark.integration
def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"


@pytest.mark.slow
@pytest.mark.integration
def test_backtest_api_small_request(monkeypatch):
    calls = {}

    def fake_run_backtest_once(symbol, timeframe, strategy_name, cfg=None, strategy_params=None):
        calls["called"] = True
        return {
            "run_id": "test_run_id",
            "metrics": {"cum_return": 0.05, "max_drawdown": -0.02},
            "trade_count": 2,
        }

    monkeypatch.setattr("finantradealgo.api.server.run_backtest_once", fake_run_backtest_once)
    app = create_app()
    client = TestClient(app)

    payload = {
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "strategy": "rule",
        "strategy_params": {"atr_mult": 1.0},
    }

    resp = client.post("/api/backtests/run", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert calls.get("called") is True
    assert data["run_id"] == "test_run_id"
    assert data["strategy"] == payload["strategy"]
    assert data["trade_count"] == 2
    assert "metrics" in data
    assert set(["cum_return", "max_drawdown"]).issubset(data["metrics"].keys())
