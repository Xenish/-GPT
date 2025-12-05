from __future__ import annotations

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from finantradealgo.api.server import create_app
from finantradealgo.system.config_loader import load_config


@pytest.fixture
def client():
    return TestClient(create_app())


def test_backtests_listing_and_trades(client: TestClient):
    resp = client.get("/api/backtests/AIAUSDT/15m")
    assert resp.status_code == 200
    runs = resp.json()
    assert isinstance(runs, list)
    if not runs:
        return
    run = runs[0]
    assert "run_id" in run
    run_id = run["run_id"]

    trades_resp = client.get(f"/api/trades/{run_id}")
    assert trades_resp.status_code == 200
    trades = trades_resp.json()
    assert isinstance(trades, list)


def test_trades_not_found_returns_404(client: TestClient):
    resp = client.get("/api/trades/nonexistent_run")
    assert resp.status_code == 404


def test_run_backtest_rule_ok(client: TestClient):
    resp = client.post(
        "/api/backtests/run",
        json={"symbol": "AIAUSDT", "timeframe": "15m", "strategy": "rule"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "run_id" in payload
    assert payload["strategy"] == "rule"
    assert payload["symbol"] == "AIAUSDT"
    assert isinstance(payload.get("metrics", {}), dict)
    assert payload.get("trade_count") is not None


def test_run_backtest_ml_without_model(monkeypatch, tmp_path):
    cfg = deepcopy(load_config("research"))
    ml_cfg = cfg.get("ml", {})
    ml_cfg.setdefault("persistence", {})
    ml_cfg["persistence"]["model_dir"] = str(tmp_path / "models_empty")
    cfg["ml"] = ml_cfg

    def _fake_load(profile="research"):
        return cfg

    monkeypatch.setattr("finantradealgo.api.server.load_config", _fake_load)
    app = create_app()
    client = TestClient(app)

    resp = client.post(
        "/api/backtests/run",
        json={"symbol": "AIAUSDT", "timeframe": "15m", "strategy": "ml"},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "No valid ML model found" in detail or "Feature mismatch" in detail


def test_run_backtest_rule_with_override(monkeypatch):
    calls = []

    def _fake_run_backtest_once(symbol, timeframe, strategy_name, cfg=None, strategy_params=None):
        calls.append(strategy_params)
        trade_count = 1
        if strategy_params and strategy_params.get("ms_trend_min") == 1.5:
            trade_count = 5
        return {
            "run_id": "mock_run",
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy_name,
            "metrics": {"cum_return": 0.1},
            "trade_count": trade_count,
        }

    monkeypatch.setattr("finantradealgo.api.server.run_backtest_once", _fake_run_backtest_once)
    client = TestClient(create_app())

    resp_default = client.post(
        "/api/backtests/run",
        json={"symbol": "AIAUSDT", "timeframe": "15m", "strategy": "rule"},
    )
    resp_override = client.post(
        "/api/backtests/run",
        json={
            "symbol": "AIAUSDT",
            "timeframe": "15m",
            "strategy": "rule",
            "strategy_params": {"ms_trend_min": 1.5},
        },
    )
    assert resp_default.status_code == 200
    assert resp_override.status_code == 200
    assert resp_default.json()["trade_count"] != resp_override.json()["trade_count"]
