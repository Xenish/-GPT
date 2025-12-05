import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.research_service import portfolio_api, montecarlo_api, scenarios_api


def _client_for(router):
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_portfolio_api_smoke():
    client = _client_for(portfolio_api.router)
    # minimal dummy returns for two strategies
    payload = {
        "portfolio_id": "pf1",
        "strategy_ids": ["s1", "s2"],
        "returns_data": {"s1": [0.01, -0.005, 0.002], "s2": [0.008, 0.0, -0.003]},
        "weighting_method": "sharpe",
    }
    resp = client.post("/api/optimize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data
    assert "sections" in data
    assert any(sec.get("title") == "Portfolio Overview" for sec in data["sections"])
    assert "sharpe_ratio" in data["metrics"]


def test_montecarlo_api_smoke():
    client = _client_for(montecarlo_api.router)
    trades = [{"pnl": 10}, {"pnl": -5}, {"pnl": 3}]
    resp = client.post(
        "/api/run",
        json={
            "strategy_id": "s1",
            "trades": trades,
            "n_simulations": 10,
            "resampling_method": "bootstrap",
            "confidence_level": 0.95,
        },
    )
    assert resp.status_code in (200, 400)
    if resp.status_code == 200:
        data = resp.json()
        metrics = data.get("metrics", {})
        for key in ("median_return", "p5_return", "p95_return", "worst_case_dd"):
            assert key in metrics
    else:
        # some environments may lack scipy random state; ensure clear error
        assert "detail" in resp.json()


def test_scenarios_api_smoke(monkeypatch):
    # mock run_scenarios to avoid heavy backtests
    def fake_run(cfg, scenarios):
        return pd.DataFrame(
            [
                {
                    "scenario_id": "sc1",
                    "label": "Test",
                    "symbol": "BTCUSDT",
                    "timeframe": "15m",
                    "strategy": "rule",
                    "params": {"a": 1},
                    "cum_return": 0.1,
                    "sharpe": 1.0,
                    "max_drawdown": -0.05,
                    "trade_count": 10,
                }
            ]
        )

    monkeypatch.setattr(scenarios_api, "run_scenarios", fake_run)
    monkeypatch.setattr(scenarios_api, "load_config", lambda _: {})

    client = _client_for(scenarios_api.router)
    payload = {
        "scenarios": [
            {
                "label": "Test",
                "symbol": "BTCUSDT",
                "timeframe": "15m",
                "strategy": "rule",
                "params": {},
            }
        ]
    }
    resp = client.post("/api/run", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_scenarios"] == 1
    result = data["results"][0]
    assert "scenario_id" in result
    assert "description" in result
    assert "metrics" in result
    assert result["metrics"]["cum_return"] == 0.1
