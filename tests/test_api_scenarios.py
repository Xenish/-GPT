from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from finantradealgo.api.server import create_app


@pytest.fixture
def client(monkeypatch):
    def _fake_run_scenario_preset(cfg, preset_name: str):
        return pd.DataFrame(
            {
                "label": ["scenario_a"],
                "strategy": ["rule"],
                "cum_return": [0.1],
                "sharpe": [1.2],
                "trade_count": [5],
            }
        )

    monkeypatch.setattr("finantradealgo.api.server.run_scenario_preset", _fake_run_scenario_preset)
    return TestClient(create_app())


def test_run_scenarios_ok(client: TestClient):
    resp = client.post(
        "/api/scenarios/run",
        json={"symbol": "AIAUSDT", "timeframe": "15m", "preset_name": "core_15m"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["preset_name"] == "core_15m"
    assert isinstance(data["rows"], list)
    assert len(data["rows"]) > 0


def test_run_scenarios_not_found(monkeypatch):
    def _raise(cfg, preset_name: str):
        raise KeyError("missing")

    monkeypatch.setattr("finantradealgo.api.server.run_scenario_preset", _raise)
    client = TestClient(create_app())
    resp = client.post(
        "/api/scenarios/run",
        json={"symbol": "AIAUSDT", "timeframe": "15m", "preset_name": "missing"},
    )
    assert resp.status_code == 404
