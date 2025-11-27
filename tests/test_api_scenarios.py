from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from finantradealgo.api import server as server_module
from finantradealgo.api.server import create_app


def _build_client(monkeypatch, tmp_path: Path) -> TestClient:
    out_dir = tmp_path / "outputs" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(server_module, "SCENARIO_GRID_DIR", out_dir)
    return TestClient(create_app()), out_dir


def test_list_scenarios_reads_latest_csv(tmp_path, monkeypatch):
    client, out_dir = _build_client(monkeypatch, tmp_path)
    csv_path = out_dir / "scenario_grid_AIAUSDT_15m.csv"
    df = pd.DataFrame(
        [
            {
                "scenario_id": "rule-0",
                "label": "rule_tp2",
                "symbol": "AIAUSDT",
                "timeframe": "15m",
                "strategy": "rule",
                "params_json": json.dumps({"tp_atr_mult": 2.0}),
                "cum_return": 1.5,
                "sharpe": 1.2,
                "max_drawdown": -0.3,
                "trade_count": 42,
            }
        ]
    )
    df.to_csv(csv_path, index=False)

    resp = client.get("/api/scenarios/AIAUSDT/15m")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    item = data[0]
    assert item["scenario_id"] == "rule-0"
    assert item["strategy"] == "rule"
    assert item["params"]["tp_atr_mult"] == 2.0
    assert item["metrics"]["cum_return"] == 1.5


def test_list_scenarios_missing_file(tmp_path, monkeypatch):
    client, _ = _build_client(monkeypatch, tmp_path)
    resp = client.get("/api/scenarios/AIAUSDT/15m")
    assert resp.status_code == 404
