from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from finantradealgo.api.server import create_app


def _write_dummy_portfolio(tmp_base):
    out_bt = tmp_base / "outputs" / "backtests"
    out_bt.mkdir(parents=True, exist_ok=True)
    path = out_bt / "portfolio_15m_test_equity.csv"
    df = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=10, freq="h"),
            "portfolio_equity": np.linspace(1000, 1100, 10),
        }
    )
    df.to_csv(path, index=False)
    trades_dir = tmp_base / "outputs" / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h"),
            "symbol": ["AIAUSDT", "BTCUSDT"],
            "pnl": [1.0, -0.5],
        }
    )
    trades_df.to_csv(trades_dir / "portfolio_15m_test_trades.csv", index=False)


def test_portfolio_backtests_and_equity(tmp_path, monkeypatch):
    _write_dummy_portfolio(tmp_path)

    # monkeypatch Path resolution inside server to use tmp_path
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("finantradealgo.api.server.load_config", lambda profile="research": {})
    client = TestClient(create_app())

    resp = client.get("/api/portfolio/backtests")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    item = data[0]
    assert item["run_id"].startswith("portfolio_15m_test")
    assert item["metrics"]["final_equity"] is not None

    run_id = item["run_id"]
    resp_eq = client.get(f"/api/portfolio/equity/{run_id}")
    assert resp_eq.status_code == 200
    points = resp_eq.json()
    assert len(points) == 10
