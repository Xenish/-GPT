from __future__ import annotations

from fastapi.testclient import TestClient

from finantradealgo.api.server import create_app


client = TestClient(create_app())


def test_backtests_listing_and_trades():
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


def test_trades_not_found_returns_404():
    resp = client.get("/api/trades/nonexistent_run")
    assert resp.status_code == 404


def test_run_backtest_endpoint():
    resp = client.post(
        "/api/backtests/run",
        json={"symbol": "AIAUSDT", "timeframe": "15m", "strategy": "rule"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "run_id" in payload
    assert payload["strategy"] == "rule"
    assert payload["symbol"] == "AIAUSDT"
