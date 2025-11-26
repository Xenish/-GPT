from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from finantradealgo.api.server import create_app


client = TestClient(create_app())
LIVE_DIR = Path("outputs") / "live"


def write_snapshot(run_id: str, filename: str) -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "symbol": "AIAUSDT",
        "timeframe": "15m",
        "strategy": "rule",
        "start_time": "2025-11-25T00:00:00Z",
        "last_bar_time": "2025-11-25T01:00:00Z",
        "equity": 10100.0,
        "realized_pnl": 100.0,
        "unrealized_pnl": 10.0,
        "daily_realized_pnl": 50.0,
        "open_positions": [],
        "risk_stats": {"blocked_entries": 0},
    }
    target = LIVE_DIR / filename
    target.write_text(__import__("json").dumps(payload), encoding="utf-8")


def test_live_status_default_snapshot(tmp_path):
    run_id = "test_live_run"
    write_snapshot(run_id, "live_state.json")
    resp = client.get("/api/live/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == run_id
    assert data["equity"] == 10100.0


def test_live_status_specific_run():
    run_id = "test_specific"
    write_snapshot(run_id, f"live_state_{run_id}.json")
    resp = client.get(f"/api/live/status?run_id={run_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["strategy"] == "rule"


def test_live_status_not_found():
    resp = client.get("/api/live/status?run_id=missing_run")
    assert resp.status_code == 404
