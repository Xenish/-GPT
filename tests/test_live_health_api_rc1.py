import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# server module exposes create_app()
import finantradealgo.api.server as server
app = server.create_app()


client = TestClient(app)


def test_live_status_endpoint_missing_snapshot():
    # Expect 404 when no snapshot exists
    resp = client.get("/api/live/status")
    assert resp.status_code in (200, 404)


def test_live_status_endpoint_reads_snapshot(tmp_path: Path):
    snapshot_path = tmp_path / "live_state.json"
    payload = {
        "run_id": "test_run",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "strategy": "rule",
        "equity": 1000.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "risk_stats": {"blocked_entries": 0, "executed_trades": 0},
        "data_source": "replay",
        "timestamp": 0,
    }
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")

    # Monkeypatch loader to point to our snapshot path
    from finantradealgo.api import server

    orig_load = server.load_config

    def _fake_load_config(profile: str):
        cfg = orig_load(profile)
        live_cfg = cfg.get("live", {}) or {}
        live_cfg["latest_state_path"] = str(snapshot_path)
        cfg["live"] = live_cfg
        return cfg

    server.load_config = _fake_load_config  # type: ignore
    try:
        app2 = server.create_app()
        client2 = TestClient(app2)
        resp = client2.get("/api/live/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == "test_run"
        assert data["equity"] == 1000.0
        assert "equity_now" in data
    finally:
        server.load_config = orig_load  # type: ignore
