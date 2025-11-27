from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from finantradealgo.api.server import create_app
from finantradealgo.live_trading.live_engine import LiveEngine
from finantradealgo.system.config_loader import LiveConfig


class DummyDataSource:
    def connect(self):
        return None

    def next_bar(self):
        return None

    def close(self):
        return None


class DummyStrategy:
    def init(self, df):
        return None

    def on_bar(self, *args, **kwargs):
        return None


class DummyRisk:
    pass


class DummyExec:
    def __init__(self, positions):
        self.positions = positions
        self.closed = []
        self.portfolio = type("P", (), {"initial_cash": 1000})()

    def get_open_positions(self):
        return list(self.positions)

    def close_position_market(self, pos):
        self.closed.append(pos)
        self.positions = []
        return pos

    def to_state_dict(self):
        return {
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "open_positions": self.positions,
        }

    def get_portfolio(self):
        return {"equity": 1000, "position": None, "cash": 1000}

    def mark_to_market(self, price, ts):
        return None

    def get_position(self):
        return None

    def submit_order(self, symbol, side, qty, order_type, **kwargs):
        return None

    def export_logs(self, timeframe: str):
        return {"equity": Path("equity.csv"), "trades": Path("trades.csv")}


def _make_system_cfg(live_cfg: LiveConfig | None = None) -> dict:
    cfg = live_cfg or LiveConfig()
    return {
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "live": {},
        "live_cfg": cfg,
    }


def test_flatten_all_removes_positions(tmp_path):
    exec_client = DummyExec(
        positions=[{"symbol": "AIAUSDT", "side": "long", "qty": 1.0, "entry_price": 10.0}]
    )
    system_cfg = _make_system_cfg()
    engine = LiveEngine(
        system_cfg=system_cfg,
        data_source=DummyDataSource(),
        strategy=DummyStrategy(),
        risk_engine=DummyRisk(),
        execution_client=exec_client,
        run_id="test",
    )
    engine.flatten_all()
    assert exec_client.positions == []
    assert len(exec_client.closed) == 1

    engine.requested_action = "stop"
    engine._apply_pending_action()
    assert engine.is_running is False


@pytest.fixture
def client(tmp_path, monkeypatch):
    live_dir = tmp_path / "outputs" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    state_path = live_dir / "live_state.json"

    def fake_load():
        return {
            "live": {
                "latest_state_path": str(state_path),
                "state_path": str(state_path),
                "paper": {"state_path": str(state_path)},
            }
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("finantradealgo.api.server.load_system_config", fake_load)
    return TestClient(create_app()), state_path


def test_live_status_and_control(client):
    test_client, state_path = client

    # status 404 when no snapshot
    resp = test_client.get("/api/live/status")
    assert resp.status_code == 404

    snapshot = {
        "run_id": "test_run",
        "symbol": "AIAUSDT",
        "timeframe": "15m",
        "strategy": "rule",
        "mode": "paper",
        "equity": 1000,
        "realized_pnl": 0,
        "unrealized_pnl": 0,
        "daily_realized_pnl": 0,
        "daily_unrealized_pnl": 0,
        "open_positions": [],
        "last_orders": [],
        "data_source": "replay",
        "stale_data_seconds": None,
        "ws_reconnect_count": 0,
        "timestamp": 0.0,
    }
    state_path.write_text(json.dumps(snapshot), encoding="utf-8")

    resp2 = test_client.get("/api/live/status")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["run_id"] == "test_run"
    assert data["equity"] == 1000

    bad = test_client.post("/api/live/control", json={"command": "foo"})
    assert bad.status_code == 400

    good = test_client.post("/api/live/control", json={"command": "flatten"})
    assert good.status_code == 200
    written = json.loads(state_path.read_text(encoding="utf-8"))
    assert written.get("requested_action") == "flatten"


def test_live_paths_from_config(tmp_path, monkeypatch):
    live_dir = tmp_path / "custom_live"
    latest_path = live_dir / "latest.json"
    run_state_path = live_dir / "run_state.json"
    live_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "live": {
            "state_dir": str(live_dir),
            "state_path": str(run_state_path),
            "latest_state_path": str(latest_path),
            "paper": {"state_path": str(live_dir / "paper_state.json")},
        }
    }

    def fake_load():
        return cfg

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("finantradealgo.api.server.load_system_config", fake_load)
    client = TestClient(create_app())

    run_snapshot = {
        "run_id": "abc",
        "symbol": "AIAUSDT",
        "timeframe": "15m",
        "strategy": "rule",
        "equity": 111,
        "realized_pnl": 0,
        "unrealized_pnl": 0,
        "daily_realized_pnl": 0,
        "open_positions": [],
    }
    run_state_path.write_text(json.dumps(run_snapshot), encoding="utf-8")
    latest_path.write_text(json.dumps(run_snapshot), encoding="utf-8")

    resp = client.get("/api/live/status")
    assert resp.status_code == 200
    assert resp.json()["equity"] == 111

    resp_run = client.get("/api/live/status", params={"run_id": "abc"})
    assert resp_run.status_code == 200
    assert resp_run.json()["run_id"] == "abc"

    client.post("/api/live/control", json={"command": "pause"})
    written = json.loads(latest_path.read_text(encoding="utf-8"))
    assert written.get("requested_action") == "pause"
