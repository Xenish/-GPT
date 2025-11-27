from __future__ import annotations

import json
import time

import pandas as pd

from finantradealgo.live_trading.live_engine import LiveEngine
from finantradealgo.system.config_loader import LiveConfig
from finantradealgo.system.kill_switch import KillSwitchConfig, KillSwitch
from tests.live_test_helpers import (
    ListDataSource,
    make_bar,
    AlwaysLongStrategy,
    AlwaysAllowRiskEngine,
    StaticExecutionClient,
)


def test_update_heartbeat_writes_file(tmp_path):
    live_cfg = LiveConfig(symbol="TEST", symbols=["TEST"], timeframe="1m", data_source="replay")
    system_cfg = {"symbol": live_cfg.symbol, "timeframe": live_cfg.timeframe, "live_cfg": live_cfg}
    engine = LiveEngine(
        system_cfg=system_cfg,
        strategy=AlwaysLongStrategy(),
        risk_engine=AlwaysAllowRiskEngine(),
        execution_client=StaticExecutionClient(),
        data_source=ListDataSource([make_bar(0)]),
        run_id="hb_test",
        kill_switch=KillSwitch(KillSwitchConfig(enabled=False)),
    )
    heartbeat_path = tmp_path / "heartbeat.json"
    engine.heartbeat_path = heartbeat_path
    ts = pd.Timestamp.utcnow()
    engine._update_heartbeat(ts)
    assert heartbeat_path.is_file()
    payload = json.loads(heartbeat_path.read_text())
    updated_at = pd.Timestamp(payload["updated_at"])
    assert abs((pd.Timestamp.utcnow() - updated_at).total_seconds()) < 5
