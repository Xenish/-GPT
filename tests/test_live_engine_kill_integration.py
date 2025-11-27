from __future__ import annotations

import pandas as pd

from finantradealgo.live_trading.live_engine import LiveEngine
from finantradealgo.system.kill_switch import KillSwitch, KillSwitchConfig
from finantradealgo.system.config_loader import LiveConfig
from tests.live_test_helpers import (
    ListDataSource,
    make_bar,
    AlwaysLongStrategy,
    AlwaysAllowRiskEngine,
    StaticExecutionClient,
)


def test_live_engine_stops_on_kill_switch():
    live_cfg = LiveConfig(symbol="TEST", symbols=["TEST"], timeframe="1m", data_source="replay")
    system_cfg = {"symbol": live_cfg.symbol, "timeframe": live_cfg.timeframe, "live_cfg": live_cfg}
    bars = [make_bar(0), make_bar(1)]
    data_source = ListDataSource(bars)
    execution_client = StaticExecutionClient(equity=1000.0)
    risk_engine = AlwaysAllowRiskEngine()
    strategy = AlwaysLongStrategy()
    kill_cfg = KillSwitchConfig(enabled=True, min_equity=5000.0)
    kill_switch = KillSwitch(kill_cfg)

    engine = LiveEngine(
        system_cfg=system_cfg,
        strategy=strategy,
        risk_engine=risk_engine,
        execution_client=execution_client,
        data_source=data_source,
        run_id="kill_test",
        kill_switch=kill_switch,
    )

    engine.run()

    assert engine.status == "STOPPED_BY_KILL_SWITCH"
    assert not execution_client.orders
