from __future__ import annotations

import threading
import time

import pandas as pd

from finantradealgo.data_engine.live_data_source import AbstractLiveDataSource, Bar
from finantradealgo.live_trading.live_engine import LiveEngine
from finantradealgo.system.config_loader import LiveConfig


class FlakySource(AbstractLiveDataSource):
    def __init__(self):
        self._connected = False
        self._calls = 0

    def connect(self) -> None:
        self._connected = True

    def next_bar(self) -> Bar | None:
        if not self._connected:
            raise RuntimeError("Source not connected")
        if self._calls == 0:
            self._calls += 1
            ts = pd.Timestamp.utcnow()
            return Bar(
                symbol="AIAUSDT",
                timeframe="1m",
                open_time=ts,
                close_time=ts,
                open=100.0,
                high=100.5,
                low=99.5,
                close=100.2,
                volume=5.0,
                extras={},
            )
        self._calls += 1
        time.sleep(0.01)
        return None

    def close(self) -> None:
        self._connected = False


class DummyStrategy:
    def on_bar(self, *_args, **_kwargs):
        return None


class DummyRisk:
    def can_open_new_trade(self, **_kwargs):
        return False

    def calc_position_size(self, **_kwargs):
        return 0.0


class DummyExec:
    def __init__(self):
        self.portfolio = type("Portfolio", (), {"initial_cash": 1000.0})()
        self._last_price = None
        self._last_timestamp = None

    def mark_to_market(self, price, ts):
        self._last_price = price
        self._last_timestamp = ts

    def get_portfolio(self):
        return {"equity": 1000.0}

    def get_position(self):
        return None

    def submit_order(self, **_kwargs):
        return {}

    def get_open_positions(self):
        return []

    def to_state_dict(self):
        return {"realized_pnl": 0.0, "unrealized_pnl": 0.0, "open_positions": []}

    def export_logs(self, **_kwargs):
        return {}


def test_live_engine_stale_alarm_triggers():
    cfg = LiveConfig(data_source="binance_ws", ws_max_stale_seconds=0)
    source = FlakySource()
    system_cfg = {
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "live": {},
        "live_cfg": cfg,
    }
    engine = LiveEngine(
        system_cfg=system_cfg,
        data_source=source,
        strategy=DummyStrategy(),
        risk_engine=DummyRisk(),
        execution_client=DummyExec(),
        strategy_name="dummy",
    )

    thread = threading.Thread(target=engine.run_loop, kwargs={"max_iterations": None})
    thread.start()
    time.sleep(0.15)
    engine.is_running = False
    thread.join(timeout=2.0)

    assert engine.stale_data_seconds is not None
    assert engine.ws_stale_alarm is True
