import datetime as dt
from pathlib import Path

import pandas as pd

from finantradealgo.live_trading.live_engine import LiveEngine
from finantradealgo.core.strategy import BaseStrategy, StrategyContext
from finantradealgo.system.kill_switch import KillSwitch, KillSwitchConfig
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.execution.client_base import ExecutionClientBase
from finantradealgo.data_engine.live_data_source import FileReplayDataSource


class DummyExecutionClient(ExecutionClientBase):
    def __init__(self):
        super().__init__()
        self._portfolio = {"equity": 100.0, "cash": 100.0}
        self._position = None

    def mark_to_market(self, price: float, ts=None):
        return

    def get_portfolio(self):
        return self._portfolio

    def get_position(self):
        return self._position

    def get_open_positions(self):
        return []

    def close_position_market(self, pos):
        return

    def submit_order(self, symbol: str, side: str, qty: float, order_type: str = "MARKET", price: float | None = None, client_order_id: str | None = None, reduce_only: bool = False):
        return

    def cancel_order(self, *args, **kwargs):
        return

    def get_open_orders(self):
        return []


class LosingStrategy(BaseStrategy):
    def init(self, df: pd.DataFrame | None = None) -> None:
        return

    def on_bar(self, row, ctx: StrategyContext):
        # Force a losing trade by setting executed_trades and daily_realized_pnl manually via execution client mocks
        return {"action": "none"}


def test_kill_switch_triggers_and_blocks_orders(tmp_path: Path):
    # Build dummy data source with 5 bars
    ts = pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {"timestamp": ts, "open": [1, 1, 1, 1, 1], "high": [1, 1, 1, 1, 1], "low": [1, 1, 1, 1, 1], "close": [1, 1, 1, 1, 1], "volume": [1, 1, 1, 1, 1]}
    )
    data_source = FileReplayDataSource(df, symbol="BTCUSDT", timeframe="15m")

    # Kill switch with tiny daily PnL limit so it triggers immediately when evaluated
    ks_cfg = KillSwitchConfig(
        enabled=True,
        daily_realized_pnl_limit=-0.0001,
        max_equity_drawdown_pct=100.0,
        max_exceptions_per_hour=999,
        min_equity=0.0,
        evaluation_interval_bars=1,
    )
    kill_switch = KillSwitch(ks_cfg)

    engine = LiveEngine(
        system_cfg={"live_cfg": None},
        strategy=LosingStrategy(),
        risk_engine=RiskEngine(RiskConfig.from_dict({})),
        execution_client=DummyExecutionClient(),
        data_source=data_source,
        run_id="kill_test",
        kill_switch=kill_switch,
    )

    # Simulate negative PnL by setting daily_realized_pnl below limit before evaluating
    engine.daily_realized_pnl[ts[0].normalize()] = -1.0

    # Process a single bar; kill switch should trigger and stop further processing
    bar = data_source.next_bar()
    engine._on_bar(bar)

    assert engine.kill_switch_triggered_flag is True
    assert engine.kill_switch_triggered_reason is not None

    # Next bar should not process signals (status STOPPED_BY_KILL_SWITCH)
    bar2 = data_source.next_bar()
    engine._on_bar(bar2)
    assert engine.status == "STOPPED_BY_KILL_SWITCH"
