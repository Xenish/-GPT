from __future__ import annotations

import pandas as pd

from finantradealgo.live_trading.live_engine import LiveEngine
from finantradealgo.core.strategy import BaseStrategy, StrategyContext, SignalType
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.data_engine.live_data_source import AbstractLiveDataSource, Bar
from finantradealgo.execution.client_base import ExecutionClientBase


class CountingExecutionClient(ExecutionClientBase):
    def __init__(self):
        super().__init__()
        self.submitted = []
        self._portfolio = {"equity": 1000.0, "cash": 1000.0}
        self._position = None
        self._open_orders = []

    def mark_to_market(self, price: float, ts=None):
        return

    def get_portfolio(self):
        return self._portfolio

    def get_position(self):
        return self._position

    def get_open_positions(self):
        return [self._position] if self._position else []

    def get_open_orders(self):
        return self._open_orders

    def submit_order(self, symbol: str, side: str, qty: float, order_type: str = "MARKET", price: float | None = None, client_order_id: str | None = None, reduce_only: bool = False):
        self.submitted.append((symbol, side, qty, order_type))
        self._open_orders.append({"symbol": symbol, "side": side, "qty": qty})
        # Simulate fill by updating position
        self._position = {"symbol": symbol, "side": side, "qty": qty}
        return {"order_id": f"mock_{len(self.submitted)}"}

    def close_position_market(self, pos):
        self._position = None
        return True

    def cancel_order(self, *args, **kwargs):
        return


class DeterministicStrategy(BaseStrategy):
    def __init__(self, open_every_n: int = 3):
        self.open_every_n = open_every_n
        self._counter = 0

    def init(self, df: pd.DataFrame) -> None:
        self._counter = 0

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        self._counter += 1
        # Open on every nth bar, close on the next bar
        if ctx.position is None and self._counter % self.open_every_n == 0:
            return "LONG"
        if ctx.position is not None:
            return "CLOSE"
        return None


class DummyReplaySource(AbstractLiveDataSource):
    def __init__(self, bars):
        self._bars = list(bars)
        self._idx = 0

    def connect(self):
        return None

    def close(self):
        return None

    def next_bar(self, timeout: float | None = None):
        if self._idx >= len(self._bars):
            return None
        bar = self._bars[self._idx]
        self._idx += 1
        return bar


def _make_bars(n=10):
    ts = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    for t in ts:
        yield Bar(
            symbol="TESTUSDT",
            timeframe="1m",
            open_time=t,
            close_time=t,
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.1,
            volume=10.0,
            extras={},
        )


def test_live_engine_replay_loop_contract():
    bars = list(_make_bars(12))
    source = DummyReplaySource(bars)
    exec_client = CountingExecutionClient()
    strat = DeterministicStrategy(open_every_n=3)
    risk_engine = RiskEngine(RiskConfig(max_notional_per_symbol=1000.0))

    engine = LiveEngine(
        system_cfg={"live_cfg": None},
        strategy=strat,
        risk_engine=risk_engine,
        execution_client=exec_client,
        data_source=source,
        run_id="contract_test",
        kill_switch=None,
    )

    # Process all bars
    while True:
        bar = source.next_bar()
        if bar is None:
            break
        engine._on_bar(bar)

    # Expect roughly floor(12/3)=4 opens; closes follow next tick after open
    assert len(exec_client.submitted) >= 3, f"Expected some orders, got {len(exec_client.submitted)}"
    # Engine should have iterated over bars
    assert engine.iteration >= len(bars)
