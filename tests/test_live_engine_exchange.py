from __future__ import annotations

import pandas as pd

from finantradealgo.core.strategy import BaseStrategy, StrategyContext
from finantradealgo.data_engine.live_data_source import AbstractLiveDataSource, Bar
from finantradealgo.execution.client_base import ExecutionClientBase
from finantradealgo.execution.execution_client import ExchangeRiskLimitError
from finantradealgo.live_trading.live_engine import LiveEngine
from finantradealgo.system.config_loader import LiveConfig
from finantradealgo.system.kill_switch import KillSwitch, KillSwitchConfig


def _build_system_cfg(mode: str = "exchange") -> dict:
    live_cfg = LiveConfig(
        mode=mode,
        symbol="TESTUSDT",
        symbols=["TESTUSDT"],
        timeframe="1m",
        data_source="replay",
    )
    return {
        "symbol": live_cfg.symbol,
        "timeframe": live_cfg.timeframe,
        "live": {},
        "live_cfg": live_cfg,
    }


def _make_bar(idx: int) -> Bar:
    ts = pd.Timestamp("2025-01-01 00:00:00+00:00") + pd.Timedelta(minutes=idx)
    return Bar(
        symbol="TESTUSDT",
        timeframe="1m",
        open_time=ts,
        close_time=ts,
        open=100 + idx,
        high=100 + idx + 1,
        low=100 + idx - 1,
        close=100 + idx,
        volume=1.0,
        extras={"timestamp": ts},
    )


class DummyDataSource(AbstractLiveDataSource):
    def __init__(self, bars):
        self._bars = iter(bars)

    def connect(self):
        return None

    def next_bar(self):
        return next(self._bars, None)

    def close(self):
        return None


class FakeExecutionClient(ExecutionClientBase):
    def __init__(self):
        self.orders: list[dict] = []
        self._position: dict | None = None
        self._last_price = None
        self._portfolio = {"equity": 1000.0}

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        *,
        price: float | None = None,
        reduce_only: bool = False,
        client_order_id: str | None = None,
        **_: dict,
    ) -> dict:
        price = price or self._last_price or 100.0
        order = {
            "symbol": symbol,
            "side": side.upper(),
            "qty": float(qty),
            "price": float(price),
            "order_type": order_type,
            "client_order_id": client_order_id,
        }
        self.orders.append(order)
        if reduce_only:
            self._position = None
        else:
            side_dir = "LONG" if side.upper() == "BUY" else "SHORT"
            self._position = {
                "symbol": symbol,
                "side": side_dir,
                "qty": float(qty),
                "entry_price": float(price),
                "pnl": 0.0,
            }
        return order

    def cancel_order(self, symbol: str, order_id: str | int) -> None:
        return None

    def get_open_positions(self):
        return [] if self._position is None else [self._position]

    def get_open_orders(self, symbol: str | None = None):
        return []

    def get_position(self):
        return self._position

    def mark_to_market(self, price: float, timestamp):
        self._last_price = price

    def get_portfolio(self):
        return self._portfolio

    def export_logs(self, timeframe: str):
        return {}

    def to_state_dict(self):
        return {"open_positions": self.get_open_positions()}

    def close(self):
        return None


class AllowRiskEngine:
    def __init__(self, size: float = 1.0):
        self.size = size

    def can_open_new_trade(self, **_kwargs):
        return True

    def calc_position_size(self, **_kwargs):
        return self.size


class BlockingRiskEngine:
    def can_open_new_trade(self, **_kwargs):
        return False

    def calc_position_size(self, **_kwargs):
        return 0.0


class AlwaysLongStrategy(BaseStrategy):
    def init(self, df):
        self._opened = False

    def on_bar(self, row, ctx: StrategyContext):
        if not self._opened:
            self._opened = True
            return "LONG"
        if ctx.position is not None:
            return "CLOSE"
        return None


class LimitExecutionClient(ExecutionClientBase):
    def __init__(self):
        self._portfolio = {"equity": 1000.0}

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        *,
        price: float | None = None,
        reduce_only: bool = False,
        client_order_id: str | None = None,
        **_: dict,
    ) -> dict:
        raise ExchangeRiskLimitError("max notional exceeded")

    def cancel_order(self, symbol: str, order_id: str | int) -> None:
        return None

    def get_open_positions(self):
        return []

    def get_open_orders(self, symbol: str | None = None):
        return []

    def get_position(self):
        return None

    def mark_to_market(self, price: float, timestamp):
        return None

    def get_portfolio(self):
        return self._portfolio

    def export_logs(self, timeframe: str):
        return {}

    def to_state_dict(self):
        return {"open_positions": []}

    def close(self):
        return None


class DummyNotifier:
    def __init__(self):
        self.warn_calls: list[str] = []

    def info(self, msg: str) -> None:
        return None

    def warn(self, msg: str) -> None:
        self.warn_calls.append(msg)

    def critical(self, msg: str) -> None:
        return None


def test_live_engine_exchange_executes_orders(monkeypatch):
    system_cfg = _build_system_cfg()
    ds = DummyDataSource([_make_bar(0), _make_bar(1)])
    exec_client = FakeExecutionClient()
    risk_engine = AllowRiskEngine(size=0.5)
    strategy = AlwaysLongStrategy()
    strategy.init(pd.DataFrame())

    engine = LiveEngine(
        system_cfg=system_cfg,
        strategy=strategy,
        risk_engine=risk_engine,
        execution_client=exec_client,
        data_source=ds,
        run_id="test_run",
    )

    engine.run()

    assert len(exec_client.orders) >= 1
    assert exec_client.get_open_positions() == []


def test_live_engine_blocks_when_risk_denies():
    system_cfg = _build_system_cfg()
    ds = DummyDataSource([_make_bar(0), _make_bar(1)])
    exec_client = FakeExecutionClient()
    risk_engine = BlockingRiskEngine()
    strategy = AlwaysLongStrategy()
    strategy.init(pd.DataFrame())

    engine = LiveEngine(
        system_cfg=system_cfg,
        strategy=strategy,
        risk_engine=risk_engine,
        execution_client=exec_client,
        data_source=ds,
        run_id="risk_block",
    )

    engine.run()

    assert len(exec_client.orders) == 0
    assert exec_client.get_open_positions() == []


def test_live_engine_warns_on_exchange_limit():
    system_cfg = _build_system_cfg()
    ds = DummyDataSource([_make_bar(0)])
    exec_client = LimitExecutionClient()
    risk_engine = AllowRiskEngine(size=1.0)
    strategy = AlwaysLongStrategy()
    strategy.init(pd.DataFrame())
    kill_switch = KillSwitch(KillSwitchConfig(enabled=True, max_exceptions_per_hour=5, min_equity=0.0))
    notifier = DummyNotifier()

    engine = LiveEngine(
        system_cfg=system_cfg,
        strategy=strategy,
        risk_engine=risk_engine,
        execution_client=exec_client,
        data_source=ds,
        run_id="limit_warn",
        kill_switch=kill_switch,
        notifier=notifier,
    )

    engine.run()

    assert notifier.warn_calls, "Notifier warn should be called for exchange limit violations"
    assert "Order blocked by exchange risk limits" in notifier.warn_calls[0]
    assert len(kill_switch.state.recent_exceptions) == 1
