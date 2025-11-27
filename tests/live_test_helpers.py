from __future__ import annotations

from typing import List, Optional

import pandas as pd

from finantradealgo.core.strategy import BaseStrategy, StrategyContext
from finantradealgo.data_engine.live_data_source import AbstractLiveDataSource, Bar
from finantradealgo.execution.client_base import ExecutionClientBase


def make_bar(idx: int, *, symbol: str = "TEST", base: float = 100.0) -> Bar:
    ts = pd.Timestamp("2025-01-01 00:00:00+00:00") + pd.Timedelta(minutes=idx)
    price = base + idx
    return Bar(
        symbol=symbol,
        timeframe="1m",
        open_time=ts,
        close_time=ts,
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=1.0,
        extras={"timestamp": ts},
    )


class ListDataSource(AbstractLiveDataSource):
    def __init__(self, bars: List[Bar]):
        self._bars = list(bars)
        self._idx = 0

    def connect(self) -> None:
        self._idx = 0

    def next_bar(self) -> Optional[Bar]:
        if self._idx >= len(self._bars):
            return None
        bar = self._bars[self._idx]
        self._idx += 1
        return bar

    def close(self) -> None:
        return None


class AlwaysLongStrategy(BaseStrategy):
    def init(self, df: pd.DataFrame) -> None:
        return None

    def on_bar(self, row, ctx: StrategyContext):
        return "LONG"


class AlwaysAllowRiskEngine:
    def can_open_new_trade(self, **_kwargs) -> bool:
        return True

    def calc_position_size(self, **_kwargs) -> float:
        return 1.0


class StaticExecutionClient(ExecutionClientBase):
    def __init__(self, equity: float = 1000.0) -> None:
        self.orders: List[dict] = []
        self._equity = equity

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        *,
        price: Optional[float] = None,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
        **_: dict,
    ) -> dict:
        order = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "reduce_only": reduce_only,
            "client_order_id": client_order_id,
        }
        self.orders.append(order)
        return order

    def cancel_order(self, symbol: str, order_id: str | int) -> None:
        return None

    def get_open_positions(self):
        return []

    def get_open_orders(self, symbol: Optional[str] = None):
        return []

    def mark_to_market(self, price: float, timestamp):
        self._last_price = price

    def get_portfolio(self):
        return {"equity": self._equity}

    def get_position(self):
        return None

    def export_logs(self, timeframe: str):
        return {}

    def to_state_dict(self):
        return {"open_positions": []}

    def close(self):
        return None
