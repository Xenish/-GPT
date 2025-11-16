from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class Position:
    """
    v1.1: LONG veya SHORT pozisyon.
    """

    side: Literal["LONG", "SHORT"]
    qty: float
    entry_price: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None


@dataclass
class Portfolio:
    initial_cash: float
    cash: float
    equity: float
    position: Optional[Position] = None
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List = field(default_factory=list)

    def update_equity(self, price: float) -> None:
        unrealized = 0.0
        if self.position is not None:
            if self.position.side == "LONG":
                unrealized = (price - self.position.entry_price) * self.position.qty
            else:
                unrealized = (self.position.entry_price - price) * self.position.qty
        self.equity = self.cash + unrealized

    def record(self, ts, price: float) -> None:
        self.update_equity(price)
        self.equity_curve.append(self.equity)
        self.timestamps.append(ts)
