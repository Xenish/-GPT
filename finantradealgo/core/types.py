from dataclasses import dataclass, field
from typing import List, Literal

import pandas as pd


@dataclass
class Bar:
    """Represents a single bar of price data (OHLCV)."""

    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    timeframe: str = ""
    open_time: pd.Timestamp = None
    close_time: pd.Timestamp = None
    ts: pd.Timestamp = None
    extras: dict = field(default_factory=dict)

    def __post_init__(self):
        # Backward compatibility: if ts is provided but open_time is not, use ts as open_time
        if self.ts is not None and self.open_time is None:
            self.open_time = self.ts
        # If open_time is provided but ts is not, use open_time as ts
        if self.open_time is not None and self.ts is None:
            self.ts = self.open_time
        # If close_time is not provided, use open_time
        if self.close_time is None and self.open_time is not None:
            self.close_time = self.open_time


@dataclass
class Trade:
    """Represents a single market trade."""

    ts: pd.Timestamp
    price: float
    size: float
    side: Literal["buy", "sell"]


@dataclass
class OrderBookLevel:
    """Represents a single level in the order book (price and size)."""

    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    """Represents a snapshot of the order book at a point in time."""

    ts: pd.Timestamp
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
