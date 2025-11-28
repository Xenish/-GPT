from dataclasses import dataclass, field
from typing import List, Literal

import pandas as pd


@dataclass
class Bar:
    """Represents a single bar of price data (OHLCV)."""

    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    timeframe: str = ""
    extras: dict = field(default_factory=dict)


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
