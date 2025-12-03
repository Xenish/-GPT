from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd


def ensure_timestamp(value: datetime | str | int | float | pd.Timestamp) -> pd.Timestamp:
    """
    Normalize any timestamp-like input into a UTC pandas Timestamp.
    """
    ts = pd.to_datetime(value, utc=True)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


@dataclass
class IngestCandle:
    ts: pd.Timestamp
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    vwap: Optional[float] = None

    @classmethod
    def from_binance_kline(cls, symbol: str, timeframe: str, kline: list) -> "IngestCandle":
        """
        Build a candle from Binance kline payload.

        Binance futures kline format:
        [ open_time, open, high, low, close, volume,
          close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore ]
        """
        ts = ensure_timestamp(kline[0])
        return cls(
            ts=ts,
            symbol=symbol,
            timeframe=timeframe,
            open=float(kline[1]),
            high=float(kline[2]),
            low=float(kline[3]),
            close=float(kline[4]),
            volume=float(kline[5]) if kline[5] is not None else None,
            vwap=float(kline[7]) / float(kline[5]) if kline[5] not in (None, 0, "0") else None,
        )


@dataclass
class FundingRate:
    ts: pd.Timestamp
    symbol: str
    funding_rate: float
    timeframe: str = "8h"
    mark_price: Optional[float] = None
    index_price: Optional[float] = None
    open_interest: Optional[float] = None


@dataclass
class OpenInterestSnapshot:
    ts: pd.Timestamp
    symbol: str
    open_interest: float
    timeframe: str = "1h"
    volume: Optional[float] = None
    turnover: Optional[float] = None


@dataclass
class FlowSnapshot:
    ts: pd.Timestamp
    symbol: str
    timeframe: str
    perp_premium: Optional[float] = None
    basis: Optional[float] = None
    oi: Optional[float] = None
    oi_change: Optional[float] = None
    liq_up: Optional[float] = None
    liq_down: Optional[float] = None


@dataclass
class SentimentSignal:
    ts: pd.Timestamp
    symbol: str
    timeframe: str
    sentiment_score: float
    volume: Optional[float] = None
    source: Optional[str] = None
