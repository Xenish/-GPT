from __future__ import annotations

"""
Factor analysis helpers for performance attribution.

Intended pipeline:
- Obtain trades as list[dict] or DataFrame from backtests.
- Obtain the OHLCV DataFrame used in the backtest.
- Call build_trade_factor_exposures(trades, ohlcv, regime_col="regime").
- Pass resulting factor DataFrame to PerformanceAttributionEngine.attach_factor_exposures.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


class FactorType(Enum):
    TREND = auto()
    VOLATILITY = auto()
    MARKET_STRUCTURE = auto()
    TIME_OF_DAY = auto()


@dataclass
class FactorExposureRow:
    trade_id: str
    trend_exposure: float | None = None
    volatility_exposure: float | None = None
    market_structure_exposure: float | None = None
    time_of_day_bucket: str | None = None
    metadata: dict[str, Any] | None = None


def time_of_day_bucket(ts: pd.Timestamp) -> str:
    """
    Map a timestamp to a coarse time-of-day bucket.

    Example buckets (UTC-based, adjust as needed):
    - "asia"        : 00:00-08:00
    - "europe"      : 08:00-14:00
    - "us_open"     : 14:00-20:00
    - "us_late"     : 20:00-24:00
    """
    hour = ts.hour
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 14:
        return "europe"
    if 14 <= hour < 20:
        return "us_open"
    return "us_late"


def compute_trend_exposure(
    price_series: pd.Series,
    *,
    window: int = 50,
) -> pd.Series:
    """
    Compute a simple trend exposure proxy based on rolling returns or MA slope.

    Returns a Series aligned with price_series.index where:
    - Positive values indicate uptrend exposure.
    - Negative values indicate downtrend exposure.
    - Magnitude encodes strength (e.g. normalized rolling return).
    """
    returns = price_series.pct_change()
    roll = returns.rolling(window=window, min_periods=window // 2).mean()
    # Normalize to [-1, 1] using a simple tanh
    exposure = np.tanh(roll * 10.0)
    exposure.name = "trend_exposure"
    return exposure


def compute_volatility_exposure(
    price_series: pd.Series,
    *,
    window: int = 50,
) -> pd.Series:
    """
    Compute a simple realized volatility proxy.

    Returns:
        Series of vol exposure (e.g. rolling std of returns).
        Can be normalized if desired.
    """
    returns = price_series.pct_change()
    vol = returns.rolling(window=window, min_periods=window // 2).std()
    vol.name = "volatility_exposure"
    return vol


def compute_market_structure_exposure(
    regime_series: pd.Series,
) -> pd.Series:
    """
    Compute market structure exposure from a categorical regime series.

    Example:
        "trend"      -> +1.0
        "mean_revert"-> -1.0
        "chop"       -> 0.0

    The concrete mapping can be tuned later to your actual regime labels.
    """
    mapping = {
        "trend": 1.0,
        "mean_revert": -1.0,
        "chop": 0.0,
    }
    exposure = regime_series.map(mapping).astype(float)
    exposure.name = "market_structure_exposure"
    return exposure


def build_trade_factor_exposures(
    trades: Iterable[Mapping[str, Any]],
    ohlcv: pd.DataFrame,
    *,
    price_col: str = "close",
    regime_col: str | None = None,
) -> pd.DataFrame:
    """
    Build a DataFrame of factor exposures per trade.

    trades:
        Iterable of dict-like with at least:
        - "trade_id"
        - "entry_ts"
        - "exit_ts"
        - possibly "entry_index"/"exit_index" if pre-aligned.

    ohlcv:
        OHLCV DataFrame indexed by timestamp (pd.DatetimeIndex) with at least:
        - price_col (default "close")
        - optionally regime_col (categorical regimes).

    Returns:
        DataFrame indexed by trade_id with columns:
        - trend_exposure
        - volatility_exposure
        - market_structure_exposure
        - time_of_day_bucket
    """
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise ValueError("ohlcv index must be a DatetimeIndex")

    price = ohlcv[price_col].astype(float)

    trend = compute_trend_exposure(price)
    vol = compute_volatility_exposure(price)
    ms = None
    if regime_col is not None and regime_col in ohlcv.columns:
        ms = compute_market_structure_exposure(ohlcv[regime_col])

    rows: list[dict[str, Any]] = []

    for t in trades:
        trade_id = str(t.get("trade_id") or t.get("id"))
        entry_ts = pd.to_datetime(t.get("entry_ts"))
        exit_ts = pd.to_datetime(t.get("exit_ts"))

        # Use entry timestamp as anchor for factor sampling.
        # Alternatively we could average over holding period.
        anchor_ts = entry_ts

        # Align to nearest index.
        try:
            anchor_loc = ohlcv.index.get_loc(anchor_ts, method="nearest")
            ts = ohlcv.index[anchor_loc]
        except Exception:
            rows.append(
                {
                    "trade_id": trade_id,
                    "trend_exposure": np.nan,
                    "volatility_exposure": np.nan,
                    "market_structure_exposure": np.nan,
                    "time_of_day_bucket": None,
                }
            )
            continue

        trend_val = (
            float(trend.loc[ts])
            if ts in trend.index and not pd.isna(trend.loc[ts])
            else np.nan
        )
        vol_val = (
            float(vol.loc[ts])
            if ts in vol.index and not pd.isna(vol.loc[ts])
            else np.nan
        )
        ms_val = (
            float(ms.loc[ts])
            if ms is not None and ts in ms.index and not pd.isna(ms.loc[ts])
            else np.nan
        )

        rows.append(
            {
                "trade_id": trade_id,
                "trend_exposure": trend_val,
                "volatility_exposure": vol_val,
                "market_structure_exposure": ms_val,
                "time_of_day_bucket": time_of_day_bucket(ts),
            }
        )

    df = pd.DataFrame.from_records(rows).set_index("trade_id")
    return df
