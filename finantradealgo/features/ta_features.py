from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class TAFeatureConfig:
    ret_periods: tuple[int, ...] = (1, 3, 5, 10, 20)
    mom_periods: tuple[int, ...] = (10, 20, 50)
    ema_periods: tuple[int, ...] = (10, 20, 50, 100, 200)
    atr_periods: tuple[int, ...] = (14, 50)
    hv_periods: tuple[int, ...] = (10, 20, 50)
    bb_period: int = 20
    bb_std: float = 2.0
    sr_periods: tuple[int, ...] = (20, 50)


def _ensure_ohlcv_columns(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing OHLCV columns: {missing}")


def _add_returns(df: pd.DataFrame, periods: Iterable[int]) -> None:
    close = df["close"]
    for n in periods:
        df[f"ret_{n}"] = close.pct_change(n)


def _add_momentum(df: pd.DataFrame, periods: Iterable[int]) -> None:
    close = df["close"]
    for n in periods:
        df[f"mom_{n}"] = close - close.shift(n)


def _add_ema(df: pd.DataFrame, periods: Iterable[int]) -> None:
    close = df["close"]
    for n in periods:
        df[f"ema_{n}"] = close.ewm(span=n, adjust=False).mean()

    if 20 in periods and 50 in periods:
        df["ema20_above_50"] = (df["ema_20"] > df["ema_50"]).astype(int)
    if 50 in periods and 200 in periods:
        df["ema50_above_200"] = (df["ema_50"] > df["ema_200"]).astype(int)


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _add_atr(df: pd.DataFrame, periods: Iterable[int]) -> None:
    tr = _true_range(df)
    for n in periods:
        df[f"atr_{n}"] = tr.rolling(n, min_periods=1).mean()
        df[f"atr_{n}_pct"] = df[f"atr_{n}"] / df["close"]


def _add_hv(df: pd.DataFrame, periods: Iterable[int]) -> None:
    close = df["close"]
    log_ret = np.log(close / close.shift(1))
    for n in periods:
        df[f"hv_{n}"] = log_ret.rolling(n, min_periods=1).std()


def _add_bollinger(df: pd.DataFrame, period: int, std_mult: float) -> None:
    close = df["close"]
    ma = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std()

    upper = ma + std_mult * std
    lower = ma - std_mult * std

    df[f"bb_mid_{period}"] = ma
    df[f"bb_upper_{period}"] = upper
    df[f"bb_lower_{period}"] = lower
    df[f"bb_width_{period}"] = (upper - lower) / close.replace(0, np.nan)


def _add_support_resistance(df: pd.DataFrame, periods: Iterable[int]) -> None:
    close = df["close"]
    high = df["high"]
    low = df["low"]

    for n in periods:
        rolling_high = high.rolling(n, min_periods=1).max()
        rolling_low = low.rolling(n, min_periods=1).min()

        df[f"rolling_high_{n}"] = rolling_high
        df[f"rolling_low_{n}"] = rolling_low

        rng = (rolling_high - rolling_low).replace(0, np.nan)
        df[f"dist_to_high_{n}"] = (close - rolling_low) / rng


def add_ta_features(df: pd.DataFrame, config: TAFeatureConfig | None = None) -> pd.DataFrame:
    if config is None:
        config = TAFeatureConfig()

    _ensure_ohlcv_columns(df)
    df = df.copy()

    _add_returns(df, config.ret_periods)
    _add_momentum(df, config.mom_periods)
    _add_ema(df, config.ema_periods)
    _add_atr(df, config.atr_periods)
    _add_hv(df, config.hv_periods)
    _add_bollinger(df, config.bb_period, config.bb_std)
    _add_support_resistance(df, config.sr_periods)

    return df
