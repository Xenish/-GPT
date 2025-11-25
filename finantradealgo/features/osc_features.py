from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class OscFeatureConfig:
    rsi_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    cci_period: int = 20
    mfi_period: int = 14


def _ensure_ohlcv_columns(df: pd.DataFrame) -> None:
    required = {"high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _add_rsi(df: pd.DataFrame, cfg: OscFeatureConfig) -> None:
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(cfg.rsi_period, min_periods=cfg.rsi_period).mean()
    roll_down = loss.rolling(cfg.rsi_period, min_periods=cfg.rsi_period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df[f"rsi_{cfg.rsi_period}"] = rsi


def _add_stoch(df: pd.DataFrame, cfg: OscFeatureConfig) -> None:
    h = df["high"]
    l = df["low"]
    c = df["close"]

    lowest_low = l.rolling(cfg.stoch_k_period, min_periods=cfg.stoch_k_period).min()
    highest_high = h.rolling(cfg.stoch_k_period, min_periods=cfg.stoch_k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = (c - lowest_low) / denom * 100.0
    stoch_d = stoch_k.rolling(cfg.stoch_d_period, min_periods=cfg.stoch_d_period).mean()

    df[f"stoch_k_{cfg.stoch_k_period}_{cfg.stoch_d_period}"] = stoch_k
    df[f"stoch_d_{cfg.stoch_k_period}_{cfg.stoch_d_period}"] = stoch_d


def _add_macd(df: pd.DataFrame, cfg: OscFeatureConfig) -> None:
    c = df["close"]
    ema_fast = c.ewm(span=cfg.macd_fast, adjust=False).mean()
    ema_slow = c.ewm(span=cfg.macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=cfg.macd_signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    df[f"macd_{cfg.macd_fast}_{cfg.macd_slow}"] = macd_line
    df[
        f"macd_signal_{cfg.macd_fast}_{cfg.macd_slow}_{cfg.macd_signal}"
    ] = macd_signal
    df[
        f"macd_hist_{cfg.macd_fast}_{cfg.macd_slow}_{cfg.macd_signal}"
    ] = macd_hist


def _add_cci(df: pd.DataFrame, cfg: OscFeatureConfig) -> None:
    h = df["high"]
    l = df["low"]
    c = df["close"]
    tp = (h + l + c) / 3.0
    sma_tp = tp.rolling(cfg.cci_period, min_periods=cfg.cci_period).mean()
    mad = (tp - sma_tp).abs().rolling(cfg.cci_period, min_periods=cfg.cci_period).mean()
    cci = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    df[f"cci_{cfg.cci_period}"] = cci


def _add_mfi(df: pd.DataFrame, cfg: OscFeatureConfig) -> None:
    h = df["high"]
    l = df["low"]
    c = df["close"]
    v = df["volume"]
    tp = (h + l + c) / 3.0
    raw_mf = tp * v
    tp_prev = tp.shift(1)
    direction = np.sign(tp - tp_prev)
    pos_mf = np.where(direction > 0, raw_mf, 0.0)
    neg_mf = np.where(direction < 0, raw_mf, 0.0)
    pos_mf_series = pd.Series(pos_mf, index=df.index)
    neg_mf_series = pd.Series(neg_mf, index=df.index)
    pos_mf_roll = pos_mf_series.rolling(cfg.mfi_period, min_periods=cfg.mfi_period).sum()
    neg_mf_roll = neg_mf_series.rolling(cfg.mfi_period, min_periods=cfg.mfi_period).sum()
    mfr = pos_mf_roll / neg_mf_roll.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    df[f"mfi_{cfg.mfi_period}"] = mfi


def add_osc_features(
    df: pd.DataFrame,
    config: Optional[OscFeatureConfig] = None,
) -> pd.DataFrame:
    if config is None:
        config = OscFeatureConfig()

    _ensure_ohlcv_columns(df)
    df = df.copy()

    _add_rsi(df, config)
    _add_stoch(df, config)
    _add_macd(df, config)
    _add_cci(df, config)
    _add_mfi(df, config)

    return df
