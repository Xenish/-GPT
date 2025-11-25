from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MarketStructureConfig:
    swing_lookback: int = 3   # fractal: 3 bar sol, 3 bar sağ
    swing_lookforward: int = 3
    fvg_min_size: float = 0.003  # %0.3 gap
    # trend state için
    min_swings_for_trend: int = 3


def _detect_swings(df: pd.DataFrame, cfg: MarketStructureConfig) -> pd.DataFrame:
    high = df["high"].values
    low = df["low"].values

    n = len(df)
    is_sw_high = np.zeros(n, dtype=int)
    is_sw_low = np.zeros(n, dtype=int)

    L = cfg.swing_lookback
    R = cfg.swing_lookforward

    for i in range(L, n - R):
        window_high = high[i - L : i + R + 1]
        window_low = low[i - L : i + R + 1]

        if high[i] == window_high.max():
            # en az bir bar ondan önce ve sonra daha düşük olsun
            if high[i] > window_high.max(initial=-np.inf) - 1e-12:
                is_sw_high[i] = 1

        if low[i] == window_low.min():
            if low[i] < window_low.min(initial=np.inf) + 1e-12:
                is_sw_low[i] = 1

    df["ms_sw_high"] = is_sw_high
    df["ms_sw_low"] = is_sw_low
    return df


def _build_trend_state(df: pd.DataFrame, cfg: MarketStructureConfig) -> pd.DataFrame:
    """
    HH-HL / LH-LL dizisine göre kaba bir trend etiketi:
      ms_trend_state: 1 = up, -1 = down, 0 = neutral
    """
    highs = df["high"]
    lows = df["low"]

    swing_idx = df.index[(df["ms_sw_high"] == 1) | (df["ms_sw_low"] == 1)].tolist()
    trend_state = np.zeros(len(df), dtype=int)

    last_trend = 0
    last_sw_high = None
    last_sw_low = None

    for idx in swing_idx:
        row = df.iloc[idx]
        if row["ms_sw_high"] == 1:
            # HH / LH kontrolü
            if last_sw_high is not None:
                if row["high"] > last_sw_high:
                    last_trend = 1  # HH -> up
                elif row["high"] < last_sw_high:
                    last_trend = -1  # LH -> down
            last_sw_high = row["high"]

        if row["ms_sw_low"] == 1:
            # HL / LL kontrolü
            if last_sw_low is not None:
                if row["low"] > last_sw_low:
                    # HL, up trend’i destekler
                    if last_trend == 0:
                        last_trend = 1
                elif row["low"] < last_sw_low:
                    # LL, down trend’i destekler
                    if last_trend == 0:
                        last_trend = -1
            last_sw_low = row["low"]

        trend_state[idx] = last_trend

    # Son swing trendini ileri propagate et
    current = 0
    for i in range(len(df)):
        if trend_state[i] != 0:
            current = trend_state[i]
        trend_state[i] = current

    df["ms_trend_state"] = trend_state
    df["ms_trend_up"] = (trend_state == 1).astype(int)
    df["ms_trend_down"] = (trend_state == -1).astype(int)
    df["ms_trend_neutral"] = (trend_state == 0).astype(int)

    return df


def _detect_fvg(df: pd.DataFrame, cfg: MarketStructureConfig) -> pd.DataFrame:
    """
    3-bar FVG mantığı (ICT tarzı):
      - Bull FVG: low[n+1] > high[n-1]
      - Bear FVG: high[n+1] < low[n-1]
    """
    high = df["high"]
    low = df["low"]

    bull_fvg = np.zeros(len(df), dtype=int)
    bear_fvg = np.zeros(len(df), dtype=int)

    for i in range(1, len(df) - 1):
        # bull
        gap_up = low.iloc[i + 1] - high.iloc[i - 1]
        if gap_up > cfg.fvg_min_size * df["close"].iloc[i]:
            bull_fvg[i] = 1

        # bear
        gap_down = low.iloc[i - 1] - high.iloc[i + 1]
        if gap_down > cfg.fvg_min_size * df["close"].iloc[i]:
            bear_fvg[i] = 1

    df["ms_fvg_up"] = bull_fvg
    df["ms_fvg_down"] = bear_fvg
    return df


def add_market_structure_features_15m(
    df: pd.DataFrame,
    config: Optional[MarketStructureConfig] = None,
) -> pd.DataFrame:
    if config is None:
        config = MarketStructureConfig()

    df = df.copy()

    # 1) Swing’ler
    df = _detect_swings(df, config)

    # 2) Trend state
    df = _build_trend_state(df, config)

    # 3) FVG
    df = _detect_fvg(df, config)

    return df
