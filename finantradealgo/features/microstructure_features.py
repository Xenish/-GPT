from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MicrostructureFeatureConfig:
    # “büyük bar” için threshold’lar
    impulse_body_mult: float = 2.0
    impulse_range_mult: float = 1.5

    # exhaustion / wick için
    wick_ratio_min: float = 0.6  # gövdeye göre wick oranı

    # chop / trend için
    atr_window: int = 14
    chop_hv_window: int = 20
    chop_bb_window: int = 20
    chop_bb_width_max: float = 0.02  # %2 civarı bant

    # burst için
    burst_ret_window: int = 20
    burst_zscore_min: float = 2.0


def add_microstructure_features_15m(
    df: pd.DataFrame,
    config: Optional[MicrostructureFeatureConfig] = None,
) -> pd.DataFrame:
    if config is None:
        config = MicrostructureFeatureConfig()

    df = df.copy()

    # Temel geometry’yi kullanacağız (zaten candle_features hesaplıyor)
    if "cs_body_abs" not in df.columns or "cs_range" not in df.columns:
        raise ValueError("Microstructure için 'cs_body_abs' ve 'cs_range' kolonları gerekli.")

    body = df["cs_body_abs"]
    rng = df["cs_range"]
    upper = df.get("cs_upper_shadow", df["high"] - df[["open", "close"]].max(axis=1))
    lower = df.get("cs_lower_shadow", df[["open", "close"]].min(axis=1) - df["low"])

    # === 1) Impulsive move (up/down) ===
    avg_body = body.rolling(50, min_periods=20).mean()
    avg_range = rng.rolling(50, min_periods=20).mean()

    cond_big_body = body > config.impulse_body_mult * avg_body
    cond_big_range = rng > config.impulse_range_mult * avg_range

    bull = df["close"] > df["open"]
    bear = ~bull

    df["ms_impulse_up"] = ((cond_big_body & cond_big_range) & bull).astype(int)
    df["ms_impulse_down"] = ((cond_big_body & cond_big_range) & bear).astype(int)

    # === 2) Exhaustion wicks (top/bottom) ===
    # üst wick gövdeye göre çok büyük + bar up veya down’a göre
    upper_ratio = upper / (rng.replace(0, np.nan))
    lower_ratio = lower / (rng.replace(0, np.nan))

    df["ms_exhaustion_top"] = (
        (upper_ratio >= config.wick_ratio_min) & bear
    ).astype(int)

    df["ms_exhaustion_bottom"] = (
        (lower_ratio >= config.wick_ratio_min) & bull
    ).astype(int)

    # === 3) Chop vs trend (regime) ===
    # ATR + BB width + HV üzerinden basit bir chop detektörü
    if "atr_14" in df.columns and "bb_width_20" in df.columns and "hv_20" in df.columns:
        atr = df["atr_14"]
        bb_width = df["bb_width_20"]
        hv20 = df["hv_20"]

        # Vol çok düşük + bant dar ise chop
        chop = (bb_width < config.chop_bb_width_max) & (hv20 < hv20.median())
        df["ms_chop"] = chop.astype(int)
        df["ms_trend"] = (~chop).astype(int)
    else:
        df["ms_chop"] = 0
        df["ms_trend"] = 0

    # === 4) Burst detector (ani patlama) ===
    ret1 = df["close"].pct_change()
    vol = ret1.rolling(config.burst_ret_window, min_periods=10).std()
    zscore = ret1 / (vol.replace(0, np.nan))

    df["ms_burst_up"] = ((zscore >= config.burst_zscore_min)).astype(int)
    df["ms_burst_down"] = ((zscore <= -config.burst_zscore_min)).astype(int)

    return df
