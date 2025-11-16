from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    return_windows: Sequence[int] = (1, 3, 5, 10)
    vol_windows: Sequence[int] = (10, 20)
    ema_fast: int = 20
    ema_slow: int = 50
    vol_regime_q: tuple[float, float] = (0.33, 0.66)
    trend_threshold: float = 0.001


def add_basic_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = FeatureConfig()

    df = df.copy()

    df["log_price"] = np.log(df["close"])
    df["ret_1"] = df["log_price"].diff()

    for window in config.return_windows:
        df[f"ret_{window}"] = df["ret_1"].rolling(window).sum()

    for window in config.vol_windows:
        df[f"vol_{window}"] = df["ret_1"].rolling(window).std()

    vol_col = f"vol_{config.vol_windows[-1]}"
    df["vol_score"] = df[vol_col]

    df["ema_fast"] = df["close"].ewm(span=config.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=config.ema_slow, adjust=False).mean()

    df["trend_raw"] = df["ema_fast"] - df["ema_slow"]
    df["trend_score"] = df["trend_raw"] / df["close"]

    q1, q2 = df["vol_score"].quantile(list(config.vol_regime_q))

    def _vol_regime(value: float) -> str:
        if np.isnan(value):
            return "VOL_UNKNOWN"
        if value < q1:
            return "VOL_LOW"
        if value < q2:
            return "VOL_MED"
        return "VOL_HIGH"

    df["regime_vol"] = df["vol_score"].apply(_vol_regime)

    threshold = config.trend_threshold

    def _trend_regime(value: float) -> str:
        if np.isnan(value):
            return "TREND_UNKNOWN"
        if value > threshold:
            return "TREND_UP"
        if value < -threshold:
            return "TREND_DOWN"
        return "RANGE"

    df["regime_trend"] = df["trend_score"].apply(_trend_regime)

    return df
