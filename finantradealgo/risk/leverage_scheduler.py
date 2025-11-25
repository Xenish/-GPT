from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class LeverageScheduleConfig:
    max_leverage: float = 5.0
    trend_strong_threshold: float = 0.3
    trend_weak_threshold: float = 0.1
    hv_high_threshold: float = 0.25
    hv_low_threshold: float = 0.05


def _get_series_value(row: pd.Series, keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        if key in row:
            try:
                value = float(row[key])
                if pd.notna(value):
                    return value
            except (TypeError, ValueError):
                continue
    return default


def compute_leverage(
    row: pd.Series,
    cfg: LeverageScheduleConfig,
) -> float:
    lev = 1.0
    trend_score = _get_series_value(
        row,
        ["ms_trend_score", "ms_trend_state", "trend_score", "regime_trend"],
        default=0.0,
    )
    hv_value = _get_series_value(
        row,
        ["hv_20", "hv20", "hv"],
        default=0.0,
    )

    if trend_score >= cfg.trend_strong_threshold and hv_value <= cfg.hv_high_threshold:
        lev = cfg.max_leverage
    elif trend_score >= cfg.trend_weak_threshold and hv_value <= cfg.hv_high_threshold:
        lev = (cfg.max_leverage + 1.0) / 2.0

    if hv_value >= cfg.hv_high_threshold:
        lev = min(lev, max(1.0, cfg.max_leverage / 2.0))
    elif hv_value <= cfg.hv_low_threshold and trend_score >= cfg.trend_strong_threshold:
        lev = min(cfg.max_leverage, lev + 1.0)

    return max(1.0, min(lev, cfg.max_leverage))
