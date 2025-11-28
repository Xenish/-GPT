from typing import Tuple

import numpy as np
import pandas as pd

from finantradealgo.microstructure.config import BurstConfig


def compute_bursts(
    close: pd.Series, cfg: BurstConfig
) -> Tuple[pd.Series, pd.Series]:
    """
    Computes upward and downward price bursts based on the z-score of returns.

    A burst is a measure of how much the recent return has deviated from its
    rolling average, normalized by its rolling standard deviation.

    Args:
        close: Series of close prices.
        cfg: Configuration object for the calculation.

    Returns:
        A tuple of two series: (burst_up, burst_down).
    """
    # 1. Calculate returns over the given window
    returns = close.pct_change(cfg.return_window)

    # 2. Calculate rolling mean and std of the returns to get the z-score
    rolling_mean = returns.rolling(
        window=cfg.z_score_window, min_periods=cfg.z_score_window
    ).mean()
    rolling_std = returns.rolling(
        window=cfg.z_score_window, min_periods=cfg.z_score_window
    ).std()

    # Calculate z-score, avoiding division by zero
    z_score = (returns - rolling_mean) / (rolling_std + 1e-9)

    # 3. Calculate up and down burst signals based on z-score thresholds
    # burst_up is the magnitude of the z-score exceeding the upper threshold
    burst_up = (z_score - cfg.z_up_threshold).clip(lower=0)

    # burst_down is the magnitude of the negative z-score exceeding the lower threshold
    burst_down = (-z_score - abs(cfg.z_down_threshold)).clip(lower=0)
    
    # Fill initial NaNs
    burst_up = burst_up.fillna(0)
    burst_down = burst_down.fillna(0)

    return burst_up, burst_down
