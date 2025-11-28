import numpy as np
import pandas as pd

from finantradealgo.microstructure.config import ParabolicConfig


def compute_parabolic_trend(close: pd.Series, cfg: ParabolicConfig) -> pd.Series:
    """
    Detects parabolic trends by measuring the curvature of the price series.

    Curvature is defined as the second discrete derivative of price, normalized
    by the rolling standard deviation.

    The trend is classified as:
    -  1: Parabolic up (strong upward acceleration)
    - -1: Parabolic down (strong downward acceleration)
    -  0: Not parabolic

    Args:
        close: Series of close prices.
        cfg: Configuration object for the calculation.

    Returns:
        A series of integers representing the parabolic trend.
    """
    # 1. Calculate the second discrete derivative of the close price
    second_diff = close.diff(2) - 2 * close.diff()
    # A more direct way:
    # second_diff = close - 2 * close.shift(1) + close.shift(2)

    # 2. Normalize by rolling standard deviation of the close price
    rolling_std = close.rolling(
        window=cfg.rolling_std_window, min_periods=cfg.rolling_std_window
    ).std()
    
    # Add epsilon to avoid division by zero
    curvature = second_diff / (rolling_std + 1e-9)

    # 3. Classify based on the curvature threshold
    trend = pd.Series(0, index=close.index, dtype=np.int8)
    trend.loc[curvature >= cfg.curvature_threshold] = 1
    trend.loc[curvature <= -cfg.curvature_threshold] = -1

    # Fill initial NaNs
    trend = trend.fillna(0)

    return trend
