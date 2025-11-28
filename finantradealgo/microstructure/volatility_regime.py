import numpy as np
import pandas as pd

from finantradealgo.microstructure.config import VolatilityRegimeConfig


def compute_volatility_regime(
    close: pd.Series, cfg: VolatilityRegimeConfig
) -> pd.Series:
    """
    Computes the volatility regime based on the z-score of rolling volatility.

    The regime is classified as:
    -  1: High volatility (z-score >= high_z_threshold)
    -  0: Normal volatility
    - -1: Low volatility (z-score <= low_z_threshold)

    Args:
        close: Series of close prices.
        cfg: Configuration object for the calculation.

    Returns:
        A series of integers representing the volatility regime.
    """
    # 1. Calculate log returns
    log_returns = np.log(close / close.shift(1))

    # 2. Calculate rolling standard deviation of log returns
    rolling_vol = log_returns.rolling(window=cfg.period, min_periods=cfg.period).std()

    # 3. Calculate z-score of the rolling volatility
    # Z-score = (value - mean) / std
    vol_mean = rolling_vol.rolling(
        window=cfg.z_score_window, min_periods=cfg.z_score_window
    ).mean()
    vol_std = rolling_vol.rolling(
        window=cfg.z_score_window, min_periods=cfg.z_score_window
    ).std()
    
    # Avoid division by zero
    z_score = (rolling_vol - vol_mean) / (vol_std + 1e-9)

    # 4. Classify regime based on z-score thresholds
    regime = pd.Series(0, index=close.index, dtype=np.int8)
    regime.loc[z_score >= cfg.high_z_threshold] = 1
    regime.loc[z_score <= cfg.low_z_threshold] = -1
    
    # Fill initial NaNs with 0 (normal regime)
    regime = regime.fillna(0)

    return regime
