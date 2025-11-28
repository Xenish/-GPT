import pandas as pd

from finantradealgo.microstructure.config import ChopConfig


def compute_chop(close: pd.Series, cfg: ChopConfig) -> pd.Series:
    """
    Computes a chop score between 0 (full trend) and 1 (full chop).

    The calculation is based on the ratio of the net price movement over a
    period to the sum of absolute price changes over the same period.

    Args:
        close: Series of close prices.
        cfg: Configuration object for the calculation.

    Returns:
        A series of floats between 0 and 1 representing the chop score.
    """
    # 1. Sum of absolute price changes over the lookback period
    diff_abs = close.diff().abs()
    range_sum = diff_abs.rolling(
        window=cfg.lookback_period, min_periods=cfg.lookback_period
    ).sum()

    # 2. Net price movement over the lookback period
    net_move = (close - close.shift(cfg.lookback_period)).abs()

    # 3. Calculate chop score
    # A small epsilon is added to the denominator to avoid division by zero
    chop = 1 - net_move / (range_sum + 1e-9)

    # 4. Clip the result to be strictly between 0 and 1
    chop = chop.clip(0, 1)
    
    # Fill initial NaNs with a neutral value, e.g., 0.5
    chop = chop.fillna(0.5)

    return chop
