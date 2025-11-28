"""
Functions for identifying Fair Value Gaps (FVGs), also known as imbalances.
"""
from typing import Tuple

import pandas as pd

from .config import FVGConfig


def detect_fvg_series(
    df: pd.DataFrame,
    cfg: FVGConfig,
) -> Tuple[pd.Series, pd.Series]:
    """
    Detects Fair Value Gaps (FVGs) in a given OHLCV DataFrame.

    This implementation uses the common 3-bar pattern definition:
    - A bullish FVG occurs at bar `i` if `df['low'].iloc[i+1] > df['high'].iloc[i-1]`.
    - A bearish FVG occurs at bar `i` if `df['high'].iloc[i+1] < df['low'].iloc[i-1]`.

    The function returns the percentage size of the gap, filtered by a minimum
    threshold.

    Args:
        df: Input OHLCV DataFrame.
        cfg: Configuration object for FVG detection.

    Returns:
        A tuple of two Series (fvg_up, fvg_down), representing the
        percentage size of the detected gaps. Values are 0 where no
        gap is detected.
    """
    high = df["high"]
    low = df["low"]

    # --- Vectorized FVG Detection ---
    # Shift series to get previous high and next low for each bar
    prev_high = high.shift(1)
    next_low = low.shift(-1)

    # Bullish FVG condition and gap calculation
    bullish_fvg_cond = next_low > prev_high
    fvg_up = (next_low - prev_high) / prev_high
    fvg_up[~bullish_fvg_cond] = 0.0

    # Filter by minimum gap size
    fvg_up[fvg_up < cfg.min_gap_pct] = 0.0

    # Shift series to get previous low and next high for each bar
    prev_low = low.shift(1)
    next_high = high.shift(-1)

    # Bearish FVG condition and gap calculation
    bearish_fvg_cond = next_high < prev_low
    fvg_down = (prev_low - next_high) / prev_low
    fvg_down[~bearish_fvg_cond] = 0.0

    # Filter by minimum gap size
    fvg_down[fvg_down < cfg.min_gap_pct] = 0.0
    
    # Fill NaNs that result from shifting
    fvg_up = fvg_up.fillna(0.0)
    fvg_down = fvg_down.fillna(0.0)

    return fvg_up, fvg_down
