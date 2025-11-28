"""
Functions for identifying Breaks of Structure (BoS) and Changes of Character (ChoCh).
"""
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import BreakConfig
from .types import SwingPoint


def detect_bos_choch(
    df: pd.DataFrame,
    swings: List[SwingPoint],
    trend_regime: pd.Series,
    cfg: BreakConfig,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Detects Breaks of Structure (BoS) and Changes of Character (ChoCh).

    - BoS: A break of a swing point in the direction of the trend.
    - ChoCh: A change in the overall trend regime.

    Args:
        df: The OHLCV DataFrame, indexed by timestamp.
        swings: A time-sorted list of detected swing points.
        trend_regime: A Series indicating the trend (-1, 0, 1) for each bar.
        cfg: Configuration object for break detection.

    Returns:
        A tuple of three Series: (bos_up, bos_down, choch).
    """
    bos_up = pd.Series(0, index=df.index, dtype=np.int8)
    bos_down = pd.Series(0, index=df.index, dtype=np.int8)

    # --- ChoCh Detection (Vectorized) ---
    # A ChoCh occurs whenever the trend regime changes value.
    choch = (trend_regime.diff() != 0).astype(np.int8)

    # --- BoS Detection (Iterative) ---
    # This part is iterative as it needs to track the last relevant swing.
    last_swing_high_price = None
    last_swing_low_price = None
    swing_iter = iter(swings)
    next_swing = next(swing_iter, None)

    for i in range(len(df)):
        # Update last swing prices as we pass them
        if next_swing and df.index[i] >= df.index[next_swing.ts]:
            if next_swing.kind == "high":
                last_swing_high_price = next_swing.price
            else:
                last_swing_low_price = next_swing.price
            next_swing = next(swing_iter, None)

        current_trend = trend_regime.iloc[i]
        current_close = df["close"].iloc[i]
        buffer_mult = 1 + cfg.swing_break_buffer_pct

        # Upward Break of Structure
        if (
            current_trend == 1
            and last_swing_high_price is not None
            and current_close > last_swing_high_price * buffer_mult
        ):
            bos_up.iloc[i] = 1
            # Invalidate the swing high so it's not used again for this trend leg
            last_swing_high_price = None

        # Downward Break of Structure
        if (
            current_trend == -1
            and last_swing_low_price is not None
            and current_close < last_swing_low_price / buffer_mult
        ):
            bos_down.iloc[i] = 1
            # Invalidate the swing low
            last_swing_low_price = None
            
    return bos_up, bos_down, choch
