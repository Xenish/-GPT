"""
Functions for identifying swing highs and swing lows in a price series.
"""
from typing import List

import numpy as np
import pandas as pd

from .config import SwingConfig
from .types import SwingPoint, SwingKind


def detect_swings(
    high: pd.Series, low: pd.Series, cfg: SwingConfig
) -> List[SwingPoint]:
    """
    Detects significant swing high and swing low points from price series.

    The process involves two main steps:
    1. Initial fractal-based detection: A point is a potential swing high if
       it's the highest high within a centered window, and vice-versa for lows.
    2. Filtering: Minor swings that don't represent a significant enough price
       change compared to the previous swing are filtered out.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        cfg: Configuration for swing detection.

    Returns:
        A list of significant SwingPoint objects, sorted by time.
    """
    if high.empty or low.empty:
        return []

    L = cfg.lookback
    window_size = 2 * L + 1

    # 1. Initial Fractal-based Detection (Vectorized)
    # A point is a swing high if it's the max in a centered window
    is_sh = high == high.rolling(
        window=window_size, center=True, min_periods=window_size
    ).max()
    # A point is a swing low if it's the min in a centered window
    is_sl = low == low.rolling(
        window=window_size, center=True, min_periods=window_size
    ).min()

    # Get the integer indices of potential swings
    sh_indices = np.where(is_sh)[0]
    sl_indices = np.where(is_sl)[0]

    # Combine and sort all potential swings
    all_swings = []
    for i in sh_indices:
        all_swings.append(SwingPoint(ts=i, price=high.iloc[i], kind="high"))
    for i in sl_indices:
        all_swings.append(SwingPoint(ts=i, price=low.iloc[i], kind="low"))

    # Sort by timestamp (integer index)
    all_swings.sort(key=lambda s: s.ts)

    if not all_swings:
        return []

    # 2. Filtering for significant swings
    significant_swings: List[SwingPoint] = [all_swings[0]]

    for i in range(1, len(all_swings)):
        current_swing = all_swings[i]
        last_sig_swing = significant_swings[-1]

        # Don't add consecutive swings of the same kind
        if current_swing.kind == last_sig_swing.kind:
            # If the new swing is "better" (higher high or lower low), replace the last one
            if (current_swing.kind == "high" and current_swing.price > last_sig_swing.price) or \
               (current_swing.kind == "low" and current_swing.price < last_sig_swing.price):
                significant_swings[-1] = current_swing
            continue

        # Check for minimum price change
        price_diff_pct = (
            abs(current_swing.price - last_sig_swing.price) / last_sig_swing.price
        )
        if price_diff_pct >= cfg.min_swing_size_pct:
            significant_swings.append(current_swing)

    return significant_swings
