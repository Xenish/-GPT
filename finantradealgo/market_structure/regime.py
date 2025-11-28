"""
Functions for identifying the current market trend regime (e.g., Bullish, Bearish).
"""
from typing import List

from .types import SwingPoint


def infer_trend_regime(swings: List[SwingPoint], min_swings: int) -> int:
    """
    Infers the current market trend based on a sequence of swing points.

    The logic is based on the classic definition of a trend:
    - Uptrend: A series of Higher Highs (HH) and Higher Lows (HL).
    - Downtrend: A series of Lower Highs (LH) and Lower Lows (LL).

    Args:
        swings: A time-sorted list of SwingPoint objects.
        min_swings: The minimum number of swings required to define a trend.

    Returns:
        An integer representing the trend: 1 for uptrend, -1 for downtrend,
        and 0 for neutral/undetermined.
    """
    if len(swings) < min_swings:
        return 0  # Not enough data to determine a trend

    # Find the last two swing highs and last two swing lows
    last_highs = [s for s in swings if s.kind == "high"][-2:]
    last_lows = [s for s in swings if s.kind == "low"][-2:]

    # Ensure we have enough of each type to make a comparison
    if len(last_highs) < 2 or len(last_lows) < 2:
        return 0

    sh1, sh2 = last_highs
    sl1, sl2 = last_lows

    is_higher_highs = sh2.price > sh1.price
    is_higher_lows = sl2.price > sl1.price
    is_lower_highs = sh2.price < sh1.price
    is_lower_lows = sl2.price < sl1.price

    # Check for a clear uptrend
    if is_higher_highs and is_higher_lows:
        return 1

    # Check for a clear downtrend
    if is_lower_highs and is_lower_lows:
        return -1

    # Otherwise, the trend is neutral or consolidating
    return 0
