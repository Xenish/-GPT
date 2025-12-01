"""
Timeframe utilities for data validation.

Task S3.E1: TIMEFRAME_TO_SECONDS map and gap detection.
"""
from typing import Dict, List, Tuple

import pandas as pd


# Task S3.E1: Timeframe to seconds mapping
TIMEFRAME_TO_SECONDS: Dict[str, int] = {
    "1s": 1,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
    "1M": 2592000,  # Approximate (30 days)
}


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Convert timeframe string to seconds.

    Args:
        timeframe: Timeframe string (e.g., "15m", "1h", "1d")

    Returns:
        Number of seconds in the timeframe

    Raises:
        ValueError: If timeframe is not recognized
    """
    if timeframe not in TIMEFRAME_TO_SECONDS:
        raise ValueError(
            f"Unknown timeframe: {timeframe}. "
            f"Supported: {list(TIMEFRAME_TO_SECONDS.keys())}"
        )
    return TIMEFRAME_TO_SECONDS[timeframe]


def detect_gaps(
    index: pd.DatetimeIndex,
    timeframe: str,
    max_gap_multiplier: float = 2.0,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, float]]:
    """
    Detect gaps in a DatetimeIndex based on expected timeframe.

    Task S3.E1: Realistic gap detection based on timeframe.

    Args:
        index: DatetimeIndex to check for gaps
        timeframe: Expected timeframe (e.g., "15m", "1h")
        max_gap_multiplier: Maximum gap size as multiple of timeframe
                           (e.g., 2.0 means gaps > 2x timeframe are reported)

    Returns:
        List of (start_time, end_time, gap_multiplier) tuples for detected gaps

    Example:
        >>> index = pd.date_range("2024-01-01", periods=100, freq="15min")
        >>> # Simulate a gap
        >>> index = index.drop(index[50:55])
        >>> gaps = detect_gaps(index, "15m", max_gap_multiplier=2.0)
        >>> len(gaps)
        1
    """
    if len(index) < 2:
        return []

    expected_seconds = timeframe_to_seconds(timeframe)
    max_gap_seconds = expected_seconds * max_gap_multiplier

    # Calculate time differences between consecutive timestamps
    diffs = index.to_series().diff()

    # Find gaps larger than threshold
    gaps = []
    for i in range(1, len(index)):
        gap_seconds = diffs.iloc[i].total_seconds()
        if gap_seconds > max_gap_seconds:
            gap_multiplier = gap_seconds / expected_seconds
            gaps.append((index[i - 1], index[i], gap_multiplier))

    return gaps


def infer_timeframe(index: pd.DatetimeIndex) -> str:
    """
    Infer timeframe from DatetimeIndex by analyzing intervals.

    Args:
        index: DatetimeIndex to analyze

    Returns:
        Inferred timeframe string (e.g., "15m", "1h")

    Raises:
        ValueError: If timeframe cannot be inferred
    """
    if len(index) < 2:
        raise ValueError("Need at least 2 timestamps to infer timeframe")

    # Calculate most common interval
    diffs = index.to_series().diff().dropna()
    most_common_seconds = diffs.mode()[0].total_seconds()

    # Find closest matching timeframe
    closest_tf = None
    min_diff = float("inf")

    for tf, seconds in TIMEFRAME_TO_SECONDS.items():
        diff = abs(seconds - most_common_seconds)
        if diff < min_diff:
            min_diff = diff
            closest_tf = tf

    # Check if the match is reasonable (within 10%)
    if closest_tf and min_diff / TIMEFRAME_TO_SECONDS[closest_tf] > 0.1:
        raise ValueError(
            f"Could not confidently infer timeframe. "
            f"Most common interval: {most_common_seconds}s, "
            f"closest match: {closest_tf} ({TIMEFRAME_TO_SECONDS[closest_tf]}s)"
        )

    return closest_tf
