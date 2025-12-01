"""
Live-specific validation functions.

Task S3.E4: Suspect bar detection for live/paper trading environments.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from finantradealgo.validation.ohlcv_validator import ValidationResult

logger = logging.getLogger(__name__)


def detect_suspect_bars(
    df: pd.DataFrame,
    volume_z_threshold: float = -3.0,
    range_z_threshold: float = -3.0,
    lookback_window: int = 100,
) -> ValidationResult:
    """
    Detect suspect bars in live/paper trading environments.

    Task S3.E4: Live-specific validation guard.

    Suspect bars are those with anomalous characteristics that might indicate:
    - Data feed issues
    - Exchange connectivity problems
    - Missing/delayed data

    Detection criteria:
    1. Extremely low volume (z-score < volume_z_threshold)
    2. Extremely narrow range (z-score < range_z_threshold)
    3. Combination of both

    Args:
        df: OHLCV DataFrame to check
        volume_z_threshold: Z-score threshold for low volume detection (negative)
        range_z_threshold: Z-score threshold for narrow range detection (negative)
        lookback_window: Rolling window for z-score calculation

    Returns:
        ValidationResult with detected suspect bars

    Example:
        >>> df = pd.read_csv("live_ohlcv.csv", index_col=0, parse_dates=True)
        >>> result = detect_suspect_bars(df)
        >>> if result.warnings_count > 0:
        ...     print(f"Found {result.warnings_count} suspect bars")
    """
    result = ValidationResult(is_valid=True)

    if df.empty:
        result.add_issue("empty_dataframe", "error", "DataFrame is empty")
        return result

    if len(df) < lookback_window:
        # Not enough data for reliable z-score calculation
        result.add_issue(
            "insufficient_data",
            "warning",
            f"Insufficient data for suspect bar detection ({len(df)} < {lookback_window})"
        )
        return result

    # Required columns
    if "volume" not in df.columns:
        result.add_issue("missing_volume", "error", "Missing 'volume' column")
        return result

    required_price_cols = ["high", "low"]
    if not all(col in df.columns for col in required_price_cols):
        result.add_issue(
            "missing_price_columns",
            "error",
            f"Missing price columns: {[c for c in required_price_cols if c not in df.columns]}"
        )
        return result

    # Calculate bar range (high - low)
    bar_range = df["high"] - df["low"]

    # Calculate rolling z-scores
    volume_rolling_mean = df["volume"].rolling(window=lookback_window).mean()
    volume_rolling_std = df["volume"].rolling(window=lookback_window).std()
    volume_z_scores = (df["volume"] - volume_rolling_mean) / volume_rolling_std

    range_rolling_mean = bar_range.rolling(window=lookback_window).mean()
    range_rolling_std = bar_range.rolling(window=lookback_window).std()
    range_z_scores = (bar_range - range_rolling_mean) / range_rolling_std

    # Detect suspect bars based on z-scores
    suspect_volume = volume_z_scores < volume_z_threshold
    suspect_range = range_z_scores < range_z_threshold
    suspect_both = suspect_volume & suspect_range

    # Count suspects
    suspect_volume_count = suspect_volume.sum()
    suspect_range_count = suspect_range.sum()
    suspect_both_count = suspect_both.sum()

    # Report findings
    if suspect_volume_count > 0:
        suspect_indices = df.index[suspect_volume].tolist()
        result.add_issue(
            "suspect_low_volume",
            "warning",
            f"Found {suspect_volume_count} bars with extremely low volume "
            f"(z < {volume_z_threshold})",
            affected_rows=suspect_indices[:10]  # Limit to first 10 for brevity
        )

    if suspect_range_count > 0:
        suspect_indices = df.index[suspect_range].tolist()
        result.add_issue(
            "suspect_narrow_range",
            "warning",
            f"Found {suspect_range_count} bars with extremely narrow range "
            f"(z < {range_z_threshold})",
            affected_rows=suspect_indices[:10]
        )

    if suspect_both_count > 0:
        suspect_indices = df.index[suspect_both].tolist()
        result.add_issue(
            "suspect_both",
            "warning",
            f"Found {suspect_both_count} bars with BOTH low volume AND narrow range "
            f"(possible data feed issue)",
            affected_rows=suspect_indices[:10]
        )

    return result


def validate_live_bar(
    bar: pd.Series,
    recent_df: pd.DataFrame,
    volume_z_threshold: float = -3.0,
    range_z_threshold: float = -3.0,
) -> ValidationResult:
    """
    Validate a single live bar against recent historical data.

    Task S3.E4: Real-time bar validation for live trading.

    This is a lightweight version of detect_suspect_bars() optimized for
    single-bar validation in live/paper trading loops.

    Args:
        bar: Single bar as pd.Series with OHLCV data
        recent_df: Recent historical bars for context (50-200 bars recommended)
        volume_z_threshold: Z-score threshold for low volume
        range_z_threshold: Z-score threshold for narrow range

    Returns:
        ValidationResult indicating if bar is suspect

    Example:
        >>> latest_bar = df.iloc[-1]
        >>> recent_bars = df.iloc[-100:]
        >>> result = validate_live_bar(latest_bar, recent_bars)
        >>> if not result.is_valid:
        ...     logger.warning("Suspect bar detected!")
    """
    result = ValidationResult(is_valid=True)

    # Basic validation
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in bar.index:
            result.add_issue(
                f"missing_{col}",
                "error",
                f"Bar missing required column: {col}"
            )
            return result

    if recent_df.empty or len(recent_df) < 20:
        result.add_issue(
            "insufficient_context",
            "warning",
            "Insufficient historical context for validation"
        )
        return result

    # Calculate bar metrics
    bar_volume = bar["volume"]
    bar_range = bar["high"] - bar["low"]

    # Calculate z-scores against recent data
    recent_volumes = recent_df["volume"]
    recent_ranges = recent_df["high"] - recent_df["low"]

    volume_mean = recent_volumes.mean()
    volume_std = recent_volumes.std()
    range_mean = recent_ranges.mean()
    range_std = recent_ranges.std()

    if volume_std > 0:
        volume_z = (bar_volume - volume_mean) / volume_std
        if volume_z < volume_z_threshold:
            result.add_issue(
                "suspect_low_volume",
                "warning",
                f"Bar has extremely low volume (z={volume_z:.2f})"
            )

    if range_std > 0:
        range_z = (bar_range - range_mean) / range_std
        if range_z < range_z_threshold:
            result.add_issue(
                "suspect_narrow_range",
                "warning",
                f"Bar has extremely narrow range (z={range_z:.2f})"
            )

    return result
