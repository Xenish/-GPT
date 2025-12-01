"""
Multi-timeframe validation functions.

Task S3.3: validate_multi_tf_alignment() for multi-TF data consistency checks.
"""
import logging
from typing import Dict, List, Optional

import pandas as pd

from finantradealgo.validation.timeframe_utils import timeframe_to_seconds
from finantradealgo.validation.ohlcv_validator import ValidationIssue, ValidationResult

logger = logging.getLogger(__name__)


def validate_multi_tf_alignment(
    dfs: Dict[str, pd.DataFrame],
    base_timeframe: str,
) -> ValidationResult:
    """
    Validate alignment and consistency across multiple timeframe DataFrames.

    Task S3.3: Multi-TF consistency validation.

    Checks:
    1. All DataFrames have DatetimeIndex
    2. Higher timeframes are proper multiples of base timeframe
    3. Timestamps align correctly (HTF bars start on expected boundaries)
    4. Price consistency (HTF OHLC matches aggregated LTF data)

    Args:
        dfs: Dictionary mapping timeframe strings to DataFrames
             Example: {"15m": df_15m, "1h": df_1h}
        base_timeframe: The base (lowest) timeframe to use as reference

    Returns:
        ValidationResult with any alignment issues detected

    Example:
        >>> dfs = {
        ...     "15m": df_15m,
        ...     "1h": df_1h,
        ... }
        >>> result = validate_multi_tf_alignment(dfs, base_timeframe="15m")
        >>> if not result.is_valid:
        ...     print(result.summary())
    """
    result = ValidationResult(is_valid=True)

    if not dfs:
        result.add_issue("empty_input", "error", "No DataFrames provided")
        return result

    if base_timeframe not in dfs:
        result.add_issue(
            "missing_base_tf",
            "error",
            f"Base timeframe '{base_timeframe}' not found in provided DataFrames"
        )
        return result

    # Validate all DataFrames have DatetimeIndex
    for tf, df in dfs.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            result.add_issue(
                f"invalid_index_{tf}",
                "error",
                f"DataFrame for timeframe '{tf}' does not have DatetimeIndex"
            )

    if not result.is_valid:
        return result  # Can't proceed without proper indices

    base_seconds = timeframe_to_seconds(base_timeframe)

    # Check timeframe relationships
    for tf in dfs.keys():
        if tf == base_timeframe:
            continue

        try:
            tf_seconds = timeframe_to_seconds(tf)
        except ValueError as e:
            result.add_issue(
                f"unknown_timeframe_{tf}",
                "error",
                f"Unknown timeframe: {tf}"
            )
            continue

        # Check if HTF is a proper multiple of base
        if tf_seconds < base_seconds:
            result.add_issue(
                f"tf_hierarchy_{tf}",
                "warning",
                f"Timeframe '{tf}' ({tf_seconds}s) is smaller than base '{base_timeframe}' ({base_seconds}s)"
            )
        elif tf_seconds % base_seconds != 0:
            result.add_issue(
                f"tf_multiple_{tf}",
                "warning",
                f"Timeframe '{tf}' ({tf_seconds}s) is not a clean multiple of base '{base_timeframe}' ({base_seconds}s)"
            )

    # Check timestamp alignment for higher timeframes
    for tf, df in dfs.items():
        if tf == base_timeframe or df.empty:
            continue

        try:
            tf_seconds = timeframe_to_seconds(tf)
        except ValueError:
            continue  # Already reported above

        # Check if HTF timestamps align to expected boundaries
        # For example, 1h bars should start at :00 minutes
        misaligned_count = 0
        for ts in df.index:
            # Calculate expected alignment based on timeframe
            # 1h should align to hour (minute=0, second=0)
            # 1d should align to day (hour=0, minute=0, second=0)
            if tf_seconds >= 3600:  # 1h or larger
                if ts.minute != 0 or ts.second != 0:
                    misaligned_count += 1
            elif tf_seconds >= 60:  # Minutes but less than hour
                if ts.second != 0:
                    misaligned_count += 1

        if misaligned_count > 0:
            result.add_issue(
                f"misaligned_timestamps_{tf}",
                "warning",
                f"Found {misaligned_count} misaligned timestamps in '{tf}' data"
            )

    # Check price consistency between base and higher timeframes
    base_df = dfs[base_timeframe]
    for tf, htf_df in dfs.items():
        if tf == base_timeframe or htf_df.empty:
            continue

        try:
            tf_seconds = timeframe_to_seconds(tf)
        except ValueError:
            continue

        if tf_seconds <= base_seconds:
            continue  # Skip LTF or equal TF

        # Check if OHLC columns exist
        required_cols = ["open", "high", "low", "close"]
        if not all(col in base_df.columns for col in required_cols):
            continue
        if not all(col in htf_df.columns for col in required_cols):
            continue

        # Sample a few HTF bars and check if they match aggregated base TF data
        sample_size = min(10, len(htf_df))
        inconsistencies = 0

        for i in range(sample_size):
            htf_bar = htf_df.iloc[i]
            htf_ts = htf_df.index[i]

            # Find corresponding base TF bars
            # HTF bar covers from htf_ts to htf_ts + tf_seconds
            bar_end = htf_ts + pd.Timedelta(seconds=tf_seconds)
            base_bars = base_df[(base_df.index >= htf_ts) & (base_df.index < bar_end)]

            if base_bars.empty:
                continue

            # Aggregate base bars and compare
            expected_open = base_bars.iloc[0]["open"]
            expected_high = base_bars["high"].max()
            expected_low = base_bars["low"].min()
            expected_close = base_bars.iloc[-1]["close"]

            # Allow small tolerance for floating point errors
            tolerance = 1e-6

            if abs(htf_bar["open"] - expected_open) > tolerance:
                inconsistencies += 1
            elif abs(htf_bar["high"] - expected_high) > tolerance:
                inconsistencies += 1
            elif abs(htf_bar["low"] - expected_low) > tolerance:
                inconsistencies += 1
            elif abs(htf_bar["close"] - expected_close) > tolerance:
                inconsistencies += 1

        if inconsistencies > 0:
            result.add_issue(
                f"price_inconsistency_{tf}",
                "error",
                f"Found {inconsistencies}/{sample_size} sampled bars with price inconsistencies "
                f"between '{base_timeframe}' and '{tf}'"
            )

    return result
