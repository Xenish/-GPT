"""
Functions for smoothing price data to identify more robust structure.
"""
import pandas as pd
from finantradealgo.market_structure.config import SmoothingConfig


def smooth_price(df: pd.DataFrame, cfg: SmoothingConfig) -> pd.DataFrame:
    """
    Apply price smoothing using a rolling moving average.

    Args:
        df: Input DataFrame with 'close' column (required)
        cfg: SmoothingConfig with smoothing parameters

    Returns:
        DataFrame with added 'price_smooth' column

    Contract:
        - Input df must have 'close' column
        - Output always contains 'price_smooth' column
        - If smoothing is disabled or window <= 1, price_smooth equals close
    """
    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain 'close' column")

    # Create a copy to avoid modifying the original
    df = df.copy()

    if not cfg.enabled or cfg.price_ma_window <= 1:
        df["price_smooth"] = df["close"]
        return df

    df["price_smooth"] = df["close"].rolling(
        cfg.price_ma_window, min_periods=1
    ).mean()
    return df


def filter_swing_points(
    df: pd.DataFrame,
    cfg: SmoothingConfig,
) -> pd.DataFrame:
    """
    Filter swing points to remove noise and micro-swings.

    Applies two filtering techniques:
    1. Distance filtering: Removes swings that are too close together
    2. Z-score filtering: Removes swings with very small range (micro-swings)

    Args:
        df: DataFrame with 'ms_swing_high' and 'ms_swing_low' columns
        cfg: SmoothingConfig with filtering parameters

    Returns:
        DataFrame with filtered swing points

    Contract:
        - Input df must have 'high', 'low', 'ms_swing_high', 'ms_swing_low' columns
        - Output preserves all columns, modifies swing columns in-place
    """
    if not cfg.enabled:
        return df

    # Check required columns
    required_cols = {"high", "low", "ms_swing_high", "ms_swing_low"}
    if not required_cols.issubset(df.columns):
        # If swing columns don't exist yet, return as-is
        return df

    df = df.copy()

    # Get swing indices (as integer positions for distance calculation)
    swing_high_idx = df.index[df["ms_swing_high"] == 1].tolist()
    swing_low_idx = df.index[df["ms_swing_low"] == 1].tolist()

    # Map index values to integer positions
    idx_to_pos = {idx: pos for pos, idx in enumerate(df.index)}

    # Combine and sort all swings with their types
    all_swings = [(idx, idx_to_pos[idx], "high", df.loc[idx, "high"]) for idx in swing_high_idx]
    all_swings += [(idx, idx_to_pos[idx], "low", df.loc[idx, "low"]) for idx in swing_low_idx]
    all_swings.sort(key=lambda x: x[1])  # Sort by integer position

    if len(all_swings) < 2:
        return df

    # 1. Distance filtering: Remove swings too close together
    filtered_swings = [all_swings[0]]
    for i in range(1, len(all_swings)):
        idx, pos, kind, price = all_swings[i]
        last_idx, last_pos, last_kind, last_price = filtered_swings[-1]

        # Calculate distance in bars (using integer positions)
        distance = pos - last_pos

        if distance < cfg.swing_min_distance:
            # Keep the more extreme swing
            if kind == last_kind:
                # Same type: keep the better one
                if (kind == "high" and price > last_price) or (kind == "low" and price < last_price):
                    filtered_swings[-1] = (idx, pos, kind, price)
            else:
                # Different types but too close: keep the one with larger range
                current_range = abs(price - last_price)
                if len(filtered_swings) > 1:
                    prev_idx, prev_pos, prev_kind, prev_price = filtered_swings[-2]
                    last_range = abs(last_price - prev_price)
                    if current_range > last_range:
                        filtered_swings[-1] = (idx, pos, kind, price)
                else:
                    filtered_swings.append((idx, pos, kind, price))
        else:
            filtered_swings.append((idx, pos, kind, price))

    # 2. Z-score filtering: Remove micro-swings
    if len(filtered_swings) > 2:
        # Calculate swing ranges
        ranges = []
        for i in range(1, len(filtered_swings)):
            idx, pos, kind, price = filtered_swings[i]
            prev_idx, prev_pos, prev_kind, prev_price = filtered_swings[i - 1]
            swing_range = abs(price - prev_price)
            ranges.append(swing_range)

        # Calculate z-score for each range
        import numpy as np
        ranges_array = np.array(ranges)
        mean_range = np.mean(ranges_array)
        std_range = np.std(ranges_array)

        if std_range > 0:
            # Filter out swings with very small ranges (below threshold z-score)
            final_swings = [filtered_swings[0]]
            for i in range(1, len(filtered_swings)):
                z_score = (ranges[i - 1] - mean_range) / std_range
                if z_score >= cfg.swing_min_zscore:
                    final_swings.append(filtered_swings[i])
        else:
            final_swings = filtered_swings
    else:
        final_swings = filtered_swings

    # Reset swing columns and apply filtered swings
    df["ms_swing_high"] = 0
    df["ms_swing_low"] = 0

    for idx, pos, kind, price in final_swings:
        if kind == "high":
            df.loc[idx, "ms_swing_high"] = 1
        else:
            df.loc[idx, "ms_swing_low"] = 1

    return df
