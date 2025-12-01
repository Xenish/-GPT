"""
Debug and health check helpers for microstructure features.

Task S2.E3: Health check and summary tools for microstructure signals.
"""
import pandas as pd
import numpy as np
from typing import Optional

from finantradealgo.microstructure.types import MicrostructureSignals


def summarize_microstructure_features(
    features_df: pd.DataFrame,
    df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Generate a text summary of microstructure features for health checking.

    This function provides a quick overview of microstructure signal statistics,
    useful for debugging and validating feature computation.

    Task S2.E3: Health check helper for microstructure.

    Args:
        features_df: DataFrame with microstructure features (ms_* columns)
        df: Optional original OHLCV DataFrame for additional context

    Returns:
        String summary of microstructure analysis

    Example:
        >>> from finantradealgo.microstructure import compute_microstructure_df
        >>> df = pd.read_csv("ohlcv.csv", index_col=0, parse_dates=True)
        >>> features = compute_microstructure_df(df)
        >>> summary = summarize_microstructure_features(features, df)
        >>> print(summary)
    """
    lines = ["=" * 60]
    lines.append("MICROSTRUCTURE FEATURES SUMMARY")
    lines.append("=" * 60)

    # Basic stats
    lines.append(f"\nTotal bars: {len(features_df)}")

    # Expected columns from MicrostructureSignals
    expected_cols = MicrostructureSignals.columns()
    found_cols = [col for col in expected_cols if col in features_df.columns]
    missing_cols = [col for col in expected_cols if col not in features_df.columns]

    lines.append(f"Expected columns: {len(expected_cols)}")
    lines.append(f"Found columns: {len(found_cols)}")

    if missing_cols:
        lines.append(f"⚠️  Missing columns: {missing_cols}")

    # --- Feature Statistics ---
    lines.append("\n--- Feature Statistics ---")

    # Volatility Regime
    if "ms_vol_regime" in features_df.columns:
        vol_regime = features_df["ms_vol_regime"]
        lines.append(f"\nVolatility Regime (ms_vol_regime):")
        lines.append(f"  Current: {vol_regime.iloc[-1]:.3f}")
        lines.append(f"  Mean: {vol_regime.mean():.3f}")
        lines.append(f"  Std: {vol_regime.std():.3f}")
        lines.append(f"  Min: {vol_regime.min():.3f}, Max: {vol_regime.max():.3f}")

    # Chop
    if "ms_chop" in features_df.columns:
        chop = features_df["ms_chop"]
        current_chop = chop.iloc[-1]
        chop_str = "CHOPPY" if current_chop > 0.6 else "TRENDING" if current_chop < 0.4 else "MIXED"
        lines.append(f"\nChop Score (ms_chop):")
        lines.append(f"  Current: {current_chop:.3f} ({chop_str})")
        lines.append(f"  Mean: {chop.mean():.3f}")
        lines.append(f"  Choppy bars (>0.6): {(chop > 0.6).sum()} ({(chop > 0.6).sum() / len(chop) * 100:.1f}%)")
        lines.append(f"  Trending bars (<0.4): {(chop < 0.4).sum()} ({(chop < 0.4).sum() / len(chop) * 100:.1f}%)")

    # Bursts
    if "ms_burst_up" in features_df.columns and "ms_burst_down" in features_df.columns:
        burst_up = features_df["ms_burst_up"]
        burst_down = features_df["ms_burst_down"]
        n_burst_up = (burst_up != 0).sum()
        n_burst_down = (burst_down != 0).sum()
        lines.append(f"\nMomentum Bursts:")
        lines.append(f"  Burst Up events: {n_burst_up}")
        lines.append(f"  Burst Down events: {n_burst_down}")
        lines.append(f"  Total burst events: {n_burst_up + n_burst_down}")

    # Exhaustion
    if "ms_exhaustion_up" in features_df.columns and "ms_exhaustion_down" in features_df.columns:
        exhaustion_up = features_df["ms_exhaustion_up"]
        exhaustion_down = features_df["ms_exhaustion_down"]
        n_exhaustion_up = (exhaustion_up != 0).sum()
        n_exhaustion_down = (exhaustion_down != 0).sum()
        lines.append(f"\nExhaustion Signals:")
        lines.append(f"  Exhaustion Up events: {n_exhaustion_up}")
        lines.append(f"  Exhaustion Down events: {n_exhaustion_down}")
        lines.append(f"  Total exhaustion events: {n_exhaustion_up + n_exhaustion_down}")

    # Parabolic Trend
    if "ms_parabolic_trend" in features_df.columns:
        parabolic = features_df["ms_parabolic_trend"]
        n_parabolic = (parabolic != 0).sum()
        lines.append(f"\nParabolic Trend:")
        lines.append(f"  Parabolic events: {n_parabolic}")
        if n_parabolic > 0:
            lines.append(f"  Current: {parabolic.iloc[-1]:.3f}")

    # Order Book Imbalance
    if "ms_imbalance" in features_df.columns:
        imbalance = features_df["ms_imbalance"]
        non_zero = imbalance[imbalance != 0]
        if len(non_zero) > 0:
            lines.append(f"\nOrder Book Imbalance:")
            lines.append(f"  Non-zero values: {len(non_zero)}")
            lines.append(f"  Current: {imbalance.iloc[-1]:.3f}")
            lines.append(f"  Mean: {non_zero.mean():.3f}")
            lines.append(f"  Min: {non_zero.min():.3f}, Max: {non_zero.max():.3f}")
        else:
            lines.append(f"\nOrder Book Imbalance: No data (book_df not provided)")

    # Liquidity Sweeps
    if "ms_sweep_up" in features_df.columns and "ms_sweep_down" in features_df.columns:
        sweep_up = features_df["ms_sweep_up"]
        sweep_down = features_df["ms_sweep_down"]
        n_sweep_up = (sweep_up != 0).sum()
        n_sweep_down = (sweep_down != 0).sum()
        if n_sweep_up > 0 or n_sweep_down > 0:
            lines.append(f"\nLiquidity Sweeps:")
            lines.append(f"  Sweep Up events: {n_sweep_up}")
            lines.append(f"  Sweep Down events: {n_sweep_down}")
            lines.append(f"  Total sweep events: {n_sweep_up + n_sweep_down}")
        else:
            lines.append(f"\nLiquidity Sweeps: No events (trades_df not provided or no sweeps detected)")

    # Price context if original df provided
    if df is not None and len(df) > 0 and "close" in df.columns:
        current_price = df["close"].iloc[-1]
        price_change = df["close"].iloc[-1] - df["close"].iloc[0]
        price_change_pct = (price_change / df["close"].iloc[0]) * 100

        lines.append(f"\n--- Price Context ---")
        lines.append(f"  Current: {current_price:.2f}")
        lines.append(f"  Change: {price_change:+.2f} ({price_change_pct:+.2f}%)")

    # Data quality check
    lines.append(f"\n--- Data Quality ---")
    for col in found_cols:
        if col in features_df.columns:
            n_nan = features_df[col].isna().sum()
            n_inf = np.isinf(features_df[col]).sum()
            if n_nan > 0 or n_inf > 0:
                lines.append(f"  ⚠️  {col}: {n_nan} NaN, {n_inf} Inf values")

    lines.append("=" * 60)

    return "\n".join(lines)


def print_microstructure_summary(
    features_df: pd.DataFrame,
    df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Print microstructure features summary to console.

    Convenience wrapper around summarize_microstructure_features().

    Args:
        features_df: DataFrame with microstructure features
        df: Optional original OHLCV DataFrame
    """
    summary = summarize_microstructure_features(features_df, df)
    print(summary)


def get_microstructure_health_metrics(features_df: pd.DataFrame) -> dict:
    """
    Get microstructure feature health metrics as a dictionary.

    Useful for automated testing and monitoring.

    Args:
        features_df: DataFrame with microstructure features

    Returns:
        Dictionary with health metrics:
        - total_bars: Number of bars
        - expected_cols: Number of expected columns
        - found_cols: Number of found columns
        - missing_cols: List of missing column names
        - nan_counts: Dict of column -> NaN count
        - inf_counts: Dict of column -> Inf count
        - burst_events: Total burst events
        - exhaustion_events: Total exhaustion events
        - sweep_events: Total sweep events

    Example:
        >>> metrics = get_microstructure_health_metrics(features)
        >>> assert metrics["missing_cols"] == []
        >>> assert metrics["total_bars"] > 0
    """
    expected_cols = MicrostructureSignals.columns()
    found_cols = [col for col in expected_cols if col in features_df.columns]
    missing_cols = [col for col in expected_cols if col not in features_df.columns]

    nan_counts = {}
    inf_counts = {}
    for col in found_cols:
        if col in features_df.columns:
            nan_counts[col] = int(features_df[col].isna().sum())
            inf_counts[col] = int(np.isinf(features_df[col]).sum())

    burst_events = 0
    if "ms_burst_up" in features_df.columns and "ms_burst_down" in features_df.columns:
        burst_events = int(
            (features_df["ms_burst_up"] != 0).sum() +
            (features_df["ms_burst_down"] != 0).sum()
        )

    exhaustion_events = 0
    if "ms_exhaustion_up" in features_df.columns and "ms_exhaustion_down" in features_df.columns:
        exhaustion_events = int(
            (features_df["ms_exhaustion_up"] != 0).sum() +
            (features_df["ms_exhaustion_down"] != 0).sum()
        )

    sweep_events = 0
    if "ms_sweep_up" in features_df.columns and "ms_sweep_down" in features_df.columns:
        sweep_events = int(
            (features_df["ms_sweep_up"] != 0).sum() +
            (features_df["ms_sweep_down"] != 0).sum()
        )

    return {
        "total_bars": len(features_df),
        "expected_cols": len(expected_cols),
        "found_cols": len(found_cols),
        "missing_cols": missing_cols,
        "nan_counts": nan_counts,
        "inf_counts": inf_counts,
        "burst_events": burst_events,
        "exhaustion_events": exhaustion_events,
        "sweep_events": sweep_events,
    }
