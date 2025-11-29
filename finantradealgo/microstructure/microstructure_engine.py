from typing import Optional

import pandas as pd

from finantradealgo.microstructure.burst_detector import compute_bursts
from finantradealgo.microstructure.chop_detector import compute_chop
from finantradealgo.microstructure.config import MicrostructureConfig
from finantradealgo.microstructure.exhaustion import compute_exhaustion
from finantradealgo.microstructure.imbalance import compute_imbalance_from_df
from finantradealgo.microstructure.liquidity_sweep import detect_liquidity_sweep
from finantradealgo.microstructure.parabolic_detector import compute_parabolic_trend
from finantradealgo.microstructure.types import MicrostructureSignals
from finantradealgo.microstructure.volatility_regime import compute_volatility_regime


def compute_microstructure_df(
    df: pd.DataFrame,
    cfg: Optional[MicrostructureConfig] = None,
    trades_df: Optional[pd.DataFrame] = None,
    book_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Computes all microstructure signals and returns them as a new DataFrame.

    This is the main entry point for offline/batch feature generation.
    Order book and trade-based features are optional and will only be computed
    if the corresponding DataFrames are provided.

    Args:
        df: Input DataFrame with OHLCV data (must have a datetime index).
        cfg: Configuration for the microstructure signals.
        trades_df: Optional DataFrame with trade data.
        book_df: Optional DataFrame with order book snapshot data.

    Returns:
        A new DataFrame containing only the microstructure signal columns.
    """
    if cfg is None:
        cfg = MicrostructureConfig()

    features_df = pd.DataFrame(index=df.index)
    all_cols = MicrostructureSignals.columns()

    # Initialize all columns to 0.0, they will be selectively overwritten
    for col in all_cols:
        features_df[col] = 0.0

    if not cfg.enabled:
        return features_df

    # --- Sprint 2: OHLCV-based features ---
    features_df["ms_vol_regime"] = compute_volatility_regime(df["close"], cfg.vol_regime)
    features_df["ms_chop"] = compute_chop(df["close"], cfg.chop)

    # --- Sprint 3: OHLCV-based features ---
    burst_up, burst_down = compute_bursts(df["close"], cfg.burst)
    features_df["ms_burst_up"] = burst_up
    features_df["ms_burst_down"] = burst_down

    exhaustion_up, exhaustion_down = compute_exhaustion(
        df["close"], df["volume"], cfg.exhaustion
    )
    features_df["ms_exhaustion_up"] = exhaustion_up
    features_df["ms_exhaustion_down"] = exhaustion_down

    features_df["ms_parabolic_trend"] = compute_parabolic_trend(
        df["close"], cfg.parabolic
    )

    # --- Sprint 4: Order book and trade features (Optional) ---
    # TODO (Option B): This is a simple integration. A more performant version
    # would use a more efficient mapping between bars and the raw trade/book data
    # instead of iterating or using a simple reindex/ffill.

    # Calculate imbalance if order book data is provided
    if book_df is not None and not book_df.empty:
        imbalance_series = compute_imbalance_from_df(book_df, cfg.imbalance.depth)
        if imbalance_series is not None:
            # Align imbalance data with the main OHLCV dataframe
            features_df["ms_imbalance"] = imbalance_series.reindex(
                df.index, method="ffill"
            ).fillna(0)

    # Calculate liquidity sweeps if trade data is provided
    if trades_df is not None and not trades_df.empty:
        # This implementation iterates over bars, which is not fully vectorized.
        # It's a starting point for integration.

        # Check if DataFrame has explicit bar timestamps (from event bars)
        has_bar_start_col = 'bar_start_ts' in df.columns
        has_bar_end_col = 'bar_end_ts' in df.columns
        has_explicit_bounds = has_bar_start_col  # bar_end_ts is typically the index

        if not has_explicit_bounds:
            # Fallback for regular time bars: infer timeframe from index differences
            timeframe_delta = df.index.to_series().diff().min()

        def sweep_for_bar(bar):
            if has_explicit_bounds:
                # Use explicit bar timestamps from event bars
                bar_start_ts = bar['bar_start_ts']
                # bar_end_ts can be either a column or the index
                bar_end_ts = bar['bar_end_ts'] if has_bar_end_col else bar.name
            else:
                # Fallback for regular time bars
                bar_start_ts = bar.name  # Assuming index is the timestamp
                bar_end_ts = bar_start_ts + timeframe_delta

            return detect_liquidity_sweep(
                bar.open, bar.close, bar_start_ts, bar_end_ts, trades_df, cfg.sweep
            )

        sweeps = df.apply(sweep_for_bar, axis=1, result_type="expand")
        features_df["ms_sweep_up"] = sweeps[0]
        features_df["ms_sweep_down"] = sweeps[1]

    # Ensure column order is consistent
    return features_df[all_cols]
