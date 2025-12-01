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

    **This is the SINGLE ENTRY-POINT for all microstructure feature computation.**

    This function serves as the main entry point for offline/batch feature generation.
    Order book and trade-based features are optional and will only be computed
    if the corresponding DataFrames are provided.

    Task S2.2: Single entry-point for microstructure features.

    Design principles:
    - All microstructure features must go through this function
    - Direct calls to individual detectors (compute_chop, compute_bursts, etc.)
      should only be made from within this function
    - Features are standardized with ms_* prefix (defined in MicrostructureSignals.columns())
    - Output contract is enforced: all columns are guaranteed to exist

    Args:
        df: Input DataFrame with OHLCV data (must have a datetime index).
        cfg: Configuration for the microstructure signals. If None, uses defaults.
        trades_df: Optional DataFrame with trade data.
                   Must have DatetimeIndex if provided.
                   Required columns: side, price, size
                   See MicrostructureInputSpec for full contract (Task S2.E1)
        book_df: Optional DataFrame with order book snapshot data.
                 Must have DatetimeIndex if provided.
                 Required columns: depends on depth (bid_price_0...N, ask_price_0...N, etc.)

    Returns:
        A new DataFrame containing only the microstructure signal columns.
        All columns from MicrostructureSignals.columns() are guaranteed to exist.

    Example:
        >>> df = pd.read_csv("ohlcv.csv", index_col=0, parse_dates=True)
        >>> cfg = MicrostructureConfig()
        >>> features = compute_microstructure_df(df, cfg)
        >>> assert "ms_chop" in features.columns
        >>> assert "ms_burst_up" in features.columns
    """
    if cfg is None:
        cfg = MicrostructureConfig()

    # --- Task S2.3: Enforce input contracts ---
    # Validate OHLCV DataFrame
    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
    assert not df.empty, "df cannot be empty"
    required_ohlcv_cols = ["open", "high", "low", "close", "volume"]
    for col in required_ohlcv_cols:
        assert col in df.columns, f"df must contain '{col}' column"

    # Validate trades_df contract if provided
    if trades_df is not None:
        assert isinstance(trades_df, pd.DataFrame), "trades_df must be a pandas DataFrame"
        assert not trades_df.empty, "trades_df cannot be empty if provided"
        required_trade_cols = ["side", "price", "size"]
        for col in required_trade_cols:
            assert col in trades_df.columns, (
                f"trades_df must contain '{col}' column. "
                f"Found columns: {list(trades_df.columns)}"
            )
        assert isinstance(trades_df.index, pd.DatetimeIndex), (
            "trades_df must have DatetimeIndex"
        )

    # Validate book_df contract if provided
    if book_df is not None:
        assert isinstance(book_df, pd.DataFrame), "book_df must be a pandas DataFrame"
        assert not book_df.empty, "book_df cannot be empty if provided"
        assert isinstance(book_df.index, pd.DatetimeIndex), (
            "book_df must have DatetimeIndex"
        )
        # Book validation is partial - full depth validation is done in compute_imbalance_from_df

    # --- Task S2.E2: Truncate trades/book data to max_lookback_seconds ---
    # This prevents excessive lookback in live/paper trading environments
    if cfg.max_lookback_seconds > 0:
        # Use the last timestamp in df as "now"
        now = df.index[-1]
        cutoff_time = now - pd.Timedelta(seconds=cfg.max_lookback_seconds)

        # Truncate trades_df if provided
        if trades_df is not None and not trades_df.empty:
            trades_df = trades_df[trades_df.index >= cutoff_time]

        # Truncate book_df if provided
        if book_df is not None and not book_df.empty:
            book_df = book_df[book_df.index >= cutoff_time]

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


# Alias for consistency with naming convention
compute_microstructure_features = compute_microstructure_df
