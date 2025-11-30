from typing import Tuple

import pandas as pd

from finantradealgo.microstructure.config import LiquiditySweepConfig


def detect_liquidity_sweep(
    bar_open: float,
    bar_close: float,
    bar_start_ts: pd.Timestamp,
    bar_end_ts: pd.Timestamp,
    trades_df: pd.DataFrame,
    cfg: LiquiditySweepConfig,
) -> Tuple[float, float]:
    """
    Detects a single liquidity sweep event for a given bar.

    A sweep is a large burst of one-sided trading that moves the price in
    the corresponding direction.

    Args:
        bar_open: The open price of the bar.
        bar_close: The close price of the bar.
        bar_start_ts: The start timestamp of the bar.
        bar_end_ts: The end timestamp of the bar.
        trades_df: A DataFrame of ALL trades, which will be filtered internally.
                   Must have 'timestamp', 'side', 'price', 'size' columns.
        cfg: Configuration for the sweep detection.

    Returns:
        A tuple of (sweep_up, sweep_down) notionals. Only one can be non-zero.
    """
    sweep_up = 0.0
    sweep_down = 0.0

    if trades_df is None or trades_df.empty:
        return sweep_up, sweep_down

    # Contract: trades_df must have DatetimeIndex
    if not isinstance(trades_df.index, pd.DatetimeIndex):
        raise ValueError(
            "trades_df must have a DatetimeIndex. "
            "Use load_trades() or ensure trades_df.set_index('timestamp') was called."
        )

    # 1. Filter trades to the relevant time window
    window_start = bar_start_ts - pd.to_timedelta(cfg.lookback_ms, unit="ms")
    relevant_trades = trades_df.loc[window_start:bar_end_ts]

    if relevant_trades.empty:
        return sweep_up, sweep_down

    # 2. Calculate buy and sell notional
    notional = relevant_trades["price"] * relevant_trades["size"]
    buy_notional = notional[relevant_trades["side"] == "buy"].sum()
    sell_notional = notional[relevant_trades["side"] == "sell"].sum()

    # 3. Check for price impact and threshold
    price_impact_up = bar_close > bar_open
    price_impact_down = bar_close < bar_open

    # Check for upward sweep
    if price_impact_up and buy_notional > cfg.notional_threshold:
        sweep_up = buy_notional

    # Check for downward sweep
    if price_impact_down and sell_notional > cfg.notional_threshold:
        sweep_down = sell_notional

    return sweep_up, sweep_down
