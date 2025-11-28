"""
Core functions for building event-based bars (e.g., volume, dollar, tick bars)
from finer-granularity data like 1-minute bars or raw trades.
"""
import pandas as pd
import numpy as np

from finantradealgo.system.config_loader import EventBarConfig


def build_event_bars(
    df: pd.DataFrame,
    cfg: EventBarConfig,
) -> pd.DataFrame:
    """
    Aggregates a DataFrame of finer-granularity bars (e.g., 1-minute OHLCV)
    into event-based bars.

    Args:
        df: Input DataFrame, must have 'timestamp', 'open', 'high', 'low',
            'close', and 'volume' columns. It is assumed to be sorted by time.
        cfg: The configuration for event bar generation.

    Returns:
        A new DataFrame with bars aggregated according to the specified mode.
        Returns the original DataFrame if mode is 'time'.
    """
    if cfg.mode == "time":
        return df

    if df.empty:
        return pd.DataFrame()

    # Ensure timestamp is a column for processing
    if df.index.name == 'timestamp':
        df = df.reset_index()

    event_bars = []
    
    threshold = 0
    metric_col = ''
    if cfg.mode == 'volume':
        threshold = cfg.target_volume
        metric_col = 'volume'
    elif cfg.mode == 'dollar':
        threshold = cfg.target_notional
        df['notional'] = df['close'] * df['volume']
        metric_col = 'notional'
    elif cfg.mode == 'tick':
        threshold = cfg.target_ticks
        df['ticks'] = 1 # Each row is one "tick"
        metric_col = 'ticks'
    else:
        raise ValueError(f"Unsupported event bar mode: {cfg.mode}")

    if not threshold or threshold <= 0:
        raise ValueError(f"Target for mode '{cfg.mode}' must be a positive number.")

    # Loop through the input DataFrame to build bars
    current_bar_start_idx = 0
    cumulative_metric = 0.0

    for i in range(len(df)):
        cumulative_metric += df[metric_col].iloc[i]

        if cumulative_metric >= threshold:
            bar_slice = df.iloc[current_bar_start_idx : i + 1]
            
            bar_open = bar_slice['open'].iloc[0]
            bar_high = bar_slice['high'].max()
            bar_low = bar_slice['low'].min()
            bar_close = bar_slice['close'].iloc[-1]
            bar_volume = bar_slice['volume'].sum()
            
            bar_start_ts = bar_slice['timestamp'].iloc[0]
            bar_end_ts = bar_slice['timestamp'].iloc[-1]

            event_bars.append({
                'timestamp': bar_end_ts,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': bar_volume,
                'bar_start_ts': bar_start_ts,
                'bar_end_ts': bar_end_ts,
            })
            
            # Reset for next bar
            current_bar_start_idx = i + 1
            cumulative_metric = 0.0

    if not event_bars:
        return pd.DataFrame()

    # Create final DataFrame and set index
    result_df = pd.DataFrame(event_bars)
    result_df.set_index('timestamp', inplace=True)
    
    return result_df
