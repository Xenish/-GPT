import pandas as pd
from typing import Literal, Optional
from finantradealgo.core.config import EventBarConfig

def build_event_bars(
    df: pd.DataFrame,  # 1m veya trade-based OHLCV
    cfg: EventBarConfig,
) -> pd.DataFrame:
    """
    Builds event-based bars (volume, dollar, tick) from a DataFrame of OHLCV data.

    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data, typically 1-minute bars
                           or trade-level data. Expected columns: 'open', 'high',
                           'low', 'close', 'volume'. The index should be a datetime.
        cfg (EventBarConfig): Configuration for event bar generation.

    Returns:
        pd.DataFrame: DataFrame of event-based bars.
    """
    if cfg.mode == "time":
        return df

    if cfg.mode == "time":
        return df
    
    if df.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'bar_start_ts', 'bar_end_ts'])

    bars_data = []
    current_bar_open = None
    current_bar_high = -float('inf')
    current_bar_low = float('inf')
    current_bar_close = None
    current_bar_volume = 0.0
    current_bar_notional = 0.0
    current_bar_ticks = 0
    bar_start_ts = None
    bar_end_ts = None

    for index, row in df.iterrows():
        # Ensure 'volume' and 'close' columns exist
        if 'volume' not in row or 'close' not in row:
            raise ValueError("DataFrame must contain 'volume' and 'close' columns for event bars.")

        if bar_start_ts is None:
            bar_start_ts = index

        current_bar_close = row['close']
        bar_end_ts = index

        if current_bar_open is None:
            current_bar_open = row['open']
        current_bar_high = max(current_bar_high, row['high'])
        current_bar_low = min(current_bar_low, row['low'])
        current_bar_volume += row['volume']
        current_bar_notional += row['close'] * row['volume'] # Assuming close * volume for notional
        current_bar_ticks += 1 # Each row is considered a tick for tick bar calculation

        should_close_bar = False
        if cfg.mode == "volume" and cfg.target_volume is not None:
            if current_bar_volume >= cfg.target_volume:
                should_close_bar = True
        elif cfg.mode == "dollar" and cfg.target_notional is not None:
            if current_bar_notional >= cfg.target_notional:
                should_close_bar = True
        elif cfg.mode == "tick" and cfg.target_ticks is not None:
            if current_bar_ticks >= cfg.target_ticks:
                should_close_bar = True

        if should_close_bar:
            bars_data.append({
                'open': current_bar_open,
                'high': current_bar_high,
                'low': current_bar_low,
                'close': current_bar_close,
                'volume': current_bar_volume,
                'bar_start_ts': bar_start_ts,
                'bar_end_ts': bar_end_ts
            })
            # Reset for next bar
            current_bar_open = None
            current_bar_high = -float('inf')
            current_bar_low = float('inf')
            current_bar_close = None
            current_bar_volume = 0.0
            current_bar_notional = 0.0
            current_bar_ticks = 0
            bar_start_ts = None
            bar_end_ts = None

    # Handle any remaining data as a final bar if it exists
    if bar_start_ts is not None and current_bar_volume > 0: # Check volume or ticks/notional
        bars_data.append({
            'open': current_bar_open,
            'high': current_bar_high,
            'low': current_bar_low,
            'close': current_bar_close,
            'volume': current_bar_volume,
            'bar_start_ts': bar_start_ts,
            'bar_end_ts': bar_end_ts
        })

    event_bars_df = pd.DataFrame(bars_data)
    if not event_bars_df.empty:
        event_bars_df.set_index('bar_end_ts', inplace=True)
    return event_bars_df