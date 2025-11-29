"""
Tests for liquidity sweep with event bars time semantics.

Ensures that liquidity sweep detection works correctly with explicit bar_start_ts and bar_end_ts.
"""
import pandas as pd
import pytest
from datetime import datetime, timedelta

from finantradealgo.microstructure.liquidity_sweep import detect_liquidity_sweep
from finantradealgo.microstructure.config import LiquiditySweepConfig
from finantradealgo.microstructure.microstructure_engine import compute_microstructure_df
from finantradealgo.microstructure.config import MicrostructureConfig


def test_liquidity_sweep_with_explicit_timestamps():
    """Test that detect_liquidity_sweep works with explicit bar_start_ts and bar_end_ts."""
    # Create sample trades
    trades_data = {
        'timestamp': pd.date_range('2023-01-01 09:00:00', periods=10, freq='1s'),
        'side': ['buy'] * 5 + ['sell'] * 5,
        'price': [100 + i * 0.1 for i in range(10)],
        'size': [10.0] * 10,
    }
    trades_df = pd.DataFrame(trades_data).set_index('timestamp')

    # Test with explicit timestamps
    bar_open = 100.0
    bar_close = 101.0  # Price went up
    bar_start_ts = pd.Timestamp('2023-01-01 09:00:00')
    bar_end_ts = pd.Timestamp('2023-01-01 09:00:10')

    cfg = LiquiditySweepConfig(
        lookback_ms=0,
        notional_threshold=100.0
    )

    sweep_up, sweep_down = detect_liquidity_sweep(
        bar_open, bar_close, bar_start_ts, bar_end_ts, trades_df, cfg
    )

    # Should detect upward sweep since price went up and there were buy trades
    assert sweep_up > 0, "Should detect upward sweep"
    assert sweep_down == 0, "Should not detect downward sweep"


def test_microstructure_with_event_bar_columns():
    """Test that microstructure engine works with event bar columns."""
    # Create OHLCV data with event bar columns
    num_bars = 10
    data = {
        'open': [100 + i for i in range(num_bars)],
        'high': [101 + i for i in range(num_bars)],
        'low': [99 + i for i in range(num_bars)],
        'close': [100.5 + i for i in range(num_bars)],
        'volume': [1000.0] * num_bars,
        'bar_start_ts': pd.date_range('2023-01-01 09:00:00', periods=num_bars, freq='1min'),
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range('2023-01-01 09:01:00', periods=num_bars, freq='1min')
    df.index.name = 'bar_end_ts'

    # Create sample trades
    trades_data = {
        'timestamp': pd.date_range('2023-01-01 09:00:00', periods=50, freq='10s'),
        'side': (['buy'] * 25 + ['sell'] * 25),
        'price': [100 + i * 0.1 for i in range(50)],
        'size': [10.0] * 50,
    }
    trades_df = pd.DataFrame(trades_data).set_index('timestamp')

    # Configure microstructure (use default config)
    cfg = MicrostructureConfig(
        sweep=LiquiditySweepConfig(lookback_ms=0, notional_threshold=0)
    )

    # Should not raise an error and should use bar_start_ts column
    result = compute_microstructure_df(df, cfg, trades_df=trades_df)

    # Verify sweep columns exist
    assert 'ms_sweep_up' in result.columns
    assert 'ms_sweep_down' in result.columns

    # Verify no NaNs (all bars should have been processed)
    assert not result['ms_sweep_up'].isna().any()
    assert not result['ms_sweep_down'].isna().any()


def test_microstructure_without_event_bar_columns():
    """Test that microstructure engine still works without event bar columns (fallback)."""
    # Create OHLCV data WITHOUT event bar columns (regular time bars)
    num_bars = 10
    data = {
        'open': [100 + i for i in range(num_bars)],
        'high': [101 + i for i in range(num_bars)],
        'low': [99 + i for i in range(num_bars)],
        'close': [100.5 + i for i in range(num_bars)],
        'volume': [1000.0] * num_bars,
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range('2023-01-01 09:00:00', periods=num_bars, freq='1min')
    df.index.name = 'timestamp'

    # Create sample trades
    trades_data = {
        'timestamp': pd.date_range('2023-01-01 09:00:00', periods=50, freq='10s'),
        'side': (['buy'] * 25 + ['sell'] * 25),
        'price': [100 + i * 0.1 for i in range(50)],
        'size': [10.0] * 50,
    }
    trades_df = pd.DataFrame(trades_data).set_index('timestamp')

    # Configure microstructure (use default config)
    cfg = MicrostructureConfig(
        sweep=LiquiditySweepConfig(lookback_ms=0, notional_threshold=0)
    )

    # Should still work with fallback timeframe_delta calculation
    result = compute_microstructure_df(df, cfg, trades_df=trades_df)

    # Verify sweep columns exist
    assert 'ms_sweep_up' in result.columns
    assert 'ms_sweep_down' in result.columns

    # Verify no NaNs (all bars should have been processed)
    assert not result['ms_sweep_up'].isna().any()
    assert not result['ms_sweep_down'].isna().any()
