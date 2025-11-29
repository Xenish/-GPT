"""
Comprehensive tests for microstructure sweep time semantics.

Tests both regular time bars and event bars to ensure sweep detection
works correctly with different bar duration patterns.
"""
import pandas as pd
import pytest
from datetime import datetime, timedelta

from finantradealgo.microstructure.microstructure_engine import compute_microstructure_df
from finantradealgo.microstructure.config import MicrostructureConfig, LiquiditySweepConfig


def test_sweep_regular_time_bars_upward():
    """Test sweep detection with regular 1m time bars - upward sweep."""
    # Create regular 1m OHLCV bars
    num_bars = 5
    data = {
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [101.0, 102.0, 103.0, 104.0, 105.0],
        'low': [99.0, 100.0, 101.0, 102.0, 103.0],
        'close': [101.0, 102.0, 103.0, 104.0, 105.0],  # All bars going up
        'volume': [1000.0] * num_bars,
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range('2023-01-01 09:00:00', periods=num_bars, freq='1min')
    df.index.name = 'timestamp'

    # Create buy-heavy trades for bar 2 (09:02:00 - 09:03:00)
    # This should trigger an upward sweep
    trades_data = []

    # Bar 0 (09:00-09:01): Light trading
    trades_data.append({'timestamp': pd.Timestamp('2023-01-01 09:00:30'), 'side': 'buy', 'price': 100.5, 'size': 10.0})

    # Bar 1 (09:01-09:02): Light trading
    trades_data.append({'timestamp': pd.Timestamp('2023-01-01 09:01:30'), 'side': 'sell', 'price': 101.5, 'size': 10.0})

    # Bar 2 (09:02-09:03): HEAVY BUY PRESSURE - should trigger sweep_up
    trades_data.extend([
        {'timestamp': pd.Timestamp('2023-01-01 09:02:10'), 'side': 'buy', 'price': 102.1, 'size': 100.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:02:20'), 'side': 'buy', 'price': 102.3, 'size': 100.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:02:30'), 'side': 'buy', 'price': 102.5, 'size': 100.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:02:40'), 'side': 'buy', 'price': 102.7, 'size': 100.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:02:50'), 'side': 'buy', 'price': 102.9, 'size': 100.0},
    ])

    # Bar 3 (09:03-09:04): Light trading
    trades_data.append({'timestamp': pd.Timestamp('2023-01-01 09:03:30'), 'side': 'buy', 'price': 103.5, 'size': 10.0})

    # Bar 4 (09:04-09:05): Light trading
    trades_data.append({'timestamp': pd.Timestamp('2023-01-01 09:04:30'), 'side': 'sell', 'price': 104.5, 'size': 10.0})

    trades_df = pd.DataFrame(trades_data).set_index('timestamp')

    # Configure sweep detection with low threshold
    cfg = MicrostructureConfig(
        sweep=LiquiditySweepConfig(lookback_ms=0, notional_threshold=10000.0)
    )

    # Compute microstructure signals
    result = compute_microstructure_df(df, cfg, trades_df=trades_df)

    # Verify sweep columns exist
    assert 'ms_sweep_up' in result.columns
    assert 'ms_sweep_down' in result.columns

    # Bar 2 should have a strong upward sweep (500 buy volume * ~102.5 price = ~51,250 notional)
    assert result['ms_sweep_up'].iloc[2] > 10000.0, "Bar 2 should detect upward sweep"

    # Other bars should not have significant sweeps
    assert result['ms_sweep_up'].iloc[0] == 0.0, "Bar 0 should not have sweep"
    assert result['ms_sweep_up'].iloc[1] == 0.0, "Bar 1 should not have sweep"
    assert result['ms_sweep_down'].iloc[2] == 0.0, "Bar 2 should not have downward sweep"


def test_sweep_regular_time_bars_downward():
    """Test sweep detection with regular 1m time bars - downward sweep."""
    # Create regular 1m OHLCV bars
    num_bars = 5
    data = {
        'open': [105.0, 104.0, 103.0, 102.0, 101.0],
        'high': [106.0, 105.0, 104.0, 103.0, 102.0],
        'low': [104.0, 103.0, 102.0, 101.0, 100.0],
        'close': [104.0, 103.0, 102.0, 101.0, 100.0],  # All bars going down
        'volume': [1000.0] * num_bars,
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range('2023-01-01 09:00:00', periods=num_bars, freq='1min')
    df.index.name = 'timestamp'

    # Create sell-heavy trades for bar 2 (09:02:00 - 09:03:00)
    # This should trigger a downward sweep
    trades_data = []

    # Bar 0 (09:00-09:01): Light trading
    trades_data.append({'timestamp': pd.Timestamp('2023-01-01 09:00:30'), 'side': 'sell', 'price': 104.5, 'size': 10.0})

    # Bar 1 (09:01-09:02): Light trading
    trades_data.append({'timestamp': pd.Timestamp('2023-01-01 09:01:30'), 'side': 'buy', 'price': 103.5, 'size': 10.0})

    # Bar 2 (09:02-09:03): HEAVY SELL PRESSURE - should trigger sweep_down
    trades_data.extend([
        {'timestamp': pd.Timestamp('2023-01-01 09:02:10'), 'side': 'sell', 'price': 102.9, 'size': 100.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:02:20'), 'side': 'sell', 'price': 102.7, 'size': 100.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:02:30'), 'side': 'sell', 'price': 102.5, 'size': 100.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:02:40'), 'side': 'sell', 'price': 102.3, 'size': 100.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:02:50'), 'side': 'sell', 'price': 102.1, 'size': 100.0},
    ])

    # Bar 3 (09:03-09:04): Light trading
    trades_data.append({'timestamp': pd.Timestamp('2023-01-01 09:03:30'), 'side': 'sell', 'price': 101.5, 'size': 10.0})

    # Bar 4 (09:04-09:05): Light trading
    trades_data.append({'timestamp': pd.Timestamp('2023-01-01 09:04:30'), 'side': 'buy', 'price': 100.5, 'size': 10.0})

    trades_df = pd.DataFrame(trades_data).set_index('timestamp')

    # Configure sweep detection with low threshold
    cfg = MicrostructureConfig(
        sweep=LiquiditySweepConfig(lookback_ms=0, notional_threshold=10000.0)
    )

    # Compute microstructure signals
    result = compute_microstructure_df(df, cfg, trades_df=trades_df)

    # Verify sweep columns exist
    assert 'ms_sweep_up' in result.columns
    assert 'ms_sweep_down' in result.columns

    # Bar 2 should have a strong downward sweep (500 sell volume * ~102.5 price = ~51,250 notional)
    assert result['ms_sweep_down'].iloc[2] > 10000.0, "Bar 2 should detect downward sweep"

    # Other bars should not have significant sweeps
    assert result['ms_sweep_down'].iloc[0] == 0.0, "Bar 0 should not have sweep"
    assert result['ms_sweep_down'].iloc[1] == 0.0, "Bar 1 should not have sweep"
    assert result['ms_sweep_up'].iloc[2] == 0.0, "Bar 2 should not have upward sweep"


def test_sweep_event_bars_variable_duration():
    """Test sweep detection with event bars of variable duration."""
    # Create event bars with DIFFERENT durations (volume-based bars)
    # Bar 1: 2 minutes long (09:00 - 09:02)
    # Bar 2: 5 minutes long (09:02 - 09:07) - BUY SWEEP HERE
    # Bar 3: 3 minutes long (09:07 - 09:10) - SELL SWEEP HERE
    # Bar 4: 1 minute long (09:10 - 09:11)

    data = {
        'open': [100.0, 102.0, 107.0, 104.0],
        'high': [102.0, 107.0, 108.0, 105.0],
        'low': [99.0, 101.0, 103.0, 103.0],
        'close': [102.0, 107.0, 104.0, 105.0],  # Bar 2 up, Bar 3 down
        'volume': [5000.0, 5000.0, 5000.0, 5000.0],
        'bar_start_ts': [
            pd.Timestamp('2023-01-01 09:00:00'),
            pd.Timestamp('2023-01-01 09:02:00'),  # Bar 2 starts
            pd.Timestamp('2023-01-01 09:07:00'),  # Bar 3 starts
            pd.Timestamp('2023-01-01 09:10:00'),
        ],
    }
    df = pd.DataFrame(data)
    df.index = pd.DatetimeIndex([
        pd.Timestamp('2023-01-01 09:02:00'),  # Bar 1 ends
        pd.Timestamp('2023-01-01 09:07:00'),  # Bar 2 ends
        pd.Timestamp('2023-01-01 09:10:00'),  # Bar 3 ends
        pd.Timestamp('2023-01-01 09:11:00'),  # Bar 4 ends
    ], name='bar_end_ts')

    # Create trades that fall specifically into different bars
    trades_data = []

    # Bar 1 (09:00-09:02): Light trading
    trades_data.append({'timestamp': pd.Timestamp('2023-01-01 09:00:30'), 'side': 'buy', 'price': 100.5, 'size': 10.0})
    trades_data.append({'timestamp': pd.Timestamp('2023-01-01 09:01:30'), 'side': 'sell', 'price': 101.5, 'size': 10.0})

    # Bar 2 (09:02-09:07): HEAVY BUY PRESSURE - 5 minutes of aggressive buying
    trades_data.extend([
        {'timestamp': pd.Timestamp('2023-01-01 09:02:30'), 'side': 'buy', 'price': 102.5, 'size': 100.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:03:00'), 'side': 'buy', 'price': 103.0, 'size': 150.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:03:30'), 'side': 'buy', 'price': 103.5, 'size': 100.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:04:00'), 'side': 'buy', 'price': 104.0, 'size': 200.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:04:30'), 'side': 'buy', 'price': 104.5, 'size': 100.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:05:00'), 'side': 'buy', 'price': 105.0, 'size': 150.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:05:30'), 'side': 'buy', 'price': 105.5, 'size': 100.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:06:00'), 'side': 'buy', 'price': 106.0, 'size': 200.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:06:30'), 'side': 'buy', 'price': 106.5, 'size': 100.0},
    ])

    # Bar 3 (09:07-09:10): HEAVY SELL PRESSURE - 3 minutes of aggressive selling
    trades_data.extend([
        {'timestamp': pd.Timestamp('2023-01-01 09:07:20'), 'side': 'sell', 'price': 106.5, 'size': 150.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:07:40'), 'side': 'sell', 'price': 106.0, 'size': 200.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:08:00'), 'side': 'sell', 'price': 105.5, 'size': 150.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:08:30'), 'side': 'sell', 'price': 105.0, 'size': 200.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:09:00'), 'side': 'sell', 'price': 104.5, 'size': 150.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:09:30'), 'side': 'sell', 'price': 104.0, 'size': 200.0},
    ])

    # Bar 4 (09:10-09:11): Light trading
    trades_data.append({'timestamp': pd.Timestamp('2023-01-01 09:10:30'), 'side': 'buy', 'price': 104.5, 'size': 10.0})

    trades_df = pd.DataFrame(trades_data).set_index('timestamp')

    # Configure sweep detection with threshold
    cfg = MicrostructureConfig(
        sweep=LiquiditySweepConfig(lookback_ms=0, notional_threshold=50000.0)
    )

    # Compute microstructure signals
    result = compute_microstructure_df(df, cfg, trades_df=trades_df)

    # Verify sweep columns exist
    assert 'ms_sweep_up' in result.columns
    assert 'ms_sweep_down' in result.columns

    # Bar 2 should have a strong upward sweep (long duration, many buy trades)
    # 1200 size * ~104.5 avg price = ~125,400 notional
    assert result['ms_sweep_up'].iloc[1] > 50000.0, f"Bar 2 should detect upward sweep, got {result['ms_sweep_up'].iloc[1]}"
    assert result['ms_sweep_down'].iloc[1] == 0.0, "Bar 2 should not have downward sweep"

    # Bar 3 should have a strong downward sweep
    # 1050 size * ~105.0 avg price = ~110,250 notional
    assert result['ms_sweep_down'].iloc[2] > 50000.0, f"Bar 3 should detect downward sweep, got {result['ms_sweep_down'].iloc[2]}"
    assert result['ms_sweep_up'].iloc[2] == 0.0, "Bar 3 should not have upward sweep"

    # Bar 1 and Bar 4 should not have significant sweeps
    assert result['ms_sweep_up'].iloc[0] == 0.0, "Bar 1 should not have upward sweep"
    assert result['ms_sweep_down'].iloc[0] == 0.0, "Bar 1 should not have downward sweep"
    assert result['ms_sweep_up'].iloc[3] == 0.0, "Bar 4 should not have upward sweep"
    assert result['ms_sweep_down'].iloc[3] == 0.0, "Bar 4 should not have downward sweep"


def test_sweep_event_bars_precise_time_boundaries():
    """Test that sweep detection respects exact bar time boundaries for event bars."""
    # Create 3 event bars with specific time boundaries
    data = {
        'open': [100.0, 102.0, 104.0],
        'high': [102.0, 104.0, 106.0],
        'low': [99.0, 101.0, 103.0],
        'close': [102.0, 104.0, 106.0],  # All going up
        'volume': [5000.0, 5000.0, 5000.0],
        'bar_start_ts': [
            pd.Timestamp('2023-01-01 09:00:00'),
            pd.Timestamp('2023-01-01 09:05:00'),  # 5-minute gap
            pd.Timestamp('2023-01-01 09:08:00'),  # 3-minute gap
        ],
    }
    df = pd.DataFrame(data)
    df.index = pd.DatetimeIndex([
        pd.Timestamp('2023-01-01 09:05:00'),  # Bar 1 ends
        pd.Timestamp('2023-01-01 09:08:00'),  # Bar 2 ends
        pd.Timestamp('2023-01-01 09:12:00'),  # Bar 3 ends (4-minute duration)
    ], name='bar_end_ts')

    # Create trades within bar boundaries (avoiding exact boundary times to prevent ambiguity)
    # Note: pandas loc[start:end] is inclusive on both ends, so trades AT bar_end_ts
    # are included in that bar
    trades_data = [
        # During Bar 1 (09:00:00 - 09:05:00)
        {'timestamp': pd.Timestamp('2023-01-01 09:02:00'), 'side': 'buy', 'price': 101.0, 'size': 100.0},

        # During Bar 2 (09:05:00 - 09:08:00) - 3 trades
        {'timestamp': pd.Timestamp('2023-01-01 09:05:30'), 'side': 'buy', 'price': 102.0, 'size': 200.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:06:00'), 'side': 'buy', 'price': 103.0, 'size': 200.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:07:30'), 'side': 'buy', 'price': 103.5, 'size': 200.0},

        # During Bar 3 (09:08:00 - 09:12:00) - 2 trades
        {'timestamp': pd.Timestamp('2023-01-01 09:09:00'), 'side': 'buy', 'price': 104.0, 'size': 150.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:10:00'), 'side': 'buy', 'price': 105.0, 'size': 150.0},
    ]

    trades_df = pd.DataFrame(trades_data).set_index('timestamp')

    # Configure sweep detection with low threshold
    cfg = MicrostructureConfig(
        sweep=LiquiditySweepConfig(lookback_ms=0, notional_threshold=5000.0)
    )

    # Compute microstructure signals
    result = compute_microstructure_df(df, cfg, trades_df=trades_df)

    # Verify sweep columns exist
    assert 'ms_sweep_up' in result.columns
    assert 'ms_sweep_down' in result.columns

    # Bar 1 should only capture 1 trade (100 * 101 = 10,100)
    assert result['ms_sweep_up'].iloc[0] > 5000.0, f"Bar 1 should detect sweep, got {result['ms_sweep_up'].iloc[0]}"
    assert result['ms_sweep_up'].iloc[0] < 15000.0, f"Bar 1 should only have 1 trade, got {result['ms_sweep_up'].iloc[0]}"

    # Bar 2 should capture 3 trades (200*102 + 200*103 + 200*103.5 = 20,400 + 20,600 + 20,700 = 61,700)
    assert result['ms_sweep_up'].iloc[1] > 60000.0, f"Bar 2 should detect large sweep, got {result['ms_sweep_up'].iloc[1]}"
    assert result['ms_sweep_up'].iloc[1] < 65000.0, f"Bar 2 should have 3 trades, got {result['ms_sweep_up'].iloc[1]}"

    # Bar 3 should capture 2 trades (150*104 + 150*105 = 15,600 + 15,750 = 31,350)
    assert result['ms_sweep_up'].iloc[2] > 30000.0, f"Bar 3 should detect sweep, got {result['ms_sweep_up'].iloc[2]}"
    assert result['ms_sweep_up'].iloc[2] < 35000.0, f"Bar 3 should have 2 trades, got {result['ms_sweep_up'].iloc[2]}"


def test_sweep_no_price_impact_no_sweep():
    """Test that sweep is NOT detected when there's no price impact (close == open)."""
    # Create bars where close == open (no price movement)
    num_bars = 3
    data = {
        'open': [100.0, 100.0, 100.0],
        'high': [101.0, 101.0, 101.0],
        'low': [99.0, 99.0, 99.0],
        'close': [100.0, 100.0, 100.0],  # No price change!
        'volume': [1000.0] * num_bars,
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range('2023-01-01 09:00:00', periods=num_bars, freq='1min')
    df.index.name = 'timestamp'

    # Create heavy buy trades (but price doesn't move, so no sweep)
    trades_data = [
        {'timestamp': pd.Timestamp('2023-01-01 09:00:30'), 'side': 'buy', 'price': 100.0, 'size': 1000.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:01:30'), 'side': 'buy', 'price': 100.0, 'size': 1000.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:02:30'), 'side': 'buy', 'price': 100.0, 'size': 1000.0},
    ]

    trades_df = pd.DataFrame(trades_data).set_index('timestamp')

    # Configure sweep detection with low threshold
    cfg = MicrostructureConfig(
        sweep=LiquiditySweepConfig(lookback_ms=0, notional_threshold=1000.0)
    )

    # Compute microstructure signals
    result = compute_microstructure_df(df, cfg, trades_df=trades_df)

    # No bars should have sweeps because there's no price impact
    assert result['ms_sweep_up'].sum() == 0.0, "No upward sweeps should be detected without price impact"
    assert result['ms_sweep_down'].sum() == 0.0, "No downward sweeps should be detected without price impact"


def test_sweep_with_lookback_window():
    """Test that lookback_ms parameter correctly extends the time window."""
    # Create a single bar
    data = {
        'open': [100.0],
        'high': [105.0],
        'low': [99.0],
        'close': [105.0],  # Strong upward move
        'volume': [1000.0],
        'bar_start_ts': [pd.Timestamp('2023-01-01 09:01:00')],  # Bar starts at 09:01
    }
    df = pd.DataFrame(data)
    df.index = pd.DatetimeIndex([pd.Timestamp('2023-01-01 09:02:00')], name='bar_end_ts')  # Bar ends at 09:02

    # Create trades BEFORE the bar starts (in lookback window)
    trades_data = [
        # These trades are at 09:00:30, which is 30 seconds before bar start
        {'timestamp': pd.Timestamp('2023-01-01 09:00:30'), 'side': 'buy', 'price': 100.0, 'size': 100.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:00:40'), 'side': 'buy', 'price': 101.0, 'size': 100.0},
        {'timestamp': pd.Timestamp('2023-01-01 09:00:50'), 'side': 'buy', 'price': 102.0, 'size': 100.0},

        # Trade during the actual bar
        {'timestamp': pd.Timestamp('2023-01-01 09:01:30'), 'side': 'buy', 'price': 103.0, 'size': 100.0},
    ]

    trades_df = pd.DataFrame(trades_data).set_index('timestamp')

    # Test WITHOUT lookback (should only capture bar-time trades)
    cfg_no_lookback = MicrostructureConfig(
        sweep=LiquiditySweepConfig(lookback_ms=0, notional_threshold=5000.0)
    )
    result_no_lookback = compute_microstructure_df(df, cfg_no_lookback, trades_df=trades_df)

    # Should only capture 1 trade (100 * 103 = 10,300)
    notional_no_lookback = result_no_lookback['ms_sweep_up'].iloc[0]
    assert 10000.0 < notional_no_lookback < 11000.0, f"Without lookback should be ~10,300, got {notional_no_lookback}"

    # Test WITH 60-second lookback (should capture all trades)
    cfg_with_lookback = MicrostructureConfig(
        sweep=LiquiditySweepConfig(lookback_ms=60000, notional_threshold=5000.0)
    )
    result_with_lookback = compute_microstructure_df(df, cfg_with_lookback, trades_df=trades_df)

    # Should capture all 4 trades (100*100 + 100*101 + 100*102 + 100*103 = 40,600)
    notional_with_lookback = result_with_lookback['ms_sweep_up'].iloc[0]
    assert 40000.0 < notional_with_lookback < 41000.0, f"With lookback should be ~40,600, got {notional_with_lookback}"

    # Verify that lookback captures more notional
    assert notional_with_lookback > notional_no_lookback * 3, "Lookback should capture significantly more trades"
