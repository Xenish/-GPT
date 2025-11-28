"""
Tests for the event bar generation logic.
"""
import pandas as pd
import pytest

from finantradealgo.system.config_loader import EventBarConfig
from finantradealgo.data_engine.event_bars import build_event_bars


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """
    Creates a sample OHLCV DataFrame for testing event bar generation.
    Total Volume = 1500. Total Notional = 151550. Total Ticks = 10.
    """
    data = {
        'timestamp': pd.to_datetime(pd.date_range('2023-01-01', periods=10, freq='1min')),
        'open':   [100, 101, 102, 103, 104, 101, 102, 103, 104, 105],
        'high':   [101, 102, 103, 104, 105, 103, 104, 105, 106, 107],
        'low':    [99,  100, 101, 102, 103, 100, 101, 102, 103, 104],
        'close':  [101, 102, 103, 104, 103, 102, 103, 104, 105, 106],
        'volume': [100, 150, 200, 50,  300, 100, 100, 100, 200, 200],
    }
    return pd.DataFrame(data)


def test_time_mode_passthrough(sample_ohlcv):
    """Tests that 'time' mode returns the original DataFrame."""
    cfg = EventBarConfig(mode='time')
    result_df = build_event_bars(sample_ohlcv.copy(), cfg)
    pd.testing.assert_frame_equal(sample_ohlcv, result_df)


def test_volume_bars(sample_ohlcv):
    """Tests volume bar generation."""
    cfg = EventBarConfig(mode='volume', target_volume=500)
    result_df = build_event_bars(sample_ohlcv.copy(), cfg)

    # Expected bars:
    # Bar 1: 0-2 (100+150+200=450) -> NO, needs one more
    # Bar 1: 0-3 (100+150+200+50=500). OHLCV = (100, 104, 99, 104, 500)
    # Bar 2: 4-7 (300+100+100+100=600). OHLCV = (104, 105, 100, 104, 600)
    # Bar 3: 8-9 (200+200=400). Leftover, but we expect it to form a bar at the end.
    # Let's adjust the logic: the last partial bar should not be returned.
    # The current implementation will only return bars that meet the threshold.
    
    assert len(result_df) == 2, "Should create 2 full volume bars"

    # Check Bar 1
    bar1 = result_df.iloc[0]
    assert bar1.name == pd.Timestamp('2023-01-01 00:03:00')
    assert bar1['open'] == 100
    assert bar1['high'] == 104
    assert bar1['low'] == 99
    assert bar1['close'] == 104
    assert bar1['volume'] == 500

    # Check Bar 2
    bar2 = result_df.iloc[1]
    assert bar2.name == pd.Timestamp('2023-01-01 00:07:00')
    assert bar2['open'] == 104
    assert bar2['high'] == 105
    assert bar2['low'] == 100
    assert bar2['close'] == 104
    assert bar2['volume'] == 600


def test_dollar_bars(sample_ohlcv):
    """Tests dollar (notional) bar generation."""
    # Notional per bar: 10100, 15300, 20600, 5200, 30900, ...
    cfg = EventBarConfig(mode='dollar', target_notional=50000)
    result_df = build_event_bars(sample_ohlcv.copy(), cfg)

    # Expected bars:
    # Bar 1: 0-3 (10100+15300+20600+5200 = 51200). OHLCV = (100, 104, 99, 104, 500)
    # Bar 2: 4-7 (30900+10200+10300+10400 = 61800). OHLCV = (104, 105, 100, 104, 600)
    
    assert len(result_df) == 2

    # Check Bar 1
    bar1 = result_df.iloc[0]
    assert bar1['open'] == 100
    assert bar1['high'] == 104
    assert bar1['low'] == 99
    assert bar1['close'] == 104
    assert bar1['volume'] == 500


def test_tick_bars(sample_ohlcv):
    """Tests tick bar generation (counting rows as ticks)."""
    cfg = EventBarConfig(mode='tick', target_ticks=4)
    result_df = build_event_bars(sample_ohlcv.copy(), cfg)

    # Expected bars:
    # Bar 1: rows 0-3. OHLCV=(100, 104, 99, 104, 500)
    # Bar 2: rows 4-7. OHLCV=(104, 105, 100, 104, 600)
    # Leftover: rows 8-9 (2 ticks)

    assert len(result_df) == 2
    
    # Check Bar 2
    bar2 = result_df.iloc[1]
    assert bar2.name == pd.Timestamp('2023-01-01 00:07:00')
    assert bar2['open'] == 104
    assert bar2['high'] == 105
    assert bar2['low'] == 100
    assert bar2['close'] == 104
    assert bar2['volume'] == 600


def test_acceptance_criteria(sample_ohlcv):
    """Tests that total volume is preserved and timestamps are correct."""
    cfg = EventBarConfig(mode='volume', target_volume=600)
    result_df = build_event_bars(sample_ohlcv.copy(), cfg)

    # Total volume of full bars generated should be a subset of original
    assert result_df['volume'].sum() <= sample_ohlcv['volume'].sum()
    assert result_df['volume'].sum() == 1100 # 500+600

    # Check timestamp boundaries
    assert result_df['bar_start_ts'].iloc[0] == sample_ohlcv['timestamp'].iloc[0]
    # The last bar in the result should end before or at the last original timestamp
    assert result_df['bar_end_ts'].iloc[-1] <= sample_ohlcv['timestamp'].iloc[-1]
    assert result_df['bar_end_ts'].iloc[-1] == pd.Timestamp('2023-01-01 00:07:00')

def test_edge_cases(sample_ohlcv):
    """Tests empty dataframes and invalid configs."""
    # Empty df
    cfg = EventBarConfig(mode='volume', target_volume=100)
    assert build_event_bars(pd.DataFrame(), cfg).empty

    # Invalid mode
    with pytest.raises(ValueError, match="Unsupported event bar mode"):
        cfg = EventBarConfig(mode='invalid')
        build_event_bars(sample_ohlcv, cfg)

    # Zero threshold
    with pytest.raises(ValueError, match="must be a positive number"):
        cfg = EventBarConfig(mode='volume', target_volume=0)
        build_event_bars(sample_ohlcv, cfg)
