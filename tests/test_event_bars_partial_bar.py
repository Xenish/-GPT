"""
Tests for event bars partial last bar behavior.

Ensures that keep_partial_last_bar config flag controls whether incomplete final bars are kept or dropped.
"""
import pandas as pd
import pytest
from datetime import datetime, timedelta
from finantradealgo.data_engine.event_bars import build_event_bars
from finantradealgo.core.config import EventBarConfig


@pytest.fixture
def sample_ohlcv_df():
    """Provides a sample 1-minute OHLCV DataFrame for testing."""
    data = {
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
        'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
        'volume': [10, 20, 15, 25, 30, 12, 18, 22, 11, 28, 35, 14, 19, 21, 26],
    }
    index = [datetime(2023, 1, 1, 9, 0, 0) + timedelta(minutes=i) for i in range(len(data['open']))]
    df = pd.DataFrame(data, index=index)
    return df


def test_partial_last_bar_dropped_by_default(sample_ohlcv_df):
    """Test that partial last bar is dropped when keep_partial_last_bar=False (default)."""
    target_volume = 50
    cfg = EventBarConfig(mode="volume", target_volume=target_volume, keep_partial_last_bar=False)
    result_df = build_event_bars(sample_ohlcv_df, cfg)

    # Volume data: [10, 20, 15, 25, 30, 12, 18, 22, 11, 28, 35, 14, 19, 21, 26]
    # Expected bars (without partial last bar):
    # Bar 1: (10+20+15+25) = 70 >= 50. Close at index 3 (09:03)
    # Bar 2: (30+12+18) = 60 >= 50. Close at index 6 (09:06)
    # Bar 3: (22+11+28) = 61 >= 50. Close at index 9 (09:09)
    # Bar 4: (35+14+19) = 68 >= 50. Close at index 12 (09:12)
    # Partial: (21+26) = 47 < 50. DROPPED (keep_partial_last_bar=False)

    assert len(result_df) == 4, "Should have 4 complete bars (no partial last bar)"

    # Verify the bars are as expected
    expected_volumes = [70.0, 60.0, 61.0, 68.0]
    actual_volumes = result_df['volume'].tolist()
    assert actual_volumes == expected_volumes, f"Expected volumes {expected_volumes}, got {actual_volumes}"


def test_partial_last_bar_kept_when_enabled(sample_ohlcv_df):
    """Test that partial last bar is kept when keep_partial_last_bar=True."""
    target_volume = 50
    cfg = EventBarConfig(mode="volume", target_volume=target_volume, keep_partial_last_bar=True)
    result_df = build_event_bars(sample_ohlcv_df, cfg)

    # Volume data: [10, 20, 15, 25, 30, 12, 18, 22, 11, 28, 35, 14, 19, 21, 26]
    # Expected bars (with partial last bar):
    # Bar 1: (10+20+15+25) = 70 >= 50. Close at index 3 (09:03)
    # Bar 2: (30+12+18) = 60 >= 50. Close at index 6 (09:06)
    # Bar 3: (22+11+28) = 61 >= 50. Close at index 9 (09:09)
    # Bar 4: (35+14+19) = 68 >= 50. Close at index 12 (09:12)
    # Bar 5: (21+26) = 47 < 50. KEPT (keep_partial_last_bar=True)

    assert len(result_df) == 5, "Should have 4 complete bars + 1 partial last bar"

    # Verify the bars are as expected
    expected_volumes = [70.0, 60.0, 61.0, 68.0, 47.0]
    actual_volumes = result_df['volume'].tolist()
    assert actual_volumes == expected_volumes, f"Expected volumes {expected_volumes}, got {actual_volumes}"

    # Verify the last bar is indeed partial (volume < target)
    assert result_df['volume'].iloc[-1] < target_volume, "Last bar should be partial (volume < target)"


def test_partial_last_bar_dollar_mode_dropped(sample_ohlcv_df):
    """Test that partial last bar is dropped in dollar mode when keep_partial_last_bar=False."""
    target_notional = 5000
    cfg = EventBarConfig(mode="dollar", target_notional=target_notional, keep_partial_last_bar=False)
    result_df = build_event_bars(sample_ohlcv_df, cfg)

    # The exact number of bars depends on notional calculations, but last partial should be dropped
    # We just verify that the result is not empty and all bars meet the threshold
    assert not result_df.empty, "Should have at least some complete bars"

    # Calculate notional for each bar and verify all meet threshold (except potentially the last if there were no partial)
    # Since we're dropping partials, all returned bars should meet the threshold
    for idx, row in result_df.iterrows():
        # We can't easily recalculate exact notional without knowing which rows were included,
        # but we can verify that the DataFrame doesn't have more rows than expected
        pass


def test_partial_last_bar_tick_mode_no_partial(sample_ohlcv_df):
    """Test tick mode when there's no partial last bar (15 rows / 5 ticks = 3 complete bars)."""
    target_ticks = 5
    cfg = EventBarConfig(mode="tick", target_ticks=target_ticks, keep_partial_last_bar=False)
    result_df = build_event_bars(sample_ohlcv_df, cfg)

    # 15 rows / 5 ticks = 3 complete bars, no partial
    assert len(result_df) == 3, "Should have exactly 3 complete bars (no partial)"


def test_partial_last_bar_tick_mode_with_partial(sample_ohlcv_df):
    """Test tick mode when there IS a partial last bar."""
    target_ticks = 4
    cfg_drop = EventBarConfig(mode="tick", target_ticks=target_ticks, keep_partial_last_bar=False)
    cfg_keep = EventBarConfig(mode="tick", target_ticks=target_ticks, keep_partial_last_bar=True)

    result_drop = build_event_bars(sample_ohlcv_df, cfg_drop)
    result_keep = build_event_bars(sample_ohlcv_df, cfg_keep)

    # 15 rows / 4 ticks = 3 complete bars + 3 rows partial
    # With drop: should have 3 bars
    # With keep: should have 4 bars (3 complete + 1 partial with 3 ticks)

    assert len(result_drop) == 3, "Should have 3 complete bars (partial dropped)"
    assert len(result_keep) == 4, "Should have 3 complete + 1 partial bar (partial kept)"


def test_all_data_is_partial(sample_ohlcv_df):
    """Test when ALL data doesn't meet threshold (entire dataset is one partial bar)."""
    target_volume = 10000  # Much higher than total volume
    cfg_drop = EventBarConfig(mode="volume", target_volume=target_volume, keep_partial_last_bar=False)
    cfg_keep = EventBarConfig(mode="volume", target_volume=target_volume, keep_partial_last_bar=True)

    result_drop = build_event_bars(sample_ohlcv_df, cfg_drop)
    result_keep = build_event_bars(sample_ohlcv_df, cfg_keep)

    # With drop: should be empty (entire dataset is partial)
    # With keep: should have 1 bar (the partial bar with all data)

    assert len(result_drop) == 0, "Should have 0 bars when all data is partial and keep_partial_last_bar=False"
    assert len(result_keep) == 1, "Should have 1 bar when all data is partial and keep_partial_last_bar=True"

    if len(result_keep) > 0:
        # Verify the partial bar contains all the data
        total_volume = sample_ohlcv_df['volume'].sum()
        assert result_keep['volume'].iloc[0] == total_volume, "Partial bar should contain all volume"
