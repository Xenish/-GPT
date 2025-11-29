"""
Tests for event bars timeframe validation.

Ensures that nonsensical combinations like "15m + volume mode" are caught.
"""
import tempfile
import pandas as pd
import pytest
from datetime import datetime, timedelta

from finantradealgo.data_engine.loader import load_ohlcv_csv
from finantradealgo.system.config_loader import DataConfig, EventBarConfig


def create_test_ohlcv_csv():
    """Create a temporary OHLCV CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write('timestamp,open,high,low,close,volume\n')
        for i in range(20):
            ts = datetime(2023, 1, 1, 9, 0) + timedelta(minutes=i)
            price = 100 + i
            f.write(f'{ts},{price},{price+1},{price-1},{price},1000\n')
        return f.name


def test_event_bars_with_1m_source_timeframe_ok():
    """Test that event bars work fine with 1m source_timeframe."""
    csv_path = create_test_ohlcv_csv()

    try:
        bars_cfg = EventBarConfig(
            mode="volume",
            target_volume=5000,
            source_timeframe="1m"
        )
        data_cfg = DataConfig(bars=bars_cfg)

        # Should not raise
        df = load_ohlcv_csv(csv_path, data_cfg)

        # Verify it's not empty and has event bar columns
        assert not df.empty
        assert 'bar_start_ts' in df.columns
        assert df.index.name == 'bar_end_ts'  # bar_end_ts is the index
    finally:
        import os
        os.unlink(csv_path)


def test_event_bars_with_15m_source_timeframe_raises_error():
    """Test that event bars with 15m source_timeframe raises ValueError."""
    csv_path = create_test_ohlcv_csv()

    try:
        bars_cfg = EventBarConfig(
            mode="volume",
            target_volume=5000,
            source_timeframe="15m"  # Invalid!
        )
        data_cfg = DataConfig(bars=bars_cfg)

        # Should raise ValueError
        with pytest.raises(ValueError, match="should be built from 1m data"):
            load_ohlcv_csv(csv_path, data_cfg)
    finally:
        import os
        os.unlink(csv_path)


def test_event_bars_dollar_mode_with_5m_raises_error():
    """Test that dollar mode with 5m source_timeframe raises ValueError."""
    csv_path = create_test_ohlcv_csv()

    try:
        bars_cfg = EventBarConfig(
            mode="dollar",
            target_notional=10000,
            source_timeframe="5m"  # Invalid!
        )
        data_cfg = DataConfig(bars=bars_cfg)

        # Should raise ValueError
        with pytest.raises(ValueError, match="should be built from 1m data"):
            load_ohlcv_csv(csv_path, data_cfg)
    finally:
        import os
        os.unlink(csv_path)


def test_event_bars_tick_mode_with_invalid_timeframe_raises_error():
    """Test that tick mode with non-1m source_timeframe raises ValueError."""
    csv_path = create_test_ohlcv_csv()

    try:
        bars_cfg = EventBarConfig(
            mode="tick",
            target_ticks=100,
            source_timeframe="1h"  # Invalid!
        )
        data_cfg = DataConfig(bars=bars_cfg)

        # Should raise ValueError
        with pytest.raises(ValueError, match="should be built from 1m data"):
            load_ohlcv_csv(csv_path, data_cfg)
    finally:
        import os
        os.unlink(csv_path)


def test_event_bars_without_source_timeframe_warns(caplog):
    """Test that event bars without source_timeframe logs a warning."""
    csv_path = create_test_ohlcv_csv()

    try:
        bars_cfg = EventBarConfig(
            mode="volume",
            target_volume=5000,
            source_timeframe=None  # Not specified
        )
        data_cfg = DataConfig(bars=bars_cfg)

        # Should not raise, but should log warning
        df = load_ohlcv_csv(csv_path, data_cfg)

        # Verify warning was logged
        assert any("source_timeframe not specified" in record.message for record in caplog.records)

        # Verify it still works
        assert not df.empty
    finally:
        import os
        os.unlink(csv_path)


def test_time_mode_with_any_timeframe_ok():
    """Test that time mode works with any source_timeframe (no validation)."""
    csv_path = create_test_ohlcv_csv()

    try:
        # Time mode should work with any timeframe
        for tf in ["1m", "5m", "15m", "1h"]:
            bars_cfg = EventBarConfig(
                mode="time",
                source_timeframe=tf
            )
            data_cfg = DataConfig(bars=bars_cfg)

            # Should not raise
            df = load_ohlcv_csv(csv_path, data_cfg)
            assert not df.empty
    finally:
        import os
        os.unlink(csv_path)
