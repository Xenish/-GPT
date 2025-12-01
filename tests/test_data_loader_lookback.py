"""
Tests for lookback_days filtering in data loader.

Verifies that:
1. lookback_days parameter correctly filters data
2. Filtering respects timezone (UTC)
3. Edge cases handled correctly (empty result, no filter, etc.)
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from finantradealgo.data_engine.loader import load_ohlcv_csv, load_ohlcv_for_symbol_tf
from finantradealgo.system.config_loader import DataConfig


class TestDataLoaderLookback:
    """Test lookback filtering in OHLCV data loader."""

    @pytest.fixture
    def ohlcv_365_days(self):
        """Create a temporary CSV with 365 days of 15m data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "TEST_15m.csv"

            # Generate 365 days of 15m data (96 bars per day)
            bars_per_day = 96
            total_bars = 365 * bars_per_day

            # Start from 365 days ago
            start_date = datetime.now() - timedelta(days=365)

            timestamps = [start_date + timedelta(minutes=15 * i) for i in range(total_bars)]
            df = pd.DataFrame({
                "timestamp": timestamps,
                "open": [100.0 + i * 0.01 for i in range(total_bars)],
                "high": [100.1 + i * 0.01 for i in range(total_bars)],
                "low": [99.9 + i * 0.01 for i in range(total_bars)],
                "close": [100.0 + i * 0.01 for i in range(total_bars)],
                "volume": [1000.0 for _ in range(total_bars)],
            })

            df.to_csv(csv_path, index=False)
            yield str(csv_path), total_bars, bars_per_day

    def test_no_lookback_filter_loads_all_data(self, ohlcv_365_days):
        """Test that without lookback filter, all data is loaded."""
        csv_path, total_bars, _ = ohlcv_365_days

        df = load_ohlcv_csv(csv_path, lookback_days=None)

        assert len(df) == total_bars, "Should load all bars when lookback_days=None"
        assert df["timestamp"].dtype == "datetime64[ns, UTC]"
        assert df["timestamp"].is_monotonic_increasing

    def test_lookback_90_days_filters_correctly(self, ohlcv_365_days):
        """Test that lookback_days=90 returns approximately 90 days of data."""
        csv_path, _, bars_per_day = ohlcv_365_days

        df = load_ohlcv_csv(csv_path, lookback_days=90)

        # Should have approximately 90 days * 96 bars/day = 8640 bars
        # Allow some tolerance for partial days
        expected_bars = 90 * bars_per_day
        assert abs(len(df) - expected_bars) < bars_per_day, \
            f"Expected ~{expected_bars} bars for 90 days, got {len(df)}"

        # Verify data is recent (within last 90 days)
        oldest_timestamp = df["timestamp"].min()
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90)

        assert oldest_timestamp >= cutoff, \
            f"Oldest timestamp {oldest_timestamp} should be >= cutoff {cutoff}"

    def test_lookback_180_days_returns_more_data_than_90(self, ohlcv_365_days):
        """Test that longer lookback returns more data."""
        csv_path, _, _ = ohlcv_365_days

        df_90 = load_ohlcv_csv(csv_path, lookback_days=90)
        df_180 = load_ohlcv_csv(csv_path, lookback_days=180)

        assert len(df_180) > len(df_90), \
            "180-day lookback should return more bars than 90-day"

    def test_lookback_365_days_returns_all_data(self, ohlcv_365_days):
        """Test that lookback_days=365 returns all data (full dataset)."""
        csv_path, total_bars, bars_per_day = ohlcv_365_days

        df = load_ohlcv_csv(csv_path, lookback_days=365)

        # Should get all or nearly all bars (allow small tolerance for cutoff timing)
        assert len(df) >= total_bars - bars_per_day, \
            f"365-day lookback should return nearly all {total_bars} bars, got {len(df)}"

    def test_lookback_500_days_handles_excess_gracefully(self, ohlcv_365_days):
        """Test that lookback > dataset length doesn't break."""
        csv_path, total_bars, _ = ohlcv_365_days

        df = load_ohlcv_csv(csv_path, lookback_days=500)

        # Should still return all available data
        assert len(df) == total_bars, \
            "Lookback > dataset should return all available data"

    def test_lookback_1_day_returns_recent_data_only(self, ohlcv_365_days):
        """Test that lookback_days=1 returns only last day."""
        csv_path, _, bars_per_day = ohlcv_365_days

        df = load_ohlcv_csv(csv_path, lookback_days=1)

        # Should have approximately 1 day of data
        # Allow some tolerance due to timestamp precision and cutoff calculation
        expected_bars = bars_per_day
        assert abs(len(df) - expected_bars) < 15, \
            f"Expected ~{expected_bars} bars for 1 day, got {len(df)}"

    def test_load_ohlcv_for_symbol_tf_applies_lookback(self):
        """Test that load_ohlcv_for_symbol_tf applies lookback from DataConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            csv_path = Path(tmpdir) / "BTCUSDT_15m.csv"
            total_bars = 30 * 96  # 30 days of 15m data

            start_date = datetime.now() - timedelta(days=30)
            timestamps = [start_date + timedelta(minutes=15 * i) for i in range(total_bars)]

            df_full = pd.DataFrame({
                "timestamp": timestamps,
                "open": [100.0 for _ in range(total_bars)],
                "high": [101.0 for _ in range(total_bars)],
                "low": [99.0 for _ in range(total_bars)],
                "close": [100.0 for _ in range(total_bars)],
                "volume": [1000.0 for _ in range(total_bars)],
            })
            df_full.to_csv(csv_path, index=False)

            # Create DataConfig with lookback
            data_cfg = DataConfig(
                ohlcv_path_template=str(tmpdir) + "/{symbol}_{timeframe}.csv",
                lookback_days={"15m": 10},
                default_lookback_days=365,
            )

            # Load with helper function
            df = load_ohlcv_for_symbol_tf("BTCUSDT", "15m", data_cfg)

            # Should have approximately 10 days of data (10 * 96 = 960 bars)
            expected_bars = 10 * 96
            assert abs(len(df) - expected_bars) < 96, \
                f"Expected ~{expected_bars} bars for 10-day lookback, got {len(df)}"

    def test_lookback_uses_default_when_timeframe_not_in_dict(self):
        """Test that default_lookback_days is used when timeframe not in lookback_days."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            csv_path = Path(tmpdir) / "BTCUSDT_1h.csv"
            total_bars = 365 * 24  # 365 days of 1h data

            start_date = datetime.now() - timedelta(days=365)
            timestamps = [start_date + timedelta(hours=i) for i in range(total_bars)]

            df_full = pd.DataFrame({
                "timestamp": timestamps,
                "open": [100.0 for _ in range(total_bars)],
                "high": [101.0 for _ in range(total_bars)],
                "low": [99.0 for _ in range(total_bars)],
                "close": [100.0 for _ in range(total_bars)],
                "volume": [1000.0 for _ in range(total_bars)],
            })
            df_full.to_csv(csv_path, index=False)

            # Create DataConfig without 1h in lookback_days
            data_cfg = DataConfig(
                ohlcv_path_template=str(tmpdir) + "/{symbol}_{timeframe}.csv",
                lookback_days={"15m": 10},  # 1h not specified
                default_lookback_days=30,   # Should use this
            )

            # Load with helper function
            df = load_ohlcv_for_symbol_tf("BTCUSDT", "1h", data_cfg)

            # Should have approximately 30 days of data (30 * 24 = 720 bars)
            expected_bars = 30 * 24
            assert abs(len(df) - expected_bars) < 24, \
                f"Expected ~{expected_bars} bars for 30-day default lookback, got {len(df)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
