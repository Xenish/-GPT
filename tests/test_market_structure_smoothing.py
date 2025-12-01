"""
Tests for market structure smoothing functionality.

Task S1.5: Test coverage for smoothing.py module.
"""
import numpy as np
import pandas as pd
import pytest

from finantradealgo.market_structure.config import SmoothingConfig
from finantradealgo.market_structure.smoothing import smooth_price, filter_swing_points


class TestSmoothPrice:
    """Test cases for smooth_price() function."""

    def test_smoothing_disabled_passthrough(self):
        """When smoothing is disabled, price_smooth should equal close."""
        df = pd.DataFrame({
            "close": [100, 105, 102, 108, 106, 110]
        })

        cfg = SmoothingConfig(enabled=False)
        result = smooth_price(df, cfg)

        assert "price_smooth" in result.columns
        pd.testing.assert_series_equal(
            result["price_smooth"],
            result["close"],
            check_names=False
        )

    def test_smoothing_window_1_passthrough(self):
        """When window=1, price_smooth should equal close."""
        df = pd.DataFrame({
            "close": [100, 105, 102, 108, 106, 110]
        })

        cfg = SmoothingConfig(enabled=True, price_ma_window=1)
        result = smooth_price(df, cfg)

        assert "price_smooth" in result.columns
        pd.testing.assert_series_equal(
            result["price_smooth"],
            result["close"],
            check_names=False
        )

    def test_price_smoothing_makes_series_smoother(self):
        """Smoothed series should have lower volatility than raw close."""
        # Create zigzag price series with high noise
        np.random.seed(42)
        n = 100
        trend = np.linspace(100, 110, n)
        noise = np.random.normal(0, 2, n)
        df = pd.DataFrame({
            "close": trend + noise
        })

        cfg = SmoothingConfig(enabled=True, price_ma_window=5)
        result = smooth_price(df, cfg)

        # Smoothed series should have lower standard deviation
        close_std = df["close"].std()
        smooth_std = result["price_smooth"].std()

        assert smooth_std < close_std, "Smoothed series should have lower volatility"

    def test_smoothing_missing_close_column(self):
        """Should raise ValueError when close column is missing."""
        df = pd.DataFrame({
            "open": [100, 105, 102]
        })

        cfg = SmoothingConfig(enabled=True)

        with pytest.raises(ValueError, match="must contain 'close' column"):
            smooth_price(df, cfg)

    def test_smoothing_preserves_index(self):
        """Smoothing should preserve DataFrame index."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        df = pd.DataFrame({
            "close": range(100, 110)
        }, index=dates)

        cfg = SmoothingConfig(enabled=True, price_ma_window=3)
        result = smooth_price(df, cfg)

        pd.testing.assert_index_equal(result.index, df.index)


class TestFilterSwingPoints:
    """Test cases for filter_swing_points() function."""

    def test_filter_disabled_passthrough(self):
        """When filtering is disabled, swing points should be unchanged."""
        df = pd.DataFrame({
            "high": [10, 12, 11, 13, 12],
            "low": [9, 10, 9, 11, 10],
            "ms_swing_high": [0, 1, 0, 1, 0],
            "ms_swing_low": [1, 0, 1, 0, 1]
        })

        cfg = SmoothingConfig(enabled=False)
        result = filter_swing_points(df, cfg)

        pd.testing.assert_series_equal(
            result["ms_swing_high"],
            df["ms_swing_high"],
            check_names=False
        )
        pd.testing.assert_series_equal(
            result["ms_swing_low"],
            df["ms_swing_low"],
            check_names=False
        )

    def test_filter_missing_columns_returns_unchanged(self):
        """When swing columns are missing, should return unchanged."""
        df = pd.DataFrame({
            "high": [10, 12, 11],
            "low": [9, 10, 9]
        })

        cfg = SmoothingConfig(enabled=True)
        result = filter_swing_points(df, cfg)

        # Should not add swing columns if they don't exist
        assert "ms_swing_high" not in result.columns
        assert "ms_swing_low" not in result.columns

    def test_swing_filter_removes_close_swings(self):
        """Filter should remove swings that are too close together."""
        # Create swings at index 1, 2, 3 (all close together)
        df = pd.DataFrame({
            "high": [10, 12, 13, 14, 12, 11, 10],
            "low": [9, 10, 11, 12, 10, 9, 8],
            "ms_swing_high": [0, 1, 1, 1, 0, 0, 0],
            "ms_swing_low": [1, 0, 0, 0, 1, 1, 1]
        })

        cfg = SmoothingConfig(
            enabled=True,
            swing_min_distance=3,  # Require at least 3 bars between swings
            swing_min_zscore=-10.0  # Disable z-score filtering for this test
        )
        result = filter_swing_points(df, cfg)

        # Should have fewer swings than before
        original_swings = df["ms_swing_high"].sum() + df["ms_swing_low"].sum()
        filtered_swings = result["ms_swing_high"].sum() + result["ms_swing_low"].sum()

        assert filtered_swings < original_swings, "Should filter out close swings"

    def test_swing_filter_preserves_index(self):
        """Filtering should preserve DataFrame index."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        df = pd.DataFrame({
            "high": range(10, 20),
            "low": range(9, 19),
            "ms_swing_high": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "ms_swing_low": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }, index=dates)

        cfg = SmoothingConfig(enabled=True)
        result = filter_swing_points(df, cfg)

        pd.testing.assert_index_equal(result.index, df.index)

    def test_swing_filter_with_few_swings(self):
        """Filter should handle cases with very few swings gracefully."""
        df = pd.DataFrame({
            "high": [10, 12, 11, 13, 12],
            "low": [9, 10, 9, 11, 10],
            "ms_swing_high": [0, 1, 0, 0, 0],
            "ms_swing_low": [1, 0, 0, 0, 0]
        })

        cfg = SmoothingConfig(enabled=True, swing_min_distance=2)
        result = filter_swing_points(df, cfg)

        # Should not crash with < 2 swings
        assert "ms_swing_high" in result.columns
        assert "ms_swing_low" in result.columns
