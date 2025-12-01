"""
Tests for microstructure engine and features.

Task S2.6: Comprehensive tests for microstructure feature computation.
"""
import pandas as pd
import pytest
import numpy as np

from finantradealgo.microstructure.config import MicrostructureConfig
from finantradealgo.microstructure.microstructure_engine import (
    compute_microstructure_df,
    compute_microstructure_features,
)
from finantradealgo.microstructure.types import MicrostructureSignals
from finantradealgo.microstructure.microstructure_debug import (
    summarize_microstructure_features,
    get_microstructure_health_metrics,
)


class TestComputeMicrostructureDF:
    """Test cases for compute_microstructure_df() entry-point."""

    def test_basic_computation(self):
        """Test basic microstructure feature computation."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(50)],
            "high": [101 + i for i in range(50)],
            "low": [99 + i for i in range(50)],
            "close": [100.5 + i for i in range(50)],
            "volume": [1000] * 50,
        }, index=pd.date_range("2024-01-01", periods=50, freq="1h"))

        cfg = MicrostructureConfig()
        features = compute_microstructure_df(df, cfg)

        # Verify output contract
        expected_cols = MicrostructureSignals.columns()
        for col in expected_cols:
            assert col in features.columns, f"Missing column: {col}"

        # Verify index matches
        assert len(features) == len(df)
        assert features.index.equals(df.index)

    def test_alias_consistency(self):
        """Test that compute_microstructure_features is an alias."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(30)],
            "high": [101 + i for i in range(30)],
            "low": [99 + i for i in range(30)],
            "close": [100.5 + i for i in range(30)],
            "volume": [1000] * 30,
        }, index=pd.date_range("2024-01-01", periods=30, freq="1h"))

        features1 = compute_microstructure_df(df)
        features2 = compute_microstructure_features(df)

        # Should be identical
        pd.testing.assert_frame_equal(features1, features2)

    def test_disabled_config(self):
        """Test that enabled=False returns zeros."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(30)],
            "high": [101 + i for i in range(30)],
            "low": [99 + i for i in range(30)],
            "close": [100.5 + i for i in range(30)],
            "volume": [1000] * 30,
        }, index=pd.date_range("2024-01-01", periods=30, freq="1h"))

        cfg = MicrostructureConfig(enabled=False)
        features = compute_microstructure_df(df, cfg)

        # All features should be 0
        for col in MicrostructureSignals.columns():
            assert (features[col] == 0).all(), f"{col} should be all zeros"

    def test_default_config(self):
        """Test computation with default config (None)."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(30)],
            "high": [101 + i for i in range(30)],
            "low": [99 + i for i in range(30)],
            "close": [100.5 + i for i in range(30)],
            "volume": [1000] * 30,
        }, index=pd.date_range("2024-01-01", periods=30, freq="1h"))

        features = compute_microstructure_df(df, None)

        # Should create all columns
        expected_cols = MicrostructureSignals.columns()
        for col in expected_cols:
            assert col in features.columns


class TestInputValidation:
    """Test cases for input validation (Task S2.3)."""

    def test_empty_dataframe_fails(self):
        """Empty DataFrame should raise assertion."""
        df = pd.DataFrame()

        with pytest.raises(AssertionError, match="df cannot be empty"):
            compute_microstructure_df(df)

    def test_missing_ohlcv_columns_fails(self):
        """Missing OHLCV columns should raise assertion."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            # Missing 'low', 'close', 'volume'
        })

        with pytest.raises(AssertionError, match="must contain"):
            compute_microstructure_df(df)

    def test_trades_df_validation(self):
        """Invalid trades_df should raise assertion."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2024-01-01", periods=3, freq="1h"))

        # trades_df without required columns
        trades_df = pd.DataFrame({
            "price": [100, 101],
            # Missing 'side', 'size'
        }, index=pd.date_range("2024-01-01", periods=2, freq="1s"))

        with pytest.raises(AssertionError, match="trades_df must contain"):
            compute_microstructure_df(df, trades_df=trades_df)

    def test_trades_df_without_datetime_index_fails(self):
        """trades_df without DatetimeIndex should fail."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2024-01-01", periods=3, freq="1h"))

        # trades_df with integer index (not DatetimeIndex)
        trades_df = pd.DataFrame({
            "side": ["buy", "sell"],
            "price": [100, 101],
            "size": [10, 20],
        })  # No DatetimeIndex

        with pytest.raises(AssertionError, match="DatetimeIndex"):
            compute_microstructure_df(df, trades_df=trades_df)


class TestFeatureComputation:
    """Test cases for individual feature computation."""

    def test_chop_computation(self):
        """Test that chop feature is computed."""
        # Create trending data
        df = pd.DataFrame({
            "open": [100 + i * 2 for i in range(50)],
            "high": [101 + i * 2 for i in range(50)],
            "low": [99 + i * 2 for i in range(50)],
            "close": [100.5 + i * 2 for i in range(50)],
            "volume": [1000] * 50,
        }, index=pd.date_range("2024-01-01", periods=50, freq="1h"))

        features = compute_microstructure_df(df)

        assert "ms_chop" in features.columns
        # Trending market should have low chop
        assert features["ms_chop"].iloc[-1] < 0.6

    def test_volatility_regime_computation(self):
        """Test that volatility regime is computed."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(50)],
            "high": [101 + i for i in range(50)],
            "low": [99 + i for i in range(50)],
            "close": [100.5 + i for i in range(50)],
            "volume": [1000] * 50,
        }, index=pd.date_range("2024-01-01", periods=50, freq="1h"))

        features = compute_microstructure_df(df)

        assert "ms_vol_regime" in features.columns
        assert not features["ms_vol_regime"].isna().all()


class TestDebugHelpers:
    """Test cases for debug and health check helpers (Task S2.E3)."""

    def test_summarize_microstructure_features(self):
        """Test text summary generation."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(50)],
            "high": [101 + i for i in range(50)],
            "low": [99 + i for i in range(50)],
            "close": [100.5 + i for i in range(50)],
            "volume": [1000] * 50,
        }, index=pd.date_range("2024-01-01", periods=50, freq="1h"))

        features = compute_microstructure_df(df)
        summary = summarize_microstructure_features(features, df)

        # Check key sections exist
        assert "MICROSTRUCTURE FEATURES SUMMARY" in summary
        assert "Total bars:" in summary
        assert "Volatility Regime" in summary
        assert "Chop Score" in summary

    def test_health_metrics(self):
        """Test health metrics extraction."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(50)],
            "high": [101 + i for i in range(50)],
            "low": [99 + i for i in range(50)],
            "close": [100.5 + i for i in range(50)],
            "volume": [1000] * 50,
        }, index=pd.date_range("2024-01-01", periods=50, freq="1h"))

        features = compute_microstructure_df(df)
        metrics = get_microstructure_health_metrics(features)

        assert metrics["total_bars"] == 50
        assert metrics["expected_cols"] == len(MicrostructureSignals.columns())
        assert metrics["found_cols"] == len(MicrostructureSignals.columns())
        assert metrics["missing_cols"] == []
        assert isinstance(metrics["burst_events"], int)
        assert isinstance(metrics["exhaustion_events"], int)
        assert isinstance(metrics["sweep_events"], int)

    def test_health_metrics_detects_missing_cols(self):
        """Test that health metrics detects missing columns."""
        # Create incomplete features DataFrame
        df = pd.DataFrame({
            "ms_chop": [0.5, 0.6, 0.4],
            "ms_vol_regime": [0.0, 0.1, -0.1],
            # Missing other columns
        })

        metrics = get_microstructure_health_metrics(df)

        assert len(metrics["missing_cols"]) > 0
        assert "ms_burst_up" in metrics["missing_cols"]


class TestOutputContract:
    """Test cases for output contract enforcement (Task S2.2)."""

    def test_all_columns_exist(self):
        """All expected columns must exist in output."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(30)],
            "high": [101 + i for i in range(30)],
            "low": [99 + i for i in range(30)],
            "close": [100.5 + i for i in range(30)],
            "volume": [1000] * 30,
        }, index=pd.date_range("2024-01-01", periods=30, freq="1h"))

        features = compute_microstructure_df(df)
        expected_cols = MicrostructureSignals.columns()

        for col in expected_cols:
            assert col in features.columns, f"Missing column: {col}"

    def test_column_order(self):
        """Columns should be in consistent order."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(30)],
            "high": [101 + i for i in range(30)],
            "low": [99 + i for i in range(30)],
            "close": [100.5 + i for i in range(30)],
            "volume": [1000] * 30,
        }, index=pd.date_range("2024-01-01", periods=30, freq="1h"))

        features = compute_microstructure_df(df)
        expected_cols = MicrostructureSignals.columns()

        assert list(features.columns) == expected_cols

    def test_no_nan_in_output(self):
        """Output should not have NaN values (initialized to 0)."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(30)],
            "high": [101 + i for i in range(30)],
            "low": [99 + i for i in range(30)],
            "close": [100.5 + i for i in range(30)],
            "volume": [1000] * 30,
        }, index=pd.date_range("2024-01-01", periods=30, freq="1h"))

        features = compute_microstructure_df(df)

        # Check no NaN values
        assert not features.isna().any().any(), "Output contains NaN values"


class TestLookbackWindowTruncation:
    """Test cases for max_lookback_seconds truncation (Task S2.E2)."""

    def test_trades_truncation_with_long_series(self):
        """Test that trades_df is truncated to max_lookback_seconds."""
        # Create OHLCV data for 1 hour
        df = pd.DataFrame({
            "open": [100 + i for i in range(4)],
            "high": [101 + i for i in range(4)],
            "low": [99 + i for i in range(4)],
            "close": [100.5 + i for i in range(4)],
            "volume": [1000] * 4,
        }, index=pd.date_range("2024-01-01 10:00:00", periods=4, freq="15min"))

        # Create trades spanning 2 hours (older than max_lookback_seconds)
        trades_df = pd.DataFrame({
            "side": ["buy"] * 100,
            "price": [100 + i * 0.1 for i in range(100)],
            "size": [10] * 100,
        }, index=pd.date_range("2024-01-01 08:00:00", periods=100, freq="1min"))

        # Config with 1800 seconds (30 minutes) lookback
        cfg = MicrostructureConfig(max_lookback_seconds=1800)

        # This should complete without error and use only recent trades
        features = compute_microstructure_df(df, cfg, trades_df=trades_df)

        # Verify output is valid
        assert len(features) == len(df)
        assert "ms_sweep_up" in features.columns
        assert "ms_sweep_down" in features.columns

    def test_book_truncation_with_long_series(self):
        """Test that book_df is truncated to max_lookback_seconds."""
        # Create OHLCV data
        df = pd.DataFrame({
            "open": [100 + i for i in range(4)],
            "high": [101 + i for i in range(4)],
            "low": [99 + i for i in range(4)],
            "close": [100.5 + i for i in range(4)],
            "volume": [1000] * 4,
        }, index=pd.date_range("2024-01-01 10:00:00", periods=4, freq="15min"))

        # Create order book data spanning 2 hours
        book_data = {}
        for i in range(5):
            book_data[f"bid_price_{i}"] = [100 - i * 0.01] * 100
            book_data[f"bid_size_{i}"] = [1000] * 100
            book_data[f"ask_price_{i}"] = [100 + i * 0.01] * 100
            book_data[f"ask_size_{i}"] = [1000] * 100

        book_df = pd.DataFrame(
            book_data,
            index=pd.date_range("2024-01-01 08:00:00", periods=100, freq="1min")
        )

        # Config with 1800 seconds (30 minutes) lookback
        cfg = MicrostructureConfig(max_lookback_seconds=1800)

        # This should complete without error and use only recent book data
        features = compute_microstructure_df(df, cfg, book_df=book_df)

        # Verify output is valid
        assert len(features) == len(df)
        assert "ms_imbalance" in features.columns

    def test_no_truncation_with_zero_max_lookback(self):
        """Test that max_lookback_seconds=0 disables truncation."""
        # Create OHLCV data
        df = pd.DataFrame({
            "open": [100 + i for i in range(4)],
            "high": [101 + i for i in range(4)],
            "low": [99 + i for i in range(4)],
            "close": [100.5 + i for i in range(4)],
            "volume": [1000] * 4,
        }, index=pd.date_range("2024-01-01 10:00:00", periods=4, freq="15min"))

        # Create trades spanning very long time
        trades_df = pd.DataFrame({
            "side": ["buy"] * 1000,
            "price": [100 + i * 0.01 for i in range(1000)],
            "size": [10] * 1000,
        }, index=pd.date_range("2024-01-01 00:00:00", periods=1000, freq="1min"))

        # Config with max_lookback_seconds=0 (no truncation)
        cfg = MicrostructureConfig(max_lookback_seconds=0)

        # All trades should be used
        features = compute_microstructure_df(df, cfg, trades_df=trades_df)

        assert len(features) == len(df)

    def test_truncation_with_default_config(self):
        """Test that default config uses 3600 seconds (1 hour) lookback."""
        # Create OHLCV data
        df = pd.DataFrame({
            "open": [100 + i for i in range(4)],
            "high": [101 + i for i in range(4)],
            "low": [99 + i for i in range(4)],
            "close": [100.5 + i for i in range(4)],
            "volume": [1000] * 4,
        }, index=pd.date_range("2024-01-01 10:00:00", periods=4, freq="15min"))

        # Create trades spanning 2 hours (some should be truncated)
        trades_df = pd.DataFrame({
            "side": ["buy"] * 120,
            "price": [100 + i * 0.1 for i in range(120)],
            "size": [10] * 120,
        }, index=pd.date_range("2024-01-01 08:00:00", periods=120, freq="1min"))

        # Use default config (should have max_lookback_seconds=3600)
        cfg = MicrostructureConfig()
        assert cfg.max_lookback_seconds == 3600

        # Should use only trades within last hour
        features = compute_microstructure_df(df, cfg, trades_df=trades_df)

        assert len(features) == len(df)
