"""
Tests for data validation module.

Task S3.4: Comprehensive tests for OHLCV validation and timeframe utilities.
"""
import pandas as pd
import pytest
import numpy as np

from finantradealgo.validation import (
    DataValidationConfig,
    OHLCVValidationConfig,
    validate_ohlcv,
    validate_ohlcv_strict,
    validate_multi_tf_alignment,
    TIMEFRAME_TO_SECONDS,
    timeframe_to_seconds,
    detect_gaps,
    infer_timeframe,
)


class TestOHLCVValidationConfig:
    """Test cases for OHLCVValidationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        cfg = OHLCVValidationConfig()

        assert cfg.required_columns == ["open", "high", "low", "close", "volume"]
        assert cfg.check_negative_prices is True
        assert cfg.check_zero_prices is True
        assert cfg.check_ohlc_relationship is True
        assert cfg.max_gap_multiplier == 2.0

    def test_from_dict(self):
        """Test config creation from dictionary."""
        data = {
            "check_negative_prices": False,
            "check_zero_prices": False,
            "max_gap_multiplier": 3.0,
        }
        cfg = OHLCVValidationConfig.from_dict(data)

        assert cfg.check_negative_prices is False
        assert cfg.check_zero_prices is False
        assert cfg.max_gap_multiplier == 3.0


class TestValidateOHLCV:
    """Test cases for validate_ohlcv() function."""

    def test_valid_ohlcv_data(self):
        """Test validation with valid OHLCV data."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(50)],
            "high": [102 + i for i in range(50)],
            "low": [99 + i for i in range(50)],
            "close": [101 + i for i in range(50)],
            "volume": [1000 + i * 10 for i in range(50)],
        }, index=pd.date_range("2024-01-01", periods=50, freq="15min"))

        result = validate_ohlcv(df, timeframe="15m")

        assert result.is_valid
        assert result.errors_count == 0
        assert len(result.issues) == 0

    def test_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()

        result = validate_ohlcv(df)

        assert not result.is_valid
        assert result.errors_count == 1
        assert any("empty" in issue.message.lower() for issue in result.issues)

    def test_missing_columns(self):
        """Test validation with missing required columns."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            # Missing low, close, volume
        }, index=pd.date_range("2024-01-01", periods=3, freq="15min"))

        result = validate_ohlcv(df)

        assert not result.is_valid
        assert result.errors_count == 1
        assert any("missing" in issue.message.lower() for issue in result.issues)

    def test_negative_prices(self):
        """Test detection of negative prices."""
        df = pd.DataFrame({
            "open": [100, -101, 102],  # Negative price
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2024-01-01", periods=3, freq="15min"))

        result = validate_ohlcv(df)

        assert not result.is_valid
        assert result.errors_count >= 1  # May have multiple errors (negative + range check)
        assert any("negative_open" in issue.check_name for issue in result.issues)

    def test_zero_prices(self):
        """Test detection of zero prices."""
        df = pd.DataFrame({
            "open": [100, 0, 102],  # Zero price
            "high": [102, 103, 104],
            "low": [99, 0, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2024-01-01", periods=3, freq="15min"))

        result = validate_ohlcv(df)

        assert not result.is_valid
        assert result.errors_count >= 1
        assert any("zero" in issue.check_name for issue in result.issues)

    def test_invalid_ohlc_relationship(self):
        """Test detection of invalid OHLC relationships."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [103, 104, 105],  # Low > High (invalid)
            "close": [101, 102, 103],
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2024-01-01", periods=3, freq="15min"))

        result = validate_ohlcv(df)

        assert not result.is_valid
        assert result.errors_count >= 1
        assert any("high_low" in issue.check_name for issue in result.issues)

    def test_open_outside_range(self):
        """Test detection of open price outside [low, high] range."""
        df = pd.DataFrame({
            "open": [100, 110, 102],  # Open > High
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2024-01-01", periods=3, freq="15min"))

        result = validate_ohlcv(df)

        assert not result.is_valid
        assert any("open_range" in issue.check_name for issue in result.issues)

    def test_close_outside_range(self):
        """Test detection of close price outside [low, high] range."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 105, 103],  # Close > High
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2024-01-01", periods=3, freq="15min"))

        result = validate_ohlcv(df)

        assert not result.is_valid
        assert any("close_range" in issue.check_name for issue in result.issues)

    def test_negative_volume(self):
        """Test detection of negative volume."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, -500, 1000],  # Negative volume
        }, index=pd.date_range("2024-01-01", periods=3, freq="15min"))

        result = validate_ohlcv(df)

        assert not result.is_valid
        assert any("negative_volume" in issue.check_name for issue in result.issues)

    def test_zero_volume_warning(self):
        """Test that zero volume generates a warning when enabled."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 0, 1000],  # Zero volume
        }, index=pd.date_range("2024-01-01", periods=3, freq="15min"))

        cfg = OHLCVValidationConfig(check_zero_volume=True)
        result = validate_ohlcv(df, cfg)

        # Zero volume is a warning, not an error
        assert result.is_valid  # Still valid
        assert result.warnings_count == 1
        assert any("zero_volume" in issue.check_name for issue in result.issues)

    def test_duplicate_timestamps(self):
        """Test detection of duplicate timestamps."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1000, 1000],
        }, index=pd.DatetimeIndex([
            "2024-01-01 00:00:00",
            "2024-01-01 00:15:00",
            "2024-01-01 00:15:00",  # Duplicate
        ]))

        result = validate_ohlcv(df)

        assert not result.is_valid
        assert any("duplicate" in issue.check_name for issue in result.issues)

    def test_non_chronological_order(self):
        """Test detection of non-chronological timestamps."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1000, 1000],
        }, index=pd.DatetimeIndex([
            "2024-01-01 00:00:00",
            "2024-01-01 00:30:00",
            "2024-01-01 00:15:00",  # Out of order
        ]))

        result = validate_ohlcv(df)

        assert not result.is_valid
        assert any("chronological" in issue.check_name for issue in result.issues)

    def test_price_spikes(self):
        """Test detection of price spikes."""
        # Create data with gradual changes and then a spike
        prices = [100.0 + i * 0.1 for i in range(100)]  # Gradual trend
        prices[50] = 120.0  # Large spike (20% jump)

        df = pd.DataFrame({
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1000] * 100,
        }, index=pd.date_range("2024-01-01", periods=100, freq="15min"))

        cfg = OHLCVValidationConfig(
            check_price_spikes=True,
            price_spike_z_threshold=3.0,
            price_spike_window=20  # Smaller window for more sensitivity
        )
        result = validate_ohlcv(df, cfg)

        # Spike should generate a warning
        assert result.warnings_count >= 1
        assert any("spike" in issue.check_name for issue in result.issues)

    def test_gap_detection(self):
        """Test detection of gaps in data."""
        # Create index with a gap
        index1 = pd.date_range("2024-01-01 00:00", periods=10, freq="15min")
        index2 = pd.date_range("2024-01-01 03:00", periods=10, freq="15min")  # 3-hour gap
        index = index1.union(index2)

        df = pd.DataFrame({
            "open": [100 + i for i in range(len(index))],
            "high": [102 + i for i in range(len(index))],
            "low": [99 + i for i in range(len(index))],
            "close": [101 + i for i in range(len(index))],
            "volume": [1000] * len(index),
        }, index=index)

        cfg = OHLCVValidationConfig(check_missing_bars=True, max_gap_multiplier=2.0)
        result = validate_ohlcv(df, cfg, timeframe="15m")

        # Gap should generate a warning
        assert result.warnings_count >= 1
        assert any("missing_bars" in issue.check_name or "gap" in issue.check_name for issue in result.issues)


class TestValidateOHLCVStrict:
    """Test cases for validate_ohlcv_strict() function."""

    def test_strict_validation_pass(self):
        """Test that strict validation passes with valid data."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2024-01-01", periods=3, freq="15min"))

        # Should not raise
        result_df = validate_ohlcv_strict(df, timeframe="15m")
        assert result_df is df

    def test_strict_validation_fail(self):
        """Test that strict validation raises on invalid data."""
        df = pd.DataFrame({
            "open": [100, -101, 102],  # Negative price
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2024-01-01", periods=3, freq="15min"))

        with pytest.raises(ValueError, match="validation failed"):
            validate_ohlcv_strict(df)


class TestTimeframeUtils:
    """Test cases for timeframe utilities."""

    def test_timeframe_to_seconds(self):
        """Test timeframe to seconds conversion."""
        assert timeframe_to_seconds("1m") == 60
        assert timeframe_to_seconds("15m") == 900
        assert timeframe_to_seconds("1h") == 3600
        assert timeframe_to_seconds("1d") == 86400

    def test_timeframe_to_seconds_invalid(self):
        """Test that invalid timeframe raises error."""
        with pytest.raises(ValueError, match="Unknown timeframe"):
            timeframe_to_seconds("invalid")

    def test_detect_gaps_no_gaps(self):
        """Test gap detection with no gaps."""
        index = pd.date_range("2024-01-01", periods=100, freq="15min")
        gaps = detect_gaps(index, "15m", max_gap_multiplier=2.0)

        assert len(gaps) == 0

    def test_detect_gaps_with_gap(self):
        """Test gap detection with a gap."""
        index1 = pd.date_range("2024-01-01 00:00", periods=10, freq="15min")
        index2 = pd.date_range("2024-01-01 03:00", periods=10, freq="15min")
        index = index1.union(index2)

        gaps = detect_gaps(index, "15m", max_gap_multiplier=2.0)

        assert len(gaps) == 1
        start, end, multiplier = gaps[0]
        assert multiplier > 2.0  # Should be much larger than threshold

    def test_detect_gaps_small_gap_ignored(self):
        """Test that small gaps below threshold are ignored."""
        # Create index with 1.5x gap (below 2.0x threshold)
        index = pd.DatetimeIndex([
            "2024-01-01 00:00:00",
            "2024-01-01 00:15:00",
            "2024-01-01 00:30:00",
            "2024-01-01 00:52:00",  # 22 minutes gap (1.47x)
            "2024-01-01 01:07:00",
        ])

        gaps = detect_gaps(index, "15m", max_gap_multiplier=2.0)

        assert len(gaps) == 0  # Should not detect 1.5x gap with 2.0x threshold

    def test_infer_timeframe(self):
        """Test timeframe inference from index."""
        index = pd.date_range("2024-01-01", periods=100, freq="15min")
        inferred = infer_timeframe(index)

        assert inferred == "15m"

    def test_infer_timeframe_hourly(self):
        """Test timeframe inference for hourly data."""
        index = pd.date_range("2024-01-01", periods=100, freq="1h")
        inferred = infer_timeframe(index)

        assert inferred == "1h"

    def test_infer_timeframe_insufficient_data(self):
        """Test that inference fails with insufficient data."""
        index = pd.DatetimeIndex(["2024-01-01 00:00:00"])

        with pytest.raises(ValueError, match="at least 2 timestamps"):
            infer_timeframe(index)


class TestValidationResult:
    """Test cases for ValidationResult class."""

    def test_summary_no_issues(self):
        """Test summary with no issues."""
        from finantradealgo.validation.ohlcv_validator import ValidationResult

        result = ValidationResult(is_valid=True)
        summary = result.summary()

        assert "PASS" in summary
        assert "Errors: 0" in summary

    def test_summary_with_issues(self):
        """Test summary with issues."""
        from finantradealgo.validation.ohlcv_validator import ValidationResult

        result = ValidationResult(is_valid=False)
        result.add_issue("test_check", "error", "Test error message")
        result.add_issue("test_warning", "warning", "Test warning message")

        summary = result.summary()

        assert "FAIL" in summary
        assert "Errors: 1" in summary
        assert "Warnings: 1" in summary
        assert "test_check" in summary
        assert "test_warning" in summary


class TestDataValidationConfig:
    """Test cases for DataValidationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        cfg = DataValidationConfig()

        assert cfg.mode == "warn"
        assert isinstance(cfg.ohlcv, OHLCVValidationConfig)

    def test_from_dict(self):
        """Test config creation from dictionary."""
        data = {
            "mode": "strict",
            "ohlcv": {
                "check_negative_prices": False,
            },
            "check_multi_tf_alignment": True,
        }
        cfg = DataValidationConfig.from_dict(data)

        assert cfg.mode == "strict"
        assert cfg.ohlcv.check_negative_prices is False
        assert cfg.check_multi_tf_alignment is True


class TestMultiTFAlignment:
    """Test cases for multi-timeframe alignment validation (Task S3.3)."""

    def test_valid_multi_tf_alignment(self):
        """Test validation with properly aligned multi-TF data."""
        # Create 15m data (4 bars per hour)
        df_15m = pd.DataFrame({
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5],
            "volume": [1000] * 16,
        }, index=pd.date_range("2024-01-01 00:00", periods=16, freq="15min"))

        # Create 1h data by properly aggregating 15m (4 bars each)
        # Hour 1 (00:00-01:00): bars 0-3 -> open=100, high=104, low=99, close=103.5
        # Hour 2 (01:00-02:00): bars 4-7 -> open=104, high=108, low=103, close=107.5
        # Hour 3 (02:00-03:00): bars 8-11 -> open=108, high=112, low=107, close=111.5
        # Hour 4 (03:00-04:00): bars 12-15 -> open=112, high=116, low=111, close=115.5
        df_1h = pd.DataFrame({
            "open": [100, 104, 108, 112],
            "high": [104, 108, 112, 116],
            "low": [99, 103, 107, 111],
            "close": [103.5, 107.5, 111.5, 115.5],
            "volume": [4000] * 4,
        }, index=pd.date_range("2024-01-01 00:00", periods=4, freq="1h"))

        result = validate_multi_tf_alignment(
            {"15m": df_15m, "1h": df_1h},
            base_timeframe="15m"
        )

        assert result.is_valid
        assert result.errors_count == 0

    def test_empty_input(self):
        """Test with empty dictionary."""
        result = validate_multi_tf_alignment({}, base_timeframe="15m")

        assert not result.is_valid
        assert any("No DataFrames" in issue.message or "empty" in issue.message.lower() for issue in result.issues)

    def test_missing_base_timeframe(self):
        """Test when base timeframe is not in provided DataFrames."""
        df_1h = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [1000, 1000],
        }, index=pd.date_range("2024-01-01", periods=2, freq="1h"))

        result = validate_multi_tf_alignment(
            {"1h": df_1h},
            base_timeframe="15m"  # Not in dict
        )

        assert not result.is_valid
        assert any("base" in issue.message.lower() for issue in result.issues)

    def test_invalid_index_type(self):
        """Test with DataFrame that doesn't have DatetimeIndex."""
        df_15m = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [1000, 1000],
        })  # No DatetimeIndex

        result = validate_multi_tf_alignment(
            {"15m": df_15m},
            base_timeframe="15m"
        )

        assert not result.is_valid
        assert any("DatetimeIndex" in issue.message for issue in result.issues)

    def test_tf_hierarchy_warning(self):
        """Test warning when HTF is smaller than base."""
        df_15m = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [1000, 1000],
        }, index=pd.date_range("2024-01-01", periods=2, freq="15min"))

        df_5m = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2024-01-01", periods=3, freq="5min"))

        result = validate_multi_tf_alignment(
            {"15m": df_15m, "5m": df_5m},
            base_timeframe="15m"
        )

        # Should warn about 5m being smaller than base
        assert result.warnings_count >= 1
        assert any("smaller" in issue.message.lower() for issue in result.issues)

    def test_misaligned_timestamps(self):
        """Test detection of misaligned timestamps."""
        df_15m = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [1000, 1000],
        }, index=pd.date_range("2024-01-01 00:00", periods=2, freq="15min"))

        # 1h data with misaligned timestamp (not on the hour)
        df_1h = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [1000, 1000],
        }, index=pd.DatetimeIndex([
            "2024-01-01 00:15:00",  # Should be 00:00:00
            "2024-01-01 01:15:00",  # Should be 01:00:00
        ]))

        result = validate_multi_tf_alignment(
            {"15m": df_15m, "1h": df_1h},
            base_timeframe="15m"
        )

        # Should warn about misaligned timestamps
        assert result.warnings_count >= 1
        assert any("misaligned" in issue.check_name for issue in result.issues)

    def test_price_consistency_check(self):
        """Test that price consistency is checked across timeframes."""
        # Create 15m data
        df_15m = pd.DataFrame({
            "open": [100, 101, 102, 103],
            "high": [101, 102, 103, 104],
            "low": [99, 100, 101, 102],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1000] * 4,
        }, index=pd.date_range("2024-01-01 00:00", periods=4, freq="15min"))

        # Create 1h data with INCONSISTENT prices
        df_1h = pd.DataFrame({
            "open": [100],
            "high": [110],  # Should be 104 based on 15m data
            "low": [99],
            "close": [103.5],
            "volume": [4000],
        }, index=pd.date_range("2024-01-01 00:00", periods=1, freq="1h"))

        result = validate_multi_tf_alignment(
            {"15m": df_15m, "1h": df_1h},
            base_timeframe="15m"
        )

        # Should detect price inconsistency
        assert result.errors_count >= 1
        assert any("inconsistency" in issue.check_name for issue in result.issues)
