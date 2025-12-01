"""
Tests for feature health summary script.

Task CRITICAL-3: Validate feature health analysis functionality.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.print_feature_health_summary import (
    analyze_column_health,
    categorize_features,
    generate_feature_health_summary,
)


class TestAnalyzeColumnHealth:
    """Test column health analysis."""

    def test_numeric_column_health(self):
        """Test health analysis for numeric column."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = analyze_column_health(series, "test_col")

        assert metrics["name"] == "test_col"
        assert metrics["count"] == 5
        assert metrics["missing_count"] == 0
        assert metrics["missing_pct"] == 0.0
        assert metrics["min"] == 1.0
        assert metrics["max"] == 5.0
        assert metrics["mean"] == 3.0
        assert not metrics["is_constant"]

    def test_missing_values_detection(self):
        """Test detection of missing values."""
        series = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        metrics = analyze_column_health(series, "test_col")

        assert metrics["missing_count"] == 2
        assert metrics["missing_pct"] == 40.0
        assert any("MISSING" in issue for issue in metrics["issues"])

    def test_high_missing_values(self):
        """Test detection of high missing percentage."""
        series = pd.Series([1.0] + [np.nan] * 99)
        metrics = analyze_column_health(series, "test_col")

        assert metrics["missing_pct"] == 99.0
        assert any("HIGH_MISSING" in issue for issue in metrics["issues"])

    def test_infinite_values_detection(self):
        """Test detection of infinite values."""
        series = pd.Series([1.0, np.inf, 3.0, -np.inf, 5.0])
        metrics = analyze_column_health(series, "test_col")

        assert metrics["inf_count"] == 2
        assert any("INF_VALUES" in issue for issue in metrics["issues"])

    def test_constant_column_detection(self):
        """Test detection of constant columns."""
        series = pd.Series([5.0] * 100)
        metrics = analyze_column_health(series, "test_col")

        assert metrics["is_constant"]
        assert metrics["unique_count"] == 1
        assert any("CONSTANT" in issue for issue in metrics["issues"])

    def test_mostly_zeros_detection(self):
        """Test detection of columns with mostly zeros."""
        series = pd.Series([0.0] * 95 + [1.0] * 5)
        metrics = analyze_column_health(series, "test_col")

        assert metrics["zero_pct"] == 95.0
        assert any("MOSTLY_ZEROS" in issue for issue in metrics["issues"])

    def test_no_issues_column(self):
        """Test healthy column with no issues."""
        series = pd.Series(np.random.randn(100))
        metrics = analyze_column_health(series, "test_col")

        assert len(metrics["issues"]) == 0
        assert not metrics["is_constant"]
        assert metrics["missing_count"] == 0


class TestCategorizeFeatures:
    """Test feature categorization."""

    def test_ohlcv_categorization(self):
        """Test OHLCV column categorization."""
        columns = ["open", "high", "low", "close", "volume"]
        categories = categorize_features(columns)

        assert len(categories["ohlcv"]) == 5
        assert "open" in categories["ohlcv"]
        assert "volume" in categories["ohlcv"]

    def test_market_structure_categorization(self):
        """Test market structure feature categorization."""
        columns = ["ms_chop", "ms_trend", "ms_impulse_up"]
        categories = categorize_features(columns)

        assert len(categories["market_structure"]) == 3
        assert all(col in categories["market_structure"] for col in columns)

    def test_microstructure_categorization(self):
        """Test microstructure feature categorization."""
        columns = ["micro_imbalance", "ms_micro_sweep"]
        categories = categorize_features(columns)

        assert len(categories["microstructure"]) == 2

    def test_htf_categorization(self):
        """Test HTF feature categorization."""
        columns = ["htf1h_rsi_14", "htf4h_trend_score"]
        categories = categorize_features(columns)

        assert len(categories["htf"]) == 2

    def test_rule_signals_categorization(self):
        """Test rule signals categorization."""
        columns = ["rule_long_entry", "rule_long_exit"]
        categories = categorize_features(columns)

        assert len(categories["rule_signals"]) == 2

    def test_mixed_categorization(self):
        """Test categorization of mixed feature types."""
        columns = [
            "open", "close",  # OHLCV
            "ms_chop",  # Market structure
            "micro_imbalance",  # Microstructure
            "htf1h_rsi_14",  # HTF
            "rule_long_entry",  # Rule signals
            "rsi_14",  # Other
        ]
        categories = categorize_features(columns)

        assert len(categories["ohlcv"]) == 2
        assert len(categories["market_structure"]) == 1
        assert len(categories["microstructure"]) == 1
        assert len(categories["htf"]) == 1
        assert len(categories["rule_signals"]) == 1
        assert len(categories["other"]) == 1


class TestGenerateFeatureHealthSummary:
    """Test feature health summary generation."""

    def test_summary_generation_basic(self):
        """Test basic summary generation."""
        df = pd.DataFrame({
            "open": [100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1000, 2000, 1500],
        }, index=pd.date_range("2025-01-01", periods=3, freq="1h"))

        report = generate_feature_health_summary(df)

        assert "FEATURE HEALTH SUMMARY" in report
        assert "Total rows: 3" in report
        assert "Total columns: 3" in report
        assert "OVERALL STATISTICS" in report
        assert "FEATURE CATEGORIES" in report

    def test_summary_detects_issues(self):
        """Test summary detects and reports issues."""
        df = pd.DataFrame({
            "open": [100.0, 101.0, 102.0],
            "bad_col": [np.nan, np.nan, np.nan],  # All NaN
            "const_col": [5.0, 5.0, 5.0],  # Constant
        }, index=pd.date_range("2025-01-01", periods=3, freq="1h"))

        report = generate_feature_health_summary(df)

        assert "Columns with issues: 2/3" in report
        assert "bad_col" in report
        assert "const_col" in report
        assert "CONSTANT COLUMNS" in report

    def test_summary_with_market_structure_features(self):
        """Test summary with market structure features."""
        df = pd.DataFrame({
            "open": [100.0, 101.0, 102.0],
            "ms_chop": [0.5, 0.6, 0.4],
            "ms_trend": [0, 1, 1],
        }, index=pd.date_range("2025-01-01", periods=3, freq="1h"))

        report = generate_feature_health_summary(df)

        assert "MARKET_STRUCTURE: 2 columns" in report

    def test_summary_categorizes_correctly(self):
        """Test summary categorizes features correctly."""
        df = pd.DataFrame({
            "open": [100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 103.0],
            "ms_chop": [0.5, 0.6, 0.4],
            "micro_imbalance": [1.0, 2.0, 1.5],
            "htf1h_rsi_14": [50.0, 55.0, 60.0],
        }, index=pd.date_range("2025-01-01", periods=3, freq="1h"))

        report = generate_feature_health_summary(df)

        assert "OHLCV: 2 columns" in report
        assert "MARKET_STRUCTURE: 1 columns" in report
        assert "MICROSTRUCTURE: 1 columns" in report
        assert "HTF: 1 columns" in report

    def test_summary_handles_empty_dataframe(self):
        """Test summary handles empty DataFrame gracefully."""
        df = pd.DataFrame()

        # Should not crash
        report = generate_feature_health_summary(df)
        assert "FEATURE HEALTH SUMMARY" in report
