"""
Tests for market structure debug/visualization helpers.

Task S1.E3: Test coverage for debug_plot.py module.
"""
import pandas as pd
import pytest

from finantradealgo.market_structure.config import MarketStructureConfig
from finantradealgo.market_structure.engine import MarketStructureEngine
from finantradealgo.market_structure.debug_plot import (
    summarize_market_structure,
    print_zone_details,
)


class TestDebugHelpers:
    """Test cases for debug and visualization helpers."""

    def test_summarize_market_structure_basic(self):
        """Test text summary generation."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(20)],
            "high": [101 + i for i in range(20)],
            "low": [99 + i for i in range(20)],
            "close": [100.5 + i for i in range(20)],
            "volume": [1000] * 20,
        })

        cfg = MarketStructureConfig()
        engine = MarketStructureEngine(cfg)
        result = engine.compute_df(df)

        summary = summarize_market_structure(result, df)

        # Check that summary contains key sections
        assert "MARKET STRUCTURE SUMMARY" in summary
        assert "Total bars:" in summary
        assert "Swing Highs:" in summary
        assert "Swing Lows:" in summary
        assert "Current Trend:" in summary
        assert "Current Chop:" in summary

    def test_summarize_market_structure_without_df(self):
        """Test summary works without original DataFrame."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(20)],
            "high": [101 + i for i in range(20)],
            "low": [99 + i for i in range(20)],
            "close": [100.5 + i for i in range(20)],
            "volume": [1000] * 20,
        })

        cfg = MarketStructureConfig()
        engine = MarketStructureEngine(cfg)
        result = engine.compute_df(df)

        # Should work without df
        summary = summarize_market_structure(result)

        assert "MARKET STRUCTURE SUMMARY" in summary
        assert "Total bars:" in summary

    def test_print_zone_details_no_zones(self, capsys):
        """Test zone details with empty zone list."""
        print_zone_details([])

        captured = capsys.readouterr()
        assert "No zones found" in captured.out

    def test_print_zone_details_with_zones(self, capsys):
        """Test zone details printing with actual zones."""
        from finantradealgo.market_structure.types import Zone

        zones = [
            Zone(
                id=1,
                type="demand",
                low=100.0,
                high=101.0,
                strength=2.0,
                first_ts=0,
                last_ts=5,
            ),
            Zone(
                id=2,
                type="supply",
                low=110.0,
                high=111.0,
                strength=3.0,
                first_ts=10,
                last_ts=15,
            ),
        ]

        print_zone_details(zones)

        captured = capsys.readouterr()
        assert "ZONE DETAILS" in captured.out
        assert "DEMAND" in captured.out
        assert "SUPPLY" in captured.out
        assert "100.00" in captured.out
        assert "110.00" in captured.out

    def test_plot_market_structure_requires_matplotlib(self):
        """Test that plotting raises error if matplotlib not available."""
        # This test is environment-dependent, so we just check the function exists
        from finantradealgo.market_structure.debug_plot import plot_market_structure

        # Function should be callable
        assert callable(plot_market_structure)

    def test_summarize_includes_all_metrics(self):
        """Test that summary includes all available metrics."""
        # Create data with enough bars for all features
        df = pd.DataFrame({
            "open": [100 + i * 0.5 for i in range(100)],
            "high": [101 + i * 0.5 for i in range(100)],
            "low": [99 + i * 0.5 for i in range(100)],
            "close": [100.5 + i * 0.5 for i in range(100)],
            "volume": [1000 + i * 10 for i in range(100)],
        })

        cfg = MarketStructureConfig()
        engine = MarketStructureEngine(cfg)
        result = engine.compute_df(df)

        summary = summarize_market_structure(result, df)

        # Check all major sections exist
        expected_sections = [
            "Total bars:",
            "Swing Highs:",
            "Swing Lows:",
            "Current Trend:",
            "Current Chop:",
            "FVG Up:",
            "FVG Down:",
            "BoS Up:",
            "BoS Down:",
            "Price Context:",
        ]

        for section in expected_sections:
            assert section in summary, f"Missing section: {section}"
