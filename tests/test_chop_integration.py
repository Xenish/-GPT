"""
Test for chop detection integration into market structure engine.

Task S1.E2: Verify chop regime is properly integrated.
"""
import pandas as pd
import pytest

from finantradealgo.market_structure.config import MarketStructureConfig
from finantradealgo.market_structure.engine import MarketStructureEngine


class TestChopIntegration:
    """Test cases for chop detection in market structure engine."""

    def test_chop_column_exists(self):
        """Verify ms_chop_regime column is created."""
        df = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000, 1100, 1050, 1200, 1150],
        })

        cfg = MarketStructureConfig()
        engine = MarketStructureEngine(cfg)
        result = engine.compute_df(df)

        assert "ms_chop_regime" in result.features.columns, \
            "ms_chop_regime column should exist in output"

    def test_chop_values_in_valid_range(self):
        """Chop values should be between 0 and 1."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(20)],
            "high": [101 + i for i in range(20)],
            "low": [99 + i for i in range(20)],
            "close": [100.5 + i for i in range(20)],
            "volume": [1000 + i * 10 for i in range(20)],
        })

        cfg = MarketStructureConfig()
        engine = MarketStructureEngine(cfg)
        result = engine.compute_df(df)

        chop_values = result.features["ms_chop_regime"]
        assert (chop_values >= 0).all(), "Chop values should be >= 0"
        assert (chop_values <= 1).all(), "Chop values should be <= 1"

    def test_chop_trending_market(self):
        """Trending market should have low chop values."""
        # Create a strong uptrend
        df = pd.DataFrame({
            "open": [100 + i * 2 for i in range(30)],
            "high": [101 + i * 2 for i in range(30)],
            "low": [99 + i * 2 for i in range(30)],
            "close": [100.5 + i * 2 for i in range(30)],
            "volume": [1000] * 30,
        })

        cfg = MarketStructureConfig()
        engine = MarketStructureEngine(cfg)
        result = engine.compute_df(df)

        # After sufficient data, chop should be low (trending)
        # Check the last half of data
        chop_values = result.features["ms_chop_regime"].iloc[15:]
        avg_chop = chop_values.mean()

        assert avg_chop < 0.5, \
            f"Strong trend should have low chop, got {avg_chop:.3f}"

    def test_chop_choppy_market(self):
        """Choppy market should have high chop values."""
        # Create a choppy market (oscillating)
        import numpy as np
        np.random.seed(42)

        base = 100
        noise = np.random.normal(0, 0.5, 30)
        oscillation = np.sin(np.arange(30) * 0.5) * 2

        close = base + noise + oscillation

        df = pd.DataFrame({
            "open": close - 0.5,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": [1000] * 30,
        })

        cfg = MarketStructureConfig()
        engine = MarketStructureEngine(cfg)
        result = engine.compute_df(df)

        # After sufficient data, chop should be high (choppy)
        chop_values = result.features["ms_chop_regime"].iloc[15:]
        avg_chop = chop_values.mean()

        assert avg_chop > 0.5, \
            f"Choppy market should have high chop, got {avg_chop:.3f}"

    def test_chop_config_respected(self):
        """Chop config lookback_period should be respected."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(50)],
            "high": [101 + i for i in range(50)],
            "low": [99 + i for i in range(50)],
            "close": [100.5 + i for i in range(50)],
            "volume": [1000] * 50,
        })

        # Test with different lookback periods
        from finantradealgo.microstructure.config import ChopConfig

        cfg1 = MarketStructureConfig()
        cfg1.chop = ChopConfig(lookback_period=5)

        cfg2 = MarketStructureConfig()
        cfg2.chop = ChopConfig(lookback_period=20)

        engine1 = MarketStructureEngine(cfg1)
        engine2 = MarketStructureEngine(cfg2)

        result1 = engine1.compute_df(df)
        result2 = engine2.compute_df(df)

        # Results should be different with different lookback periods
        chop1 = result1.features["ms_chop_regime"]
        chop2 = result2.features["ms_chop_regime"]

        # They should not be identical
        assert not chop1.equals(chop2), \
            "Different lookback periods should produce different chop values"
