"""
Integration tests for market structure features in the feature pipeline.

This test ensures that the market_structure → feature_pipeline integration
works correctly and prevents the tuple bug from reoccurring.
"""
import tempfile
import numpy as np
import pandas as pd
import pytest

from finantradealgo.features.feature_pipeline import (
    FeaturePipelineConfig,
    FeaturePipelineResult,
    build_feature_pipeline,
)
from finantradealgo.market_structure.config import MarketStructureConfig


def create_test_ohlcv_csv():
    """Create a temporary OHLCV CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write('timestamp,open,high,low,close,volume\n')
        np.random.seed(42)
        for i in range(100):
            ts = pd.Timestamp('2023-01-01') + pd.Timedelta(minutes=15 * i)
            price = 100 + np.random.randn() * 5
            f.write(f'{ts},{price},{price + 1},{price - 1},{price},1000\n')
        return f.name


def test_market_structure_integration_without_zones():
    """
    Test that market structure features are added correctly
    when market_structure_return_zones=False.
    """
    csv_path = create_test_ohlcv_csv()

    try:
        cfg = FeaturePipelineConfig(
            use_base=True,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_microstructure=False,
            use_market_structure=True,  # Enable market structure
            market_structure_return_zones=False,  # Don't return zones
            use_external=False,
            use_rule_signals=False,
            drop_na=False,
        )

        result = build_feature_pipeline(csv_path, cfg)

        # Assert result type
        assert isinstance(result, FeaturePipelineResult), \
            "Pipeline should return FeaturePipelineResult"

        # Assert df is DataFrame (not tuple!)
        assert isinstance(result.df, pd.DataFrame), \
            "result.df should be a DataFrame, not a tuple"

        # Assert market structure columns exist
        ms_cols = [col for col in result.df.columns if col.startswith("ms_")]
        assert len(ms_cols) > 0, \
            "Market structure columns (ms_*) should be present"

        expected_ms_cols = {
            "ms_swing_high", "ms_swing_low", "ms_trend_regime",
            "ms_fvg_up", "ms_fvg_down", "ms_zone_demand", "ms_zone_supply"
        }
        assert expected_ms_cols.issubset(result.df.columns), \
            f"Expected MS columns missing. Found: {ms_cols}"

        # Assert zones NOT in meta
        assert "market_structure_zones" not in result.meta, \
            "Zones should NOT be in meta when market_structure_return_zones=False"

    finally:
        import os
        os.unlink(csv_path)


def test_market_structure_integration_with_zones():
    """
    Test that market structure zones are correctly added to meta
    when market_structure_return_zones=True.
    """
    csv_path = create_test_ohlcv_csv()

    try:
        cfg = FeaturePipelineConfig(
            use_base=True,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_microstructure=False,
            use_market_structure=True,  # Enable market structure
            market_structure_return_zones=True,  # Return zones in meta
            use_external=False,
            use_rule_signals=False,
            drop_na=False,
        )

        result = build_feature_pipeline(csv_path, cfg)

        # Assert result type
        assert isinstance(result, FeaturePipelineResult), \
            "Pipeline should return FeaturePipelineResult"

        # Assert df is DataFrame (not tuple!)
        assert isinstance(result.df, pd.DataFrame), \
            "result.df should be a DataFrame, not a tuple"

        # Assert market structure columns exist
        assert any(col.startswith("ms_") for col in result.df.columns), \
            "Market structure columns (ms_*) should be present"

        # Assert zones ARE in meta
        assert "market_structure_zones" in result.meta, \
            "Zones should be in meta when market_structure_return_zones=True"

        # Assert zones is a list
        zones = result.meta["market_structure_zones"]
        assert isinstance(zones, list), \
            "market_structure_zones should be a list"

    finally:
        import os
        os.unlink(csv_path)


def test_tuple_bug_does_not_return():
    """
    Regression test: Ensure the pipeline NEVER returns a tuple.

    This test exists to catch the bug where market structure integration
    would return (df, zones) instead of a proper result object.
    """
    csv_path = create_test_ohlcv_csv()

    try:
        cfg = FeaturePipelineConfig(
            use_market_structure=True,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_external=False,
            use_rule_signals=False,
            drop_na=False,
        )

        result = build_feature_pipeline(csv_path, cfg)

        # THE KEY ASSERTION: Result should NOT be a plain tuple
        assert not isinstance(result, tuple) or isinstance(result, FeaturePipelineResult), \
            "REGRESSION: Pipeline returned a plain tuple! The tuple bug has returned."

        # Result should be FeaturePipelineResult
        assert isinstance(result, FeaturePipelineResult), \
            "Pipeline should return FeaturePipelineResult, not a plain tuple"

        # But it should still support tuple unpacking for backward compat
        df, meta = result
        assert isinstance(df, pd.DataFrame), \
            "Tuple unpacking should work (backward compat)"
        assert isinstance(meta, dict), \
            "Tuple unpacking should yield (DataFrame, dict)"

    finally:
        import os
        os.unlink(csv_path)


def test_backward_compatibility_tuple_unpacking():
    """
    Test that backward compatibility is maintained via tuple unpacking.
    """
    csv_path = create_test_ohlcv_csv()

    try:
        cfg = FeaturePipelineConfig(
            use_market_structure=True,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_external=False,
            use_rule_signals=False,
            drop_na=False,
        )

        # Old style: tuple unpacking should still work
        df, meta = build_feature_pipeline(csv_path, cfg)

        assert isinstance(df, pd.DataFrame), \
            "Tuple unpacking should yield DataFrame as first element"
        assert isinstance(meta, dict), \
            "Tuple unpacking should yield dict as second element"
        assert "feature_cols" in meta, \
            "Meta should contain feature_cols"

    finally:
        import os
        os.unlink(csv_path)
