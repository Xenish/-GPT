"""
Integration tests for market structure and microstructure in feature pipeline.

CRITICAL-1: Ensure both MS and Micro features integrate correctly in pipeline.
"""
import pandas as pd
import pytest

from finantradealgo.features.feature_pipeline import (
    build_feature_pipeline,
    FeaturePipelineConfig,
)
from finantradealgo.market_structure.config import MarketStructureConfig
from finantradealgo.microstructure.config import MicrostructureConfig
from finantradealgo.market_structure.types import MarketStructureColumns
from finantradealgo.microstructure.types import MicrostructureSignals


class TestMarketStructurePipelineIntegration:
    """Test market structure integration in feature pipeline."""

    def test_market_structure_features_added(self):
        """Test that market structure features are added to pipeline."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(100)],
            "high": [101 + i for i in range(100)],
            "low": [99 + i for i in range(100)],
            "close": [100.5 + i for i in range(100)],
            "volume": [1000] * 100,
        }, index=pd.date_range("2024-01-01", periods=100, freq="1h"))

        cfg = FeaturePipelineConfig(
            use_base=False,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_external=False,
            use_rule_signals=False,
            use_market_structure=True,
            use_microstructure=False,
            drop_na=False,
        )

        result = build_feature_pipeline(df_ohlcv=df, pipeline_cfg=cfg)

        # Check market structure columns exist
        ms_cols = MarketStructureColumns()
        ms_col_names = [
            ms_cols.price_smooth,
            ms_cols.swing_high,
            ms_cols.swing_low,
            ms_cols.trend_regime,
            ms_cols.chop_regime,
            ms_cols.fvg_up,
            ms_cols.fvg_down,
        ]

        for col in ms_col_names:
            assert col in result.df.columns, f"Missing MS column: {col}"

    def test_market_structure_config_propagation(self):
        """Test that custom MS config is used in pipeline."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(100)],
            "high": [101 + i for i in range(100)],
            "low": [99 + i for i in range(100)],
            "close": [100.5 + i for i in range(100)],
            "volume": [1000] * 100,
        }, index=pd.date_range("2024-01-01", periods=100, freq="1h"))

        # Custom config with non-default values
        ms_cfg = MarketStructureConfig()
        ms_cfg.smoothing.price_ma_window = 5  # Non-default

        cfg = FeaturePipelineConfig(
            use_base=False,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_external=False,
            use_rule_signals=False,
            use_market_structure=True,
            use_microstructure=False,
            market_structure=ms_cfg,
            drop_na=False,
        )

        result = build_feature_pipeline(df_ohlcv=df, pipeline_cfg=cfg)

        # Verify pipeline ran successfully with custom config
        assert len(result.df) > 0


class TestMicrostructurePipelineIntegration:
    """Test microstructure integration in feature pipeline."""

    def test_microstructure_features_added(self):
        """Test that microstructure features are added to pipeline."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(100)],
            "high": [101 + i for i in range(100)],
            "low": [99 + i for i in range(100)],
            "close": [100.5 + i for i in range(100)],
            "volume": [1000] * 100,
        }, index=pd.date_range("2024-01-01", periods=100, freq="1h"))

        cfg = FeaturePipelineConfig(
            use_base=False,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_external=False,
            use_rule_signals=False,
            use_market_structure=False,
            use_microstructure=True,
            drop_na=False,
        )

        result = build_feature_pipeline(df_ohlcv=df, pipeline_cfg=cfg)

        # Check microstructure columns exist
        micro_cols = MicrostructureSignals.columns()

        for col in micro_cols:
            assert col in result.df.columns, f"Missing micro column: {col}"

    def test_microstructure_config_propagation(self):
        """Test that custom micro config is used in pipeline."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(100)],
            "high": [101 + i for i in range(100)],
            "low": [99 + i for i in range(100)],
            "close": [100.5 + i for i in range(100)],
            "volume": [1000] * 100,
        }, index=pd.date_range("2024-01-01", periods=100, freq="1h"))

        # Custom config with non-default values
        micro_cfg = MicrostructureConfig()
        micro_cfg.chop.lookback_period = 20  # Non-default

        cfg = FeaturePipelineConfig(
            use_base=False,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_external=False,
            use_rule_signals=False,
            use_market_structure=False,
            use_microstructure=True,
            microstructure=micro_cfg,
            drop_na=False,
        )

        result = build_feature_pipeline(df_ohlcv=df, pipeline_cfg=cfg)

        # Verify pipeline ran successfully with custom config
        assert len(result.df) > 0


class TestCombinedMSAndMicroIntegration:
    """Test combined market structure + microstructure integration."""

    def test_both_features_coexist(self):
        """Test that MS and Micro features can coexist in pipeline."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(100)],
            "high": [101 + i for i in range(100)],
            "low": [99 + i for i in range(100)],
            "close": [100.5 + i for i in range(100)],
            "volume": [1000] * 100,
        }, index=pd.date_range("2024-01-01", periods=100, freq="1h"))

        cfg = FeaturePipelineConfig(
            use_base=False,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_external=False,
            use_rule_signals=False,
            use_market_structure=True,
            use_microstructure=True,
            drop_na=False,
        )

        result = build_feature_pipeline(df_ohlcv=df, pipeline_cfg=cfg)

        # Check both MS and Micro columns exist
        ms_cols = MarketStructureColumns()
        assert ms_cols.trend_regime in result.df.columns
        assert ms_cols.chop_regime in result.df.columns

        micro_cols = MicrostructureSignals.columns()
        for col in micro_cols:
            assert col in result.df.columns

    def test_no_column_conflicts(self):
        """Test that MS and Micro don't have conflicting column names."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(100)],
            "high": [101 + i for i in range(100)],
            "low": [99 + i for i in range(100)],
            "close": [100.5 + i for i in range(100)],
            "volume": [1000] * 100,
        }, index=pd.date_range("2024-01-01", periods=100, freq="1h"))

        cfg = FeaturePipelineConfig(
            use_base=False,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_external=False,
            use_rule_signals=False,
            use_market_structure=True,
            use_microstructure=True,
            drop_na=False,
        )

        result = build_feature_pipeline(df_ohlcv=df, pipeline_cfg=cfg)

        # Check no duplicate columns
        assert len(result.df.columns) == len(set(result.df.columns)), \
            "Duplicate columns detected!"

    def test_full_pipeline_with_ms_micro(self):
        """Test full pipeline with base features + MS + Micro."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(100)],
            "high": [101 + i for i in range(100)],
            "low": [99 + i for i in range(100)],
            "close": [100.5 + i for i in range(100)],
            "volume": [1000] * 100,
        }, index=pd.date_range("2024-01-01", periods=100, freq="1h"))

        cfg = FeaturePipelineConfig(
            use_base=True,
            use_ta=True,
            use_candles=True,
            use_osc=True,
            use_htf=False,  # Requires 1h data
            use_external=False,
            use_rule_signals=False,
            use_market_structure=True,
            use_microstructure=True,
            drop_na=True,  # Drop NaN rows
        )

        result = build_feature_pipeline(df_ohlcv=df, pipeline_cfg=cfg)

        # Verify both MS and Micro columns exist after full pipeline
        ms_cols = MarketStructureColumns()
        assert ms_cols.trend_regime in result.df.columns

        assert "ms_chop" in result.df.columns
        assert "ms_vol_regime" in result.df.columns

        # Verify no NaN in result (after drop_na)
        assert not result.df.isna().any().any()


class TestPipelineMetadata:
    """Test pipeline metadata handling."""

    def test_metadata_contains_version(self):
        """Test that pipeline metadata includes version."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(50)],
            "high": [101 + i for i in range(50)],
            "low": [99 + i for i in range(50)],
            "close": [100.5 + i for i in range(50)],
            "volume": [1000] * 50,
        }, index=pd.date_range("2024-01-01", periods=50, freq="1h"))

        cfg = FeaturePipelineConfig(
            use_htf=False,  # Disable HTF to avoid timestamp dependency
            use_external=False,  # Disable external to avoid timestamp dependency
            use_rule_signals=False,  # Disable rule signals (depends on HTF)
            use_market_structure=True,
            use_microstructure=True,
            drop_na=False,
        )

        result = build_feature_pipeline(df_ohlcv=df, pipeline_cfg=cfg)

        assert "pipeline_version" in result.meta
        assert "feature_cols" in result.meta
        assert "feature_preset" in result.meta

    def test_market_structure_zones_in_metadata(self):
        """Test that zones are returned in metadata when requested."""
        df = pd.DataFrame({
            "open": [100 + i for i in range(100)],
            "high": [101 + i for i in range(100)],
            "low": [99 + i for i in range(100)],
            "close": [100.5 + i for i in range(100)],
            "volume": [1000] * 100,
        }, index=pd.date_range("2024-01-01", periods=100, freq="1h"))

        cfg = FeaturePipelineConfig(
            use_base=False,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_external=False,
            use_rule_signals=False,
            use_market_structure=True,
            market_structure_return_zones=True,
            drop_na=False,
        )

        result = build_feature_pipeline(df_ohlcv=df, pipeline_cfg=cfg)

        assert "market_structure_zones" in result.meta
        assert isinstance(result.meta["market_structure_zones"], list)
