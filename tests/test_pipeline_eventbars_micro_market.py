"""
Integration test for the full pipeline with event bars, microstructure, and market structure.

This test verifies that:
1. Event bars (volume mode) are correctly applied
2. Microstructure features are computed on event bars
3. Market structure features are computed on event bars
4. The entire pipeline runs without crashes
5. Output contains all expected columns with reasonable data
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
from finantradealgo.system.config_loader import DataConfig, EventBarConfig
from finantradealgo.microstructure.config import MicrostructureConfig
from finantradealgo.market_structure.config import MarketStructureConfig


def create_test_1m_ohlcv_csv():
    """
    Create a temporary 1-minute OHLCV CSV file with enough data for event bars.

    Returns 200 bars of synthetic 1m data with realistic price movement
    for testing event bar aggregation.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write('timestamp,open,high,low,close,volume\n')

        np.random.seed(42)
        base_price = 100.0

        for i in range(200):
            ts = pd.Timestamp('2023-01-01 09:00:00') + pd.Timedelta(minutes=i)

            # Create realistic price movement with some trend
            trend = 0.02 * (i // 20)  # Slight upward trend every 20 bars
            noise = np.random.randn() * 0.5
            price = base_price + trend + noise

            # OHLC with some intra-bar movement
            open_price = price
            high_price = price + abs(np.random.randn() * 0.3)
            low_price = price - abs(np.random.randn() * 0.3)
            close_price = price + np.random.randn() * 0.2

            # Variable volume for more realistic event bars
            volume = 500 + np.random.randint(-200, 300)

            f.write(f'{ts},{open_price:.2f},{high_price:.2f},{low_price:.2f},{close_price:.2f},{volume}\n')

        return f.name


def test_pipeline_with_event_bars_and_all_features():
    """
    Full integration test: event bars + microstructure + market structure.

    This is the primary integration test that ensures the entire pipeline
    works correctly when event bars are used as the base timeframe.
    """
    csv_path = create_test_1m_ohlcv_csv()

    try:
        # Configure event bars: volume mode with target of 5000
        # This should aggregate roughly 10 bars per event bar (500*10 = 5000)
        data_cfg = DataConfig(
            bars=EventBarConfig(
                mode="volume",
                target_volume=5000.0,
                source_timeframe="1m",
                keep_partial_last_bar=True
            )
        )

        # Configure pipeline with all features enabled
        pipeline_cfg = FeaturePipelineConfig(
            use_base=True,
            use_ta=False,  # Keep test simple, focus on micro/market structure
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_microstructure=True,  # Enable microstructure
            use_market_structure=True,  # Enable market structure
            market_structure_return_zones=True,  # Request zones in meta
            use_external=False,
            use_rule_signals=False,
            drop_na=False,  # Keep NaNs to verify data quality
        )

        # Run the pipeline
        result = build_feature_pipeline(csv_path, pipeline_cfg, data_cfg=data_cfg)

        # ===== ASSERTIONS =====

        # 1. Verify result type
        assert isinstance(result, FeaturePipelineResult), \
            "Pipeline should return FeaturePipelineResult"

        assert isinstance(result.df, pd.DataFrame), \
            "result.df should be a DataFrame"

        # 2. Verify we have event bars (fewer bars than original 200)
        assert len(result.df) < 200, \
            f"Event bars should reduce bar count from 200, got {len(result.df)}"

        assert len(result.df) > 10, \
            f"Should have reasonable number of event bars (>10), got {len(result.df)}"

        # 3. Verify event bar metadata columns exist
        assert 'bar_start_ts' in result.df.columns, \
            "Event bars should add bar_start_ts column"

        assert result.df.index.name == 'bar_end_ts', \
            f"Index should be bar_end_ts for event bars, got {result.df.index.name}"

        # 4. Verify microstructure columns exist
        expected_microstructure_cols = {
            'ms_imbalance',
            'ms_sweep_up',
            'ms_sweep_down',
            'ms_chop',
            'ms_burst_up',
            'ms_burst_down',
            'ms_vol_regime',
            'ms_exhaustion_up',
            'ms_exhaustion_down',
            'ms_parabolic_trend',
        }

        actual_microstructure_cols = {col for col in result.df.columns
                                      if col in expected_microstructure_cols}

        assert actual_microstructure_cols == expected_microstructure_cols, \
            f"Missing microstructure columns: {expected_microstructure_cols - actual_microstructure_cols}"

        # 5. Verify market structure columns exist
        expected_market_structure_cols = {
            'ms_swing_high',
            'ms_swing_low',
            'ms_trend_regime',
            'ms_fvg_up',
            'ms_fvg_down',
            'ms_zone_demand',
            'ms_zone_supply',
        }

        actual_market_structure_cols = {col for col in result.df.columns
                                        if col in expected_market_structure_cols}

        assert actual_market_structure_cols == expected_market_structure_cols, \
            f"Missing market structure columns: {expected_market_structure_cols - actual_market_structure_cols}"

        # 6. Verify market structure zones in meta
        assert 'market_structure_zones' in result.meta, \
            "market_structure_zones should be in meta when market_structure_return_zones=True"

        assert isinstance(result.meta['market_structure_zones'], list), \
            "market_structure_zones should be a list"

        # 7. Verify NaN ratios are reasonable (not all NaN)
        # Microstructure features should have SOME valid values
        for col in expected_microstructure_cols:
            nan_ratio = result.df[col].isna().sum() / len(result.df)
            assert nan_ratio < 1.0, \
                f"Microstructure column {col} is all NaN - calculation failed"

        # Market structure features should have SOME valid values
        for col in expected_market_structure_cols:
            nan_ratio = result.df[col].isna().sum() / len(result.df)
            assert nan_ratio < 1.0, \
                f"Market structure column {col} is all NaN - calculation failed"

        # 8. Verify basic sanity of microstructure values
        # vol_regime should be in [0, 1, 2]
        vol_regime_values = result.df['ms_vol_regime'].dropna().unique()
        assert all(v in [0.0, 1.0, 2.0] for v in vol_regime_values), \
            f"ms_vol_regime should only have values [0, 1, 2], got {vol_regime_values}"

        # chop should be in [0, 100]
        chop_values = result.df['ms_chop'].dropna()
        if len(chop_values) > 0:
            assert chop_values.min() >= 0.0 and chop_values.max() <= 100.0, \
                f"ms_chop should be in [0, 100], got range [{chop_values.min()}, {chop_values.max()}]"

        # 9. Verify basic sanity of market structure values
        # trend_regime should be in [-1, 0, 1]
        trend_regime_values = result.df['ms_trend_regime'].dropna().unique()
        assert all(v in [-1.0, 0.0, 1.0] for v in trend_regime_values), \
            f"ms_trend_regime should only have values [-1, 0, 1], got {trend_regime_values}"

        # FVG and zone columns should be 0 or 1 (boolean flags)
        for col in ['ms_fvg_up', 'ms_fvg_down', 'ms_zone_demand', 'ms_zone_supply']:
            unique_values = result.df[col].dropna().unique()
            assert all(v in [0.0, 1.0] for v in unique_values), \
                f"{col} should only have values [0, 1], got {unique_values}"

    finally:
        import os
        os.unlink(csv_path)


def test_pipeline_event_bars_without_market_structure():
    """
    Test event bars + microstructure only (no market structure).

    This ensures microstructure works independently on event bars.
    """
    csv_path = create_test_1m_ohlcv_csv()

    try:
        # Configure event bars
        data_cfg = DataConfig(
            bars=EventBarConfig(
                mode="volume",
                target_volume=5000.0,
                source_timeframe="1m",
                keep_partial_last_bar=True
            )
        )

        # Configure pipeline with only microstructure
        pipeline_cfg = FeaturePipelineConfig(
            use_base=True,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_microstructure=True,  # Enable microstructure
            use_market_structure=False,  # Disable market structure
            use_external=False,
            use_rule_signals=False,
            drop_na=False,
        )

        # Run the pipeline
        result = build_feature_pipeline(csv_path, pipeline_cfg, data_cfg=data_cfg)

        # Verify microstructure columns exist
        expected_microstructure_cols = {
            'ms_imbalance', 'ms_sweep_up', 'ms_sweep_down',
            'ms_chop', 'ms_burst_up', 'ms_burst_down',
            'ms_vol_regime', 'ms_exhaustion_up', 'ms_exhaustion_down',
            'ms_parabolic_trend',
        }

        actual_microstructure_cols = {col for col in result.df.columns
                                      if col in expected_microstructure_cols}

        assert actual_microstructure_cols == expected_microstructure_cols, \
            f"Missing microstructure columns: {expected_microstructure_cols - actual_microstructure_cols}"

        # Verify NO market structure columns
        market_structure_cols = {col for col in result.df.columns
                                 if col.startswith('ms_swing') or col.startswith('ms_fvg')
                                 or col.startswith('ms_zone') or col.startswith('ms_trend')}

        # Note: ms_trend_regime is market structure, not microstructure
        # But all microstructure cols start with ms_ too, so we need to be specific
        unexpected_ms_cols = market_structure_cols - expected_microstructure_cols
        assert len(unexpected_ms_cols) == 0, \
            f"Should not have market structure columns when disabled, found: {unexpected_ms_cols}"

    finally:
        import os
        os.unlink(csv_path)


def test_pipeline_event_bars_different_modes():
    """
    Test that different event bar modes work with the pipeline.

    Tests dollar and tick modes in addition to volume mode.
    """
    csv_path = create_test_1m_ohlcv_csv()

    try:
        # Test 1: Dollar mode
        data_cfg_dollar = DataConfig(
            bars=EventBarConfig(
                mode="dollar",
                target_notional=50000.0,
                source_timeframe="1m",
                keep_partial_last_bar=True
            )
        )

        pipeline_cfg = FeaturePipelineConfig(
            use_base=True,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_microstructure=True,
            use_market_structure=True,
            market_structure_return_zones=False,
            use_external=False,
            use_rule_signals=False,
            drop_na=False,
        )

        result_dollar = build_feature_pipeline(csv_path, pipeline_cfg, data_cfg=data_cfg_dollar)

        assert isinstance(result_dollar.df, pd.DataFrame), "Dollar mode should work"
        assert 'bar_start_ts' in result_dollar.df.columns, "Dollar mode should have event bar metadata"
        assert len(result_dollar.df) < 200, "Dollar mode should reduce bar count"

        # Test 2: Tick mode
        data_cfg_tick = DataConfig(
            bars=EventBarConfig(
                mode="tick",
                target_ticks=10,
                source_timeframe="1m",
                keep_partial_last_bar=True
            )
        )

        result_tick = build_feature_pipeline(csv_path, pipeline_cfg, data_cfg=data_cfg_tick)

        assert isinstance(result_tick.df, pd.DataFrame), "Tick mode should work"
        assert 'bar_start_ts' in result_tick.df.columns, "Tick mode should have event bar metadata"
        assert len(result_tick.df) < 200, "Tick mode should reduce bar count"

    finally:
        import os
        os.unlink(csv_path)
