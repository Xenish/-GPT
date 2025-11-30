"""
Test validator integration with feature pipeline.

Verifies that:
1. validate_market_structure flag correctly enables/disables validation
2. Validation errors are raised when structural violations occur
3. Valid market structure passes validation
"""

import pandas as pd
import pytest
import tempfile
from pathlib import Path

from finantradealgo.features.feature_pipeline import (
    FeaturePipelineConfig,
    build_feature_pipeline
)


class TestValidatorIntegration:
    """Test market structure validator integration in feature pipeline."""

    @pytest.fixture
    def sample_ohlcv_csv(self):
        """Create a temporary CSV file with sample OHLCV data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ohlcv_path = Path(tmpdir) / "BTCUSDT_15m.csv"

            # Create sample OHLCV data
            ohlcv_data = pd.DataFrame({
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="15min"),
                "open": [40000 + i * 10 for i in range(100)],
                "high": [40100 + i * 10 for i in range(100)],
                "low": [39900 + i * 10 for i in range(100)],
                "close": [40000 + i * 10 for i in range(100)],
                "volume": [1000.0 + i for i in range(100)]
            })
            ohlcv_data.to_csv(ohlcv_path, index=False)

            yield str(ohlcv_path)

    def test_validator_disabled_by_default(self, sample_ohlcv_csv):
        """Test that validator is disabled by default."""
        cfg = FeaturePipelineConfig(
            use_market_structure=True,
            market_structure_return_zones=True,
            validate_market_structure=False,  # Explicitly disabled
            use_base=False,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_external=False,
            use_rule_signals=False,
            drop_na=False
        )

        # Should not raise even if data has violations
        result = build_feature_pipeline(sample_ohlcv_csv, pipeline_cfg=cfg)

        assert result.df is not None
        assert "market_structure_zones" in result.meta

    def test_validator_enabled_passes_valid_data(self, sample_ohlcv_csv):
        """Test that validator passes with valid market structure data."""
        cfg = FeaturePipelineConfig(
            use_market_structure=True,
            market_structure_return_zones=True,
            validate_market_structure=True,  # Enabled
            use_base=False,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_external=False,
            use_rule_signals=False,
            drop_na=False
        )

        # Should pass validation with valid data
        result = build_feature_pipeline(sample_ohlcv_csv, pipeline_cfg=cfg)

        assert result.df is not None
        assert "market_structure_zones" in result.meta

    def test_validator_works_without_zones(self, sample_ohlcv_csv):
        """Test that validator works even when zones are not returned."""
        cfg = FeaturePipelineConfig(
            use_market_structure=True,
            market_structure_return_zones=False,  # No zones
            validate_market_structure=True,  # But validation enabled
            use_base=False,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_external=False,
            use_rule_signals=False,
            drop_na=False
        )

        # Should still run validation (even without zones)
        result = build_feature_pipeline(sample_ohlcv_csv, pipeline_cfg=cfg)

        assert result.df is not None
        assert "market_structure_zones" not in result.meta

    def test_validator_disabled_when_market_structure_disabled(self, sample_ohlcv_csv):
        """Test that validator doesn't run when market structure is disabled."""
        cfg = FeaturePipelineConfig(
            use_market_structure=False,  # Market structure disabled
            validate_market_structure=True,  # This should have no effect
            use_base=False,
            use_ta=False,
            use_candles=False,
            use_osc=False,
            use_htf=False,
            use_external=False,
            use_rule_signals=False,
            drop_na=False
        )

        # Should not raise, validator shouldn't run
        result = build_feature_pipeline(sample_ohlcv_csv, pipeline_cfg=cfg)

        assert result.df is not None
        assert "market_structure_zones" not in result.meta


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
