"""
Config round-trip tests for YAML serialization/deserialization.

Task CRITICAL-2: Ensure all config classes can safely round-trip through YAML
without data loss or corruption.
"""
import pytest
import yaml

from finantradealgo.market_structure.config import (
    MarketStructureConfig,
    SmoothingConfig,
    ChopConfig as MSChopConfig,
)
from finantradealgo.microstructure.config import (
    MicrostructureConfig,
    ImbalanceConfig,
    LiquiditySweepConfig,
    ChopConfig as MicroChopConfig,
    VolatilityRegimeConfig,
    BurstConfig,
    ExhaustionConfig,
    ParabolicConfig,
)
from finantradealgo.validation.config import (
    DataValidationConfig,
    OHLCVValidationConfig,
    ExternalValidationConfig,
)


class TestMarketStructureConfigRoundtrip:
    """Test Market Structure config round-trip through YAML."""

    def test_smoothing_config_roundtrip(self):
        """Test SmoothingConfig YAML round-trip."""
        original = SmoothingConfig(
            enabled=True,
            price_ma_window=5,
            swing_min_distance=4,
            swing_min_zscore=0.7,
        )

        # Serialize to dict
        data = {
            "enabled": original.enabled,
            "price_ma_window": original.price_ma_window,
            "swing_min_distance": original.swing_min_distance,
            "swing_min_zscore": original.swing_min_zscore,
        }

        # Simulate YAML round-trip
        yaml_str = yaml.dump(data)
        loaded_data = yaml.safe_load(yaml_str)

        # Deserialize
        restored = SmoothingConfig.from_dict(loaded_data)

        # Verify equality
        assert restored.enabled == original.enabled
        assert restored.price_ma_window == original.price_ma_window
        assert restored.swing_min_distance == original.swing_min_distance
        assert restored.swing_min_zscore == original.swing_min_zscore

    def test_ms_chop_config_roundtrip(self):
        """Test MS ChopConfig YAML round-trip."""
        original = MSChopConfig(lookback_period=20)

        data = {"lookback_period": original.lookback_period}
        yaml_str = yaml.dump(data)
        loaded_data = yaml.safe_load(yaml_str)

        restored = MSChopConfig.from_dict(loaded_data)

        assert restored.lookback_period == original.lookback_period

    def test_market_structure_config_roundtrip(self):
        """Test MarketStructureConfig YAML round-trip."""
        original = MarketStructureConfig(
            smoothing=SmoothingConfig(enabled=True, price_ma_window=5),
            chop=MSChopConfig(lookback_period=20),
        )

        # Serialize to dict
        data = {
            "smoothing": {
                "enabled": original.smoothing.enabled,
                "price_ma_window": original.smoothing.price_ma_window,
                "swing_min_distance": original.smoothing.swing_min_distance,
                "swing_min_zscore": original.smoothing.swing_min_zscore,
            },
            "chop": {
                "lookback_period": original.chop.lookback_period,
            },
        }

        # YAML round-trip
        yaml_str = yaml.dump(data)
        loaded_data = yaml.safe_load(yaml_str)

        # Deserialize
        restored = MarketStructureConfig.from_dict(loaded_data)

        # Verify
        assert restored.smoothing.enabled == original.smoothing.enabled
        assert restored.smoothing.price_ma_window == original.smoothing.price_ma_window
        assert restored.chop.lookback_period == original.chop.lookback_period


class TestMicrostructureConfigRoundtrip:
    """Test Microstructure config round-trip through YAML."""

    def test_imbalance_config_roundtrip(self):
        """Test ImbalanceConfig YAML round-trip."""
        original = ImbalanceConfig(depth=10, threshold=3.0)

        data = {"depth": original.depth, "threshold": original.threshold}
        yaml_str = yaml.dump(data)
        loaded_data = yaml.safe_load(yaml_str)

        restored = ImbalanceConfig.from_dict(loaded_data)

        assert restored.depth == original.depth
        assert restored.threshold == original.threshold

    def test_liquidity_sweep_config_roundtrip(self):
        """Test LiquiditySweepConfig YAML round-trip."""
        original = LiquiditySweepConfig(
            lookback_ms=10000,
            notional_threshold=100000.0,
        )

        data = {
            "lookback_ms": original.lookback_ms,
            "notional_threshold": original.notional_threshold,
        }
        yaml_str = yaml.dump(data)
        loaded_data = yaml.safe_load(yaml_str)

        restored = LiquiditySweepConfig.from_dict(loaded_data)

        assert restored.lookback_ms == original.lookback_ms
        assert restored.notional_threshold == original.notional_threshold

    def test_microstructure_config_roundtrip(self):
        """Test MicrostructureConfig YAML round-trip."""
        original = MicrostructureConfig(
            imbalance=ImbalanceConfig(depth=10, threshold=3.0),
            sweep=LiquiditySweepConfig(lookback_ms=10000),
            chop=MicroChopConfig(lookback_period=20),
            vol_regime=VolatilityRegimeConfig(period=30),
            burst=BurstConfig(return_window=10),
            exhaustion=ExhaustionConfig(min_consecutive_bars=10),
            parabolic=ParabolicConfig(rolling_std_window=30),
            enabled=True,
            max_lookback_seconds=7200,
        )

        # Serialize to dict (comprehensive)
        data = {
            "enabled": original.enabled,
            "max_lookback_seconds": original.max_lookback_seconds,
            "imbalance": {
                "depth": original.imbalance.depth,
                "threshold": original.imbalance.threshold,
            },
            "sweep": {
                "lookback_ms": original.sweep.lookback_ms,
                "notional_threshold": original.sweep.notional_threshold,
            },
            "chop": {"lookback_period": original.chop.lookback_period},
            "vol_regime": {
                "period": original.vol_regime.period,
                "z_score_window": original.vol_regime.z_score_window,
                "low_z_threshold": original.vol_regime.low_z_threshold,
                "high_z_threshold": original.vol_regime.high_z_threshold,
            },
            "burst": {
                "return_window": original.burst.return_window,
                "z_score_window": original.burst.z_score_window,
                "z_up_threshold": original.burst.z_up_threshold,
                "z_down_threshold": original.burst.z_down_threshold,
            },
            "exhaustion": {
                "min_consecutive_bars": original.exhaustion.min_consecutive_bars,
                "volume_z_score_window": original.exhaustion.volume_z_score_window,
                "volume_z_threshold": original.exhaustion.volume_z_threshold,
            },
            "parabolic": {
                "rolling_std_window": original.parabolic.rolling_std_window,
                "curvature_threshold": original.parabolic.curvature_threshold,
            },
        }

        # YAML round-trip
        yaml_str = yaml.dump(data)
        loaded_data = yaml.safe_load(yaml_str)

        # Deserialize
        restored = MicrostructureConfig.from_dict(loaded_data)

        # Verify all fields
        assert restored.enabled == original.enabled
        assert restored.max_lookback_seconds == original.max_lookback_seconds
        assert restored.imbalance.depth == original.imbalance.depth
        assert restored.sweep.lookback_ms == original.sweep.lookback_ms
        assert restored.chop.lookback_period == original.chop.lookback_period
        assert restored.vol_regime.period == original.vol_regime.period
        assert restored.burst.return_window == original.burst.return_window
        assert restored.exhaustion.min_consecutive_bars == original.exhaustion.min_consecutive_bars
        assert restored.parabolic.rolling_std_window == original.parabolic.rolling_std_window


class TestValidationConfigRoundtrip:
    """Test Validation config round-trip through YAML."""

    def test_ohlcv_validation_config_roundtrip(self):
        """Test OHLCVValidationConfig YAML round-trip."""
        original = OHLCVValidationConfig(
            check_negative_prices=False,
            check_zero_prices=False,
            check_ohlc_relationship=True,
            max_gap_multiplier=3.0,
            price_spike_z_threshold=4.0,
        )

        # Serialize
        data = {
            "check_negative_prices": original.check_negative_prices,
            "check_zero_prices": original.check_zero_prices,
            "check_ohlc_relationship": original.check_ohlc_relationship,
            "max_gap_multiplier": original.max_gap_multiplier,
            "price_spike_z_threshold": original.price_spike_z_threshold,
        }

        # YAML round-trip
        yaml_str = yaml.dump(data)
        loaded_data = yaml.safe_load(yaml_str)

        # Deserialize
        restored = OHLCVValidationConfig.from_dict(loaded_data)

        # Verify
        assert restored.check_negative_prices == original.check_negative_prices
        assert restored.check_zero_prices == original.check_zero_prices
        assert restored.check_ohlc_relationship == original.check_ohlc_relationship
        assert restored.max_gap_multiplier == original.max_gap_multiplier
        assert restored.price_spike_z_threshold == original.price_spike_z_threshold

    def test_external_validation_config_roundtrip(self):
        """Test ExternalValidationConfig YAML round-trip."""
        original = ExternalValidationConfig(
            check_missing_data=True,
            max_missing_pct=0.05,
            check_value_range=True,
            min_value=-1.0,
            max_value=1.0,
        )

        data = {
            "check_missing_data": original.check_missing_data,
            "max_missing_pct": original.max_missing_pct,
            "check_value_range": original.check_value_range,
            "min_value": original.min_value,
            "max_value": original.max_value,
        }

        yaml_str = yaml.dump(data)
        loaded_data = yaml.safe_load(yaml_str)

        restored = ExternalValidationConfig.from_dict(loaded_data)

        assert restored.check_missing_data == original.check_missing_data
        assert restored.max_missing_pct == original.max_missing_pct
        assert restored.check_value_range == original.check_value_range
        assert restored.min_value == original.min_value
        assert restored.max_value == original.max_value

    def test_data_validation_config_roundtrip(self):
        """Test DataValidationConfig YAML round-trip."""
        original = DataValidationConfig(
            mode="strict",
            ohlcv=OHLCVValidationConfig(check_negative_prices=False),
            external=ExternalValidationConfig(max_missing_pct=0.05),
            check_multi_tf_alignment=True,
            check_suspect_bars=True,
        )

        # Serialize (nested)
        data = {
            "mode": original.mode,
            "ohlcv": {
                "check_negative_prices": original.ohlcv.check_negative_prices,
                "check_zero_prices": original.ohlcv.check_zero_prices,
            },
            "external": {
                "check_missing_data": original.external.check_missing_data,
                "max_missing_pct": original.external.max_missing_pct,
            },
            "check_multi_tf_alignment": original.check_multi_tf_alignment,
            "check_suspect_bars": original.check_suspect_bars,
        }

        # YAML round-trip
        yaml_str = yaml.dump(data)
        loaded_data = yaml.safe_load(yaml_str)

        # Deserialize
        restored = DataValidationConfig.from_dict(loaded_data)

        # Verify
        assert restored.mode == original.mode
        assert restored.ohlcv.check_negative_prices == original.ohlcv.check_negative_prices
        assert restored.external.max_missing_pct == original.external.max_missing_pct
        assert restored.check_multi_tf_alignment == original.check_multi_tf_alignment
        assert restored.check_suspect_bars == original.check_suspect_bars


class TestConfigDefaultValues:
    """Test that configs with None/empty dicts use proper defaults."""

    def test_microstructure_config_with_none(self):
        """Test MicrostructureConfig.from_dict(None) uses defaults."""
        cfg = MicrostructureConfig.from_dict(None)

        assert cfg.enabled is True
        assert cfg.max_lookback_seconds == 3600
        assert cfg.imbalance.depth == 5
        assert cfg.sweep.lookback_ms == 5000

    def test_market_structure_config_with_none(self):
        """Test MarketStructureConfig.from_dict(None) uses defaults."""
        cfg = MarketStructureConfig.from_dict(None)

        assert cfg.smoothing.enabled is True
        assert cfg.smoothing.price_ma_window == 3
        assert cfg.chop.lookback_period == 14

    def test_validation_config_with_empty_dict(self):
        """Test DataValidationConfig.from_dict({}) uses defaults."""
        cfg = DataValidationConfig.from_dict({})

        assert cfg.mode == "warn"
        assert cfg.ohlcv.check_negative_prices is True
        assert cfg.external.check_missing_data is True


class TestConfigPartialUpdates:
    """Test that partial config updates preserve defaults for unspecified fields."""

    def test_microstructure_partial_update(self):
        """Test partial update of MicrostructureConfig."""
        # Only override imbalance.depth
        data = {
            "imbalance": {"depth": 20},  # Override only depth
        }

        cfg = MicrostructureConfig.from_dict(data)

        # Overridden value
        assert cfg.imbalance.depth == 20

        # Default values preserved
        assert cfg.imbalance.threshold == 2.0
        assert cfg.enabled is True
        assert cfg.max_lookback_seconds == 3600

    def test_validation_partial_update(self):
        """Test partial update of DataValidationConfig."""
        data = {
            "mode": "strict",
            "ohlcv": {"check_negative_prices": False},
        }

        cfg = DataValidationConfig.from_dict(data)

        # Overridden values
        assert cfg.mode == "strict"
        assert cfg.ohlcv.check_negative_prices is False

        # Default values preserved
        assert cfg.ohlcv.check_zero_prices is True
        assert cfg.external.check_missing_data is True
