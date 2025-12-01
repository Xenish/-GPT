"""
Tests for MicrostructureConfig and from_dict() functionality.

Task S2.1: Test config round-trip for microstructure.
"""
import pytest

from finantradealgo.microstructure.config import (
    MicrostructureConfig,
    ImbalanceConfig,
    LiquiditySweepConfig,
    ChopConfig,
    VolatilityRegimeConfig,
    BurstConfig,
    ExhaustionConfig,
    ParabolicConfig,
)


class TestConfigFromDict:
    """Test from_dict() methods for all microstructure config classes."""

    def test_imbalance_config_from_dict(self):
        """Test ImbalanceConfig.from_dict() with custom values."""
        data = {"depth": 10, "threshold": 3.0}
        cfg = ImbalanceConfig.from_dict(data)

        assert cfg.depth == 10
        assert cfg.threshold == 3.0

    def test_imbalance_config_from_dict_defaults(self):
        """Test ImbalanceConfig.from_dict() with defaults."""
        cfg = ImbalanceConfig.from_dict(None)

        assert cfg.depth == ImbalanceConfig.depth
        assert cfg.threshold == ImbalanceConfig.threshold

    def test_liquidity_sweep_config_from_dict(self):
        """Test LiquiditySweepConfig.from_dict() with custom values."""
        data = {"lookback_ms": 10000, "notional_threshold": 100000.0}
        cfg = LiquiditySweepConfig.from_dict(data)

        assert cfg.lookback_ms == 10000
        assert cfg.notional_threshold == 100000.0

    def test_chop_config_from_dict(self):
        """Test ChopConfig.from_dict() with custom values."""
        data = {"lookback_period": 20}
        cfg = ChopConfig.from_dict(data)

        assert cfg.lookback_period == 20

    def test_volatility_regime_config_from_dict(self):
        """Test VolatilityRegimeConfig.from_dict() with custom values."""
        data = {
            "period": 30,
            "z_score_window": 200,
            "low_z_threshold": -2.0,
            "high_z_threshold": 2.0,
        }
        cfg = VolatilityRegimeConfig.from_dict(data)

        assert cfg.period == 30
        assert cfg.z_score_window == 200
        assert cfg.low_z_threshold == -2.0
        assert cfg.high_z_threshold == 2.0

    def test_burst_config_from_dict(self):
        """Test BurstConfig.from_dict() with custom values."""
        data = {
            "return_window": 10,
            "z_score_window": 150,
            "z_up_threshold": 2.5,
            "z_down_threshold": 2.5,
        }
        cfg = BurstConfig.from_dict(data)

        assert cfg.return_window == 10
        assert cfg.z_score_window == 150
        assert cfg.z_up_threshold == 2.5
        assert cfg.z_down_threshold == 2.5

    def test_exhaustion_config_from_dict(self):
        """Test ExhaustionConfig.from_dict() with custom values."""
        data = {
            "min_consecutive_bars": 7,
            "volume_z_score_window": 60,
            "volume_z_threshold": -0.7,
        }
        cfg = ExhaustionConfig.from_dict(data)

        assert cfg.min_consecutive_bars == 7
        assert cfg.volume_z_score_window == 60
        assert cfg.volume_z_threshold == -0.7

    def test_parabolic_config_from_dict(self):
        """Test ParabolicConfig.from_dict() with custom values."""
        data = {"rolling_std_window": 30, "curvature_threshold": 2.0}
        cfg = ParabolicConfig.from_dict(data)

        assert cfg.rolling_std_window == 30
        assert cfg.curvature_threshold == 2.0

    def test_microstructure_config_from_dict_full(self):
        """Test MicrostructureConfig.from_dict() with full nested config."""
        data = {
            "enabled": True,
            "imbalance": {"depth": 10, "threshold": 3.0},
            "sweep": {"lookback_ms": 10000, "notional_threshold": 100000.0},
            "chop": {"lookback_period": 20},
            "vol_regime": {
                "period": 30,
                "z_score_window": 200,
                "low_z_threshold": -2.0,
                "high_z_threshold": 2.0,
            },
            "burst": {
                "return_window": 10,
                "z_score_window": 150,
                "z_up_threshold": 2.5,
                "z_down_threshold": 2.5,
            },
            "exhaustion": {
                "min_consecutive_bars": 7,
                "volume_z_score_window": 60,
                "volume_z_threshold": -0.7,
            },
            "parabolic": {"rolling_std_window": 30, "curvature_threshold": 2.0},
        }

        cfg = MicrostructureConfig.from_dict(data)

        assert cfg.enabled is True
        assert cfg.imbalance.depth == 10
        assert cfg.sweep.lookback_ms == 10000
        assert cfg.chop.lookback_period == 20
        assert cfg.vol_regime.period == 30
        assert cfg.burst.return_window == 10
        assert cfg.exhaustion.min_consecutive_bars == 7
        assert cfg.parabolic.rolling_std_window == 30

    def test_microstructure_config_from_dict_defaults(self):
        """Test MicrostructureConfig.from_dict() with all defaults."""
        cfg = MicrostructureConfig.from_dict(None)

        assert cfg.enabled is True
        assert cfg.imbalance.depth == 5
        assert cfg.sweep.lookback_ms == 5000
        assert cfg.chop.lookback_period == 14
        assert cfg.vol_regime.period == 20
        assert cfg.burst.return_window == 5
        assert cfg.exhaustion.min_consecutive_bars == 5
        assert cfg.parabolic.rolling_std_window == 20

    def test_microstructure_config_from_dict_partial(self):
        """Test MicrostructureConfig.from_dict() with partial config."""
        data = {
            "enabled": False,
            "imbalance": {"depth": 7},  # Only override depth
            "chop": {"lookback_period": 21},  # Only override lookback
        }

        cfg = MicrostructureConfig.from_dict(data)

        assert cfg.enabled is False
        assert cfg.imbalance.depth == 7
        assert cfg.imbalance.threshold == 2.0  # Should use default
        assert cfg.chop.lookback_period == 21
        assert cfg.sweep.lookback_ms == 5000  # Should use default


class TestConfigYAMLCompatibility:
    """Test that config works with YAML-like dictionaries."""

    def test_yaml_like_config(self):
        """Test config from dict that looks like YAML."""
        yaml_dict = {
            "enabled": True,
            "imbalance": {
                "depth": 5,
                "threshold": 2.0,
            },
            "sweep": {
                "lookback_ms": 5000,
                "notional_threshold": 50000.0,
            },
            "chop": {
                "lookback_period": 14,
            },
            "vol_regime": {
                "period": 20,
                "z_score_window": 100,
                "low_z_threshold": -1.5,
                "high_z_threshold": 1.5,
            },
            "burst": {
                "return_window": 5,
                "z_score_window": 100,
                "z_up_threshold": 2.0,
                "z_down_threshold": 2.0,
            },
            "exhaustion": {
                "min_consecutive_bars": 5,
                "volume_z_score_window": 50,
                "volume_z_threshold": -0.5,
            },
            "parabolic": {
                "rolling_std_window": 20,
                "curvature_threshold": 1.5,
            },
        }

        cfg = MicrostructureConfig.from_dict(yaml_dict)

        assert cfg.enabled is True
        assert cfg.imbalance.depth == 5
        assert cfg.imbalance.threshold == 2.0
        assert cfg.sweep.lookback_ms == 5000
        assert cfg.sweep.notional_threshold == 50000.0
        assert cfg.chop.lookback_period == 14
        assert cfg.vol_regime.period == 20
        assert cfg.vol_regime.z_score_window == 100
        assert cfg.vol_regime.low_z_threshold == -1.5
        assert cfg.vol_regime.high_z_threshold == 1.5
        assert cfg.burst.return_window == 5
        assert cfg.burst.z_score_window == 100
        assert cfg.burst.z_up_threshold == 2.0
        assert cfg.burst.z_down_threshold == 2.0
        assert cfg.exhaustion.min_consecutive_bars == 5
        assert cfg.exhaustion.volume_z_score_window == 50
        assert cfg.exhaustion.volume_z_threshold == -0.5
        assert cfg.parabolic.rolling_std_window == 20
        assert cfg.parabolic.curvature_threshold == 1.5
