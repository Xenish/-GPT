"""
Tests for ML targets configuration.

Verifies that:
1. get_ml_targets() returns correct targets from config
2. Fallback to first symbol/timeframe works when targets not specified
3. is_ml_enabled() correctly identifies ML status
4. Invalid target configurations are handled gracefully
"""

import pytest

from finantradealgo.ml.ml_utils import get_ml_targets, is_ml_enabled
from finantradealgo.system.config_loader import DataConfig


class TestMLTargetsConfig:
    """Test ML targets configuration parsing and fallback logic."""

    def test_get_ml_targets_with_explicit_targets(self):
        """Test that explicit ml.targets are returned correctly."""
        cfg = {
            "ml": {
                "enabled": True,
                "targets": [
                    {"symbol": "BTCUSDT", "timeframe": "15m"},
                    {"symbol": "AIAUSDT", "timeframe": "15m"},
                    {"symbol": "BTCUSDT", "timeframe": "1h"},
                ],
            },
        }

        targets = get_ml_targets(cfg)

        assert len(targets) == 3, "Should return all 3 targets"
        assert ("BTCUSDT", "15m") in targets
        assert ("AIAUSDT", "15m") in targets
        assert ("BTCUSDT", "1h") in targets

    def test_get_ml_targets_falls_back_to_first_symbol_timeframe(self):
        """Test fallback when ml.targets is empty or not specified."""
        # Create DataConfig
        data_cfg = DataConfig(
            symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            timeframes=["1m", "5m", "15m", "1h"],
        )

        cfg = {
            "ml": {
                "enabled": True,
                "targets": [],  # Empty targets
            },
            "data_cfg": data_cfg,
        }

        targets = get_ml_targets(cfg)

        assert len(targets) == 1, "Should fallback to single target"
        assert targets[0] == ("BTCUSDT", "1m"), \
            "Should use first symbol and first timeframe"

    def test_get_ml_targets_falls_back_when_ml_section_missing(self):
        """Test fallback when ml section doesn't exist."""
        data_cfg = DataConfig(
            symbols=["AIAUSDT"],
            timeframes=["15m"],
        )

        cfg = {
            "data_cfg": data_cfg,
        }

        targets = get_ml_targets(cfg)

        assert len(targets) == 1
        assert targets[0] == ("AIAUSDT", "15m")

    def test_get_ml_targets_from_data_section_when_data_cfg_missing(self):
        """Test that data section is used when data_cfg not available."""
        cfg = {
            "ml": {"targets": []},
            "data": {
                "symbols": ["ETHUSDT", "BTCUSDT"],
                "timeframes": ["5m", "15m"],
            },
        }

        targets = get_ml_targets(cfg)

        assert len(targets) == 1
        assert targets[0] == ("ETHUSDT", "5m"), \
            "Should use first from data.symbols and data.timeframes"

    def test_get_ml_targets_legacy_fallback(self):
        """Test fallback to legacy single symbol/timeframe config."""
        cfg = {
            "ml": {"targets": []},
            "symbol": "XRPUSDT",
            "timeframe": "1h",
        }

        targets = get_ml_targets(cfg)

        assert len(targets) == 1
        assert targets[0] == ("XRPUSDT", "1h"), \
            "Should fallback to legacy symbol/timeframe fields"

    def test_get_ml_targets_skips_invalid_entries(self):
        """Test that invalid target entries are skipped with warning."""
        cfg = {
            "ml": {
                "targets": [
                    {"symbol": "BTCUSDT", "timeframe": "15m"},  # Valid
                    {"symbol": "ETHUSDT"},  # Invalid: missing timeframe
                    {"timeframe": "1h"},  # Invalid: missing symbol
                    {"symbol": "AIAUSDT", "timeframe": "5m"},  # Valid
                ],
            },
        }

        targets = get_ml_targets(cfg)

        assert len(targets) == 2, "Should skip 2 invalid entries"
        assert ("BTCUSDT", "15m") in targets
        assert ("AIAUSDT", "5m") in targets
        assert ("ETHUSDT", None) not in targets
        assert (None, "1h") not in targets

    def test_is_ml_enabled_returns_true_when_enabled(self):
        """Test that is_ml_enabled returns True when ml.enabled=true."""
        cfg = {
            "ml": {
                "enabled": True,
            },
        }

        assert is_ml_enabled(cfg) is True

    def test_is_ml_enabled_returns_false_when_disabled(self):
        """Test that is_ml_enabled returns False when ml.enabled=false."""
        cfg = {
            "ml": {
                "enabled": False,
            },
        }

        assert is_ml_enabled(cfg) is False

    def test_is_ml_enabled_defaults_to_true(self):
        """Test that is_ml_enabled defaults to True for backward compatibility."""
        cfg = {
            "ml": {},  # No 'enabled' key
        }

        assert is_ml_enabled(cfg) is True, \
            "Should default to enabled for backward compatibility"

    def test_is_ml_enabled_when_ml_section_missing(self):
        """Test that is_ml_enabled defaults to True when ml section missing."""
        cfg = {}

        assert is_ml_enabled(cfg) is True

    def test_targets_with_multiple_timeframes_for_same_symbol(self):
        """Test that same symbol can have multiple timeframes."""
        cfg = {
            "ml": {
                "targets": [
                    {"symbol": "BTCUSDT", "timeframe": "15m"},
                    {"symbol": "BTCUSDT", "timeframe": "1h"},
                    {"symbol": "BTCUSDT", "timeframe": "4h"},
                ],
            },
        }

        targets = get_ml_targets(cfg)

        assert len(targets) == 3
        # All targets are for BTCUSDT but different timeframes
        symbols = [s for s, _ in targets]
        assert all(s == "BTCUSDT" for s in symbols)

        timeframes = [tf for _, tf in targets]
        assert set(timeframes) == {"15m", "1h", "4h"}

    def test_targets_preserves_order(self):
        """Test that target order is preserved from config."""
        cfg = {
            "ml": {
                "targets": [
                    {"symbol": "ETHUSDT", "timeframe": "1h"},
                    {"symbol": "BTCUSDT", "timeframe": "15m"},
                    {"symbol": "AIAUSDT", "timeframe": "5m"},
                ],
            },
        }

        targets = get_ml_targets(cfg)

        # Order should match config order
        assert targets[0] == ("ETHUSDT", "1h")
        assert targets[1] == ("BTCUSDT", "15m")
        assert targets[2] == ("AIAUSDT", "5m")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
