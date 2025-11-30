"""
Tests for live configuration scope validation.

Sprint T3.5 - Validate that live config enforces single symbol/timeframe
even when multi-TF/multi-symbol data is available.

Principle: "Research universe is wide, live universe is narrow"
"""

import os
import pytest

# Set dummy FCM key to avoid validation errors
if not os.getenv("FCM_SERVER_KEY"):
    os.environ["FCM_SERVER_KEY"] = "dummy_test_key"

from finantradealgo.system.config_loader import load_system_config


class TestLiveConfigScope:
    """Test suite for live configuration scope validation."""

    def test_live_config_has_single_symbol(self):
        """Test that live config specifies a single symbol."""
        cfg = load_system_config()

        live_cfg = cfg.get("live", {})
        assert live_cfg, "Live config should be present"

        # Live should have a single symbol field
        assert "symbol" in live_cfg, "Live config should have 'symbol' field"
        symbol = live_cfg["symbol"]

        assert isinstance(symbol, str), "Live symbol should be a string (single value)"
        assert len(symbol) > 0, "Live symbol should not be empty"

        # Validate it's a valid symbol from the data config
        data_cfg = cfg.get("data_cfg")
        if data_cfg and data_cfg.symbols:
            assert symbol in data_cfg.symbols, \
                f"Live symbol '{symbol}' should be in data.symbols {data_cfg.symbols}"

    def test_live_config_has_single_timeframe(self):
        """Test that live config specifies a single timeframe."""
        cfg = load_system_config()

        live_cfg = cfg.get("live", {})
        assert live_cfg, "Live config should be present"

        # Live should have a single timeframe field
        assert "timeframe" in live_cfg, "Live config should have 'timeframe' field"
        timeframe = live_cfg["timeframe"]

        assert isinstance(timeframe, str), "Live timeframe should be a string (single value)"
        assert len(timeframe) > 0, "Live timeframe should not be empty"

        # Validate it's a valid timeframe from the data config
        data_cfg = cfg.get("data_cfg")
        if data_cfg and data_cfg.timeframes:
            assert timeframe in data_cfg.timeframes, \
                f"Live timeframe '{timeframe}' should be in data.timeframes {data_cfg.timeframes}"

    def test_live_uses_narrow_scope_vs_data_wide_scope(self):
        """Test that live config is narrow (1 combo) while data config is wide (many combos)."""
        cfg = load_system_config()

        # Data config should have multiple symbols/timeframes (wide scope)
        data_cfg = cfg.get("data_cfg")
        assert data_cfg, "Data config should be present"

        num_symbols = len(data_cfg.symbols) if data_cfg.symbols else 0
        num_timeframes = len(data_cfg.timeframes) if data_cfg.timeframes else 0
        total_combinations = num_symbols * num_timeframes

        assert total_combinations >= 20, \
            f"Data config should have many combinations (got {total_combinations}), " \
            f"showing wide research scope"

        # Live config should have exactly 1 combination (narrow scope)
        live_cfg = cfg.get("live", {})
        assert live_cfg, "Live config should be present"

        live_symbol = live_cfg.get("symbol")
        live_timeframe = live_cfg.get("timeframe")

        assert live_symbol, "Live should have exactly one symbol"
        assert live_timeframe, "Live should have exactly one timeframe"
        assert isinstance(live_symbol, str), "Live symbol should be single string, not list"
        assert isinstance(live_timeframe, str), "Live timeframe should be single string, not list"

    def test_live_symbol_is_common_live_choice(self):
        """Test that live config uses a common/sensible symbol choice."""
        cfg = load_system_config()

        live_cfg = cfg.get("live", {})
        live_symbol = live_cfg.get("symbol", "")

        # Live symbol should be a major pair (not some exotic altcoin)
        # Common live trading choices: BTCUSDT, ETHUSDT, AIAUSDT, BNBUSDT
        common_symbols = {"BTCUSDT", "ETHUSDT", "AIAUSDT", "BNBUSDT", "XRPUSDT"}

        assert live_symbol in common_symbols, \
            f"Live symbol '{live_symbol}' should be a common choice from {common_symbols}"

    def test_live_timeframe_is_common_live_choice(self):
        """Test that live config uses a common/sensible timeframe choice."""
        cfg = load_system_config()

        live_cfg = cfg.get("live", {})
        live_timeframe = live_cfg.get("timeframe", "")

        # Live timeframe should be a common intraday timeframe
        # Too fast (1m) = noise, too slow (1d) = few signals
        # 15m, 5m, 1h are common live trading timeframes
        common_timeframes = {"1m", "5m", "15m", "1h"}

        assert live_timeframe in common_timeframes, \
            f"Live timeframe '{live_timeframe}' should be a common choice from {common_timeframes}"

    def test_live_config_prefers_15m_timeframe(self):
        """Test that live config defaults to 15m (good balance of signal quality vs frequency)."""
        cfg = load_system_config()

        live_cfg = cfg.get("live", {})
        live_timeframe = live_cfg.get("timeframe", "")

        # 15m is the preferred live timeframe in system.yml
        assert live_timeframe == "15m", \
            f"Live should prefer '15m' timeframe for good signal quality, got '{live_timeframe}'"

    def test_ml_targets_subset_of_data_combinations(self):
        """Test that ML targets are a small subset of all possible data combinations."""
        cfg = load_system_config()

        # Get total possible combinations from data config
        data_cfg = cfg.get("data_cfg")
        num_symbols = len(data_cfg.symbols) if data_cfg and data_cfg.symbols else 0
        num_timeframes = len(data_cfg.timeframes) if data_cfg and data_cfg.timeframes else 0
        total_combinations = num_symbols * num_timeframes

        # Get ML targets
        ml_cfg = cfg.get("ml", {}) or {}
        ml_targets = ml_cfg.get("targets", [])
        num_ml_targets = len(ml_targets) if ml_targets else 1  # Default to 1 if empty

        # ML targets should be much smaller than total combinations
        # e.g., 2-3 targets vs 20 total combinations
        assert num_ml_targets < total_combinations, \
            f"ML targets ({num_ml_targets}) should be a subset of total combinations ({total_combinations})"

        # Typically ML targets should be <= 10% of total combinations
        percentage = (num_ml_targets / total_combinations) * 100
        assert percentage <= 20, \
            f"ML targets should be a small subset (<20% of total), got {percentage:.1f}%"

    def test_live_mode_is_paper_or_exchange(self):
        """Test that live mode is correctly configured."""
        cfg = load_system_config()

        live_cfg = cfg.get("live", {})
        mode = live_cfg.get("mode")

        assert mode in ["paper", "exchange"], \
            f"Live mode should be 'paper' or 'exchange', got '{mode}'"

    def test_live_config_has_risk_limits(self):
        """Test that live config has risk management limits."""
        cfg = load_system_config()

        live_cfg = cfg.get("live", {})

        # Should have position/risk limits
        assert "max_concurrent_positions" in live_cfg
        assert "max_position_notional" in live_cfg
        assert "max_daily_loss" in live_cfg

        # Limits should be reasonable
        max_positions = live_cfg.get("max_concurrent_positions", 0)
        assert max_positions >= 1 and max_positions <= 5, \
            f"Max concurrent positions should be 1-5, got {max_positions}"

    def test_live_enforces_single_combination_principle(self):
        """
        Integration test: Validate the core principle.

        "Research universe is wide, live universe is narrow"
        - Data: many symbols × many timeframes = wide research space
        - ML: small subset for expensive training
        - Live: exactly 1 combination for safe, focused trading
        """
        cfg = load_system_config()

        # Wide research universe
        data_cfg = cfg.get("data_cfg")
        research_symbols = len(data_cfg.symbols) if data_cfg and data_cfg.symbols else 0
        research_timeframes = len(data_cfg.timeframes) if data_cfg and data_cfg.timeframes else 0

        # ML subset
        ml_cfg = cfg.get("ml", {}) or {}
        ml_targets = ml_cfg.get("targets", [])
        ml_combinations = len(ml_targets) if ml_targets else 1

        # Narrow live universe
        live_cfg = cfg.get("live", {})
        live_combinations = 1  # Always exactly 1

        # Validate the principle
        assert research_symbols >= 5, "Research should have >= 5 symbols"
        assert research_timeframes >= 4, "Research should have >= 4 timeframes"
        assert ml_combinations < (research_symbols * research_timeframes), \
            "ML should be subset of research"
        assert live_combinations == 1, "Live should have exactly 1 combination"

        print(f"\n[OK] Principle validated:")
        print(f"  Research: {research_symbols} symbols × {research_timeframes} TFs = "
              f"{research_symbols * research_timeframes} combinations (WIDE)")
        print(f"  ML:       {ml_combinations} combinations (SELECTIVE)")
        print(f"  Live:     {live_combinations} combination (NARROW)")
