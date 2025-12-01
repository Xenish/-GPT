"""
Tests for strategy feature dependency checking.

Task S1.E4: Test coverage for dependency_check.py module.
"""
import pandas as pd
import pytest

from finantradealgo.features.dependency_check import (
    check_market_structure_dependencies,
    check_microstructure_dependencies,
    validate_strategy_dependencies,
    get_missing_market_structure_features,
    has_market_structure_features,
)
from finantradealgo.market_structure.types import MarketStructureColumns


class TestMarketStructureDependencies:
    """Test cases for market structure dependency checking."""

    def test_check_with_all_features_present(self):
        """Should pass when all market structure features are present."""
        cols = MarketStructureColumns()
        df = pd.DataFrame({
            "close": [100, 101, 102],
            cols.swing_high: [0, 1, 0],
            cols.swing_low: [1, 0, 0],
            cols.trend_regime: [1, 1, 1],
            cols.fvg_up: [0, 0, 1],
            cols.fvg_down: [0, 0, 0],
        })

        result = check_market_structure_dependencies(
            df, "test_strategy", requires_market_structure=True, strict=False
        )

        assert result is True

    def test_check_with_missing_features_non_strict(self):
        """Should return False but not raise when features missing in non-strict mode."""
        df = pd.DataFrame({"close": [100, 101, 102]})

        result = check_market_structure_dependencies(
            df, "test_strategy", requires_market_structure=True, strict=False
        )

        assert result is False

    def test_check_with_missing_features_strict(self):
        """Should raise ValueError when features missing in strict mode."""
        df = pd.DataFrame({"close": [100, 101, 102]})

        with pytest.raises(ValueError, match="requires market structure features"):
            check_market_structure_dependencies(
                df, "test_strategy", requires_market_structure=True, strict=True
            )

    def test_check_when_not_required(self):
        """Should pass when market structure not required."""
        df = pd.DataFrame({"close": [100, 101, 102]})

        result = check_market_structure_dependencies(
            df, "test_strategy", requires_market_structure=False, strict=True
        )

        assert result is True


class TestMicrostructureDependencies:
    """Test cases for microstructure dependency checking."""

    def test_check_with_features_present(self):
        """Should pass when microstructure features are present."""
        df = pd.DataFrame({
            "close": [100, 101, 102],
            "imb_depth_5": [0.5, 0.6, 0.4],
            "sweep_notional": [1000, 1100, 900],
        })

        result = check_microstructure_dependencies(
            df, "test_strategy", requires_microstructure=True, strict=False
        )

        assert result is True

    def test_check_with_missing_features_non_strict(self):
        """Should return False but not raise when features missing."""
        df = pd.DataFrame({"close": [100, 101, 102]})

        result = check_microstructure_dependencies(
            df, "test_strategy", requires_microstructure=True, strict=False
        )

        assert result is False

    def test_check_with_missing_features_strict(self):
        """Should raise ValueError when features missing in strict mode."""
        df = pd.DataFrame({"close": [100, 101, 102]})

        with pytest.raises(ValueError, match="requires microstructure features"):
            check_microstructure_dependencies(
                df, "test_strategy", requires_microstructure=True, strict=True
            )

    def test_check_when_not_required(self):
        """Should pass when microstructure not required."""
        df = pd.DataFrame({"close": [100, 101, 102]})

        result = check_microstructure_dependencies(
            df, "test_strategy", requires_microstructure=False, strict=True
        )

        assert result is True


class TestValidateStrategyDependencies:
    """Test cases for combined strategy dependency validation."""

    def test_validate_all_dependencies_present(self):
        """Should pass when all dependencies are met."""
        cols = MarketStructureColumns()
        df = pd.DataFrame({
            "close": [100, 101, 102],
            cols.swing_high: [0, 1, 0],
            cols.swing_low: [1, 0, 0],
            cols.trend_regime: [1, 1, 1],
            cols.fvg_up: [0, 0, 1],
            cols.fvg_down: [0, 0, 0],
            "imb_depth_5": [0.5, 0.6, 0.4],
        })

        result = validate_strategy_dependencies(
            df,
            "test_strategy",
            uses_market_structure=True,
            uses_microstructure=True,
            strict=False,
        )

        assert result is True

    def test_validate_missing_market_structure(self):
        """Should return False when market structure is missing."""
        df = pd.DataFrame({
            "close": [100, 101, 102],
            "imb_depth_5": [0.5, 0.6, 0.4],
        })

        result = validate_strategy_dependencies(
            df,
            "test_strategy",
            uses_market_structure=True,
            uses_microstructure=True,
            strict=False,
        )

        assert result is False

    def test_validate_no_requirements(self):
        """Should pass when strategy has no special requirements."""
        df = pd.DataFrame({"close": [100, 101, 102]})

        result = validate_strategy_dependencies(
            df,
            "simple_strategy",
            uses_market_structure=False,
            uses_microstructure=False,
            strict=True,
        )

        assert result is True


class TestMarketStructureHelpers:
    """Test cases for market structure helper functions."""

    def test_get_missing_features_all_present(self):
        """Should return empty list when all features present."""
        cols = MarketStructureColumns()
        df = pd.DataFrame({
            cols.price_smooth: [100.0, 101.0],
            cols.swing_high: [0, 1],
            cols.swing_low: [1, 0],
            cols.trend_regime: [1, 1],
            cols.chop_regime: [0.3, 0.4],
            cols.fvg_up: [0, 1],
            cols.fvg_down: [0, 0],
            cols.zone_demand: [0.0, 0.0],
            cols.zone_supply: [0.0, 0.0],
            cols.bos_up: [0, 0],
            cols.bos_down: [0, 0],
            cols.choch: [0, 0],
        })

        missing = get_missing_market_structure_features(df)

        assert missing == []

    def test_get_missing_features_some_missing(self):
        """Should return list of missing features."""
        cols = MarketStructureColumns()
        df = pd.DataFrame({
            cols.swing_high: [0, 1],
            cols.swing_low: [1, 0],
        })

        missing = get_missing_market_structure_features(df)

        assert len(missing) > 0
        assert cols.trend_regime in missing
        assert cols.fvg_up in missing

    def test_has_market_structure_minimal(self):
        """Should check for minimal market structure features."""
        cols = MarketStructureColumns()
        df = pd.DataFrame({
            cols.swing_high: [0, 1],
            cols.swing_low: [1, 0],
            cols.trend_regime: [1, 1],
        })

        assert has_market_structure_features(df, minimal=True) is True

    def test_has_market_structure_minimal_missing(self):
        """Should return False when minimal features are missing."""
        cols = MarketStructureColumns()
        df = pd.DataFrame({
            cols.swing_high: [0, 1],
            # Missing swing_low and trend_regime
        })

        assert has_market_structure_features(df, minimal=True) is False

    def test_has_market_structure_all(self):
        """Should check for all market structure features."""
        cols = MarketStructureColumns()
        df_partial = pd.DataFrame({
            cols.swing_high: [0, 1],
            cols.swing_low: [1, 0],
            cols.trend_regime: [1, 1],
        })

        assert has_market_structure_features(df_partial, minimal=False) is False
