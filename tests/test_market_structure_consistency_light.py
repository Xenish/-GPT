"""
Tests for market structure consistency validation.

These tests ensure the structure validator catches logical inconsistencies
and prevents regressions during refactoring.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from finantradealgo.validators.structure_validator import (
    validate_market_structure,
    validate_and_raise,
    ValidationViolation,
)
from finantradealgo.market_structure.engine import MarketStructureEngine
from finantradealgo.market_structure.config import MarketStructureConfig


def create_valid_market_structure_df():
    """Create a valid market structure DataFrame for testing."""
    # Create synthetic OHLCV data with clear market structure
    dates = pd.date_range('2023-01-01', periods=50, freq='1h')

    # Create a trending pattern: down -> up -> down
    prices = []
    base = 100.0

    # Downtrend (0-15)
    for i in range(15):
        prices.append(base - i * 0.5)

    # Uptrend (15-35)
    for i in range(20):
        prices.append(prices[-1] + i * 0.3)

    # Downtrend (35-50)
    for i in range(15):
        prices.append(prices[-1] - i * 0.4)

    df = pd.DataFrame({
        'open': prices,
        'high': [p + 0.5 for p in prices],
        'low': [p - 0.5 for p in prices],
        'close': prices,
        'volume': [1000] * len(prices),
    }, index=dates)

    return df


def test_validate_valid_structure():
    """Test that validator passes on correctly computed market structure."""
    df = create_valid_market_structure_df()

    # Compute real market structure
    engine = MarketStructureEngine(cfg=MarketStructureConfig())
    result = engine.compute_df(df)

    # Validate - should have no violations
    violations = validate_market_structure(
        result.features,
        zones=result.zones,
        cfg=MarketStructureConfig()
    )

    # Filter only errors (warnings are OK)
    errors = [v for v in violations if v.severity == 'error']

    assert len(errors) == 0, \
        f"Valid market structure should have no errors, got:\n" + \
        "\n".join(str(e) for e in errors)


def test_validate_swing_overlap_violation():
    """Test that validator catches swing high and low on same bar."""
    df = pd.DataFrame({
        'ms_swing_high': [0, 1, 0, 0],  # Bar 1 is swing high
        'ms_swing_low': [0, 1, 0, 0],   # Bar 1 is also swing low - VIOLATION!
        'ms_trend_regime': [0, 0, 0, 0],
        'ms_bos_up': [0, 0, 0, 0],
        'ms_bos_down': [0, 0, 0, 0],
        'ms_choch': [0, 0, 0, 0],
        'ms_fvg_up': [0, 0, 0, 0],
        'ms_fvg_down': [0, 0, 0, 0],
        'ms_zone_demand': [0.0, 0.0, 0.0, 0.0],
        'ms_zone_supply': [0.0, 0.0, 0.0, 0.0],
    }, index=pd.date_range('2023-01-01', periods=4, freq='1h'))

    violations = validate_market_structure(df)

    # Should find swing_overlap violation
    swing_violations = [v for v in violations if v.rule == 'swing_overlap']
    assert len(swing_violations) == 1, "Should detect swing overlap"


def test_validate_fvg_overlap_violation():
    """Test that validator catches FVG up and down on same bar."""
    df = pd.DataFrame({
        'ms_swing_high': [0, 0, 0, 0],
        'ms_swing_low': [0, 0, 0, 0],
        'ms_trend_regime': [1, 1, 1, 1],
        'ms_bos_up': [0, 0, 0, 0],
        'ms_bos_down': [0, 0, 0, 0],
        'ms_choch': [0, 0, 0, 0],
        'ms_fvg_up': [0.0, 0.05, 0.0, 0.0],    # Bar 1 has FVG up (5% gap)
        'ms_fvg_down': [0.0, 0.03, 0.0, 0.0],  # Bar 1 also has FVG down - VIOLATION!
        'ms_zone_demand': [0.0, 0.0, 0.0, 0.0],
        'ms_zone_supply': [0.0, 0.0, 0.0, 0.0],
    }, index=pd.date_range('2023-01-01', periods=4, freq='1h'))

    violations = validate_market_structure(df)

    fvg_violations = [v for v in violations if v.rule == 'fvg_overlap']
    assert len(fvg_violations) == 1, "Should detect FVG overlap"


def test_validate_bos_up_trend_violation():
    """Test that validator catches BOS up without uptrend."""
    df = pd.DataFrame({
        'ms_swing_high': [0, 0, 0, 0],
        'ms_swing_low': [0, 0, 0, 0],
        'ms_trend_regime': [0, -1, 0, 0],  # Bar 1 is downtrend
        'ms_bos_up': [0, 1, 0, 0],         # But has BOS up - VIOLATION!
        'ms_bos_down': [0, 0, 0, 0],
        'ms_choch': [0, 0, 0, 0],
        'ms_fvg_up': [0, 0, 0, 0],
        'ms_fvg_down': [0, 0, 0, 0],
        'ms_zone_demand': [0.0, 0.0, 0.0, 0.0],
        'ms_zone_supply': [0.0, 0.0, 0.0, 0.0],
    }, index=pd.date_range('2023-01-01', periods=4, freq='1h'))

    violations = validate_market_structure(df)

    bos_violations = [v for v in violations if v.rule == 'bos_up_trend']
    assert len(bos_violations) == 1, "Should detect BOS up without uptrend"


def test_validate_bos_down_trend_violation():
    """Test that validator catches BOS down without downtrend."""
    df = pd.DataFrame({
        'ms_swing_high': [0, 0, 0, 0],
        'ms_swing_low': [0, 0, 0, 0],
        'ms_trend_regime': [0, 1, 0, 0],  # Bar 1 is uptrend
        'ms_bos_up': [0, 0, 0, 0],
        'ms_bos_down': [0, 1, 0, 0],      # But has BOS down - VIOLATION!
        'ms_choch': [0, 0, 0, 0],
        'ms_fvg_up': [0, 0, 0, 0],
        'ms_fvg_down': [0, 0, 0, 0],
        'ms_zone_demand': [0.0, 0.0, 0.0, 0.0],
        'ms_zone_supply': [0.0, 0.0, 0.0, 0.0],
    }, index=pd.date_range('2023-01-01', periods=4, freq='1h'))

    violations = validate_market_structure(df)

    bos_violations = [v for v in violations if v.rule == 'bos_down_trend']
    assert len(bos_violations) == 1, "Should detect BOS down without downtrend"


def test_validate_bos_overlap_violation():
    """Test that validator catches BOS up and down on same bar."""
    df = pd.DataFrame({
        'ms_swing_high': [0, 0, 0, 0],
        'ms_swing_low': [0, 0, 0, 0],
        'ms_trend_regime': [0, 0, 0, 0],
        'ms_bos_up': [0, 1, 0, 0],    # Bar 1 has BOS up
        'ms_bos_down': [0, 1, 0, 0],  # Bar 1 also has BOS down - VIOLATION!
        'ms_choch': [0, 0, 0, 0],
        'ms_fvg_up': [0, 0, 0, 0],
        'ms_fvg_down': [0, 0, 0, 0],
        'ms_zone_demand': [0.0, 0.0, 0.0, 0.0],
        'ms_zone_supply': [0.0, 0.0, 0.0, 0.0],
    }, index=pd.date_range('2023-01-01', periods=4, freq='1h'))

    violations = validate_market_structure(df)

    # Should find both bos_overlap AND bos_up_trend AND bos_down_trend violations
    bos_overlap_violations = [v for v in violations if v.rule == 'bos_overlap']
    assert len(bos_overlap_violations) == 1, "Should detect BOS overlap"


def test_validate_choch_trend_change_violation():
    """Test that validator catches ChoCh without trend change."""
    df = pd.DataFrame({
        'ms_swing_high': [0, 0, 0, 0],
        'ms_swing_low': [0, 0, 0, 0],
        'ms_trend_regime': [1, 1, 1, 1],  # Trend stays constant
        'ms_bos_up': [0, 0, 0, 0],
        'ms_bos_down': [0, 0, 0, 0],
        'ms_choch': [0, 1, 0, 0],         # But ChoCh fires - VIOLATION!
        'ms_fvg_up': [0, 0, 0, 0],
        'ms_fvg_down': [0, 0, 0, 0],
        'ms_zone_demand': [0.0, 0.0, 0.0, 0.0],
        'ms_zone_supply': [0.0, 0.0, 0.0, 0.0],
    }, index=pd.date_range('2023-01-01', periods=4, freq='1h'))

    violations = validate_market_structure(df)

    choch_violations = [v for v in violations if v.rule == 'choch_trend_change']
    assert len(choch_violations) == 1, "Should detect ChoCh without trend change"


def test_validate_invalid_trend_regime():
    """Test that validator catches invalid trend_regime values."""
    df = pd.DataFrame({
        'ms_swing_high': [0, 0, 0, 0],
        'ms_swing_low': [0, 0, 0, 0],
        'ms_trend_regime': [1, 2, -1, 0],  # 2 is invalid!
        'ms_bos_up': [0, 0, 0, 0],
        'ms_bos_down': [0, 0, 0, 0],
        'ms_choch': [0, 0, 0, 0],
        'ms_fvg_up': [0, 0, 0, 0],
        'ms_fvg_down': [0, 0, 0, 0],
        'ms_zone_demand': [0.0, 0.0, 0.0, 0.0],
        'ms_zone_supply': [0.0, 0.0, 0.0, 0.0],
    }, index=pd.date_range('2023-01-01', periods=4, freq='1h'))

    violations = validate_market_structure(df)

    trend_violations = [v for v in violations if v.rule == 'trend_regime_values']
    assert len(trend_violations) == 1, "Should detect invalid trend_regime value"


def test_validate_invalid_binary_flag():
    """Test that validator catches invalid binary flag values."""
    df = pd.DataFrame({
        'ms_swing_high': [0, 2, 0, 0],  # 2 is invalid!
        'ms_swing_low': [0, 0, 0, 0],
        'ms_trend_regime': [0, 0, 0, 0],
        'ms_bos_up': [0, 0, 0, 0],
        'ms_bos_down': [0, 0, 0, 0],
        'ms_choch': [0, 0, 0, 0],
        'ms_fvg_up': [0, 0, 0, 0],
        'ms_fvg_down': [0, 0, 0, 0],
        'ms_zone_demand': [0.0, 0.0, 0.0, 0.0],
        'ms_zone_supply': [0.0, 0.0, 0.0, 0.0],
    }, index=pd.date_range('2023-01-01', periods=4, freq='1h'))

    violations = validate_market_structure(df)

    flag_violations = [v for v in violations if v.rule == 'binary_flag_values']
    assert len(flag_violations) == 1, "Should detect invalid binary flag value"


def test_validate_negative_zone_strength():
    """Test that validator catches negative zone strengths."""
    df = pd.DataFrame({
        'ms_swing_high': [0, 0, 0, 0],
        'ms_swing_low': [0, 0, 0, 0],
        'ms_trend_regime': [0, 0, 0, 0],
        'ms_bos_up': [0, 0, 0, 0],
        'ms_bos_down': [0, 0, 0, 0],
        'ms_choch': [0, 0, 0, 0],
        'ms_fvg_up': [0, 0, 0, 0],
        'ms_fvg_down': [0, 0, 0, 0],
        'ms_zone_demand': [0.0, -1.5, 0.0, 0.0],  # Negative strength - VIOLATION!
        'ms_zone_supply': [0.0, 0.0, 0.0, 0.0],
    }, index=pd.date_range('2023-01-01', periods=4, freq='1h'))

    violations = validate_market_structure(df)

    zone_violations = [v for v in violations if v.rule == 'zone_strength_positive']
    assert len(zone_violations) == 1, "Should detect negative zone strength"


def test_validate_missing_columns():
    """Test that validator catches missing required columns."""
    # DataFrame missing ms_bos_up column
    df = pd.DataFrame({
        'ms_swing_high': [0, 0, 0, 0],
        'ms_swing_low': [0, 0, 0, 0],
        'ms_trend_regime': [0, 0, 0, 0],
        # ms_bos_up missing!
        'ms_bos_down': [0, 0, 0, 0],
        'ms_choch': [0, 0, 0, 0],
        'ms_fvg_up': [0, 0, 0, 0],
        'ms_fvg_down': [0, 0, 0, 0],
        'ms_zone_demand': [0.0, 0.0, 0.0, 0.0],
        'ms_zone_supply': [0.0, 0.0, 0.0, 0.0],
    }, index=pd.date_range('2023-01-01', periods=4, freq='1h'))

    violations = validate_market_structure(df)

    col_violations = [v for v in violations if v.rule == 'required_columns']
    assert len(col_violations) == 1, "Should detect missing columns"
    assert 'ms_bos_up' in col_violations[0].message


def test_validate_and_raise_with_errors():
    """Test that validate_and_raise raises exception on errors."""
    # Create invalid data (swing overlap)
    df = pd.DataFrame({
        'ms_swing_high': [0, 1, 0, 0],
        'ms_swing_low': [0, 1, 0, 0],  # Same bar - violation!
        'ms_trend_regime': [0, 0, 0, 0],
        'ms_bos_up': [0, 0, 0, 0],
        'ms_bos_down': [0, 0, 0, 0],
        'ms_choch': [0, 0, 0, 0],
        'ms_fvg_up': [0, 0, 0, 0],
        'ms_fvg_down': [0, 0, 0, 0],
        'ms_zone_demand': [0.0, 0.0, 0.0, 0.0],
        'ms_zone_supply': [0.0, 0.0, 0.0, 0.0],
    }, index=pd.date_range('2023-01-01', periods=4, freq='1h'))

    with pytest.raises(ValueError, match="validation error"):
        validate_and_raise(df)


def test_validate_and_raise_with_warnings_allowed():
    """Test that validate_and_raise allows warnings by default."""
    # Create data with too many swing points (warning only)
    n = 100
    df = pd.DataFrame({
        'ms_swing_high': [1] * n,  # Every bar is swing high - warning!
        'ms_swing_low': [0] * n,
        'ms_trend_regime': [0] * n,
        'ms_bos_up': [0] * n,
        'ms_bos_down': [0] * n,
        'ms_choch': [0] * n,
        'ms_fvg_up': [0] * n,
        'ms_fvg_down': [0] * n,
        'ms_zone_demand': [0.0] * n,
        'ms_zone_supply': [0.0] * n,
    }, index=pd.date_range('2023-01-01', periods=n, freq='1h'))

    # Should not raise (warnings allowed by default)
    validate_and_raise(df, allow_warnings=True)


def test_validate_on_real_fixture():
    """Test validator on a real market structure computation."""
    # Create more realistic data with clear trends
    dates = pd.date_range('2023-01-01', periods=100, freq='1h')

    # Create a wave pattern
    import math
    prices = [100 + 20 * math.sin(i * 0.1) + i * 0.05 for i in range(100)]

    df = pd.DataFrame({
        'open': prices,
        'high': [p + abs(np.random.randn() * 0.5) for p in prices],
        'low': [p - abs(np.random.randn() * 0.5) for p in prices],
        'close': [p + np.random.randn() * 0.2 for p in prices],
        'volume': [1000 + np.random.randint(-200, 200) for _ in prices],
    }, index=dates)

    # Compute market structure
    engine = MarketStructureEngine(cfg=MarketStructureConfig())
    result = engine.compute_df(df)

    # Validate - should pass with at most warnings
    violations = validate_market_structure(
        result.features,
        zones=result.zones,
        cfg=MarketStructureConfig()
    )

    errors = [v for v in violations if v.severity == 'error']

    assert len(errors) == 0, \
        f"Real fixture should have no validation errors, got:\n" + \
        "\n".join(str(e) for e in errors)

    # Print any warnings for debugging
    warnings = [v for v in violations if v.severity == 'warning']
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")
