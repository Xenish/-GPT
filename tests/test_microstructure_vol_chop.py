import numpy as np
import pandas as pd
import pytest

from finantradealgo.microstructure.chop_detector import compute_chop
from finantradealgo.microstructure.config import ChopConfig, VolatilityRegimeConfig
from finantradealgo.microstructure.volatility_regime import compute_volatility_regime


@pytest.fixture
def base_config_vol():
    return VolatilityRegimeConfig(period=10, z_score_window=20)


@pytest.fixture
def base_config_chop():
    return ChopConfig(lookback_period=14)


def test_vol_regime_constant_vol(base_config_vol):
    """Test that volatility regime is 0 for a series with constant volatility."""
    np.random.seed(42)
    close = pd.Series(100 + np.random.randn(100).cumsum() * 0.1)
    regime = compute_volatility_regime(close, base_config_vol)

    # In a constant vol series, most of the regimes should be normal (0)
    # Allowing for some noise at the edges of the rolling windows
    assert regime.value_counts().get(0, 0) > 80


def test_vol_regime_high_vol_spike(base_config_vol):
    """Test that volatility regime detects a spike in volatility."""
    np.random.seed(42)
    # Stable series
    stable_returns = np.random.randn(50) * 0.1
    # High volatility series
    volatile_returns = np.random.randn(50) * 0.5
    
    returns = np.concatenate([stable_returns, volatile_returns])
    close = pd.Series(100 + returns.cumsum())
    
    regime = compute_volatility_regime(close, base_config_vol)

    # There should be a significant number of high-volatility regimes (1)
    # in the second half of the series.
    assert regime.iloc[60:].value_counts().get(1, 0) > 10
    assert regime.iloc[:40].value_counts().get(1, 0) < 5


def test_chop_strong_trend(base_config_chop):
    """Test that chop score is low for a strong trend."""
    close = pd.Series(np.arange(100, 200, 1))  # Monotonically increasing
    chop = compute_chop(close, base_config_chop)

    # For a pure trend, chop score should be very close to 0
    # We check the average value, ignoring initial NaNs
    assert chop.iloc[base_config_chop.lookback_period :].mean() < 0.1


def test_chop_choppy_market(base_config_chop):
    """Test that chop score is high for a sideways/choppy market."""
    np.random.seed(42)
    # Random walk within a range
    returns = np.random.choice([-0.5, 0.5], 100)
    close = pd.Series(100 + returns.cumsum())
    
    # Ensure it stays within a range
    close = close.clip(lower=95, upper=105)

    chop = compute_chop(close, base_config_chop)
    
    # For a choppy series, chop score should be high (closer to 1)
    assert chop.iloc[base_config_chop.lookback_period :].mean() > 0.7
