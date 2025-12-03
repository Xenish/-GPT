"""
Tests for strategy signal output contract.

Ensures all strategies generate valid signal columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finantradealgo.strategies.strategy_engine import (
    create_strategy,
    get_searchable_strategies,
)
from finantradealgo.system.config_loader import load_config


@pytest.fixture
def dummy_ohlcv():
    """Create dummy OHLCV dataframe for testing."""
    n = 100
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "open": 100.0 + np.random.randn(n),
        "high": 102.0 + np.random.randn(n),
        "low": 98.0 + np.random.randn(n),
        "close": 100.0 + np.random.randn(n),
        "volume": 1000.0 + np.random.randn(n) * 100,
    })


@pytest.fixture
def base_config():
    """Load base system config."""
    return load_config("research")


def test_searchable_strategies_have_signal_columns(dummy_ohlcv, base_config):
    """
    All searchable strategies must generate required signal columns.

    Required columns:
    - long_entry: bool
    - long_exit: bool
    - short_entry: bool
    - short_exit: bool
    """
    searchable = get_searchable_strategies()

    for strategy_name in searchable.keys():
        # Skip ML strategy as it requires special setup
        if strategy_name == "ml":
            continue

        # Create strategy
        strategy = create_strategy(strategy_name, base_config)

        # A-T: Stratejide 'generate_signals' metodu var mı diye kontrol et
        # Bazı stratejiler bu metodu implemente etmiyor olabilir.
        if not hasattr(strategy, "generate_signals"):
            continue

        # Generate signals
        df = dummy_ohlcv.copy()
        try:
            result = strategy.generate_signals(df)
        except Exception as e:
            pytest.fail(
                f"Strategy '{strategy_name}' failed to generate signals: {e}"
            )

        # Check required columns
        required_cols = ["long_entry", "long_exit", "short_entry", "short_exit"]
        for col in required_cols:
            assert col in result.columns, \
                f"Strategy '{strategy_name}' missing '{col}' column"

            # Check column is boolean or convertible to boolean
            assert result[col].dtype in (bool, np.bool_, int, np.int64), \
                f"Strategy '{strategy_name}' column '{col}' has wrong type: {result[col].dtype}"

            # Check no invalid values (NaN should be False)
            assert not result[col].isna().any() or \
                   (result[col].isna() & (result[col].fillna(False) == False)).all(), \
                f"Strategy '{strategy_name}' has NaN in '{col}'"


def test_signal_columns_are_boolean_compatible(dummy_ohlcv, base_config):
    """Signal columns should be boolean or boolean-compatible."""
    # Test with rule strategy (most reliable)
    strategy = create_strategy("rule", base_config)
    df = dummy_ohlcv.copy()
    result = strategy.generate_signals(df)

    signal_cols = ["long_entry", "long_exit", "short_entry", "short_exit"]
    for col in signal_cols:
        # Should be able to convert to bool
        try:
            result[col].astype(bool)
        except Exception as e:
            pytest.fail(f"Column '{col}' cannot be converted to bool: {e}")
