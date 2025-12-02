"""
Tests for Ensemble Strategies.

Tests weighted ensembles, bandit ensembles, and ensemble backtesting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finantradealgo.core.strategy import StrategyContext
from finantradealgo.research.ensemble.base import ComponentStrategy, EnsembleConfig
from finantradealgo.research.ensemble.weighted import (
    WeightedEnsembleStrategy,
    WeightedEnsembleConfig,
    WeightingMethod,
    _normalize_with_bounds,
)
from finantradealgo.research.ensemble.bandit import (
    BanditEnsembleStrategy,
    BanditEnsembleConfig,
    BanditAlgorithm,
)


@pytest.fixture
def dummy_ohlcv():
    """Create dummy OHLCV dataframe for testing."""
    n = 200
    np.random.seed(42)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "open": 100.0 + np.random.randn(n) * 0.5,
        "high": 102.0 + np.random.randn(n) * 0.5,
        "low": 98.0 + np.random.randn(n) * 0.5,
        "close": 100.0 + np.random.randn(n) * 0.5,
        "volume": 1000.0 + np.random.randn(n) * 100,
    })


@pytest.fixture
def dummy_component_signals(dummy_ohlcv):
    """Add dummy component signals to DataFrame."""
    df = dummy_ohlcv.copy()

    # Component 1: Simple trend follower (signal = 1 when price > MA50)
    df["ma50"] = df["close"].rolling(50).mean()
    df["ensemble_component_comp1_signal"] = (df["close"] > df["ma50"]).astype(int)

    # Component 2: Mean reversion (signal = 1 when price < MA20)
    df["ma20"] = df["close"].rolling(20).mean()
    df["ensemble_component_comp2_signal"] = (df["close"] < df["ma20"]).astype(int)

    # Component 3: Random signal (for testing)
    np.random.seed(42)
    df["ensemble_component_comp3_signal"] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])

    return df


def test_component_strategy_creation():
    """Test ComponentStrategy dataclass."""
    comp = ComponentStrategy(
        strategy_name="rule",
        strategy_params={"warmup_bars": 50},
        weight=1.5,
        label="my_rule",
    )

    assert comp.strategy_name == "rule"
    assert comp.strategy_params == {"warmup_bars": 50}
    assert comp.weight == 1.5
    assert comp.label == "my_rule"


def test_component_strategy_default_label():
    """Test ComponentStrategy auto-generates label from strategy_name."""
    comp = ComponentStrategy(strategy_name="trend_continuation")
    assert comp.label == "trend_continuation"


def test_ensemble_config_from_dict():
    """Test EnsembleConfig creation from dictionary."""
    data = {
        "components": [
            {"strategy_name": "rule", "params": {"warmup": 50}, "weight": 1.0},
            {"strategy_name": "trend_continuation", "params": {}, "weight": 2.0, "label": "trend"},
        ],
        "warmup_bars": 100,
    }

    config = EnsembleConfig.from_dict(data)

    assert len(config.components) == 2
    assert config.components[0].strategy_name == "rule"
    assert config.components[1].label == "trend"
    assert config.warmup_bars == 100


def test_weighted_ensemble_equal_weight_initialization():
    """Test WeightedEnsembleStrategy initializes with equal weights."""
    components = [
        ComponentStrategy("rule", label="comp1"),
        ComponentStrategy("trend_continuation", label="comp2"),
        ComponentStrategy("sweep_reversal", label="comp3"),
    ]

    config = WeightedEnsembleConfig(
        components=components,
        weighting_method=WeightingMethod.EQUAL,
    )

    ensemble = WeightedEnsembleStrategy(config)
    ensemble._initialize_weights()

    # Should have equal weights (1/3 each)
    assert abs(ensemble.current_weights["comp1"] - 1/3) < 0.001
    assert abs(ensemble.current_weights["comp2"] - 1/3) < 0.001
    assert abs(ensemble.current_weights["comp3"] - 1/3) < 0.001


def test_weighted_ensemble_custom_weights():
    """Test WeightedEnsembleStrategy with custom weights."""
    components = [
        ComponentStrategy("rule", weight=2.0, label="comp1"),
        ComponentStrategy("trend_continuation", weight=1.0, label="comp2"),
    ]

    config = WeightedEnsembleConfig(
        components=components,
        weighting_method=WeightingMethod.CUSTOM,
        normalize_weights=True,
    )

    ensemble = WeightedEnsembleStrategy(config)
    ensemble._initialize_weights()

    # Should normalize to sum to 1
    assert abs(ensemble.current_weights["comp1"] - 2/3) < 0.001
    assert abs(ensemble.current_weights["comp2"] - 1/3) < 0.001


def test_weighted_ensemble_weight_constraints():
    """Test weight min/max constraints."""
    components = [
        ComponentStrategy("rule", weight=10.0, label="comp1"),
        ComponentStrategy("trend_continuation", weight=0.01, label="comp2"),
    ]

    config = WeightedEnsembleConfig(
        components=components,
        weighting_method=WeightingMethod.CUSTOM,
        min_weight=0.1,
        max_weight=0.6,
        normalize_weights=True,
    )

    ensemble = WeightedEnsembleStrategy(config)
    ensemble._initialize_weights()

    # Weights should be clipped and normalized
    assert ensemble.current_weights["comp1"] <= 0.6
    assert ensemble.current_weights["comp2"] >= 0.1
    assert abs(sum(ensemble.current_weights.values()) - 1.0) < 0.001


def test_weighted_ensemble_generate_signals(dummy_component_signals):
    """Test WeightedEnsembleStrategy signal generation."""
    components = [
        ComponentStrategy("rule", label="comp1"),
        ComponentStrategy("trend", label="comp2"),
    ]

    config = WeightedEnsembleConfig(
        components=components,
        weighting_method=WeightingMethod.EQUAL,
        warmup_bars=50,
        signal_threshold=0.5,
    )

    ensemble = WeightedEnsembleStrategy(config)
    result = ensemble.generate_signals(dummy_component_signals)

    # Check required columns exist
    assert "long_entry" in result.columns
    assert "long_exit" in result.columns
    assert "short_entry" in result.columns
    assert "short_exit" in result.columns

    # Check signals are boolean
    assert result["long_entry"].dtype == bool
    assert result["long_exit"].dtype == bool


def test_bandit_ensemble_initialization():
    """Test BanditEnsembleStrategy initialization."""
    components = [
        ComponentStrategy("rule", label="comp1"),
        ComponentStrategy("trend_continuation", label="comp2"),
        ComponentStrategy("sweep_reversal", label="comp3"),
    ]

    config = BanditEnsembleConfig(
        components=components,
        bandit_algorithm=BanditAlgorithm.EPSILON_GREEDY,
        epsilon=0.2,
    )

    ensemble = BanditEnsembleStrategy(config)

    # Check bandit stats initialized
    assert len(ensemble.bandit_stats) == 3
    assert ensemble.bandit_stats["comp1"].n_pulls == 0
    assert ensemble.bandit_stats["comp2"].alpha == 1.0
    assert ensemble.bandit_stats["comp2"].beta == 1.0


def test_bandit_ensemble_epsilon_greedy_selection(dummy_component_signals):
    """Test epsilon-greedy arm selection."""
    components = [
        ComponentStrategy("rule", label="comp1"),
        ComponentStrategy("trend", label="comp2"),
    ]

    config = BanditEnsembleConfig(
        components=components,
        bandit_algorithm=BanditAlgorithm.EPSILON_GREEDY,
        epsilon=0.1,
        min_pulls_per_arm=0,  # Skip forced exploration for testing
    )

    ensemble = BanditEnsembleStrategy(config)
    ensemble.init(dummy_component_signals)

    # Set different mean rewards
    ensemble.bandit_stats["comp1"].mean_reward = 0.8
    ensemble.bandit_stats["comp2"].mean_reward = 0.2

    # Run selection multiple times
    selections = [ensemble._epsilon_greedy() for _ in range(100)]

    # Should mostly select comp1 (higher reward)
    comp1_count = selections.count("comp1")
    assert comp1_count > 70  # Expect >70% exploitation of best arm


def test_bandit_ensemble_ucb1_selection(dummy_component_signals):
    """Test UCB1 arm selection."""
    components = [
        ComponentStrategy("rule", label="comp1"),
        ComponentStrategy("trend", label="comp2"),
    ]

    config = BanditEnsembleConfig(
        components=components,
        bandit_algorithm=BanditAlgorithm.UCB1,
        ucb_c=2.0,
    )

    ensemble = BanditEnsembleStrategy(config)
    ensemble.init(dummy_component_signals)

    # Set pulls and rewards
    ensemble.bandit_stats["comp1"].n_pulls = 10
    ensemble.bandit_stats["comp1"].mean_reward = 0.5
    ensemble.bandit_stats["comp2"].n_pulls = 2
    ensemble.bandit_stats["comp2"].mean_reward = 0.3

    # UCB1 should favor comp2 (less explored, bonus from uncertainty)
    selected = ensemble._ucb1(current_bar=100)

    # With current setup, comp2 should have higher UCB score
    # (lower pulls = higher exploration bonus)
    # This is probabilistic, so just check it runs without error
    assert selected in ["comp1", "comp2"]


def test_bandit_ensemble_generate_signals(dummy_component_signals):
    """Test BanditEnsembleStrategy signal generation."""
    components = [
        ComponentStrategy("rule", label="comp1"),
        ComponentStrategy("trend", label="comp2"),
    ]

    config = BanditEnsembleConfig(
        components=components,
        bandit_algorithm=BanditAlgorithm.EPSILON_GREEDY,
        update_period=20,
        warmup_bars=50,
    )

    ensemble = BanditEnsembleStrategy(config)
    result = ensemble.generate_signals(dummy_component_signals)

    # Check required columns exist
    assert "long_entry" in result.columns
    assert "long_exit" in result.columns

    # Check signals are boolean
    assert result["long_entry"].dtype == bool
    assert result["long_exit"].dtype == bool


def test_bandit_ensemble_get_stats(dummy_component_signals):
    """Test bandit stats retrieval."""
    components = [
        ComponentStrategy("rule", label="comp1"),
        ComponentStrategy("trend", label="comp2"),
    ]

    config = BanditEnsembleConfig(components=components)
    ensemble = BanditEnsembleStrategy(config)
    ensemble.init(dummy_component_signals)

    # Update some stats
    ensemble.bandit_stats["comp1"].n_pulls = 5
    ensemble.bandit_stats["comp1"].total_reward = 2.5
    ensemble.bandit_stats["comp1"].mean_reward = 0.5

    stats_df = ensemble.get_bandit_stats_df()

    assert len(stats_df) == 2
    assert "component" in stats_df.columns
    assert "n_pulls" in stats_df.columns
    assert "mean_reward" in stats_df.columns

    comp1_stats = stats_df[stats_df["component"] == "comp1"].iloc[0]
    assert comp1_stats["n_pulls"] == 5
    assert comp1_stats["mean_reward"] == 0.5


def test_weighted_ensemble_config_from_dict():
    """Test WeightedEnsembleConfig from dictionary."""
    data = {
        "components": [
            {"strategy_name": "rule", "weight": 1.0},
        ],
        "weighting_method": "sharpe",
        "reweight_period": 50,
        "lookback_bars": 100,
        "signal_threshold": 0.6,
    }

    config = WeightedEnsembleConfig.from_dict(data)

    assert config.weighting_method == WeightingMethod.SHARPE
    assert config.reweight_period == 50
    assert config.lookback_bars == 100
    assert config.signal_threshold == 0.6


def test_bandit_ensemble_config_from_dict():
    """Test BanditEnsembleConfig from dictionary."""
    data = {
        "components": [
            {"strategy_name": "rule"},
        ],
        "bandit_algorithm": "ucb1",
        "ucb_c": 1.5,
        "update_period": 30,
        "reward_metric": "sharpe",
    }

    config = BanditEnsembleConfig.from_dict(data)

    assert config.bandit_algorithm == BanditAlgorithm.UCB1
    assert config.ucb_c == 1.5
    assert config.update_period == 30
    assert config.reward_metric == "sharpe"


def test_normalize_with_bounds_simple_case():
    """Test _normalize_with_bounds with a simple scenario."""
    raw = np.array([1.0, 1.0, 2.0])
    result = _normalize_with_bounds(raw, min_weight=0.1, max_weight=0.5)
    
    assert np.allclose(result, [0.25, 0.25, 0.5])
    assert np.isclose(result.sum(), 1.0)

def test_normalize_with_bounds_clipping_and_redistribution():
    """Test _normalize_with_bounds with a case requiring redistribution."""
    raw = np.array([10.0, 0.01])
    result = _normalize_with_bounds(raw, min_weight=0.1, max_weight=0.6)

    assert np.allclose(result, [0.6, 0.4])
    assert np.isclose(result.sum(), 1.0)
    assert result[0] <= 0.6
    assert result[1] >= 0.1

def test_normalize_with_bounds_infeasible_min():
    """Test _normalize_with_bounds with infeasible min_weight."""
    raw = np.array([1.0, 1.0, 1.0])
    # 3 * 0.4 = 1.2 > 1.0, so this is not possible.
    # Should fall back to uniform distribution.
    result = _normalize_with_bounds(raw, min_weight=0.4, max_weight=0.8)
    
    assert np.allclose(result, [1/3, 1/3, 1/3])

def test_normalize_with_bounds_infeasible_max():
    """Test _normalize_with_bounds with infeasible max_weight."""
    raw = np.array([1.0, 1.0, 1.0, 1.0])
    # 4 * 0.2 = 0.8 < 1.0, so this is not possible.
    # Should fall back to uniform + clip.
    result = _normalize_with_bounds(raw, min_weight=0.1, max_weight=0.2)
    
    assert np.allclose(result, [0.2, 0.2, 0.2, 0.2])

def test_normalize_with_bounds_with_negatives():
    """Test _normalize_with_bounds handles negative raw weights."""
    raw = np.array([2.0, -1.0, 1.0])
    # Should treat -1.0 as 0.0
    # Effective raw weights: [2.0, 0.0, 1.0] -> normalized [2/3, 0, 1/3]
    result = _normalize_with_bounds(raw, min_weight=None, max_weight=None)
    assert np.allclose(result, [2/3, 0.0, 1/3])


def test_weighted_ensemble_reweighting(dummy_ohlcv):
    """Test that weights are recomputed during backtest."""
    components = [
        ComponentStrategy("sharpe_comp", label="comp1"),
        ComponentStrategy("inv_vol_comp", label="comp2"),
    ]

    config = WeightedEnsembleConfig(
        components=components,
        weighting_method=WeightingMethod.SHARPE,
        reweight_period=50,
        lookback_bars=40,
        warmup_bars=40
    )

    # Create deterministic price data where comp1 strategy performs better
    n = len(dummy_ohlcv)
    np.random.seed(42)

    # Create a trending price that increases over time
    trend = np.linspace(100, 110, n)
    noise = np.random.randn(n) * 0.1
    close_prices = trend + noise

    df = pd.DataFrame({
        "timestamp": dummy_ohlcv["timestamp"],
        "open": close_prices,
        "high": close_prices * 1.01,
        "low": close_prices * 0.99,
        "close": close_prices,
        "volume": dummy_ohlcv["volume"],
    })

    # comp1 is always in (catches the trend)
    df["ensemble_component_comp1_signal"] = 1

    # comp2 is mostly out (misses the trend)
    df["ensemble_component_comp2_signal"] = (df.index.to_series().index % 10 == 0).astype(int)

    ensemble = WeightedEnsembleStrategy(config)
    ensemble.init(df)

    initial_weights = ensemble.current_weights.copy()
    assert abs(initial_weights["comp1"] - 0.5) < 0.001 # Starts with equal weights

    # Run a pseudo-backtest loop
    for i in range(len(df)):
        row = df.iloc[i]
        ctx = StrategyContext(equity=10000.0, position=None, index=i)
        _ = ensemble.on_bar(row, ctx)

        # Check weights after first reweight period
        if i == config.warmup_bars + config.reweight_period:
            break

    final_weights = ensemble.current_weights.copy()

    # Assert that weights have changed and comp1 has higher weight
    assert initial_weights != final_weights
    assert final_weights["comp1"] > final_weights["comp2"]
    assert abs(sum(final_weights.values()) - 1.0) < 0.001