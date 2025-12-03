import pytest

from finantradealgo.core.strategy import BaseStrategy
from finantradealgo.strategies.strategy_engine import (
    STRATEGY_REGISTRY,
    create_strategy,
    get_searchable_strategies,
    get_strategy_meta,
    list_strategies,
)


def test_list_strategies_includes_core_names():
    names = set(list_strategies().keys())
    for required in ["rule", "trend_continuation", "sweep_reversal", "volatility_breakout"]:
        assert required in names


def test_get_strategy_meta_rule():
    meta = get_strategy_meta("rule")
    assert meta.family == "rule"
    assert meta.param_space is not None
    assert meta.is_searchable is True


def test_get_searchable_strategies_filters_by_param_space():
    searchable = get_searchable_strategies()
    assert "rule" in searchable
    assert "trend_continuation" in searchable
    assert "sweep_reversal" in searchable
    assert "volatility_breakout" in searchable
    assert "ml" not in searchable  # ml has no param_space


def test_create_strategy_returns_instance():
    cfg = {"strategy": {}}  # minimal
    strategy = create_strategy("rule", cfg)
    assert isinstance(strategy, BaseStrategy)
