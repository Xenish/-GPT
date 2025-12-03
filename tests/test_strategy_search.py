from __future__ import annotations

import pandas as pd
import pytest

from finantradealgo.strategies.param_space import ParamSpec
from research.strategy_search.search_engine import (
    evaluate_strategy_once,
    random_search,
)


def test_evaluate_strategy_once(monkeypatch):
    trades = pd.DataFrame({"pnl": [1.0, -0.5]})

    def fake_run_backtest_once(symbol, timeframe, strategy_name, cfg):
        return {
            "metrics": {
                "cum_return": 0.1,
                "sharpe": 0.5,
                "max_drawdown": -0.2,
                "trade_count": 2,
            },
            "trades": trades,
        }

    monkeypatch.setattr(
        "research.strategy_search.search_engine.run_backtest_once",
        fake_run_backtest_once,
    )
    monkeypatch.setattr(
        "research.strategy_search.search_engine.load_config",
        lambda profile="research": {"symbol": "AIAUSDT", "timeframe": "15m"},
    )

    result = evaluate_strategy_once("rule", {"tp_atr_mult": 2.0})
    assert result["params"]["tp_atr_mult"] == 2.0
    assert result["win_rate"] == pytest.approx(0.5)
    assert result["trade_count"] == 2


def test_random_search_uses_param_space(monkeypatch):
    trades = pd.DataFrame({"pnl": [1.0]})

    def fake_run_backtest_once(symbol, timeframe, strategy_name, cfg):
        return {
            "metrics": {
                "cum_return": 0.1,
                "sharpe": 0.3,
                "max_drawdown": -0.1,
                "trade_count": 1,
            },
            "trades": trades,
        }

    monkeypatch.setattr(
        "research.strategy_search.search_engine.run_backtest_once",
        fake_run_backtest_once,
    )
    monkeypatch.setattr(
        "research.strategy_search.search_engine.load_config",
        lambda profile="research": {"symbol": "AIAUSDT", "timeframe": "15m"},
    )

    dummy_space = {"alpha": ParamSpec(name="alpha", type="float", low=0.1, high=0.2)}

    class DummyMeta:
        def __init__(self):
            self.param_space = dummy_space

    monkeypatch.setattr(
        "research.strategy_search.search_engine.get_strategy_meta",
        lambda name: DummyMeta(),
    )

    samples = iter([{"alpha": 0.1}, {"alpha": 0.15}])

    def fake_sample_params(space):
        return next(samples)

    monkeypatch.setattr(
        "research.strategy_search.search_engine.sample_params",
        fake_sample_params,
    )

    results = random_search("rule", n_samples=2)
    assert len(results) == 2
    assert results[0]["params"]["alpha"] == 0.1
    assert results[1]["params"]["alpha"] == 0.15
