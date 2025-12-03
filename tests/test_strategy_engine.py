from __future__ import annotations

import pandas as pd
import pytest

from finantradealgo.core.portfolio import Position
from finantradealgo.core.strategy import StrategyContext
from finantradealgo.strategies.strategy_engine import create_strategy
from finantradealgo.strategies.trend_continuation import (
    TrendContinuationConfig,
    TrendContinuationStrategy,
)
from finantradealgo.system.config_loader import load_config


def test_create_strategy_rule_returns_expected_class():
    cfg = load_config("research")
    strategy = create_strategy("rule", cfg)
    from finantradealgo.strategies.rule_signals import RuleSignalStrategy

    assert isinstance(strategy, RuleSignalStrategy)


def test_create_strategy_invalid_name():
    cfg = load_config("research")
    with pytest.raises(ValueError):
        create_strategy("does_not_exist", cfg)


def test_trend_continuation_strategy_signals():
    cfg = TrendContinuationConfig(
        ema_fast_col="ema_fast",
        ema_slow_col="ema_slow",
        rsi_col="rsi",
        htf_trend_col="trend",
        ms_trend_col="ms_trend",
        warmup_bars=0,
        use_ms_trend_filter=True,
    )
    strat = TrendContinuationStrategy(cfg)
    df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-01-01 00:00:00"),
                "close": 100,
                "ema_fast": 101,
                "ema_slow": 99,
                "rsi": 60,
                "trend": 0.5,
                "ms_trend": 0.1,
            },
            {
                "timestamp": pd.Timestamp("2025-01-01 00:15:00"),
                "close": 95,
                "ema_fast": 100,
                "ema_slow": 100,
                "rsi": 40,
                "trend": 0.4,
                "ms_trend": 0.0,
            },
        ]
    )
    strat.init(df)

    ctx_entry = StrategyContext(equity=10_000, position=None, index=0)
    signal_entry = strat.on_bar(df.iloc[0], ctx_entry)
    assert signal_entry == "LONG"

    ctx_exit = StrategyContext(
        equity=10_000,
        position=Position(side="LONG", qty=1, entry_price=100),
        index=1,
    )
    signal_exit = strat.on_bar(df.iloc[1], ctx_exit)
    assert signal_exit == "CLOSE"
