from __future__ import annotations

import pandas as pd
import numpy as np

from finantradealgo.backtester.backtest_engine import BacktestEngine, BacktestConfig
from finantradealgo.risk.risk_engine import RiskEngine, RiskConfig
from finantradealgo.strategies.ema_cross import EMACrossStrategy


def _make_price_series():
    idx = pd.date_range("2025-01-01 00:00:00", periods=200, freq="1min", tz="UTC")
    prices = pd.Series(100 + np.sin(np.linspace(0, 10, len(idx))) * 2 + np.linspace(0, 5, len(idx)), index=idx)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices + 0.1,
            "volume": 1000,
        }
    )
    return df


def test_backtest_basic_run_smoke():
    df = _make_price_series()
    df = df.reset_index(drop=True)  # ensure integer index for backtester/strategy loop
    strat = EMACrossStrategy(fast=5, slow=15)
    risk = RiskEngine(RiskConfig())
    engine = BacktestEngine(strategy=strat, risk_engine=risk, config=BacktestConfig(initial_cash=10_000.0))

    result = engine.run(df)

    metrics = result.get("metrics", {})
    assert metrics, "Missing metrics in result"
    assert metrics["final_equity"] is not None
    assert metrics["cum_return"] is not None
    assert metrics["max_drawdown"] is not None

    # Trades may be zero on synthetic data; still ensure fields are present and non-NaN
    assert "trade_count" in metrics
    assert metrics["trade_count"] >= 0
    assert result.get("equity_curve") is not None
