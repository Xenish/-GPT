from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finantradealgo.backtester.backtest_engine import BacktestEngine, BacktestConfig
from finantradealgo.risk.risk_engine import RiskEngine, RiskConfig
from finantradealgo.strategies.ema_cross import EMACrossStrategy

GOLDEN_DIR = Path(__file__).parent / "golden"
EMA_GOLDEN = GOLDEN_DIR / "ema_cross_synthetic.json"


def _make_price_series():
    idx = pd.date_range("2025-01-01 00:00:00", periods=200, freq="1min", tz="UTC")
    prices = 100 + np.sin(np.linspace(0, 10, len(idx))) * 2 + np.linspace(0, 5, len(idx))
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
    return df.reset_index(drop=True)


def test_ema_cross_golden_regression():
    if not EMA_GOLDEN.exists():
        raise FileNotFoundError(f"Golden file missing: {EMA_GOLDEN}. Run scripts/generate_backtest_golden.py")

    with EMA_GOLDEN.open("r", encoding="utf-8") as f:
        golden = json.load(f)

    df = _make_price_series()
    engine = BacktestEngine(
        strategy=EMACrossStrategy(fast=5, slow=15),
        risk_engine=RiskEngine(RiskConfig()),
        config=BacktestConfig(initial_cash=10_000.0),
    )
    res = engine.run(df)
    metrics = res["metrics"]
    golden_metrics = golden["metrics"]

    for key in ("final_equity", "cum_return", "max_drawdown", "sharpe", "trade_count"):
        assert key in metrics
        assert key in golden_metrics
        assert metrics[key] == pytest.approx(golden_metrics[key], rel=1e-6, abs=1e-6)
