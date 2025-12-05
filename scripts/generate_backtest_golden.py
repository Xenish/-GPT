"""
Generate golden backtest fixtures for regression tests.

Usage:
    python scripts/generate_backtest_golden.py

Outputs:
    tests/golden/ema_cross_synthetic.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from finantradealgo.backtester.backtest_engine import BacktestEngine, BacktestConfig
from finantradealgo.risk.risk_engine import RiskEngine, RiskConfig
from finantradealgo.strategies.ema_cross import EMACrossStrategy

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "tests" / "golden"
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)


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


def generate_ema_cross_synthetic():
    df = _make_price_series()
    engine = BacktestEngine(
        strategy=EMACrossStrategy(fast=5, slow=15),
        risk_engine=RiskEngine(RiskConfig()),
        config=BacktestConfig(initial_cash=10_000.0),
    )
    res = engine.run(df)
    payload = {
        "metrics": res["metrics"],
        "meta": {
            "strategy": "ema_cross",
            "symbol": "SYNTH",
            "timeframe": "1m",
            "bars": len(df),
            "generated_by": "scripts/generate_backtest_golden.py",
        },
    }
    path = GOLDEN_DIR / "ema_cross_synthetic.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {path}")


def main():
    generate_ema_cross_synthetic()


if __name__ == "__main__":
    main()
