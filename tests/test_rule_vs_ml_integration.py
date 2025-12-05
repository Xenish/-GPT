from __future__ import annotations

import numpy as np
import pandas as pd

from finantradealgo.backtester.backtest_engine import BacktestEngine
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.strategies.ml_strategy import MLSignalStrategy, MLStrategyConfig
from finantradealgo.strategies.signal_column import ColumnSignalStrategy, ColumnSignalStrategyConfig


def _make_test_df() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=50, freq="15min")
    close = np.linspace(100, 105, len(ts))
    open_ = close - 0.1
    high = close + 0.2
    low = close - 0.2
    proba = np.concatenate([np.linspace(0.2, 0.9, 25), np.linspace(0.9, 0.3, 25)])
    rule_sig = np.where(np.arange(len(ts)) % 10 == 0, 1.0, np.nan)
    rule_sig[rule_sig == 1.0] = 1.0
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "ml_long_proba": proba,
            "signal_rule": rule_sig,
        }
    )
    return df


def test_rule_vs_ml_backtest_metrics_shape():
    df = _make_test_df()
    risk_engine = RiskEngine(RiskConfig())

    ml_cfg = MLStrategyConfig(proba_col="ml_long_proba", entry_threshold=0.6, exit_threshold=0.4, warmup_bars=0)
    ml_strategy = MLSignalStrategy(ml_cfg)
    ml_engine = BacktestEngine(strategy=ml_strategy, risk_engine=risk_engine, price_col="close", timestamp_col="timestamp")
    ml_result = ml_engine.run(df.copy())

    rule_strategy = ColumnSignalStrategy(ColumnSignalStrategyConfig(signal_col="signal_rule", warmup_bars=0))
    rule_engine = BacktestEngine(
        strategy=rule_strategy,
        risk_engine=risk_engine,
        price_col="close",
        timestamp_col="timestamp",
    )
    rule_result = rule_engine.run(df.copy())

    assert set(ml_result["metrics"].keys()) == set(rule_result["metrics"].keys())
    assert "equity_curve" in ml_result and "equity_curve" in rule_result
