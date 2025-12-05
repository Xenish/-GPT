from __future__ import annotations

import pandas as pd

from finantradealgo.features.feature_pipeline import build_feature_pipeline_from_system_config
from finantradealgo.strategies.ema_cross import EMACrossStrategy
from finantradealgo.backtester.backtest_engine import BacktestEngine, BacktestConfig
from finantradealgo.risk.risk_engine import RiskEngine, RiskConfig
from finantradealgo.system.config_loader import load_config


def _make_df():
    ts = pd.date_range("2025-01-01", periods=120, freq="1min", tz="UTC")
    prices = pd.Series(range(120), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": prices + 1,
            "high": prices + 2,
            "low": prices,
            "close": prices + 1.5,
            "volume": 1000,
        }
    )


def test_market_micro_features_compatible_with_backtest():
    cfg = load_config("research")
    cfg_local = dict(cfg)
    features_cfg = dict(cfg_local.get("features", {}))
    features_cfg["use_market_structure"] = True
    features_cfg["use_microstructure"] = True
    cfg_local["features"] = features_cfg

    df = _make_df()
    df_feat, meta = build_feature_pipeline_from_system_config(cfg_local, df_ohlcv_override=df)

    # Run a simple strategy on the enriched feature set; should not raise
    engine = BacktestEngine(
        strategy=EMACrossStrategy(fast=5, slow=15),
        risk_engine=RiskEngine(RiskConfig()),
        config=BacktestConfig(initial_cash=10_000.0),
    )
    result = engine.run(df_feat)

    assert result["metrics"]["final_equity"] is not None
    assert result["equity_curve"] is not None
