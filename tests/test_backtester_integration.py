from __future__ import annotations

import pandas as pd

from finantradealgo.backtester.backtest_engine import BacktestEngine
from finantradealgo.features.feature_pipeline import (
    build_feature_pipeline_from_system_config,
)
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.strategies.rule_signals import RuleSignalStrategy, RuleStrategyConfig
from finantradealgo.system.config_loader import load_config


def build_small_feature_df(window: int = 300):
    """
    Helper that mirrors the production feature builder but trims
    the result so integration tests remain fast.
    """
    cfg = load_config("research")
    df, meta = build_feature_pipeline_from_system_config(cfg)
    df_small = df.tail(window).reset_index(drop=True)
    return df_small, meta, cfg


def test_backtester_runs_with_rule_and_risk():
    df, meta, cfg = build_small_feature_df()
    assert not df.empty
    assert {"timestamp", "close"}.issubset(df.columns)
    assert meta["symbol"] == cfg["symbol"]

    risk_cfg = RiskConfig.from_dict(cfg.get("risk", {}))
    risk_engine = RiskEngine(risk_cfg)

    strat_cfg = RuleStrategyConfig.from_dict(cfg.get("rule", {}))
    strategy = RuleSignalStrategy(strat_cfg)

    engine = BacktestEngine(
        strategy=strategy,
        risk_engine=risk_engine,
        price_col="close",
        timestamp_col="timestamp",
    )

    result = engine.run(df)

    assert isinstance(result, dict)
    for key in ("equity_curve", "trades", "metrics", "risk_stats"):
        assert key in result

    equity_curve = result["equity_curve"]
    assert isinstance(equity_curve, pd.Series)
    assert len(equity_curve) > 0
    assert int(equity_curve.isna().sum()) == 0

    trades = result["trades"]
    assert isinstance(trades, pd.DataFrame)
    if not trades.empty:
        required_cols = {"timestamp", "side", "entry_price", "exit_price", "pnl"}
        assert required_cols.issubset(trades.columns)

    metrics = result["metrics"]
    assert metrics["final_equity"] is not None
    assert metrics["cum_return"] is not None

    risk_stats = result["risk_stats"]
    assert "blocked_entries" in risk_stats
    blocked = risk_stats["blocked_entries"]
    if isinstance(blocked, dict):
        blocked_total = sum(blocked.values())
    else:
        blocked_total = float(blocked or 0)
    assert blocked_total >= 0
