from __future__ import annotations

import pandas as pd
import pytest

from finantradealgo.backtester.backtest_engine import BacktestEngine
from finantradealgo.features.feature_pipeline import build_feature_pipeline_from_system_config
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.strategies.ml_strategy import MLSignalStrategy, MLStrategyConfig
from finantradealgo.strategies.rule_signals import RuleSignalStrategy, RuleStrategyConfig
from finantradealgo.system.config_loader import load_system_config
from tests.utils_ml import prepare_ml_eval_df


@pytest.mark.slow
def test_rule_vs_ml_integration_pipeline():
    cfg = load_system_config()
    df_all, meta = build_feature_pipeline_from_system_config(cfg)
    df_all = df_all.reset_index(drop=True)

    assert meta["symbol"] == cfg["symbol"]
    assert meta["timeframe"] == cfg["timeframe"]

    df_eval, proba_col = prepare_ml_eval_df(df_all, meta, cfg)
    assert proba_col in df_eval.columns

    risk_cfg = RiskConfig.from_dict(cfg.get("risk", {}))

    rule_strategy = RuleSignalStrategy(RuleStrategyConfig.from_dict(cfg.get("rule", {})))
    rule_engine = BacktestEngine(
        strategy=rule_strategy,
        risk_engine=RiskEngine(risk_cfg),
        price_col="close",
        timestamp_col="timestamp",
    )

    ml_strategy_cfg = MLStrategyConfig.from_dict(cfg.get("ml", {}).get("backtest", {}))
    assert ml_strategy_cfg.proba_col == proba_col
    ml_strategy = MLSignalStrategy(ml_strategy_cfg)
    ml_engine = BacktestEngine(
        strategy=ml_strategy,
        risk_engine=RiskEngine(risk_cfg),
        price_col="close",
        timestamp_col="timestamp",
    )

    rule_result = rule_engine.run(df_eval)
    ml_result = ml_engine.run(df_eval)

    assert rule_result["metrics"]["trade_count"] >= 0
    assert ml_result["metrics"]["trade_count"] >= 0

    for res in (rule_result, ml_result):
        metrics = res["metrics"]
        assert metrics["final_equity"] is not None
        assert not pd.isna(metrics["final_equity"])
        assert not pd.isna(metrics["cum_return"])
