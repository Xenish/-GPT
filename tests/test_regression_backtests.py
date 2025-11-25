from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from finantradealgo.backtester.backtest_engine import BacktestEngine
from finantradealgo.features.feature_pipeline_15m import build_feature_pipeline_from_system_config
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.strategies.ml_strategy import MLSignalStrategy, MLStrategyConfig
from finantradealgo.strategies.rule_signals import RuleSignalStrategy, RuleStrategyConfig
from finantradealgo.system.config_loader import load_system_config
from tests.utils_ml import prepare_ml_eval_df

GOLDEN_PATH = Path(__file__).parent / "golden" / "regression_rule_ml_15m.json"
REGRESSION_WINDOW = 1000
FINAL_EQUITY_TOL = 0.02  # 2%
TRADE_COUNT_TOL = 3


def _load_golden_payload() -> dict:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden regression file missing at {GOLDEN_PATH}. "
            "Run the documented golden refresh process."
        )
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compare_relative(actual: float, expected: float, tol: float) -> None:
    if expected == 0:
        assert abs(actual) <= tol
        return
    delta = abs(actual - expected) / abs(expected)
    assert delta <= tol


@pytest.mark.slow
def test_regression_backtests_against_golden():
    cfg = load_system_config()
    df_full, meta = build_feature_pipeline_from_system_config(cfg)
    df_window = df_full.tail(REGRESSION_WINDOW).reset_index(drop=True)

    df_eval, proba_col = prepare_ml_eval_df(df_window, meta, cfg)
    assert proba_col in df_eval.columns

    golden = _load_golden_payload()

    risk_cfg = RiskConfig.from_dict(cfg.get("risk", {}))

    rule_strategy = RuleSignalStrategy(RuleStrategyConfig.from_dict(cfg.get("rule", {})))
    rule_engine = BacktestEngine(
        strategy=rule_strategy,
        risk_engine=RiskEngine(risk_cfg),
        price_col="close",
        timestamp_col="timestamp",
    )
    rule_result = rule_engine.run(df_eval)

    ml_strategy_cfg = MLStrategyConfig.from_dict(cfg.get("ml", {}).get("backtest", {}))
    assert ml_strategy_cfg.proba_col == proba_col
    ml_strategy = MLSignalStrategy(ml_strategy_cfg)
    ml_engine = BacktestEngine(
        strategy=ml_strategy,
        risk_engine=RiskEngine(risk_cfg),
        price_col="close",
        timestamp_col="timestamp",
    )
    ml_result = ml_engine.run(df_eval)

    rule_metrics = rule_result["metrics"]
    ml_metrics = ml_result["metrics"]

    _compare_relative(rule_metrics["final_equity"], golden["rule"]["final_equity"], FINAL_EQUITY_TOL)
    _compare_relative(ml_metrics["final_equity"], golden["ml"]["final_equity"], FINAL_EQUITY_TOL)

    assert abs(rule_metrics["trade_count"] - golden["rule"]["trade_count"]) <= TRADE_COUNT_TOL
    assert abs(ml_metrics["trade_count"] - golden["ml"]["trade_count"]) <= TRADE_COUNT_TOL
