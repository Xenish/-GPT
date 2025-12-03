"""
Generates the golden file for backtest regression tests.
Run this after making algorithmic changes that affect backtest results.
"""
import json
import os
import sys
from pathlib import Path

# Set dummy FCM key for testing if not already set
if "FCM_SERVER_KEY" not in os.environ:
    os.environ["FCM_SERVER_KEY"] = "dummy_key_for_testing"

# Ensure project root is in Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.backtester.backtest_engine import BacktestEngine
from finantradealgo.features.feature_pipeline import build_feature_pipeline_from_system_config
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.strategies.ml_strategy import MLSignalStrategy, MLStrategyConfig
from finantradealgo.strategies.rule_signals import RuleSignalStrategy, RuleStrategyConfig
from finantradealgo.system.config_loader import load_config
from tests.utils_ml import prepare_ml_eval_df

GOLDEN_PATH = ROOT / "tests" / "golden" / "regression_rule_ml_15m.json"
REGRESSION_WINDOW = 1000


def generate_golden_file():
    """Generate golden file with current backtest results."""
    print("Loading system config...")
    cfg = load_config("research")

    print("Building feature pipeline...")
    df_full, meta = build_feature_pipeline_from_system_config(cfg)
    df_window = df_full.tail(REGRESSION_WINDOW).reset_index(drop=True)

    print("Preparing ML evaluation data...")
    df_eval, proba_col = prepare_ml_eval_df(df_window, meta, cfg)
    assert proba_col in df_eval.columns

    risk_cfg = RiskConfig.from_dict(cfg.get("risk", {}))

    # Run Rule Strategy Backtest
    print("Running rule strategy backtest...")
    rule_strategy = RuleSignalStrategy(RuleStrategyConfig.from_dict(cfg.get("rule", {})))
    rule_engine = BacktestEngine(
        strategy=rule_strategy,
        risk_engine=RiskEngine(risk_cfg),
        price_col="close",
        timestamp_col="timestamp",
    )
    rule_result = rule_engine.run(df_eval)

    # Run ML Strategy Backtest
    print("Running ML strategy backtest...")
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

    # Extract metrics
    rule_metrics = rule_result["metrics"]
    ml_metrics = ml_result["metrics"]

    # Create golden payload
    golden = {
        "rule": {
            "final_equity": rule_metrics["final_equity"],
            "trade_count": rule_metrics["trade_count"],
        },
        "ml": {
            "final_equity": ml_metrics["final_equity"],
            "trade_count": ml_metrics["trade_count"],
        },
    }

    # Save to golden file
    print(f"Saving golden file to: {GOLDEN_PATH}")
    GOLDEN_PATH.parent.mkdir(exist_ok=True)
    with GOLDEN_PATH.open("w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2)

    print("\nGolden file generated successfully!")
    print(f"Rule - Final Equity: {rule_metrics['final_equity']:.2f}, Trade Count: {rule_metrics['trade_count']}")
    print(f"ML   - Final Equity: {ml_metrics['final_equity']:.2f}, Trade Count: {ml_metrics['trade_count']}")


if __name__ == "__main__":
    generate_golden_file()
