from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.backtester.backtest_engine import BacktestConfig, Backtester
from finantradealgo.core.report import ReportConfig, generate_report
from finantradealgo.features.feature_pipeline import (
    PIPELINE_VERSION,
    build_feature_pipeline_from_system_config,
)
from finantradealgo.ml.model_registry import (
    get_latest_model,
    load_model_by_id,
)
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.strategies.ml_strategy import MLSignalStrategy, MLStrategyConfig
from finantradealgo.system.config_loader import load_config


def log_run_header(symbol: str, timeframe: str, preset: str, pipeline_version: str, extra: str | None = None) -> None:
    msg = (
        f"[RUN] symbol={symbol} timeframe={timeframe} "
        f"feature_preset={preset} pipeline_version={pipeline_version}"
    )
    if extra:
        msg += f" {extra}"
    print(msg)


def _ensure_output_dirs() -> tuple[Path, Path]:
    bt_dir = Path("outputs") / "backtests"
    tr_dir = Path("outputs") / "trades"
    bt_dir.mkdir(parents=True, exist_ok=True)
    tr_dir.mkdir(parents=True, exist_ok=True)
    return bt_dir, tr_dir


def _load_model(sys_cfg: dict):
    ml_cfg = sys_cfg.get("ml", {})
    persistence_cfg = ml_cfg.get("persistence", {}) or {}
    registry_cfg = ml_cfg.get("registry", {}) or {}
    model_dir = persistence_cfg.get("model_dir", "outputs/ml_models")
    selected_id = registry_cfg.get("selected_id")

    if selected_id:
        model, meta = load_model_by_id(model_dir, selected_id)
        return model, meta

    model_type = ml_cfg.get("model", {}).get("type", "RandomForest")
    symbol = sys_cfg.get("symbol", "BTCUSDT")
    timeframe = sys_cfg.get("timeframe", "15m")
    if not registry_cfg.get("use_latest", True):
        raise ValueError("No model_id provided and registry.use_latest is False.")
    entry = get_latest_model(model_dir, symbol=symbol, timeframe=timeframe, model_type=model_type)
    if entry is None:
        raise ValueError("No saved ML model found in registry.")
    model, meta = load_model_by_id(model_dir, entry.model_id)
    return model, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML inference backtest using a saved model.")
    parser.add_argument("--allow-version-mismatch", action="store_true", help="Allow mismatched pipeline versions.")
    args = parser.parse_args()

    sys_cfg = load_config("research")
    ml_cfg = sys_cfg.get("ml", {})
    model, meta = _load_model(sys_cfg)

    df, pipeline_meta = build_feature_pipeline_from_system_config(sys_cfg)
    missing = [c for c in meta.feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature mismatch between model and pipeline: missing {missing}")
    current_version = pipeline_meta.get("pipeline_version", PIPELINE_VERSION)
    if meta.pipeline_version and meta.pipeline_version != current_version:
        if not args.allow_version_mismatch:
            raise ValueError(
                f"Model {meta.model_id} trained with pipeline_version {meta.pipeline_version} "
                f"but current pipeline is {current_version}. Re-train or pass --allow-version-mismatch."
            )
        else:
            print(
                f"[WARN] Model {meta.model_id} trained with pipeline_version "
                f"{meta.pipeline_version} but current pipeline is {current_version}. Feature definitions may differ."
            )

    symbol = pipeline_meta.get("symbol", sys_cfg.get("symbol", "BTCUSDT"))
    timeframe = pipeline_meta.get("timeframe", sys_cfg.get("timeframe", "15m"))
    preset = pipeline_meta.get("feature_preset", sys_cfg.get("features", {}).get("feature_preset", "extended"))
    log_run_header(symbol, timeframe, preset, current_version, extra=f"model_id={meta.model_id}")

    X = df[meta.feature_cols].to_numpy()
    proba = model.predict_proba(X)
    if proba.shape[1] < 2:
        raise ValueError("Loaded model does not provide binary probabilities.")
    proba_col = ml_cfg.get("backtest", {}).get("proba_column", "ml_proba_long")
    df[proba_col] = proba[:, 1]

    ml_bt_cfg = MLStrategyConfig.from_dict(ml_cfg.get("backtest"))
    risk_engine = RiskEngine(RiskConfig.from_dict(sys_cfg.get("risk")))
    bt_config = BacktestConfig(
        initial_cash=10_000.0,
        fee_pct=0.0004,
        slippage_pct=0.0005,
    )
    strategy = MLSignalStrategy(ml_bt_cfg)
    backtester = Backtester(strategy=strategy, risk_engine=risk_engine, config=bt_config)
    result = backtester.run(df)

    report = generate_report(
        result,
        df=df,
        config=ReportConfig(regime_columns=["regime_trend", "regime_vol"]),
    )
    report["risk_stats"] = result.get("risk_stats", {})

    eq = report["equity_metrics"]
    ts = report["trade_stats"]

    print(f"[INFO] Using model: {meta.model_id}")
    print("\n=== ML Inference Backtest (15m) ===")
    print("Equity:")
    print(f"  Initial cash : {eq['initial_cash']}")
    print(f"  Final equity : {eq['final_equity']}")
    print(f"  Cumulative R.: {eq['cum_return']}")
    print(f"  Max drawdown : {eq['max_drawdown']}")
    print(f"  Sharpe       : {eq['sharpe']}")

    print("\nTrades:")
    print(f"  Count        : {ts['trade_count']}")
    print(f"  Win rate     : {ts['win_rate']}")
    print(f"  Avg PnL      : {ts['avg_pnl']}")

    risk_stats = report.get("risk_stats", {}) or {}
    blocked_total = sum(risk_stats.get("blocked_entries", {}).values())
    print(f"[RISK] blocked_entries_total={blocked_total}")

    bt_dir, tr_dir = _ensure_output_dirs()
    eq_path = bt_dir / "ml_infer_equity_15m.csv"
    trades_path = tr_dir / "ml_infer_trades_15m.csv"
    report["equity_curve"].to_csv(eq_path, header=True)
    report["trades"].to_csv(trades_path, index=False)
    print(f"\n[INFO] Saved {eq_path} and {trades_path}")


if __name__ == "__main__":
    main()
