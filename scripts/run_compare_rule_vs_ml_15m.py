from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from finantradealgo.features.feature_pipeline_15m import (
    PIPELINE_VERSION_15M,
    build_feature_pipeline_from_system_config,
)
from finantradealgo.ml.model_registry import get_latest_model, load_model_by_id
from finantradealgo.system.config_loader import load_system_config
from scripts.run_ml_backtest_15m import run_ml_backtest
from scripts.run_rule_backtest_15m import run_rule_backtest


def log_run_header(symbol: str, timeframe: str, preset: str, pipeline_version: str, extra: str | None = None) -> None:
    msg = (
        f"[RUN] symbol={symbol} timeframe={timeframe} "
        f"feature_preset={preset} pipeline_version={pipeline_version}"
    )
    if extra:
        msg += f" {extra}"
    print(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rule vs ML strategies on 15m data.")
    parser.add_argument("--model-id", help="Evaluate a specific model_id from the registry.")
    parser.add_argument("--allow-version-mismatch", action="store_true", help="Allow pipeline version mismatches.")
    args = parser.parse_args()

    sys_cfg = load_system_config()
    cfg_eval = copy.deepcopy(sys_cfg)

    ml_cfg = cfg_eval.get("ml", {})
    persistence_cfg = ml_cfg.get("persistence", {}) or {}
    registry_cfg = ml_cfg.get("registry", {}) or {}
    model_dir = persistence_cfg.get("model_dir", "outputs/ml_models")

    model_id = args.model_id or registry_cfg.get("selected_id")
    if not model_id:
        latest = get_latest_model(
            model_dir,
            symbol=cfg_eval.get("symbol", "BTCUSDT"),
            timeframe=cfg_eval.get("timeframe", "15m"),
            model_type=ml_cfg.get("model", {}).get("type", "RandomForest"),
        )
        if latest is None:
            raise ValueError("No saved ML model found in registry for comparison.")
        model_id = latest.model_id

    _, meta = load_model_by_id(model_dir, model_id)

    if meta.pipeline_version and meta.pipeline_version != PIPELINE_VERSION_15M and not args.allow_version_mismatch:
        raise ValueError(
            f"Model {model_id} trained with pipeline_version {meta.pipeline_version} "
            f"but current pipeline is {PIPELINE_VERSION_15M}. Re-train or re-run with --allow-version-mismatch."
        )
    elif meta.pipeline_version and meta.pipeline_version != PIPELINE_VERSION_15M:
        print(
            f"[WARN] Model {model_id} trained with pipeline_version {meta.pipeline_version} "
            f"but current pipeline is {PIPELINE_VERSION_15M}. Feature definitions may differ."
        )

    cfg_eval.setdefault("features", {})
    cfg_eval["features"]["feature_preset"] = meta.feature_preset
    cfg_eval["ml"]["backtest"]["use_saved_model"] = True
    cfg_eval["ml"]["backtest"]["model_id"] = model_id
    cfg_eval["ml"]["backtest"]["allow_pipeline_mismatch"] = args.allow_version_mismatch
    cfg_eval["ml"]["persistence"]["save_model"] = False

    df, pipeline_meta = build_feature_pipeline_from_system_config(cfg_eval)
    symbol = pipeline_meta.get("symbol", cfg_eval.get("symbol", "BTCUSDT"))
    timeframe = pipeline_meta.get("timeframe", cfg_eval.get("timeframe", "15m"))
    preset = pipeline_meta.get("feature_preset", cfg_eval.get("features", {}).get("feature_preset", "extended"))
    pipeline_version = pipeline_meta.get("pipeline_version", "unknown")
    log_run_header(symbol, timeframe, preset, pipeline_version, extra=f"model_id={model_id}")

    ml_report, df_eval, aux_ml = run_ml_backtest(
        sys_cfg=cfg_eval,
        df=df,
        pipeline_meta=pipeline_meta,
    )

    rule_report, _, _ = run_rule_backtest(sys_cfg=cfg_eval, df=df_eval.copy())

    rows = []
    rows.append(
        {
            "strategy": "rule",
            "final_equity": rule_report["equity_metrics"]["final_equity"],
            "cum_return": rule_report["equity_metrics"]["cum_return"],
            "max_drawdown": rule_report["equity_metrics"]["max_drawdown"],
            "sharpe": rule_report["equity_metrics"]["sharpe"],
            "trade_count": rule_report["trade_stats"]["trade_count"],
            "win_rate": rule_report["trade_stats"]["win_rate"],
        }
    )
    rows.append(
        {
            "strategy": "ml",
            "model_id": model_id,
            "final_equity": ml_report["equity_metrics"]["final_equity"],
            "cum_return": ml_report["equity_metrics"]["cum_return"],
            "max_drawdown": ml_report["equity_metrics"]["max_drawdown"],
            "sharpe": ml_report["equity_metrics"]["sharpe"],
            "trade_count": ml_report["trade_stats"]["trade_count"],
            "win_rate": ml_report["trade_stats"]["win_rate"],
        }
    )

    df_cmp = pd.DataFrame(rows)
    print("\n=== Rule vs ML Comparison (15m) ===")
    print(df_cmp)

    out_dir = Path("outputs") / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compare_rule_vs_ml_15m.csv"
    df_cmp.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved comparison table -> {out_path}")

    rule_risk = rule_report.get("risk_stats", {}) or {}
    ml_risk = ml_report.get("risk_stats", {}) or {}
    print(
        f"[RISK] rule_blocked_entries_total={sum(rule_risk.get('blocked_entries', {}).values())} | "
        f"ml_blocked_entries_total={sum(ml_risk.get('blocked_entries', {}).values())}"
    )

    rule_eq_path = Path("outputs") / "backtests" / "rule_equity_15m.csv"
    rule_trades_path = Path("outputs") / "trades" / "rule_trades_15m.csv"
    ml_eq_path = Path("outputs") / "backtests" / "ml_equity_15m.csv"
    ml_trades_path = Path("outputs") / "trades" / "ml_trades_15m.csv"
    print(f"[INFO] rule_equity_path={rule_eq_path} rule_trades_path={rule_trades_path}")
    print(f"[INFO] ml_equity_path={ml_eq_path} ml_trades_path={ml_trades_path}")


if __name__ == "__main__":
    main()
