from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from finantradealgo.backtester.backtest_engine import BacktestConfig, Backtester
from finantradealgo.core.report import ReportConfig, generate_report
from finantradealgo.features.feature_pipeline import (
    PIPELINE_VERSION,
    build_feature_pipeline_from_system_config,
    get_feature_cols,
)
from finantradealgo.ml.labels import LabelConfig, add_long_only_labels
from finantradealgo.ml.model import (
    SklearnLongModel,
    SklearnModelConfig,
    save_sklearn_model,
)
from finantradealgo.ml.model_registry import (
    get_latest_model,
    load_model_by_id,
    register_model,
)
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.strategies.ml_strategy import MLSignalStrategy, MLStrategyConfig
from finantradealgo.system.config_loader import load_system_config


def log_run_header(symbol: str, timeframe: str, preset: str, pipeline_version: str, extra: str | None = None) -> None:
    msg = (
        f"[RUN] symbol={symbol} timeframe={timeframe} "
        f"feature_preset={preset} pipeline_version={pipeline_version}"
    )
    if extra:
        msg += f" {extra}"
    print(msg)


def run_ml_backtest(
    sys_cfg: Optional[Dict[str, Any]] = None,
    df: Optional[pd.DataFrame] = None,
    pipeline_meta: Optional[Dict[str, Any]] = None,
    split_ratio: float = 0.7,
) -> Tuple[dict, pd.DataFrame, Dict[str, Any]]:
    sys_cfg = sys_cfg or load_system_config()

    if df is None or pipeline_meta is None:
        df, pipeline_meta = build_feature_pipeline_from_system_config(sys_cfg)
    else:
        df = df.copy()

    symbol = pipeline_meta.get("symbol", sys_cfg.get("symbol", "BTCUSDT"))
    timeframe = pipeline_meta.get("timeframe", sys_cfg.get("timeframe", "15m"))
    pipeline_version = pipeline_meta.get("pipeline_version", PIPELINE_VERSION)

    ml_cfg = sys_cfg.get("ml", {})
    backtest_cfg = ml_cfg.get("backtest", {}) or {}
    persistence_cfg = ml_cfg.get("persistence", {}) or {}
    registry_cfg = ml_cfg.get("registry", {}) or {}

    label_cfg = LabelConfig.from_dict(ml_cfg.get("label"))
    df_lab = add_long_only_labels(df, label_cfg)
    target_col = "label_long"
    if target_col not in df_lab.columns:
        raise ValueError(f"{target_col} column missing after labeling.")

    df_lab = df_lab.dropna(subset=[target_col]).reset_index(drop=True)
    if len(df_lab) < 10:
        raise ValueError("Not enough labeled rows for ML backtest.")

    split_idx = max(int(len(df_lab) * split_ratio), 1)
    if split_idx >= len(df_lab):
        split_idx = len(df_lab) - 1

    df_train = df_lab.iloc[:split_idx].copy()
    df_eval = df_lab.iloc[split_idx:].copy().reset_index(drop=True)

    feature_preset = pipeline_meta.get(
        "feature_preset",
        sys_cfg.get("features", {}).get("feature_preset", "extended"),
    )

    feature_cols = pipeline_meta.get("feature_cols")
    if feature_cols is None:
        feature_cols = get_feature_cols(df_lab, preset=feature_preset)

    missing_cols = [c for c in feature_cols if c not in df_lab.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns for ML backtest: {missing_cols}")

    X_eval = df_eval[feature_cols].to_numpy()
    y_eval = df_eval[target_col].to_numpy(dtype=int)

    use_saved_model = backtest_cfg.get("use_saved_model", False)
    model_obj = None
    loaded_meta = None
    model_cfg = None

    if use_saved_model:
        model_dir = persistence_cfg.get("model_dir", "outputs/ml_models")
        model_id = backtest_cfg.get("model_id") or registry_cfg.get("selected_id")
        if not model_id:
            latest = get_latest_model(
                model_dir,
                symbol=symbol,
                timeframe=timeframe,
                model_type=ml_cfg.get("model", {}).get("type", "RandomForest"),
            )
            if latest is None:
                raise ValueError("No saved model found in registry for inference.")
            model_id = latest.model_id
        model_obj, loaded_meta = load_model_by_id(model_dir, model_id)
        feature_cols = loaded_meta.feature_cols
        missing_cols = [c for c in feature_cols if c not in df_lab.columns]
        if missing_cols:
            raise ValueError(f"Current pipeline missing columns from model: {missing_cols}")
        X_eval = df_eval[feature_cols].to_numpy()
        saved_version = getattr(loaded_meta, "pipeline_version", None)
        if saved_version and saved_version != pipeline_version:
            if not backtest_cfg.get("allow_pipeline_mismatch", False):
                raise ValueError(
                    f"Model {loaded_meta.model_id} trained with pipeline_version "
                    f"{saved_version} but current pipeline is {pipeline_version}. "
                    "Re-train the model or set ml.backtest.allow_pipeline_mismatch=true to override."
                )
            else:
                print(
                    f"[WARN] Model {loaded_meta.model_id} trained with pipeline_version "
                    f"{saved_version} but current pipeline is {pipeline_version}. "
                    "Feature definitions may have drifted."
                )
    else:
        model_cfg = SklearnModelConfig.from_dict(ml_cfg.get("model"))
        model_obj = SklearnLongModel(model_cfg)
        X_train = df_train[feature_cols].to_numpy()
        y_train = df_train[target_col].to_numpy(dtype=int)
        model_obj.fit(X_train, y_train)

    proba = model_obj.predict_proba(X_eval)[:, 1]

    ml_bt_cfg = MLStrategyConfig.from_dict(backtest_cfg)
    df_eval[ml_bt_cfg.proba_col] = proba

    y_pred = (proba >= 0.5).astype(int)
    cls_metrics = {
        "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
        "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
        "f1": float(f1_score(y_eval, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_eval, y_pred)),
    }

    risk_engine = RiskEngine(RiskConfig.from_dict(sys_cfg.get("risk")))
    bt_config = BacktestConfig(
        initial_cash=10_000.0,
        fee_pct=0.0004,
        slippage_pct=0.0005,
    )
    strategy = MLSignalStrategy(ml_bt_cfg)
    backtester = Backtester(strategy=strategy, risk_engine=risk_engine, config=bt_config)
    result = backtester.run(df_eval)

    report = generate_report(
        result,
        df=df_eval,
        config=ReportConfig(regime_columns=["regime_trend", "regime_vol"]),
    )
    report["risk_stats"] = result.get("risk_stats", {})

    train_size = len(df_train) if not use_saved_model else len(df_lab) - len(df_eval)
    aux = {
        "train_size": train_size,
        "test_size": len(df_eval),
        "classification_metrics": cls_metrics,
        "feature_count": len(feature_cols),
        "model_meta": loaded_meta,
        "pipeline_meta": pipeline_meta,
        "mode": "saved_model" if use_saved_model else "train_fit",
    }

    eq = report["equity_metrics"]
    ts_stats = report["trade_stats"]
    metrics = {
        **cls_metrics,
        "cum_return": eq.get("cum_return"),
        "sharpe": eq.get("sharpe"),
        "final_equity": eq.get("final_equity"),
        "trade_count": ts_stats.get("trade_count"),
        "win_rate": ts_stats.get("win_rate"),
    }

    if (not use_saved_model) and persistence_cfg.get("save_model", False):
        if "timestamp" not in df_train.columns:
            raise ValueError("timestamp column missing for model metadata.")
        train_start = pd.to_datetime(df_train["timestamp"].iloc[0])
        train_end = pd.to_datetime(df_train["timestamp"].iloc[-1])
        model_dir = persistence_cfg.get("model_dir", "outputs/ml_models")
        meta = save_sklearn_model(
            model=model_obj.clf,
            symbol=symbol,
            timeframe=timeframe,
            model_cfg=model_cfg,
            label_cfg=label_cfg,
            feature_preset=feature_preset,
            feature_cols=feature_cols,
            train_start=train_start,
            train_end=train_end,
            metrics=metrics,
            base_dir=model_dir,
            pipeline_version=pipeline_version,
        )
        aux["model_meta"] = meta
        if persistence_cfg.get("use_registry", True):
            register_model(
                meta,
                base_dir=model_dir,
                status="success",
                max_models=persistence_cfg.get("max_models_per_symbol_tf"),
            )

    return report, df_eval, aux


def _ensure_output_dirs() -> tuple[Path, Path]:
    bt_dir = Path("outputs") / "backtests"
    tr_dir = Path("outputs") / "trades"
    bt_dir.mkdir(parents=True, exist_ok=True)
    tr_dir.mkdir(parents=True, exist_ok=True)
    return bt_dir, tr_dir


def main() -> None:
    sys_cfg = load_system_config()
    df, pipeline_meta = build_feature_pipeline_from_system_config(sys_cfg)
    report, df_eval, aux = run_ml_backtest(
        sys_cfg=sys_cfg,
        df=df,
        pipeline_meta=pipeline_meta,
    )
    meta = aux.get("pipeline_meta", pipeline_meta)
    symbol = meta.get("symbol", sys_cfg.get("symbol", "BTCUSDT"))
    timeframe = meta.get("timeframe", sys_cfg.get("timeframe", "15m"))
    preset = meta.get("feature_preset", sys_cfg.get("features", {}).get("feature_preset", "extended"))
    pipeline_version = meta.get("pipeline_version", PIPELINE_VERSION)
    model_meta = aux.get("model_meta")
    mode_label = aux.get("mode", "train_fit")
    if model_meta:
        extra = f"{mode_label} model_id={model_meta.model_id}"
    else:
        extra = mode_label
    log_run_header(symbol, timeframe, preset, pipeline_version, extra=extra)

    eq = report["equity_metrics"]
    ts = report["trade_stats"]
    cls = aux["classification_metrics"]

    print(f"[INFO] Train size: {aux['train_size']}, Test size: {aux['test_size']}")
    print("\n=== ML-based strategy backtest (15m) ===")
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
    print(f"  Avg Win      : {ts['avg_win']}")
    print(f"  Avg Loss     : {ts['avg_loss']}")
    print(f"  ProfitFactor : {ts['profit_factor']}")
    print(f"  Median hold  : {ts['median_hold_time']}")

    print("\nClassification metrics (threshold=0.50):")
    print(f"  precision: {cls['precision']:.4f}")
    print(f"  recall   : {cls['recall']:.4f}")
    print(f"  f1       : {cls['f1']:.4f}")
    print(f"  accuracy : {cls['accuracy']:.4f}")
    risk_stats = report.get("risk_stats", {}) or {}
    blocked_total = sum(risk_stats.get("blocked_entries", {}).values())
    print(f"[RISK] blocked_entries_total={blocked_total}")

    bt_dir, tr_dir = _ensure_output_dirs()
    eq_path = bt_dir / "ml_equity_15m.csv"
    trades_path = tr_dir / "ml_trades_15m.csv"
    report["equity_curve"].to_csv(eq_path, header=True)
    report["trades"].to_csv(trades_path, index=False)
    backtest_cfg = sys_cfg.get("ml", {}).get("backtest", {}) or {}
    if aux.get("model_meta"):
        if backtest_cfg.get("use_saved_model"):
            print(f"[INFO] Evaluated saved model -> {aux['model_meta'].model_id}")
        else:
            print(f"[INFO] Saved model -> {aux['model_meta'].model_id}")
    print(f"\n[INFO] Saved {eq_path} and {trades_path}")


if __name__ == "__main__":
    main()
