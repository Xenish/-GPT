from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from typing import Dict, Tuple

from sklearn.metrics import classification_report

from finantradealgo.backtester.backtest_engine import BacktestConfig, Backtester
from finantradealgo.data_engine.loader import load_ohlcv_csv
from finantradealgo.features.base_features import FeatureConfig, add_basic_features
from finantradealgo.core.report import ReportConfig, generate_report
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.ml.labels import LabelConfig, add_long_only_labels
from finantradealgo.ml.model import SklearnModelConfig
from finantradealgo.ml.walkforward import WalkForwardConfig, add_walkforward_ml_signals
from finantradealgo.strategies.signal_column import (
    ColumnSignalStrategy,
    ColumnSignalStrategyConfig,
)


def build_model_configs() -> Dict[str, SklearnModelConfig]:
    return {
        "gbm": SklearnModelConfig(
            model_type="gradient_boosting",
            params={
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 3,
            },
        ),
        "logreg": SklearnModelConfig(
            model_type="logreg",
            params={
                "C": 0.1,
                "class_weight": "balanced",
                "max_iter": 200,
            },
        ),
        "rf": SklearnModelConfig(
            model_type="random_forest",
            params={
                "n_estimators": 300,
                "max_depth": 5,
                "min_samples_leaf": 5,
                "class_weight": "balanced_subsample",
            },
        ),
        "xgb": SklearnModelConfig(
            model_type="xgboost",
            params={
                "n_estimators": 400,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        ),
    }


def run_for_model(
    model_name: str,
    model_cfg: SklearnModelConfig,
    df_base,
    feature_cols,
) -> Tuple[dict | None, dict | None]:
    print(f"\n\n================ {model_name} ================")

    df_clean = df_base.dropna(subset=["label_long"] + feature_cols).copy()

    wf_cfg = WalkForwardConfig(
        initial_train_size=3000,
        train_window=3000,
        retrain_every=50,
        proba_entry=0.55,
        model_config=model_cfg,
    )

    df_wf, wf_metrics = add_walkforward_ml_signals(
        df_clean,
        feature_cols=feature_cols,
        label_col="label_long",
        config=wf_cfg,
        log_metrics=True,
    )

    if wf_metrics is not None and not wf_metrics.empty:
        print("Walk-forward ML metrics (avg):")
        print(
            wf_metrics[["precision", "recall", "accuracy", "f1"]]
            .mean()
            .round(4)
        )

    mask = ~df_wf["ml_signal_long"].isna()
    y_true = df_wf.loc[mask, "label_long"].astype(int)
    y_pred = df_wf.loc[mask, "ml_signal_long"].astype(int)

    if not y_true.empty:
        print("\nOverall classification report:")
        print(classification_report(y_true, y_pred, digits=3))
    else:
        print("\nNo predictions to evaluate.")

    df_bt = df_wf.loc[mask].copy()
    if len(df_bt) == 0:
        print("  WARNING: No tradable bars for this model.")
        return None, None

    risk_engine = RiskEngine(
        RiskConfig(
            risk_per_trade=0.01,
            stop_loss_pct=0.01,
            max_leverage=1.0,
        )
    )

    bt_config = BacktestConfig(
        initial_cash=10_000.0,
        fee_pct=0.0004,
        slippage_pct=0.0005,
    )

    strategy = ColumnSignalStrategy(
        ColumnSignalStrategyConfig(signal_col="ml_signal_long", warmup_bars=0)
    )

    backtester = Backtester(
        strategy=strategy,
        risk_engine=risk_engine,
        config=bt_config,
    )

    result = backtester.run(df_bt)

    report_cfg = ReportConfig(regime_columns=["regime_trend", "regime_vol"])
    report = generate_report(result, df=df_bt, config=report_cfg)

    eq = report["equity_metrics"]
    ts = report["trade_stats"]

    print("\nBacktest metrics:")
    print(f"  Final equity : {eq['final_equity']}")
    print(f"  Cum return   : {eq['cum_return']}")
    print(f"  MaxDD        : {eq['max_drawdown']}")
    print(f"  Sharpe       : {eq['sharpe']}")
    print(f"  Trades       : {ts['trade_count']}")
    print(f"  Win rate     : {ts['win_rate']}")
    print(f"  ProfitFactor : {ts['profit_factor']}")

    return wf_metrics, report


def main() -> None:
    df = load_ohlcv_csv("data/BTCUSDT_P_15m.csv")
    df = df.tail(10_000).copy()

    feat_config = FeatureConfig()
    df_feat = add_basic_features(df, feat_config)

    label_config = LabelConfig(
        horizon=5,
        pos_threshold=0.003,
        fee_slippage=0.001,
    )
    df_lab = add_long_only_labels(df_feat, label_config)

    feature_cols = [
        "ret_1",
        "ret_3",
        "ret_5",
        "ret_10",
        "vol_10",
        "vol_20",
        "trend_score",
    ]

    model_cfgs = build_model_configs()

    summary_rows = []
    for name, cfg in model_cfgs.items():
        wf_metrics, report = run_for_model(name, cfg, df_lab, feature_cols)
        if wf_metrics is None or report is None:
            continue

        eq = report["equity_metrics"]
        ts = report["trade_stats"]

        summary_rows.append(
            {
                "model": name,
                "wf_precision_mean": wf_metrics["precision"].mean(),
                "wf_recall_mean": wf_metrics["recall"].mean(),
                "wf_f1_mean": wf_metrics["f1"].mean(),
                "final_equity": eq["final_equity"],
                "cum_return": eq["cum_return"],
                "max_drawdown": eq["max_drawdown"],
                "sharpe": eq["sharpe"],
                "trade_count": ts["trade_count"],
                "win_rate": ts["win_rate"],
                "profit_factor": ts["profit_factor"],
            }
        )

    if summary_rows:
        import pandas as pd

        df_summary = pd.DataFrame(summary_rows)
        print("\n\n================ MODEL COMPARISON SUMMARY ================")
        print(
            df_summary[
                [
                    "model",
                    "wf_precision_mean",
                    "wf_recall_mean",
                    "wf_f1_mean",
                    "final_equity",
                    "cum_return",
                    "max_drawdown",
                    "sharpe",
                    "trade_count",
                    "win_rate",
                    "profit_factor",
                ]
            ]
        )
        df_summary.to_csv("ml_model_comparison_15m.csv", index=False)


if __name__ == "__main__":
    main()
