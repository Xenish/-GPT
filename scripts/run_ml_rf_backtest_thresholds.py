"""
Run a simple RF backtest across multiple entry thresholds using the standard
ML proba/signal column names (`ml_long_proba`).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.backtester.backtest_engine import BacktestConfig, Backtester
from finantradealgo.core.report import ReportConfig, generate_report
from finantradealgo.data_engine.loader import load_ohlcv_csv
from finantradealgo.features.base_features import FeatureConfig, add_basic_features
from finantradealgo.ml.labels import LabelConfig, add_long_only_labels
from finantradealgo.ml.model import SklearnLongModel, SklearnModelConfig
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.strategies.ml_strategy import MLSignalStrategy, MLStrategyConfig


def train_rf_model_15m(csv_path: str = "data/ohlcv/BTCUSDT_15m.csv"):
    """
    Prepare features/labels from a 15m CSV and train a RandomForest.
    Returns the trained model, the out-of-sample DataFrame, and feature columns.
    """
    df = load_ohlcv_csv(csv_path)

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

    df_ml = df_lab.dropna(subset=["label_long"] + feature_cols).copy()

    X = df_ml[feature_cols]
    y = df_ml["label_long"].astype(int)

    split_idx = int(len(df_ml) * 0.7)
    X_train, _ = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]

    print(f"[INFO] Train size: {len(X_train)}, Backtest size: {len(df_ml) - len(X_train)}")

    model_config = SklearnModelConfig(model_type="random_forest")
    model = SklearnLongModel(model_config)
    model.fit(X_train, y_train)

    df_bt = df_ml.iloc[split_idx:].copy()

    return model, df_bt, feature_cols


def run_backtest_for_thresholds(
    thresholds: Sequence[float],
    initial_cash: float = 10_000.0,
    csv_path: str = "data/ohlcv/BTCUSDT_15m.csv",
    out_csv: str = "ml_rf_backtest_thresholds_15m.csv",
) -> None:
    """
    Run backtests for a list of probability entry thresholds and export summary.
    """
    model, df_bt, feature_cols = train_rf_model_15m(csv_path=csv_path)
    df_bt = df_bt.copy()
    df_bt["ml_long_proba"] = model.predict_proba(df_bt[feature_cols])[:, 1].astype(float)

    results = []

    for th in thresholds:
        print(f"\n=== Backtest for proba_entry={th:.2f} ===")

        risk_engine = RiskEngine(
            RiskConfig(
                risk_per_trade=0.01,
                stop_loss_pct=0.01,
                max_leverage=1.0,
            )
        )

        bt_config = BacktestConfig(
            initial_cash=initial_cash,
            fee_pct=0.0004,
            slippage_pct=0.0005,
        )

        strategy_cfg = MLStrategyConfig(
            proba_col="ml_long_proba",
            entry_threshold=th,
            exit_threshold=0.50,
            warmup_bars=0,
            side="long_only",
        )
        strategy = MLSignalStrategy(strategy_cfg)

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

        print("Equity:")
        print(f"  Final equity : {eq['final_equity']}")
        print(f"  Cumulative R.: {eq['cum_return']}")
        print(f"  Max drawdown : {eq['max_drawdown']}")
        print(f"  Sharpe       : {eq['sharpe']}")

        print("Trades:")
        print(f"  Count        : {ts['trade_count']}")
        print(f"  Win rate     : {ts['win_rate']}")
        print(f"  Avg PnL      : {ts['avg_pnl']}")
        print(f"  Avg Win      : {ts['avg_win']}")
        print(f"  Avg Loss     : {ts['avg_loss']}")
        print(f"  ProfitFactor : {ts['profit_factor']}")

        results.append(
            {
                "proba_entry": th,
                "final_equity": eq["final_equity"],
                "cum_return": eq["cum_return"],
                "max_drawdown": eq["max_drawdown"],
                "sharpe": eq["sharpe"],
                "trade_count": ts["trade_count"],
                "win_rate": ts["win_rate"],
                "avg_pnl": ts["avg_pnl"],
                "avg_win": ts["avg_win"],
                "avg_loss": ts["avg_loss"],
                "profit_factor": ts["profit_factor"],
            }
        )

    df_summary = pd.DataFrame(results)
    df_summary.to_csv(out_csv, index=False)
    print(f"\nSaved RF backtest threshold summary to: {out_csv}")


def main() -> None:
    thresholds = [0.50, 0.55, 0.60]
    run_backtest_for_thresholds(
        thresholds=thresholds,
        initial_cash=10_000.0,
        csv_path="data/ohlcv/BTCUSDT_15m.csv",
        out_csv="ml_rf_backtest_thresholds_15m.csv",
    )


if __name__ == "__main__":
    main()
