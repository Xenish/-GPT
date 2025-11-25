from __future__ import annotations

from typing import Sequence

import pandas as pd

from finantradealgo.core.backtest import BacktestConfig, Backtester
from finantradealgo.core.data import load_ohlcv_csv
from finantradealgo.core.features import FeatureConfig, add_basic_features
from finantradealgo.core.report import ReportConfig, generate_report
from finantradealgo.core.risk import RiskConfig, RiskEngine
from finantradealgo.ml.labels import LabelConfig, add_long_only_labels
from finantradealgo.ml.model import SklearnLongModel, SklearnModelConfig
from finantradealgo.strategies.ml_signal import MLSignalStrategy


def train_rf_model_15m(csv_path: str = "data/BTCUSDT_P_15m.csv"):
    """
    15 dakikalık veriden:
      - feature'ları ve label'ları üretir
      - %70 train, %30 backtest split yapar
      - RandomForest modeli train eder
    Geriye:
      - eğitilmiş model
      - sadece backtest kısmına ait df (label + feature'lı)
      - feature kolon isimleri
    döner.
    """
    # 1) Data + feature + label
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

    # Label + feature için NaN satırları at
    df_ml = df_lab.dropna(subset=["label_long"] + feature_cols).copy()

    X = df_ml[feature_cols]
    y = df_ml["label_long"].astype(int)

    # 70/30 split
    split_idx = int(len(df_ml) * 0.7)
    X_train, X_bt = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]

    print(f"[INFO] Train size: {len(X_train)}, Backtest size: {len(X_bt)}")

    # 2) Kendi wrapper üzerinden RandomForest seç
    #    (run_ml_model_comparison_15m.py ile aynı mantık)
    model_config = SklearnModelConfig(model_type="rf")
    model = SklearnLongModel(model_config)
    model.fit(X_train, y_train)

    # Backtest'te kullanılacak df: out-of-sample kısmı
    df_bt = df_ml.iloc[split_idx:].copy()

    return model, df_bt, feature_cols


def run_backtest_for_thresholds(
    thresholds: Sequence[float],
    initial_cash: float = 10_000.0,
    csv_path: str = "data/BTCUSDT_P_15m.csv",
    out_csv: str = "ml_rf_backtest_thresholds_15m.csv",
) -> None:
    """
    Aynı RF modelini kullanarak farklı proba_entry threshold değerleri için
    backtest çalıştırır, özet metrikleri CSV'ye yazar.
    """
    model, df_bt, feature_cols = train_rf_model_15m(csv_path=csv_path)

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

        strategy = MLSignalStrategy(
            model=model,
            feature_cols=feature_cols,
            proba_entry=th,
            proba_exit=0.50,  # sabit exit threshold (istersek sonra oynarız)
            warmup_bars=100,
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
        csv_path="data/BTCUSDT_P_15m.csv",
        out_csv="ml_rf_backtest_thresholds_15m.csv",
    )


if __name__ == "__main__":
    main()
