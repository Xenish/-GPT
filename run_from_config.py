from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

from finantradealgo.core.backtest import BacktestConfig, Backtester
from finantradealgo.core.data import load_ohlcv_csv
from finantradealgo.core.features import FeatureConfig, add_basic_features
from finantradealgo.core.report import ReportConfig, generate_report
from finantradealgo.core.risk import RiskConfig, RiskEngine
from finantradealgo.data_sources import BinanceKlinesConfig, fetch_klines
from finantradealgo.ml.labels import LabelConfig, add_long_only_labels
from finantradealgo.ml.model import SklearnLongModel, SklearnModelConfig
from finantradealgo.strategies.ema_cross import EMACrossStrategy
from finantradealgo.strategies.ml_signal import MLSignalStrategy
from finantradealgo.strategies.rsi_reversion import RSIReversionStrategy


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config structure in {path}")
    return cfg


def build_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    data_cfg = cfg.get("data", {})
    source = data_cfg.get("source", "csv")

    if source == "csv":
        path = data_cfg.get("path")
        if not path:
            raise ValueError("data.path is required for source=csv")
        return load_ohlcv_csv(path)

    if source == "binance":
        symbol = data_cfg.get("symbol", "BTCUSDT")
        interval = data_cfg.get("interval", "1h")
        limit = int(data_cfg.get("limit", 1000))
        bcfg = BinanceKlinesConfig(symbol=symbol, interval=interval, limit=limit)
        return fetch_klines(bcfg)

    raise ValueError(f"Unknown data.source: {source}")


def apply_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    feat_cfg = cfg.get("features", {})
    if not feat_cfg.get("enabled", True):
        return df

    fc = FeatureConfig(
        return_windows=tuple(feat_cfg.get("return_windows", (1, 3, 5, 10))),
        vol_windows=tuple(feat_cfg.get("vol_windows", (10, 20))),
        ema_fast=int(feat_cfg.get("ema_fast", 20)),
        ema_slow=int(feat_cfg.get("ema_slow", 50)),
        trend_threshold=float(feat_cfg.get("trend_threshold", 0.001)),
    )
    return add_basic_features(df, fc)


def apply_labels(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    label_cfg = cfg.get("labels", {})
    if not label_cfg.get("enabled", False):
        return df

    lc = LabelConfig(
        horizon=int(label_cfg.get("horizon", 5)),
        pos_threshold=float(label_cfg.get("pos_threshold", 0.003)),
        fee_slippage=float(label_cfg.get("fee_slippage", 0.001)),
    )
    return add_long_only_labels(df, lc)


def train_model_if_needed(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[SklearnLongModel | None, List[str] | None]:
    model_cfg = cfg.get("model", {})
    if not model_cfg.get("enabled", False):
        return None, None

    feature_cols: List[str] = model_cfg.get(
        "feature_cols",
        [
            "ret_1",
            "ret_3",
            "ret_5",
            "ret_10",
            "vol_10",
            "vol_20",
            "trend_score",
        ],
    )

    if "label_long" not in df.columns:
        raise ValueError("label_long column not found; enable labels to train model.")

    df_ml = df.dropna(subset=["label_long"] + feature_cols).copy()
    if df_ml.empty:
        raise ValueError("No rows with non-NaN label_long to train model.")

    X = df_ml[feature_cols]
    y = df_ml["label_long"].astype(int)

    train_ratio = float(model_cfg.get("train_ratio", 0.7))
    split_idx = max(int(len(df_ml) * train_ratio), 1)
    split_idx = min(split_idx, len(df_ml) - 1)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    sk_cfg = SklearnModelConfig(
        n_estimators=int(model_cfg.get("n_estimators", 200)),
        learning_rate=float(model_cfg.get("learning_rate", 0.05)),
        max_depth=int(model_cfg.get("max_depth", 3)),
        random_state=int(model_cfg.get("random_state", 42)),
    )

    model = SklearnLongModel(sk_cfg)
    model.fit(X_train, y_train)

    if len(X_test) > 0:
        metrics = model.evaluate(X_test, y_test)
        print("=== ML Classification Metrics ===")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    return model, feature_cols


def build_strategy(cfg: Dict[str, Any], model, feature_cols):
    strat_cfg = cfg.get("strategy", {})
    strat_type = strat_cfg.get("type")

    if strat_type == "ema_cross":
        params = strat_cfg.get("params", {})
        fast = int(params.get("fast", 20))
        slow = int(params.get("slow", 50))
        return EMACrossStrategy(fast=fast, slow=slow)

    if strat_type == "ml_long":
        if model is None or feature_cols is None:
            raise ValueError("strategy.type=ml_long requires model.enabled=true.")
        params = strat_cfg.get("params", {})
        proba_entry = float(params.get("proba_entry", 0.55))
        proba_exit = float(params.get("proba_exit", 0.50))
        warmup_bars = int(params.get("warmup_bars", 100))
        return MLSignalStrategy(
            model=model,
            feature_cols=feature_cols,
            proba_entry=proba_entry,
            proba_exit=proba_exit,
            warmup_bars=warmup_bars,
        )

    if strat_type == "rsi_reversion":
        params = strat_cfg.get("params", {})
        period = int(params.get("period", 14))
        oversold = float(params.get("oversold", 30.0))
        overbought = float(params.get("overbought", 70.0))
        warmup_bars = int(params.get("warmup_bars", 50))
        return RSIReversionStrategy(
            period=period,
            oversold=oversold,
            overbought=overbought,
            warmup_bars=warmup_bars,
        )

    raise ValueError(f"Unknown strategy.type: {strat_type}")


def build_risk_engine(cfg: Dict[str, Any]) -> RiskEngine:
    risk_cfg = cfg.get("risk", {})
    rc = RiskConfig(
        risk_per_trade=float(risk_cfg.get("risk_per_trade", 0.01)),
        stop_loss_pct=float(risk_cfg.get("stop_loss_pct", 0.01)),
        max_leverage=float(risk_cfg.get("max_leverage", 1.0)),
    )
    return RiskEngine(rc)


def build_backtester(cfg: Dict[str, Any], strategy, risk_engine: RiskEngine) -> Backtester:
    bt_cfg = cfg.get("backtest", {})
    bc = BacktestConfig(
        initial_cash=float(bt_cfg.get("initial_cash", 10_000.0)),
        fee_pct=float(bt_cfg.get("fee_pct", 0.0004)),
        slippage_pct=float(bt_cfg.get("slippage_pct", 0.0005)),
    )
    return Backtester(strategy=strategy, risk_engine=risk_engine, config=bc)


def run_from_config(path: str | Path) -> None:
    cfg = load_yaml_config(path)

    df = build_data(cfg)
    df = apply_features(df, cfg)
    df = apply_labels(df, cfg)

    model, feature_cols = train_model_if_needed(df, cfg)
    strategy = build_strategy(cfg, model, feature_cols)

    risk_engine = build_risk_engine(cfg)
    backtester = build_backtester(cfg, strategy, risk_engine)

    result = backtester.run(df)

    report_cfg = cfg.get("report", {})
    regime_cols = report_cfg.get("regime_columns", [])
    report = generate_report(
        backtest_result=result,
        df=df,
        config=ReportConfig(regime_columns=regime_cols),
    )

    eq = report["equity_metrics"]
    ts = report["trade_stats"]

    print("\n=== Backtest Report ===")
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

    if report["regime_stats"]:
        print("\n=== Regime stats ===")
        for col, regimes in report["regime_stats"].items():
            print(f"\n  Regime column: {col}")
            for regime_value, stats in regimes.items():
                print(
                    f"    {regime_value}: trades={stats['trade_count']}, "
                    f"win_rate={stats['win_rate']:.2f}, avg_pnl={stats['avg_pnl']:.4f}"
                )

    equity_output = report_cfg.get("equity_output")
    if equity_output:
        Path(equity_output).parent.mkdir(parents=True, exist_ok=True)
        report["equity_curve"].to_csv(equity_output, header=True)

    trades_output = report_cfg.get("trades_output")
    if trades_output:
        Path(trades_output).parent.mkdir(parents=True, exist_ok=True)
        report["trades"].to_csv(trades_output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinanTradeAlgo from YAML config.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/ema_example.yml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    run_from_config(args.config)


if __name__ == "__main__":
    main()
