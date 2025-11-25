from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from finantradealgo.backtester.backtest_engine import BacktestConfig, Backtester
from finantradealgo.features.feature_pipeline_15m import (
    FeaturePipelineConfig,
    build_feature_pipeline_15m,
)
from finantradealgo.core.report import ReportConfig, generate_report
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.strategies.rule_signals import RuleSignalStrategy, RuleStrategyConfig


def run_rule_backtest_15m() -> None:
    symbol = "BTCUSDT"
    ohlcv_path = Path("data/ohlcv") / f"{symbol}_15m.csv"
    funding_path = Path("data/external/funding") / f"{symbol}_funding_15m.csv"
    oi_path = Path("data/external/open_interest") / f"{symbol}_oi_15m.csv"

    pipe_cfg = FeaturePipelineConfig(
        rule_allowed_hours=list(range(8, 18)),
        rule_allowed_weekdays=[0, 1, 2, 3, 4],
        use_rule_signals=True,
    )
    df, _ = build_feature_pipeline_15m(
        csv_ohlcv_path=str(ohlcv_path),
        pipeline_cfg=pipe_cfg,
        csv_funding_path=str(funding_path) if funding_path.exists() else None,
        csv_oi_path=str(oi_path) if oi_path.exists() else None,
    )
    print(f"[INFO] Prepared DF shape: {df.shape}")

    strat_cfg = RuleStrategyConfig(
        entry_col="rule_long_entry",
        exit_col="rule_long_exit",
        warmup_bars=pipe_cfg.rule_allowed_hours[0] if pipe_cfg.rule_allowed_hours else 50,
    )
    strategy = RuleSignalStrategy(strat_cfg)

    risk_engine = RiskEngine(
        RiskConfig(
            risk_per_trade=0.01,
            stop_loss_pct=0.02,
            max_leverage=1.0,
        )
    )

    bt_config = BacktestConfig(
        initial_cash=10_000.0,
        fee_pct=0.0004,
        slippage_pct=0.0005,
    )

    backtester = Backtester(
        strategy=strategy,
        risk_engine=risk_engine,
        config=bt_config,
    )

    result = backtester.run(df)

    report_cfg = ReportConfig(regime_columns=["regime_trend", "regime_vol"])
    report = generate_report(result, df=df, config=report_cfg)

    eq = report["equity_metrics"]
    ts = report["trade_stats"]

    print("\n=== Rule-based strategy backtest (15m) ===")
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

    report["equity_curve"].to_csv("rule_equity_curve_15m.csv", header=True)
    report["trades"].to_csv("rule_trades_15m.csv", index=False)
    print("\n[INFO] Saved rule_equity_curve_15m.csv and rule_trades_15m.csv")


if __name__ == "__main__":
    run_rule_backtest_15m()
