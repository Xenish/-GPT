from __future__ import annotations

from finantradealgo.core.backtest import BacktestConfig, Backtester
from finantradealgo.core.data import load_ohlcv_csv

from finantradealgo.core.report import ReportConfig, generate_report
from finantradealgo.core.risk import RiskConfig, RiskEngine
from finantradealgo.strategies.ema_cross import EMACrossStrategy
from finantradealgo.core.ta_features import TAFeatureConfig, add_ta_features




def main() -> None:
    df = load_ohlcv_csv("data/AIAUSDT_P_15m.csv")

    feat_config = TAFeatureConfig()
    df_feat = add_ta_features(df, feat_config)

    strategy = EMACrossStrategy(fast=20, slow=50)
    risk_engine = RiskEngine(
        RiskConfig(risk_per_trade=0.01, stop_loss_pct=0.01, max_leverage=1.0)
    )
    bt_config = BacktestConfig(
        initial_cash=10_000.0,
        fee_pct=0.0004,
        slippage_pct=0.0005,
    )
    backtester = Backtester(strategy=strategy, risk_engine=risk_engine, config=bt_config)

    result = backtester.run(df_feat)

    report_cfg = ReportConfig(regime_columns=["regime_trend", "regime_vol"])
    report = generate_report(result, df=df_feat, config=report_cfg)

    eq = report["equity_metrics"]
    ts = report["trade_stats"]

    print("=== EMA Cross Backtest Report ===")
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

    report["equity_curve"].to_csv("equity_curve_ema.csv", header=True)
    report["trades"].to_csv("trades_ema.csv", index=False)


if __name__ == "__main__":
    main()
