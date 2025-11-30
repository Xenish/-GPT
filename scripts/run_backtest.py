from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from finantradealgo.backtester.backtest_engine import BacktestConfig, Backtester
from finantradealgo.data_engine.loader import load_ohlcv_csv
from finantradealgo.features.base_features import FeatureConfig, add_basic_features
from finantradealgo.core.report import ReportConfig, generate_report
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.strategies.ema_cross import EMACrossStrategy
from finantradealgo.system.config_loader import load_system_config



def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest with EMA Cross strategy")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: from FT_CONFIG_PATH env or config/system.yml)"
    )
    args = parser.parse_args()

    # Load config with profile support
    sys_cfg = load_system_config(path=args.config)

    # SAFETY: Assert research mode for backtest
    cfg_mode = sys_cfg.get("mode", "unknown")
    if cfg_mode != "research":
        raise RuntimeError(
            f"Backtest must run with mode='research' config. Got mode='{cfg_mode}'. "
            f"Use --config config/system.research.yml or set FT_CONFIG_PATH=config/system.research.yml"
        )
    data_cfg = sys_cfg.get("data", {})
    symbol = sys_cfg.get("symbol", "BTCUSDT")
    timeframe = sys_cfg.get("timeframe", "15m")
    ohlcv_path = Path(data_cfg.get("ohlcv_dir", "data/ohlcv")) / f"{symbol}_{timeframe}.csv"
    df = load_ohlcv_csv(str(ohlcv_path))

    feat_config = FeatureConfig()
    df_feat = add_basic_features(df, feat_config)

    strategy = EMACrossStrategy(fast=20, slow=50)
    risk_engine = RiskEngine(RiskConfig.from_dict(sys_cfg.get("risk")))
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
