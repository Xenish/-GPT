from __future__ import annotations

import pandas as pd

from finantradealgo.core.data import load_ohlcv_csv
from finantradealgo.core.features import FeatureConfig, add_basic_features
from finantradealgo.core.ta_features import TAFeatureConfig, add_ta_features
from finantradealgo.core.candle_features import CandleFeatureConfig, add_candlestick_features
from finantradealgo.core.osc_features import OscFeatureConfig, add_osc_features
from finantradealgo.core.multi_tf_features import MultiTFConfig, add_multitf_1h_features
from finantradealgo.core.rule_signals import RuleSignalConfig, add_rule_signals_v1

from finantradealgo.core.risk import RiskConfig, RiskEngine
from finantradealgo.core.backtest import BacktestConfig, Backtester
from finantradealgo.core.report import ReportConfig, generate_report

from finantradealgo.strategies.rule_signals import RuleSignalStrategy, RuleStrategyConfig


def prepare_rule_df_15m(csv_path: str) -> pd.DataFrame:
    """
    15m AIAUSDT datasından:
      - basic features
      - TA features
      - candle features
      - oscillator features
      - multi-TF (1h) features
      - rule-based entry/exit sinyalleri
    ekleyip, NaN'leri temizleyip DataFrame döndürür.
    """
    df = load_ohlcv_csv(csv_path)

    # 1) Basic features
    feat_cfg = FeatureConfig()
    df = add_basic_features(df, feat_cfg)

    # 2) TA features
    ta_cfg = TAFeatureConfig()
    df = add_ta_features(df, ta_cfg)

    # 3) Candlestick geometry + patterns
    c_cfg = CandleFeatureConfig()
    df = add_candlestick_features(df, c_cfg)

    # 4) Oscillators
    o_cfg = OscFeatureConfig()
    df = add_osc_features(df, o_cfg)

    # 5) Higher timeframe (1h) features
    mtf_cfg = MultiTFConfig()
    df = add_multitf_1h_features(df, mtf_cfg)

    # 6) Rule-based sinyaller (rule_long_entry / rule_long_exit)
    rule_cfg = RuleSignalConfig(
        allowed_hours=list(range(8, 18)),   
        allowed_weekdays=[0, 1, 2, 3, 4],  
    )
    df = add_rule_signals_v1(df, rule_cfg)

    # Bütün pipeline sonrası NaN temizliği
    df = df.dropna().reset_index(drop=True)

    return df


def run_rule_backtest_15m() -> None:
    csv_path = "data/AIAUSDT_P_15m.csv"

    df = prepare_rule_df_15m(csv_path)
    print(f"[INFO] Prepared DF shape: {df.shape}")
    print(f"[INFO] Columns include rule signals? "
          f"entry='rule_long_entry' in cols: {'rule_long_entry' in df.columns}, "
          f"exit='rule_long_exit' in cols: {'rule_long_exit' in df.columns}")

    # 1) Rule strategy
    strat_cfg = RuleStrategyConfig(
        entry_col="rule_long_entry",
        exit_col="rule_long_exit",
        warmup_bars=100,
    )
    strategy = RuleSignalStrategy(strat_cfg)

    # 2) Risk engine
    risk_engine = RiskEngine(
        RiskConfig(
            risk_per_trade=0.01,   # sermayenin %1'i risk
            stop_loss_pct=0.02,    # %2 stop
            max_leverage=1.0,
        )
    )

    # 3) Backtest config
    bt_config = BacktestConfig(
        initial_cash=10_000.0,
        fee_pct=0.0004,
        slippage_pct=0.0005,
    )

    # 4) Backtester
    backtester = Backtester(
        strategy=strategy,
        risk_engine=risk_engine,
        config=bt_config,
    )

    result = backtester.run(df)

    # 5) Rapor
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

    # İstersen CSV'ye de at
    report["equity_curve"].to_csv("rule_equity_curve_15m.csv", header=True)
    report["trades"].to_csv("rule_trades_15m.csv", index=False)
    print("\n[INFO] Saved rule_equity_curve_15m.csv and rule_trades_15m.csv")


if __name__ == "__main__":
    run_rule_backtest_15m()
