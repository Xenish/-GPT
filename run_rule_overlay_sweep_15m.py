from __future__ import annotations

import itertools
from typing import List, Dict, Any

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


def prepare_df_15m(csv_path: str) -> pd.DataFrame:
    """
    15m datası için:
      - basic features
      - TA features
      - candle features
      - oscillators
      - HTF 1h features
      - rule_long_entry / rule_long_exit
    hepsini uygular.
    """

    df = load_ohlcv_csv(csv_path)

    # 1) Basic
    feat_cfg = FeatureConfig()
    df = add_basic_features(df, feat_cfg)

    # 2) TA
    ta_cfg = TAFeatureConfig()
    df = add_ta_features(df, ta_cfg)

    # 3) Candlestick
    c_cfg = CandleFeatureConfig()
    df = add_candlestick_features(df, c_cfg)

    # 4) Oscillators
    o_cfg = OscFeatureConfig()
    df = add_osc_features(df, o_cfg)

    # 5) HTF 1h
    m_cfg = MultiTFConfig()
    df = add_multitf_1h_features(df, m_cfg)

    # 6) Rule signals (entry/exit)
    rule_cfg = RuleSignalConfig(
        allowed_hours=list(range(8, 18)),   # 08:00–17:45 arası
        allowed_weekdays=[0, 1, 2, 3, 4],   # Mon–Fri
    )
    df = add_rule_signals_v1(df, rule_cfg)

    # NaN temizliği (özellikle indicator warmup dönemleri)
    df = df.dropna().reset_index(drop=True)

    return df


def run_one_backtest(df: pd.DataFrame, cfg: RuleStrategyConfig) -> Dict[str, Any]:
    """
    Verilen RuleStrategyConfig ile tek bir backtest çalıştırır, metrikleri döner.
    """
    # Risk Engine
    risk_engine = RiskEngine(
        RiskConfig(
            risk_per_trade=0.01,
            stop_loss_pct=0.01,   # burada gerçek SL'i ATR overlay yapıyor zaten
            max_leverage=1.0,
        )
    )

    # Backtest config
    bt_cfg = BacktestConfig(
        initial_cash=10_000.0,
        fee_pct=0.0004,
        slippage_pct=0.0005,
    )

    # Strategy
    strategy = RuleSignalStrategy(config=cfg)

    backtester = Backtester(
        strategy=strategy,
        risk_engine=risk_engine,
        config=bt_cfg,
    )

    result = backtester.run(df)

    report_cfg = ReportConfig(regime_columns=["regime_trend", "regime_vol"])
    report = generate_report(result, df=df, config=report_cfg)

    eq = report["equity_metrics"]
    ts = report["trade_stats"]

    out: Dict[str, Any] = {
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
        "median_hold_time": ts["median_hold_time"],
    }

    return out


def main() -> None:
    csv_path = "data/BTCUSDT_P_15m.csv"

    print("[INFO] Preparing DF with all features + rule signals...")
    df = prepare_df_15m(csv_path)
    print(f"[INFO] Prepared DF shape: {df.shape}")

    # Sweep edilecek parametre grid'i
    tp_atr_list = [1.0, 1.5, 2.0, 2.5]
    sl_atr_list = [0.5, 1.0, 1.5]
    max_hold_list = [4 * 4, 8 * 4, 24 * 4]  # 1 saat, 2 saat, 1 gün (15m bar)

    records: List[Dict[str, Any]] = []

    for tp_mult, sl_mult, max_hold in itertools.product(tp_atr_list, sl_atr_list, max_hold_list):
        cfg = RuleStrategyConfig(
            entry_col="rule_long_entry",
            exit_col="rule_long_exit",
            warmup_bars=50,
            max_hold_bars=max_hold,
            use_rule_exit=True,
            use_atr_tp_sl=True,
            atr_col="atr_14_pct",
            tp_atr_mult=tp_mult,
            sl_atr_mult=sl_mult,
        )

        print(
            f"\n[INFO] Running backtest for "
            f"tp_atr_mult={tp_mult}, sl_atr_mult={sl_mult}, max_hold_bars={max_hold}..."
        )

        metrics = run_one_backtest(df, cfg)

        row = {
            "tp_atr_mult": tp_mult,
            "sl_atr_mult": sl_mult,
            "max_hold_bars": max_hold,
        }
        row.update(metrics)
        records.append(row)

        print(
            f"  -> cum_return={metrics['cum_return']:.4f}, "
            f"sharpe={metrics['sharpe']:.3f}, "
            f"PF={metrics['profit_factor']:.3f}, "
            f"trades={metrics['trade_count']}"
        )

    df_res = pd.DataFrame(records)
    out_csv = "rule_overlay_sweep_15m.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved overlay sweep results to: {out_csv}")
    print(df_res.sort_values("sharpe", ascending=False).head(10))


if __name__ == "__main__":
    main()
