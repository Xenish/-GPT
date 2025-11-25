from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from finantradealgo.data_engine.loader import load_ohlcv_csv
from finantradealgo.features.base_features import FeatureConfig, add_basic_features
from finantradealgo.features.ta_features import TAFeatureConfig, add_ta_features
from finantradealgo.features.candle_features import CandleFeatureConfig, add_candlestick_features
from finantradealgo.features.osc_features import OscFeatureConfig, add_osc_features
from finantradealgo.features.multi_tf_features import MultiTFConfig, add_multitf_1h_features
from finantradealgo.features.rule_signals import RuleSignalConfig, add_rule_signals_v1


def main() -> None:
    df = load_ohlcv_csv("data/ohlcv/BTCUSDT_15m.csv")

    # A: basic
    feat_cfg = FeatureConfig()
    df_basic = add_basic_features(df, feat_cfg)

    # A+: TA
    ta_cfg = TAFeatureConfig()
    df_ta = add_ta_features(df_basic, ta_cfg)

    # B: candlestick
    c_cfg = CandleFeatureConfig()
    df_c = add_candlestick_features(df_ta, c_cfg)

    # C: oscillators
    o_cfg = OscFeatureConfig()
    df_osc = add_osc_features(df_c, o_cfg)

    # D: HTF 1h
    mtf_cfg = MultiTFConfig()
    df_mtf = add_multitf_1h_features(df_osc, mtf_cfg)

    # E: rule-based sinyaller
    r_cfg = RuleSignalConfig()
    df_rule = add_rule_signals_v1(df_mtf, r_cfg)

    print("Columns with rule signals:")
    print([c for c in df_rule.columns if c.startswith("rule_")])

    # Entry / exit görünen son 40 bar
    mask = (df_rule["rule_long_entry"] == 1) | (df_rule["rule_long_exit"] == 1)

    cols_to_show = [
        "timestamp",
        "close",
        "rule_long_entry",
        "rule_long_exit",
        "htf1h_trend_score",
        "htf1h_rsi_14",
        "htf1h_hv_20",
        "rsi_14",
        "stoch_k_14_3",
        "stoch_d_14_3",
        "macd_hist_12_26_9",
        "cs_body_pct",
        "cs_bull",
        "cdl_bull_engulf",
        "cdl_hammer",
        "cdl_marubozu",
    ]

    print("\nRecent bars with any rule signal (entry or exit):")
    print(df_rule.loc[mask, cols_to_show].tail(40)),

# ==== Genel istatistikler ====
    total_bars = len(df_rule)
    n_entry = int(df_rule["rule_long_entry"].sum())
    n_exit = int(df_rule["rule_long_exit"].sum())

    print("\n=== Rule v1 signal stats ===")
    print(f"Total bars : {total_bars}")
    print(f"Long entries: {n_entry}")
    print(f"Long exits  : {n_exit}")

    if n_entry > 0:
        print("\nSample entry bars:")
        print(
            df_rule.loc[df_rule["rule_long_entry"] == 1, cols_to_show]
            .head(10)
        )

    if n_exit > 0:
        print("\nSample exit bars:")
        print(
            df_rule.loc[df_rule["rule_long_exit"] == 1, cols_to_show]
            .head(10)
        )


if __name__ == "__main__":
    main()
