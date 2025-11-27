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

    # D: 1h HTF features
    mtf_cfg = MultiTFConfig()
    df_all = add_multitf_1h_features(df_osc, mtf_cfg)

    htf_cols = [c for c in df_all.columns if c.startswith("htf1h_")]

    print("HTF (1h) columns:")
    print(htf_cols)

    print("\nTail with HTF features:")
    cols_to_show = ["timestamp", "close"] + htf_cols
    print(df_all[cols_to_show].tail(10))


if __name__ == "__main__":
    main()
