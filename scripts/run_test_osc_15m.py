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


def main() -> None:
    df = load_ohlcv_csv("data/ohlcv/BTCUSDT_15m.csv")

    # A grubu: basic
    feat_cfg = FeatureConfig()
    df_basic = add_basic_features(df, feat_cfg)

    # A+: TA (trend/vol/ATR/bbands vs.)
    ta_cfg = TAFeatureConfig()
    df_ta = add_ta_features(df_basic, ta_cfg)

    # B: candlestick
    c_cfg = CandleFeatureConfig()
    df_c = add_candlestick_features(df_ta, c_cfg)

    # C: oscillators
    o_cfg = OscFeatureConfig()
    df_all = add_osc_features(df_c, o_cfg)

    osc_cols = [
        c
        for c in df_all.columns
        if c.startswith("rsi_")
        or c.startswith("stoch_")
        or c.startswith("macd_")
        or c.startswith("cci_")
        or c.startswith("mfi_")
    ]

    print("Oscillator columns:")
    print(osc_cols)

    print("\nTail with oscillators:")
    print(df_all[["timestamp", "close"] + osc_cols].tail(10))


if __name__ == "__main__":
    main()
