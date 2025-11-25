from __future__ import annotations

import pandas as pd

from finantradealgo.core.data import load_ohlcv_csv
from finantradealgo.core.features import FeatureConfig, add_basic_features
from finantradealgo.core.ta_features import TAFeatureConfig, add_ta_features


def main() -> None:
    df = load_ohlcv_csv("data/BTCUSDT_P_15m.csv")

    # 1) Mevcut basic feature’ların eklenmesi (varsa)
    feat_cfg = FeatureConfig()
    df_basic = add_basic_features(df, feat_cfg)

    # 2) Yeni TA feature’larının eklenmesi
    ta_cfg = TAFeatureConfig()
    df_ta = add_ta_features(df_basic, ta_cfg)

    print(df_ta.columns)
    print(df_ta.tail())


if __name__ == "__main__":
    main()
