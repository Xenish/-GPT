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


def main() -> None:
    df = load_ohlcv_csv("data/ohlcv/BTCUSDT_15m.csv")

    # 1) Mevcut basic feature'lar
    feat_cfg = FeatureConfig()
    df_basic = add_basic_features(df, feat_cfg)

    # 2) TA feature'lar (A grubu)
    ta_cfg = TAFeatureConfig()
    df_ta = add_ta_features(df_basic, ta_cfg)

    # 3) Candlestick pattern feature'lar (B grubu)
    c_cfg = CandleFeatureConfig()
    df_all = add_candlestick_features(df_ta, c_cfg)

    # Kontrol: hangi cdl_* sütunları var?
    cdl_cols = [c for c in df_all.columns if c.startswith("cdl_")]
    cs_cols = [c for c in df_all.columns if c.startswith("cs_")]

    print("Candlestick geometry columns (cs_*):")
    print(cs_cols)

    print("\nPattern columns (cdl_*):")
    print(cdl_cols)

    # Son birkaç barı göster (pattern'lerle birlikte)
    print(
        df_all[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
            ]
            + cs_cols
            + cdl_cols
        ].tail(20)
    )


if __name__ == "__main__":
    main()
