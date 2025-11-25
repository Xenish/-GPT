from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import os

import pandas as pd

from finantradealgo.core.features import FeatureConfig, add_basic_features
from finantradealgo.core.ta_features import TAFeatureConfig, add_ta_features
from finantradealgo.core.candle_features import CandleFeatureConfig, add_candlestick_features
from finantradealgo.core.osc_features import OscFeatureConfig, add_osc_features
from finantradealgo.core.multi_tf_features import MultiTFConfig, add_multitf_1h_features
from finantradealgo.core.rule_signals import RuleSignalConfig, add_rule_signals_v1


@dataclass
class FeaturePipelineConfig:
    """
    15m feature pipeline ayarları.

    Hepsini True bırak → full feature set.
    İleride ablation yapmak istersen buradan kapatıp açarsın.
    """
    use_basic: bool = True
    use_ta: bool = True
    use_candles: bool = True
    use_osc: bool = True
    use_multitf: bool = True
    use_external: bool = True   # funding + OI CSV'leri
    use_rule_signals: bool = True

    # rule-based strateji için zaman filtreleri
    rule_allowed_hours: Optional[List[int]] = None      # ör: list(range(8,18))
    rule_allowed_weekdays: Optional[List[int]] = None   # ör: [0,1,2,3,4]


def add_external_features_15m(
    df: pd.DataFrame,
    symbol: str,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Funding + OI CSV'lerini timestamp üzerinden merge eder.
    Kolon isimleri neyse aynen DF'e gelir, isimle uğraşmıyoruz.
    """
    ts_col = "timestamp"
    if ts_col not in df.columns:
        return df

    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df.sort_values(ts_col, inplace=True)

    # --- Funding ---
    funding_path = os.path.join(data_dir, f"{symbol}_funding_15m.csv")
    if os.path.exists(funding_path):
        df_f = pd.read_csv(funding_path)
        if ts_col in df_f.columns:
            df_f[ts_col] = pd.to_datetime(df_f[ts_col], utc=True)
            df_f.sort_values(ts_col, inplace=True)

            df = pd.merge_asof(
                df,
                df_f,
                on=ts_col,
                direction="backward",
                suffixes=("", "_fund"),
            )
        else:
            print(f"[WARN] Funding CSV'de '{ts_col}' yok: {funding_path}")
    else:
        print(f"[WARN] Funding CSV bulunamadı: {funding_path}")

    # --- OI ---
    oi_path = os.path.join(data_dir, f"{symbol}_oi_15m.csv")
    if os.path.exists(oi_path):
        df_oi = pd.read_csv(oi_path)
        if ts_col in df_oi.columns:
            df_oi[ts_col] = pd.to_datetime(df_oi[ts_col], utc=True)
            df_oi.sort_values(ts_col, inplace=True)

            df = pd.merge_asof(
                df,
                df_oi,
                on=ts_col,
                direction="backward",
                suffixes=("", "_oi"),
            )
        else:
            print(f"[WARN] OI CSV'de '{ts_col}' yok: {oi_path}")
    else:
        # Şu an durum bu: OI yok, sorun değil. Sonradan düzelince otomatik eklenecek.
        print(f"[WARN] OI CSV bulunamadı: {oi_path}")

    return df


def build_feature_pipeline_15m(
    df_ohlcv: pd.DataFrame,
    symbol: str,
    cfg: Optional[FeaturePipelineConfig] = None,
) -> pd.DataFrame:
    """
    15 dakikalık OHLCV DF'ini alır, üzerine sırayla:

    - basic features
    - TA features
    - candlestick geometry + pattern
    - oscillators
    - multi-TF (1h) özellikler
    - funding + OI varsa dış veri
    - rule_long_entry / rule_long_exit (rule_signals)

    ekler ve tek bir DF döner.
    """

    if cfg is None:
        cfg = FeaturePipelineConfig(
            rule_allowed_hours=list(range(8, 18)),   # 08:00–17:45
            rule_allowed_weekdays=[0, 1, 2, 3, 4],  # Mon–Fri
        )

    df = df_ohlcv.copy()

    # 1) Basic
    if cfg.use_basic:
        feat_cfg = FeatureConfig()
        df = add_basic_features(df, feat_cfg)

    # 2) TA (trend/vol)
    if cfg.use_ta:
        ta_cfg = TAFeatureConfig()
        df = add_ta_features(df, ta_cfg)

    # 3) Candlestick geometry + pattern
    if cfg.use_candles:
        c_cfg = CandleFeatureConfig()
        df = add_candlestick_features(df, c_cfg)

    # 4) Oscillators (RSI, Stoch, MACD, CCI, MFI)
    if cfg.use_osc:
        o_cfg = OscFeatureConfig()
        df = add_osc_features(df, o_cfg)

    # 5) HTF (1h) özellikleri
    if cfg.use_multitf:
        mtf_cfg = MultiTFConfig()
        df = add_multitf_1h_features(df, mtf_cfg)

    # 6) Funding + OI
    if cfg.use_external:
        df = add_external_features_15m(df, symbol)

    # 7) Rule-based entry/exit sinyalleri
    if cfg.use_rule_signals:
        rule_cfg = RuleSignalConfig(
            allowed_hours=cfg.rule_allowed_hours,
            allowed_weekdays=cfg.rule_allowed_weekdays,
        )
        df = add_rule_signals_v1(df, rule_cfg)

    return df
