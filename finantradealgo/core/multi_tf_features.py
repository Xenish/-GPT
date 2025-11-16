from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from finantradealgo.core.features import FeatureConfig, add_basic_features
from finantradealgo.core.ta_features import TAFeatureConfig, add_ta_features
from finantradealgo.core.osc_features import OscFeatureConfig, add_osc_features


@dataclass
class MultiTFConfig:
    """
    Multi-timeframe feature ayarları.

    Şimdilik tek bir TF: 1h
    İleride istersen 4h, 1d vs. de ekleyebiliriz.
    """
    rule_1h: str = "1H"  # pandas resample rule


def _resample_ohlcv_1h(df_15m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    15m OHLCV'yi (timestamp, open, high, low, close, volume) alıp
    verilen rule'a göre resample eder (örn. '1H').
    """
    if "timestamp" not in df_15m.columns:
        raise ValueError("DataFrame must have a 'timestamp' column")

    df = df_15m.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    df_res = (
        df.set_index("timestamp")
        .resample(rule)
        .agg(ohlc_dict)
        .dropna()
        .reset_index()
    )

    return df_res


def add_multitf_1h_features(df_15m: pd.DataFrame, config: MultiTFConfig | None = None) -> pd.DataFrame:
    """
    15m dataframe'e 1h timeframe'den gelen trend / vol / momentum feature'larını ekler.

    - 15m -> 1h resample
    - 1h üzerinde basic + TA + oscillator pipeline'ı çalıştır
    - Seçtiğimiz önemli kolonları prefix'leyip (htf1h_*) 15m'e merge_asof ile bind et
    """
    if config is None:
        config = MultiTFConfig()

    df_15 = df_15m.copy()
    df_15["timestamp"] = pd.to_datetime(df_15["timestamp"])
    df_15 = df_15.sort_values("timestamp")

    # 1) 15m -> 1H
    df_1h = _resample_ohlcv_1h(df_15, config.rule_1h)

    # 2) 1H üzerinde mevcut pipeline:
    feat_cfg = FeatureConfig()
    df_1h_feat = add_basic_features(df_1h, feat_cfg)

    ta_cfg = TAFeatureConfig()
    df_1h_ta = add_ta_features(df_1h_feat, ta_cfg)

    osc_cfg = OscFeatureConfig()
    df_1h_all = add_osc_features(df_1h_ta, osc_cfg)

    # 3) 1H'tan 15m'e taşıyacağımız kolonlar:
    #    - trend / regime
    #    - vol / ATR / hv / bb_width
    #    - EMA50
    #    - RSI14
    select_cols = [
        "timestamp",
        "trend_score",
        "regime_trend",
        "regime_vol",
        "ema_50",
        "atr_14_pct",
        "hv_20",
        "bb_width_20",
        "rsi_14",
    ]

    for c in select_cols:
        if c not in df_1h_all.columns:
            raise ValueError(f"Column {c} not found in 1H feature dataframe")

    df_1h_sel = df_1h_all[select_cols].copy()

    # 4) Prefix'le
    rename_map = {
        "trend_score": "htf1h_trend_score",
        "regime_trend": "htf1h_regime_trend",
        "regime_vol": "htf1h_regime_vol",
        "ema_50": "htf1h_ema_50",
        "atr_14_pct": "htf1h_atr_14_pct",
        "hv_20": "htf1h_hv_20",
        "bb_width_20": "htf1h_bb_width_20",
        "rsi_14": "htf1h_rsi_14",
    }

    df_1h_sel = df_1h_sel.rename(columns=rename_map)

    # 5) merge_asof ile "son oluşmuş 1H barı" 15m bara bind et
    df_1h_sel = df_1h_sel.sort_values("timestamp")

    df_out = pd.merge_asof(
        df_15.sort_values("timestamp"),
        df_1h_sel,
        on="timestamp",
        direction="backward",
    )

    return df_out
