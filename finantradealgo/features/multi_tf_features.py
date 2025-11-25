from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from finantradealgo.features.base_features import FeatureConfig, add_basic_features
from finantradealgo.features.osc_features import OscFeatureConfig, add_osc_features
from finantradealgo.features.ta_features import TAFeatureConfig, add_ta_features


@dataclass
class MultiTFConfig:
    rule_1h: str = "1h"


def _resample_ohlcv_1h(df_15m: pd.DataFrame, rule: str) -> pd.DataFrame:
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
    if config is None:
        config = MultiTFConfig()

    df_15 = df_15m.copy()
    df_15["timestamp"] = pd.to_datetime(df_15["timestamp"])
    df_15 = df_15.sort_values("timestamp")

    df_1h = _resample_ohlcv_1h(df_15, config.rule_1h)

    feat_cfg = FeatureConfig()
    df_1h_feat = add_basic_features(df_1h, feat_cfg)

    ta_cfg = TAFeatureConfig()
    df_1h_ta = add_ta_features(df_1h_feat, ta_cfg)

    osc_cfg = OscFeatureConfig()
    df_1h_all = add_osc_features(df_1h_ta, osc_cfg)

    select_cols: List[str] = [
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
    df_1h_sel = df_1h_sel.sort_values("timestamp")

    df_out = pd.merge_asof(
        df_15.sort_values("timestamp"),
        df_1h_sel,
        on="timestamp",
        direction="backward",
    )

    return df_out
