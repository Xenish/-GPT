from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import pandas as pd


@dataclass
class RuleSignalConfig:
    htf_trend_min: float = -0.05
    htf_rsi_min: float = 40.0
    htf_rsi_max: float = 75.0
    atr_pct_min: float = 0.002
    atr_pct_max: float = 0.08
    rsi_entry_min: float = 45.0
    rsi_entry_max: float = 75.0
    macd_entry_min: float = -0.001
    stoch_k_entry_min: float = 20.0
    stoch_k_entry_max: float = 85.0
    min_body_pct: float = 0.08
    trend_exit_max: float = -0.10
    rsi_exit_max: float = 45.0
    use_patterns: bool = False
    allowed_hours: Optional[List[int]] = None
    allowed_weekdays: Optional[List[int]] = None


def build_rule_signals(
    df: pd.DataFrame,
    cfg: Optional[RuleSignalConfig] = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = RuleSignalConfig()

    df = df.copy()

    cond_htf_trend = df["htf1h_trend_score"] > cfg.htf_trend_min
    cond_htf_rsi = df["htf1h_rsi_14"].between(cfg.htf_rsi_min, cfg.htf_rsi_max)
    cond_atr = df["atr_14_pct"].between(cfg.atr_pct_min, cfg.atr_pct_max)
    cond_ema = (df["ema20_above_50"] == 1) & (df["ema50_above_200"] == 1)
    cond_rsi_ltf = df["rsi_14"].between(cfg.rsi_entry_min, cfg.rsi_entry_max)
    cond_macd = df["macd_hist_12_26_9"] > cfg.macd_entry_min
    cond_stoch = df["stoch_k_14_3"].between(cfg.stoch_k_entry_min, cfg.stoch_k_entry_max)
    cond_body = df["cs_body_pct"] >= cfg.min_body_pct

    entry_core = (
        cond_htf_trend
        & cond_htf_rsi
        & cond_atr
        & cond_ema
        & cond_rsi_ltf
        & cond_macd
        & cond_stoch
        & cond_body
    )

    if cfg.use_patterns:
        patt = (
            (df["cdl_bull_engulf"] == 1)
            | (df["cdl_hammer"] == 1)
            | (df["cdl_marubozu"] == 1)
        )
        entry = entry_core & patt
    else:
        entry = entry_core

    ts_col = "timestamp"
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col])

        if cfg.allowed_hours is not None:
            hours = df[ts_col].dt.hour
            entry = entry & hours.isin(cfg.allowed_hours)

        if cfg.allowed_weekdays is not None:
            wdays = df[ts_col].dt.dayofweek
            entry = entry & wdays.isin(cfg.allowed_weekdays)

    df["rule_long_entry"] = entry.astype(int)

    cond_exit_trend = df["htf1h_trend_score"] < cfg.trend_exit_max
    cond_exit_weak = (df["rsi_14"] < cfg.rsi_exit_max) & (df["close"] < df["ema_20"])
    exit_core = cond_exit_trend | cond_exit_weak

    df["rule_long_exit"] = exit_core.astype(int)

    return df


def add_rule_signals_v1(
    df: pd.DataFrame,
    config: Optional[RuleSignalConfig] = None,
) -> pd.DataFrame:
    return build_rule_signals(df, config)
