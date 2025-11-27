from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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
    use_ms_trend_filter: bool = False
    ms_trend_min: float = -0.5
    ms_trend_max: float = 1.5
    use_ms_chop_filter: bool = False
    allow_chop: bool = False
    use_fvg_filter: bool = False
    min_sentiment: Optional[float] = None
    max_sentiment: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "RuleSignalConfig":
        data = data or {}
        return cls(
            htf_trend_min=data.get("htf_trend_min", cls.htf_trend_min),
            htf_rsi_min=data.get("htf_rsi_min", cls.htf_rsi_min),
            htf_rsi_max=data.get("htf_rsi_max", cls.htf_rsi_max),
            atr_pct_min=data.get("atr_pct_min", cls.atr_pct_min),
            atr_pct_max=data.get("atr_pct_max", cls.atr_pct_max),
            rsi_entry_min=data.get("rsi_entry_min", cls.rsi_entry_min),
            rsi_entry_max=data.get("rsi_entry_max", cls.rsi_entry_max),
            macd_entry_min=data.get("macd_entry_min", cls.macd_entry_min),
            stoch_k_entry_min=data.get("stoch_k_entry_min", cls.stoch_k_entry_min),
            stoch_k_entry_max=data.get("stoch_k_entry_max", cls.stoch_k_entry_max),
            min_body_pct=data.get("min_body_pct", cls.min_body_pct),
            trend_exit_max=data.get("trend_exit_max", cls.trend_exit_max),
            rsi_exit_max=data.get("rsi_exit_max", cls.rsi_exit_max),
            use_patterns=data.get("use_patterns", cls.use_patterns),
            allowed_hours=data.get("allowed_hours"),
            allowed_weekdays=data.get("allowed_weekdays"),
            use_ms_trend_filter=data.get("use_ms_trend_filter", cls.use_ms_trend_filter),
            ms_trend_min=data.get("ms_trend_min", cls.ms_trend_min),
            ms_trend_max=data.get("ms_trend_max", cls.ms_trend_max),
            use_ms_chop_filter=data.get("use_ms_chop_filter", cls.use_ms_chop_filter),
            allow_chop=data.get("allow_chop", cls.allow_chop),
            use_fvg_filter=data.get("use_fvg_filter", cls.use_fvg_filter),
            min_sentiment=data.get("min_sentiment", cls.min_sentiment),
            max_sentiment=data.get("max_sentiment", cls.max_sentiment),
        )


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

    if cfg.use_ms_trend_filter:
        trend_col = _resolve_ms_column(
            df,
            ["ms_trend_state", "ms_trend_regime", "ms_trend"],
            "Microstructure trend",
        )
        cond_ms_trend = df[trend_col].between(cfg.ms_trend_min, cfg.ms_trend_max)
        entry_core = entry_core & cond_ms_trend

    if cfg.use_ms_chop_filter:
        chop_col = _resolve_ms_column(
            df,
            ["ms_chop_flag", "ms_chop"],
            "Microstructure chop",
        )
        chop_series = df[chop_col].fillna(0).astype(int)
        if cfg.allow_chop:
            cond_chop = pd.Series(True, index=df.index)
        else:
            cond_chop = chop_series == 0
        entry_core = entry_core & cond_chop

    if cfg.use_fvg_filter:
        fvg_col = _resolve_ms_column(
            df,
            ["ms_fvg_up"],
            "FVG (up)",
        )
        entry_core = entry_core & (df[fvg_col] == 1)

    if cfg.min_sentiment is not None or cfg.max_sentiment is not None:
        if "sentiment_score" not in df.columns:
            raise ValueError(
                "RuleSignalConfig requires sentiment filtering but 'sentiment_score' column is missing. "
                "Enable sentiment features in the pipeline."
            )
        lower = cfg.min_sentiment if cfg.min_sentiment is not None else float("-inf")
        upper = cfg.max_sentiment if cfg.max_sentiment is not None else float("inf")
        sentiment_mask = df["sentiment_score"].between(lower, upper)
        entry_core = entry_core & sentiment_mask

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


def _resolve_ms_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"{label} kolonu bulunamadı. Aranan kolonlar: {candidates}")
