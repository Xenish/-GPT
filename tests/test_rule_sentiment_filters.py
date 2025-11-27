from __future__ import annotations

import pandas as pd
import pytest

from finantradealgo.features.rule_signals import RuleSignalConfig, build_rule_signals


def _base_df(sentiments):
    size = len(sentiments)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=size, freq="15min"),
            "htf1h_trend_score": 0.5,
            "htf1h_rsi_14": 50,
            "atr_14_pct": 0.01,
            "ema20_above_50": 1,
            "ema50_above_200": 1,
            "rsi_14": 60,
            "macd_hist_12_26_9": 0.01,
            "stoch_k_14_3": 50,
            "cs_body_pct": 0.2,
            "close": 100,
            "ema_20": 99,
            "sentiment_score": sentiments,
        }
    )


def test_rule_sentiment_filter_clamps_entries():
    df = _base_df([-0.5, 0.0, 0.3, 0.8])
    cfg = RuleSignalConfig(
        enable_sentiment_filters=True,
        min_sentiment=0.0,
        max_sentiment=0.5,
    )
    result = build_rule_signals(df, cfg)
    entries = result["rule_long_entry"].tolist()
    assert entries == [0, 1, 1, 0]


def test_rule_sentiment_filter_missing_column(caplog):
    df = _base_df([0.1, 0.2])
    df = df.drop(columns=["sentiment_score"])
    cfg = RuleSignalConfig(enable_sentiment_filters=True, min_sentiment=0.0, max_sentiment=0.5)
    result = build_rule_signals(df, cfg)
    assert result["rule_long_entry"].sum() == len(result)
    assert any("[RULE]" in rec.message for rec in caplog.records)
