from __future__ import annotations

import pandas as pd

from finantradealgo.features.sentiment_features import add_sentiment_features


def test_add_sentiment_features_merges_scores():
    ohlcv = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"),
            "close": 100,
        }
    )
    sentiment = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC"),
            "sentiment_score": [0.2, 0.4],
        }
    )
    merged = add_sentiment_features(ohlcv, sentiment)
    assert "sentiment_score" in merged.columns
    assert len(merged) == len(ohlcv)


def test_add_sentiment_features_handles_empty():
    ohlcv = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC")})
    merged = add_sentiment_features(ohlcv, pd.DataFrame())
    assert merged.equals(ohlcv)
