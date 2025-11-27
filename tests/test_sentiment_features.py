from __future__ import annotations

import numpy as np
import pandas as pd

from finantradealgo.features.sentiment_features import add_sentiment_features


def test_add_sentiment_features_creates_columns():
    ts = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
    df_ohlcv = pd.DataFrame(
        {
            "timestamp": ts,
            "open": np.linspace(100, 110, len(ts)),
            "high": np.linspace(100, 110, len(ts)),
            "low": np.linspace(100, 110, len(ts)),
            "close": np.linspace(100, 110, len(ts)),
            "volume": np.random.rand(len(ts)),
        }
    )
    df_sent = pd.DataFrame(
        {
            "timestamp": ts,
            "sentiment_score": np.linspace(-1, 1, len(ts)),
        }
    )

    df = add_sentiment_features(df_ohlcv, df_sent, rolling_window=4)
    assert "sentiment_score" in df.columns
    assert "sentiment_zscore" in df.columns
    assert df["sentiment_zscore"].notna().any()
