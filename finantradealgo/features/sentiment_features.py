from __future__ import annotations

import pandas as pd


def add_sentiment_features(
    df_ohlcv: pd.DataFrame,
    df_sent: pd.DataFrame,
    tolerance: str = "1d",
    rolling_window: int = 96,
) -> pd.DataFrame:
    if df_sent is None or df_sent.empty:
        return df_ohlcv

    df_ohlcv = df_ohlcv.sort_values("timestamp")
    df_sent = df_sent.sort_values("timestamp")

    merged = pd.merge_asof(
        df_ohlcv,
        df_sent,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )

    if "sentiment_score" in merged.columns and rolling_window > 0:
        s = merged["sentiment_score"]
        roll = s.rolling(rolling_window, min_periods=max(1, rolling_window // 4))
        mean = roll.mean()
        std = roll.std(ddof=0)
        merged["sentiment_zscore"] = (s - mean) / std.replace(0, pd.NA)

    return merged
