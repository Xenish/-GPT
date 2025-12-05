from __future__ import annotations

import pandas as pd

from finantradealgo.features.market_structure_features import add_market_structure_features
from finantradealgo.market_structure.config import MarketStructureConfig


def _make_ohlcv(n: int = 30) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    prices = pd.Series(range(n), dtype=float)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": prices + 1,
            "high": prices + 2,
            "low": prices,
            "close": prices + 1.5,
            "volume": 1000,
        }
    )
    return df


def test_market_structure_features_contract():
    df = _make_ohlcv()
    cfg = MarketStructureConfig()
    out = add_market_structure_features(df, cfg)

    expected_cols = {
        "ms_swing_high",
        "ms_swing_low",
        "ms_trend_regime",
        "ms_fvg_up",
        "ms_fvg_down",
    }

    assert expected_cols.issubset(set(out.columns)), f"Missing MS cols: {expected_cols - set(out.columns)}"

    # Ensure no index/length corruption
    assert len(out) == len(df)
    assert out["timestamp"].is_monotonic_increasing

    # Critical columns should not be entirely NaN and should be numeric/bool-ish
    for col in expected_cols:
        series = out[col]
        assert series.notna().any(), f"{col} is all NaN"
        assert pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series), f"{col} not numeric/bool"
