from __future__ import annotations

import pandas as pd

from finantradealgo.features.microstructure_features import add_microstructure_features
from finantradealgo.microstructure.config import MicrostructureConfig


def _make_ohlcv(n: int = 30) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    prices = pd.Series(range(n), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": prices + 1,
            "high": prices + 2,
            "low": prices,
            "close": prices + 1.5,
            "volume": 1000,
        }
    )


def test_microstructure_features_contract():
    df = _make_ohlcv()
    cfg = MicrostructureConfig()
    out = add_microstructure_features(df, cfg)

    expected_cols = {
        "ms_chop",
        "ms_burst_up",
        "ms_burst_down",
        "ms_vol_regime",
        "ms_imbalance",
        "ms_sweep_up",
        "ms_sweep_down",
        "ms_exhaustion_up",
        "ms_exhaustion_down",
        "ms_parabolic_trend",
    }

    assert expected_cols.issubset(set(out.columns)), f"Missing micro cols: {expected_cols - set(out.columns)}"
    assert len(out) == len(df)
    assert out["timestamp"].is_monotonic_increasing

    for col in expected_cols:
        series = out[col]
        assert series.notna().any(), f"{col} is all NaN"
        assert pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series), f"{col} not numeric/bool"
