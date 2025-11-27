from __future__ import annotations

import pandas as pd

from finantradealgo.features.flow_features import add_flow_features


def test_add_flow_features_merges_and_prefixes():
    ohlcv = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"),
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100,
        }
    )
    flow = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC"),
            "perp_premium": [0.1, 0.2],
            "basis": [0.01, 0.02],
            "oi": [1e6, 1.1e6],
        }
    )
    merged = add_flow_features(ohlcv, flow)
    assert "flow_perp_premium" in merged.columns
    assert "flow_basis" in merged.columns
    assert len(merged) == len(ohlcv)


def test_add_flow_features_handles_empty_flow():
    ohlcv = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC")})
    flow = pd.DataFrame()
    merged = add_flow_features(ohlcv, flow)
    assert merged.equals(ohlcv)
