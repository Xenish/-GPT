from __future__ import annotations

import numpy as np
import pandas as pd

from finantradealgo.features.flow_features import add_flow_features


def test_add_flow_features_merges_and_handles_nan():
    ts = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    df_ohlcv = pd.DataFrame(
        {
            "timestamp": ts,
            "open": np.linspace(100, 104, 5),
            "high": np.linspace(101, 105, 5),
            "low": np.linspace(99, 103, 5),
            "close": np.linspace(100, 104, 5),
            "volume": np.random.rand(5),
        }
    )
    df_flow = pd.DataFrame(
        {
            "timestamp": ts[::2],
            "perp_premium": [0.1, 0.2, 0.3],
            "basis": [5.0, 6.0, 7.0],
        }
    )

    df = add_flow_features(df_ohlcv, df_flow)
    assert "flow_perp_premium" in df.columns
    assert "flow_basis" in df.columns
    assert df["flow_perp_premium"].isna().sum() <= 2
    assert df["flow_perp_premium"].dtype == float
