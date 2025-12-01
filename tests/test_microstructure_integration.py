from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finantradealgo.features.feature_pipeline import (
    FeaturePipelineConfig,
    build_feature_pipeline,
)
from finantradealgo.microstructure.types import MicrostructureSignals


def test_pipeline_adds_microstructure_columns(tmp_path: Path):
    """
    Tests that the feature pipeline correctly adds microstructure feature columns
    when the flag is enabled. This is an integration test for the feature pipeline.
    """
    # 1. Create a deterministic random OHLCV DataFrame
    np.random.seed(42)
    num_bars = 100
    data = {
        "timestamp": pd.to_datetime(
            pd.date_range("2023-01-01", periods=num_bars, freq="15min")
        ),
        "open": 1000 + np.random.uniform(-1, 1, num_bars).cumsum(),
        "close": 1000 + np.random.uniform(-1, 1, num_bars).cumsum(),
        "volume": np.random.uniform(10, 100, num_bars),
    }
    df_ohlcv = pd.DataFrame(data)
    df_ohlcv["high"] = (
        df_ohlcv[["open", "close"]].max(axis=1) + np.random.uniform(0, 1, num_bars)
    )
    df_ohlcv["low"] = (
        df_ohlcv[["open", "close"]].min(axis=1) - np.random.uniform(0, 1, num_bars)
    )

    # 2. Save to a temporary file for the pipeline to read
    temp_csv_path = tmp_path / "temp_ohlcv.csv"
    df_ohlcv.to_csv(temp_csv_path, index=False)

    # 3. Configure pipeline to *only* run the microstructure part
    pipeline_cfg = FeaturePipelineConfig(
        use_microstructure=True,
        # Disable all other features to isolate the test
        use_base=False,
        use_ta=False,
        use_candles=False,
        use_osc=False,
        use_htf=False,
        use_market_structure=False,
        use_external=False,
        use_rule_signals=False,
        use_flow_features=False,
        use_sentiment_features=False,
        drop_na=False,  # Keep all rows to match length
    )

    # 4. Run the feature pipeline
    df_out, _ = build_feature_pipeline(
        csv_ohlcv_path=str(temp_csv_path),
        pipeline_cfg=pipeline_cfg,
    )

    # 5. Assertions
    ms_cols = MicrostructureSignals.columns()

    # Check that output has the same number of rows
    assert len(df_out) == num_bars

    # Check that all microstructure columns were added
    assert all(col in df_out.columns for col in ms_cols)

    # Check that the columns have no NaNs (values can be non-zero if features are implemented)
    assert df_out[ms_cols].isnull().sum().sum() == 0

    # Check that original columns are still there
    assert "close" in df_out.columns
