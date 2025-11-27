from __future__ import annotations

import pandas as pd

from finantradealgo.features.feature_pipeline import (
    PIPELINE_VERSION,
    FeaturePipelineConfig,
    build_feature_pipeline,
    get_feature_cols,
)


def test_pipeline_metadata_round_trip(tmp_path):
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="15min"),
            "open": [1, 2, 3, 4, 5],
            "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
            "volume": [10, 10, 10, 10, 10],
        }
    )
    csv_path = tmp_path / "ohlcv.csv"
    data.to_csv(csv_path, index=False)

    cfg = FeaturePipelineConfig(
        use_base=False,
        use_ta=False,
        use_candles=False,
        use_osc=False,
        use_htf=False,
        use_microstructure=False,
        use_market_structure=False,
        use_external=False,
        use_rule_signals=False,
        drop_na=False,
        feature_preset="extended",
    )
    df_feat, meta = build_feature_pipeline(str(csv_path), cfg)
    assert meta["pipeline_version"] == PIPELINE_VERSION
    expected_cols = get_feature_cols(df_feat, preset=cfg.feature_preset)
    assert meta["feature_cols"] == expected_cols
