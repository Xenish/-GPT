from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pytest

from finantradealgo.features.feature_pipeline import build_feature_pipeline_from_system_config, FeaturePipelineConfig


@pytest.fixture
def dummy_cfg(tmp_path) -> Dict[str, Any]:
    data_dir = tmp_path / "data"
    ohlcv_dir = data_dir / "ohlcv"
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ohlcv_dir / "AIAUSDT_15m.csv"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="15min"),
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100,
            "volume": 1,
        }
    )
    df.to_csv(csv_path, index=False)
    cfg = {
        "symbol": "AIAUSDT",
        "timeframe": "15m",
        "data": {
            "ohlcv_dir": str(ohlcv_dir),
            "external_dir": str(tmp_path / "external"),
            "flow_dir": str(tmp_path / "flow"),
            "sentiment_dir": str(tmp_path / "sentiment"),
        },
        "features": {
            "use_flow_features": True,
            "use_sentiment_features": True,
            "use_microstructure": False,
            "use_market_structure": False,
            "drop_na": False,
        },
    }
    return cfg


def test_pipeline_includes_flow_and_sentiment(monkeypatch, dummy_cfg):
    def fake_flow(*args, **kwargs):
        ts = pd.date_range("2024-01-01", periods=5, freq="30min", tz="UTC")
        return pd.DataFrame({
            "timestamp": ts,
            "perp_premium": 0.1,
            "basis": 0.01,
            "oi": 1e6,
        })

    def fake_sent(*args, **kwargs):
        ts = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "sentiment_score": 0.5})

    monkeypatch.setattr(
        "finantradealgo.features.feature_pipeline.load_flow_features", fake_flow
    )
    monkeypatch.setattr(
        "finantradealgo.features.feature_pipeline.load_sentiment_features", fake_sent
    )

    df, _ = build_feature_pipeline_from_system_config(dummy_cfg, None)
    assert "flow_perp_premium" in df.columns
    assert "flow_basis" in df.columns
    assert "sentiment_score" in df.columns


def test_pipeline_disables_missing_flow(monkeypatch, dummy_cfg):
    monkeypatch.setattr(
        "finantradealgo.features.feature_pipeline.load_flow_features", lambda *args, **kwargs: None
    )

    def fake_sent(*args, **kwargs):
        ts = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "sentiment_score": 0.1})

    monkeypatch.setattr(
        "finantradealgo.features.feature_pipeline.load_sentiment_features", fake_sent
    )

    df, _ = build_feature_pipeline_from_system_config(dummy_cfg, None)
    assert "sentiment_score" in df.columns
    assert not any(col.startswith("flow_") for col in df.columns)
