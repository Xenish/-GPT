from __future__ import annotations

import pandas as pd
import pytest

from finantradealgo.data_engine.loader import load_flow_features, load_sentiment_features


def _write_csv(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_load_flow_features_combined(tmp_path):
    base = tmp_path / "flow"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "perp_premium": [0.1, 0.2, 0.3],
            "basis": [0.01, 0.02, 0.03],
            "oi": [1e6, 1.1e6, 1.2e6],
        }
    )
    _write_csv(base / "flow_AIAUSDT_15m.csv", df)

    result = load_flow_features("AIAUSDT", "15m", base_dir=tmp_path)
    assert result is not None
    assert set(["timestamp", "perp_premium", "basis"]).issubset(result.columns)


def test_load_flow_features_missing_returns_none(caplog, tmp_path):
    result = load_flow_features("AIAUSDT", "15m", base_dir=tmp_path)
    assert result is None
    assert any("[FLOW]" in rec.message for rec in caplog.records)


def test_load_flow_features_missing_required(caplog, tmp_path):
    base = tmp_path / "flow"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "perp_premium": [0.1, 0.2, 0.3],
        }
    )
    _write_csv(base / "flow_AIAUSDT_15m.csv", df)

    result = load_flow_features("AIAUSDT", "15m", base_dir=tmp_path)
    assert result is None
    assert any("Missing required" in rec.message for rec in caplog.records)


def test_load_sentiment_features_defaults(tmp_path):
    base = tmp_path / "sentiment"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC"),
            "sentiment_score": [0.1, -0.2],
        }
    )
    _write_csv(base / "sentiment_AIAUSDT_15m.csv", df)

    result = load_sentiment_features("AIAUSDT", "15m", base_dir=tmp_path)
    assert result is not None
    assert "volume" in result.columns
    assert "source" in result.columns


def test_load_sentiment_features_missing_file(caplog, tmp_path):
    result = load_sentiment_features("AIAUSDT", "15m", base_dir=tmp_path)
    assert result is None
    assert any("[SENTIMENT]" in rec.message for rec in caplog.records)
