import pandas as pd

from finantradealgo.features.feature_pipeline import build_feature_pipeline_from_system_config
from finantradealgo.system.config_loader import load_config


def test_feature_pipeline_accepts_preloaded_ohlcv(monkeypatch, tmp_path):
    cfg = load_config("research")
    # make tiny ohlcv df
    ts = pd.date_range("2024-01-01", periods=5, freq="1H", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    )
    feat_df, meta = build_feature_pipeline_from_system_config(
        cfg,
        symbol=cfg.get("symbol", "BTCUSDT"),
        timeframe=cfg.get("timeframe", "15m"),
        df_ohlcv_override=df,
    )
    assert len(feat_df) == len(df)
    assert "pipeline_version" in meta
