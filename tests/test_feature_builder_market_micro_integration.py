from __future__ import annotations

import pandas as pd

from finantradealgo.features.feature_pipeline import build_feature_pipeline_from_system_config
from finantradealgo.system.config_loader import load_config

MS_COLS = {"ms_swing_high", "ms_swing_low", "ms_trend_regime", "ms_fvg_up", "ms_fvg_down"}
MICRO_COLS = {
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


def _make_df():
    ts = pd.date_range("2025-01-01", periods=30, freq="1min", tz="UTC")
    prices = pd.Series(range(30), dtype=float)
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


def test_feature_builder_adds_market_and_micro_when_enabled():
    cfg = load_config("research")
    cfg_local = dict(cfg)
    features_cfg = dict(cfg_local.get("features", {}))
    features_cfg["use_market_structure"] = True
    features_cfg["use_microstructure"] = True
    cfg_local["features"] = features_cfg

    df = _make_df()
    df_out, meta = build_feature_pipeline_from_system_config(cfg_local, df_ohlcv_override=df)

    cols = set(df_out.columns)
    assert MS_COLS.issubset(cols), f"Missing market structure columns: {MS_COLS - cols}"
    assert MICRO_COLS.issubset(cols), f"Missing microstructure columns: {MICRO_COLS - cols}"


def test_feature_builder_excludes_market_and_micro_when_disabled():
    cfg = load_config("research")
    cfg_local = dict(cfg)
    features_cfg = dict(cfg_local.get("features", {}))
    features_cfg["use_market_structure"] = False
    features_cfg["use_microstructure"] = False
    # Also disable rule signals to avoid ms_* dependency in rule signals
    features_cfg["use_rule_signals"] = False
    cfg_local["features"] = features_cfg

    df = _make_df()
    df_out, meta = build_feature_pipeline_from_system_config(cfg_local, df_ohlcv_override=df)
    cols = set(df_out.columns)
    assert MS_COLS.isdisjoint(cols), f"Market structure columns should be absent when disabled"
    assert MICRO_COLS.isdisjoint(cols), f"Microstructure columns should be absent when disabled"
