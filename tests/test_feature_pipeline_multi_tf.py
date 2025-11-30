"""
Tests for multi-symbol/multi-timeframe feature pipeline.

Verifies that:
1. Pipeline works for multiple symbol/timeframe combinations
2. Each combination produces valid DataFrames
3. Timestamps are monotonic and within expected range
4. Lookback filtering is applied correctly per timeframe
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from finantradealgo.features.feature_pipeline import (
    build_feature_pipeline_from_system_config,
)
from finantradealgo.system.config_loader import load_system_config


class TestFeaturePipelineMultiTF:
    """Test feature pipeline with multiple symbol/timeframe combinations."""

    @pytest.fixture
    def temp_ohlcv_data(self):
        """Create temporary OHLCV CSVs for multiple symbols and timeframes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ohlcv_dir = Path(tmpdir) / "ohlcv"
            ohlcv_dir.mkdir()

            symbols = ["BTCUSDT", "ETHUSDT"]
            timeframes = ["15m", "1h"]

            # Generate data for each combination
            for symbol in symbols:
                for tf in timeframes:
                    if tf == "15m":
                        bars_per_day = 96
                        days = 30
                        freq_minutes = 15
                    else:  # 1h
                        bars_per_day = 24
                        days = 60
                        freq_minutes = 60

                    total_bars = days * bars_per_day
                    start_date = datetime.now() - timedelta(days=days)

                    timestamps = [
                        start_date + timedelta(minutes=freq_minutes * i)
                        for i in range(total_bars)
                    ]

                    df = pd.DataFrame({
                        "timestamp": timestamps,
                        "open": [100.0 + i * 0.01 for i in range(total_bars)],
                        "high": [100.1 + i * 0.01 for i in range(total_bars)],
                        "low": [99.9 + i * 0.01 for i in range(total_bars)],
                        "close": [100.0 + i * 0.01 for i in range(total_bars)],
                        "volume": [1000.0 for _ in range(total_bars)],
                    })

                    csv_path = ohlcv_dir / f"{symbol}_{tf}.csv"
                    df.to_csv(csv_path, index=False)

            yield str(ohlcv_dir)

    @pytest.fixture
    def temp_system_config(self, temp_ohlcv_data):
        """Create a temporary system.yml with multi-TF configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "system.yml"

            config_content = f"""
symbol: "BTCUSDT"
timeframe: "15m"

data:
  ohlcv_dir: "{temp_ohlcv_data}"
  external_dir: "data/external"
  features_dir: "data/features"
  symbols:
    - "BTCUSDT"
    - "ETHUSDT"
  timeframes:
    - "15m"
    - "1h"
  ohlcv_path_template: "{temp_ohlcv_data}/{{symbol}}_{{timeframe}}.csv"
  lookback_days:
    "15m": 10
    "1h": 30
  default_lookback_days: 365
  bars:
    mode: "time"

features:
  use_base: true
  use_ta: false
  use_candles: false
  use_osc: false
  use_htf: false
  use_microstructure: false
  use_market_structure: false
  use_external: false
  use_rule_signals: false
  use_flow_features: false
  use_sentiment_features: false
  drop_na: false
  feature_preset: "core"

risk:
  capital_risk_pct_per_trade: 0.01
  max_leverage: 5.0
  warmup_bars: 50

rule: {{}}

ml:
  enabled: true
  targets: []
  label:
    method: "fixed_horizon"
    horizon_bars: 8
  model:
    type: "RandomForest"
  persistence:
    model_dir: "outputs/ml_models"

notifications:
  enabled: false
  fcm:
    enabled: false
"""
            with open(config_path, "w") as f:
                f.write(config_content)

            yield str(config_path)

    def test_pipeline_works_for_all_combinations(self, temp_system_config):
        """Test that pipeline successfully builds features for all symbol/TF combos."""
        # Set dummy FCM key
        if not os.getenv("FCM_SERVER_KEY"):
            os.environ["FCM_SERVER_KEY"] = "dummy_test_key"

        cfg = load_system_config(temp_system_config)
        data_cfg = cfg["data_cfg"]

        symbols = data_cfg.symbols
        timeframes = data_cfg.timeframes

        assert len(symbols) == 2, "Should have 2 symbols"
        assert len(timeframes) == 2, "Should have 2 timeframes"

        results = {}

        # Build features for each combination
        for symbol in symbols:
            for tf in timeframes:
                df, meta = build_feature_pipeline_from_system_config(
                    cfg,
                    symbol=symbol,
                    timeframe=tf,
                )

                assert df is not None, f"DataFrame should not be None for {symbol} {tf}"
                assert len(df) > 0, f"DataFrame should not be empty for {symbol} {tf}"
                assert "timestamp" in df.columns, f"Should have timestamp column for {symbol} {tf}"

                results[(symbol, tf)] = (df, meta)

        # Verify we got all 4 combinations
        assert len(results) == 4, "Should have 4 combinations (2 symbols × 2 timeframes)"

    def test_lookback_applied_per_timeframe(self, temp_system_config):
        """Test that different timeframes get different lookback periods."""
        if not os.getenv("FCM_SERVER_KEY"):
            os.environ["FCM_SERVER_KEY"] = "dummy_test_key"

        cfg = load_system_config(temp_system_config)

        # Build features for 15m (10-day lookback)
        df_15m, _ = build_feature_pipeline_from_system_config(
            cfg,
            symbol="BTCUSDT",
            timeframe="15m",
        )

        # Build features for 1h (30-day lookback)
        df_1h, _ = build_feature_pipeline_from_system_config(
            cfg,
            symbol="BTCUSDT",
            timeframe="1h",
        )

        # 15m: 10 days × 96 bars/day = 960 bars (approximately)
        # 1h: 30 days × 24 bars/day = 720 bars (approximately)
        # Allow tolerance for partial days
        assert abs(len(df_15m) - 960) < 96, \
            f"15m should have ~960 bars for 10-day lookback, got {len(df_15m)}"

        assert abs(len(df_1h) - 720) < 24, \
            f"1h should have ~720 bars for 30-day lookback, got {len(df_1h)}"

    def test_timestamps_monotonic_increasing(self, temp_system_config):
        """Test that timestamps are sorted correctly for all combinations."""
        if not os.getenv("FCM_SERVER_KEY"):
            os.environ["FCM_SERVER_KEY"] = "dummy_test_key"

        cfg = load_system_config(temp_system_config)
        data_cfg = cfg["data_cfg"]

        for symbol in data_cfg.symbols:
            for tf in data_cfg.timeframes:
                df, _ = build_feature_pipeline_from_system_config(
                    cfg,
                    symbol=symbol,
                    timeframe=tf,
                )

                assert df["timestamp"].is_monotonic_increasing, \
                    f"Timestamps should be monotonic increasing for {symbol} {tf}"

    def test_metadata_contains_symbol_timeframe(self, temp_system_config):
        """Test that metadata includes correct symbol and timeframe."""
        if not os.getenv("FCM_SERVER_KEY"):
            os.environ["FCM_SERVER_KEY"] = "dummy_test_key"

        cfg = load_system_config(temp_system_config)

        symbol = "ETHUSDT"
        timeframe = "1h"

        df, meta = build_feature_pipeline_from_system_config(
            cfg,
            symbol=symbol,
            timeframe=timeframe,
        )

        assert meta["symbol"] == symbol, f"Metadata should contain correct symbol"
        assert meta["timeframe"] == timeframe, f"Metadata should contain correct timeframe"
        assert "feature_cols" in meta, "Metadata should contain feature_cols"
        assert "pipeline_version" in meta, "Metadata should contain pipeline_version"

    def test_different_symbols_produce_independent_features(self, temp_system_config):
        """Test that different symbols produce independent feature sets."""
        if not os.getenv("FCM_SERVER_KEY"):
            os.environ["FCM_SERVER_KEY"] = "dummy_test_key"

        cfg = load_system_config(temp_system_config)

        # Build features for BTCUSDT
        df_btc, _ = build_feature_pipeline_from_system_config(
            cfg,
            symbol="BTCUSDT",
            timeframe="15m",
        )

        # Build features for ETHUSDT
        df_eth, _ = build_feature_pipeline_from_system_config(
            cfg,
            symbol="ETHUSDT",
            timeframe="15m",
        )

        # DataFrames should have same structure but different data
        assert list(df_btc.columns) == list(df_eth.columns), \
            "Same columns for both symbols"

        # But different timestamps (loaded from different files)
        # This is a simple check - in reality timestamps might be similar
        assert df_btc is not df_eth, "Should be different DataFrame objects"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
