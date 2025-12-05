from __future__ import annotations

import pandas as pd
import pytest

from finantradealgo.live_trading.factories import create_data_source
from finantradealgo.system.config_loader import LiveConfig, WarehouseConfig
from finantradealgo.data_engine.live_data_source import FileReplayDataSource, DataBackendReplaySource
from finantradealgo.data_engine.binance_ws_source import BinanceWsDataSource


def _base_cfg():
    return {
        "data_cfg": type("DC", (), {"backend": "none"})(),  # minimal stub
        "warehouse_cfg": WarehouseConfig(backend="none"),
        "exchange_cfg": type("EC", (), {})(),
    }


def test_create_data_source_replay_file():
    cfg = _base_cfg()
    live_cfg = LiveConfig.from_dict({"data_source": "replay", "symbol": "BTCUSDT", "symbols": ["BTCUSDT"], "timeframe": "1m"})
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=1), "close": [1.0]})
    src = create_data_source(cfg, live_cfg, df)
    assert isinstance(src, FileReplayDataSource)


def test_create_data_source_backend_replay():
    cfg = _base_cfg()
    cfg["data_cfg"].backend = "timescale"
    live_cfg = LiveConfig.from_dict({"data_source": "db", "symbol": "BTCUSDT", "symbols": ["BTCUSDT"], "timeframe": "1m"})
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=1), "close": [1.0]})
    src = create_data_source(cfg, live_cfg, df)
    assert isinstance(src, DataBackendReplaySource)


def test_create_data_source_binance_ws(monkeypatch):
    cfg = _base_cfg()
    live_cfg = LiveConfig.from_dict({"data_source": "binance_ws", "symbol": "BTCUSDT", "symbols": ["BTCUSDT"], "timeframe": "1m"})
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=1), "close": [1.0]})
    src = create_data_source(cfg, live_cfg, df)
    assert isinstance(src, BinanceWsDataSource)


def test_create_data_source_invalid():
    cfg = _base_cfg()
    live_cfg = LiveConfig.from_dict({"data_source": "invalid", "symbol": "BTCUSDT", "symbols": ["BTCUSDT"], "timeframe": "1m"})
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=1), "close": [1.0]})
    with pytest.raises(ValueError):
        create_data_source(cfg, live_cfg, df)
