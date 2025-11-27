from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import finantradealgo.data_engine.binance_ws_source as ws_module
from finantradealgo.data_engine.binance_ws_source import BinanceWsDataSource


class DummyRestClient:
    def __init__(self, *args, **kwargs):
        self.klines_response = []

    def get_klines(self, *_, **__):
        return list(self.klines_response)


@pytest.fixture(autouse=True)
def patch_rest_client(monkeypatch):
    monkeypatch.setattr(ws_module, "BinanceFuturesClient", DummyRestClient)


def _minimal_cfg():
    return {
        "symbol": "AIAUSDT",
        "timeframe": "15m",
        "exchange": {
            "symbol_mapping": {"AIAUSDT": "AIAUSDT"},
            "testnet": True,
            "time_sync": False,
        },
        "live": {
            "symbol": "AIAUSDT",
            "timeframe": "15m",
            "ws": {
                "use_1m_stream": True,
                "aggregate_to_tf": "15m",
                "resync_lookback_bars": 10,
            },
        },
    }


def test_binance_ws_backfill_gap(monkeypatch):
    cfg = _minimal_cfg()
    data_source = BinanceWsDataSource(cfg, ["AIAUSDT"])
    fake_klines = [
        {
            "t": int(pd.Timestamp("2025-01-01 00:05:00+00:00").timestamp() * 1000),
            "T": int(pd.Timestamp("2025-01-01 00:06:00+00:00").timestamp() * 1000),
            "i": "1m",
            "o": "100.0",
            "h": "101.0",
            "l": "99.5",
            "c": "100.5",
            "v": "10",
        },
        {
            "t": int(pd.Timestamp("2025-01-01 00:06:00+00:00").timestamp() * 1000),
            "T": int(pd.Timestamp("2025-01-01 00:07:00+00:00").timestamp() * 1000),
            "i": "1m",
            "o": "100.6",
            "h": "101.1",
            "l": "99.9",
            "c": "100.9",
            "v": "8",
        },
    ]
    data_source._rest_client.klines_response = fake_klines

    start = pd.Timestamp("2025-01-01 00:05:00+00:00")
    end = pd.Timestamp("2025-01-01 00:07:00+00:00")
    bars = data_source._backfill_gap("AIAUSDT", start, end)
    assert len(bars) == 2
    assert bars[0].open_time == pd.to_datetime(fake_klines[0]["t"], unit="ms")
    assert bars[-1].close_time == pd.to_datetime(fake_klines[-1]["T"], unit="ms")
