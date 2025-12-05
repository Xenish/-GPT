from __future__ import annotations

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import scripts.ingest_marketdata as ingest_cli


def test_ingest_cli_uses_build_factory(monkeypatch):
    called = {}

    def fake_load_config_from_env():
        called["load_cfg"] = True
        return {
            "profile": "research",
            "data_cfg": SimpleNamespace(symbols=["BTCUSDT"], timeframes=["1m"]),
            "exchange_cfg": SimpleNamespace(),
        }

    class DummyWarehouse:
        def __init__(self):
            called["warehouse_built"] = True

        def upsert_ohlcv(self, candles):
            called["upsert"] = len(candles)
            return len(candles)

        def get_latest_ts(self, symbol, timeframe):
            return None

        def detect_gaps(self, *args, **kwargs):
            return []

    def fake_build_warehouse(cfg):
        called["build_cfg_profile"] = cfg.get("profile")
        return DummyWarehouse()

    class DummySource:
        def fetch(self, symbol, timeframe, start_ts, end_ts, limit=None):
            return []

    monkeypatch.setattr(ingest_cli, "load_config_from_env", fake_load_config_from_env)
    monkeypatch.setattr(ingest_cli, "_build_warehouse", fake_build_warehouse)
    monkeypatch.setattr(ingest_cli, "BinanceRESTCandleSource", lambda *a, **k: DummySource())
    monkeypatch.setattr(ingest_cli, "load_exchange_credentials", lambda *a, **k: ("k", "s"))
    monkeypatch.setattr(ingest_cli.pd.Timestamp, "utcnow", staticmethod(lambda: ingest_cli.pd.Timestamp("2025-01-01 00:00:00")))
    class DummyIngestor:
        def __init__(self, source, warehouse, *args, **kwargs):
            called["ingestor_init"] = True
        def backfill_range(self, *args, **kwargs):
            called["backfill"] = True
            return 0
    monkeypatch.setattr(ingest_cli, "HistoricalOHLCVIngestor", DummyIngestor)

    runner = CliRunner()
    result = runner.invoke(ingest_cli.cli, ["historical"])
    assert result.exit_code == 0, result.output
    assert called.get("load_cfg") is True
    assert called.get("build_cfg_profile") == "research"
    assert called.get("warehouse_built") is True
    assert called.get("ingestor_init") is True
    assert called.get("backfill") is True
