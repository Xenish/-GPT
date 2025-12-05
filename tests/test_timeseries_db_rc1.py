import pandas as pd

from finantradealgo.storage.timeseries_db import (
    TimeSeriesBackend,
    TimeSeriesDBClient,
    TimeSeriesDBConfig,
    build_timeseries_client_from_warehouse,
)
from finantradealgo.system.config_loader import WarehouseConfig


def test_mock_timeseries_write_and_query():
    cfg = TimeSeriesDBConfig(backend=TimeSeriesBackend.MOCK)
    client = TimeSeriesDBClient(cfg)

    idx = pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:15:00"])
    df = pd.DataFrame(
        {"open": [1, 2], "high": [2, 3], "low": [0.5, 1.5], "close": [1.5, 2.5], "volume": [10, 20]},
        index=idx,
    )
    client.write_ohlcv("BTCUSDT", "15m", df)

    out = client.query_ohlcv("BTCUSDT", "15m", start_ts=idx[0])
    assert len(out) == 2
    assert "open" in out.columns


def test_build_from_warehouse_none_returns_mock():
    wcfg = WarehouseConfig(backend="none")
    client = build_timeseries_client_from_warehouse(wcfg)
    assert isinstance(client, TimeSeriesDBClient)
