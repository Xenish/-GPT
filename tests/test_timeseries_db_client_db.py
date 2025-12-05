from __future__ import annotations

import os
from uuid import uuid4

import pandas as pd
import pytest

from finantradealgo.storage.timeseries_db import TimeSeriesBackend, TimeSeriesDBClient, TimeSeriesDBConfig


@pytest.mark.db
def test_timeseries_db_client_roundtrip(monkeypatch):
    dsn = os.getenv("FT_TIMESCALE_DSN") or os.getenv("FT_POSTGRES_DSN")
    if not dsn:
        pytest.skip("No Timescale/Postgres DSN set in FT_TIMESCALE_DSN/FT_POSTGRES_DSN")

    pytest.importorskip("psycopg2")

    table = f"ohlcv_test_{uuid4().hex[:8]}"
    cfg = TimeSeriesDBConfig(
        backend=TimeSeriesBackend.TIMESCALE,
        dsn=dsn,
        ohlcv_table=table,
        metrics_table="metrics",
    )
    client = TimeSeriesDBClient(cfg)

    idx = pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:15:00"]).tz_localize("UTC")
    df = pd.DataFrame(
        {"open": [1, 2], "high": [2, 3], "low": [0.5, 1.5], "close": [1.5, 2.5], "volume": [10, 20]},
        index=idx,
    )

    # Ensure table exists
    import psycopg2

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                ts timestamptz NOT NULL,
                symbol text NOT NULL,
                timeframe text NOT NULL,
                open double precision,
                high double precision,
                low double precision,
                close double precision,
                volume double precision,
                PRIMARY KEY (symbol, timeframe, ts)
            );
            """
        )

    client.write_ohlcv("TESTUSDT", "15m", df)
    out = client.query_ohlcv("TESTUSDT", "15m", start_ts=idx[0], end_ts=idx[-1])

    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table};")
    conn.close()

    assert len(out) == 2
    assert out.iloc[0]["close"] == 1.5
    assert out.iloc[1]["close"] == 2.5
