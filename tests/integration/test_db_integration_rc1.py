import os

import pandas as pd
import pytest

from finantradealgo.storage.timeseries_db import TimeSeriesBackend, TimeSeriesDBClient, TimeSeriesDBConfig


DSN = os.getenv("FT_TIMESCALE_DSN") or os.getenv("FT_POSTGRES_DSN")
PSYCOPG2 = pytest.importorskip("psycopg2") if DSN else None


if not DSN:
    pytest.skip("DB DSN not set; skipping DB integration tests", allow_module_level=True)


def _create_test_table(conn, table_name: str):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                ts timestamptz NOT NULL,
                symbol text NOT NULL,
                timeframe text NOT NULL,
                open double precision NOT NULL,
                high double precision NOT NULL,
                low double precision NOT NULL,
                close double precision NOT NULL,
                volume double precision NULL,
                PRIMARY KEY (symbol, timeframe, ts)
            );
            """
        )
    conn.commit()


def _drop_test_table(conn, table_name: str):
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.commit()
    except Exception:
        conn.rollback()


def test_timeseries_roundtrip():
    table = "test_raw_ohlcv"
    conn = PSYCOPG2.connect(DSN)
    _create_test_table(conn, table)

    cfg = TimeSeriesDBConfig(
        backend=TimeSeriesBackend.TIMESCALE,
        dsn=DSN,
        ohlcv_table=table,
    )
    client = TimeSeriesDBClient(cfg)

    idx = pd.to_datetime(["2025-01-01 00:00:00Z", "2025-01-01 00:15:00Z"])
    df = pd.DataFrame(
        {"open": [1, 2], "high": [2, 3], "low": [0.5, 1.5], "close": [1.5, 2.5], "volume": [10, 20]},
        index=idx,
    )
    client.write_ohlcv("BTCUSDT", "15m", df)

    out = client.query_ohlcv("BTCUSDT", "15m", start_ts=idx[0])
    assert len(out) == 2

    _drop_test_table(conn, table)
    conn.close()
