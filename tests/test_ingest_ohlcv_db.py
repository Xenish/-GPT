from __future__ import annotations

import os
from uuid import uuid4

import pandas as pd
import pytest

from finantradealgo.data_engine.ingestion.models import IngestCandle
from finantradealgo.data_engine.ingestion.ohlcv import HistoricalOHLCVIngestor, timeframe_to_seconds
from finantradealgo.data_engine.ingestion.writer import build_warehouse_writer
from finantradealgo.system.config_loader import WarehouseConfig


@pytest.mark.db
def test_ingest_ohlcv_writes_to_db(monkeypatch):
    dsn = os.getenv("FT_TIMESCALE_DSN") or os.getenv("FT_POSTGRES_DSN")
    if not dsn:
        pytest.skip("No Timescale/Postgres DSN set in FT_TIMESCALE_DSN/FT_POSTGRES_DSN")

    psycopg2 = pytest.importorskip("psycopg2")

    table_name = f"raw_ohlcv_test_{uuid4().hex[:8]}"
    cfg = WarehouseConfig(
        backend="timescale",
        dsn_env="TEST_DSN",
        ohlcv_table=table_name,
    )
    monkeypatch.setenv("TEST_DSN", dsn)

    # Prepare table
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
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

    writer = build_warehouse_writer(cfg)

    class DummySource:
        def fetch(self, symbol, timeframe, start_ts, end_ts, limit=None):
            step = timeframe_to_seconds(timeframe)
            idx = pd.date_range(start_ts, end_ts, freq=f"{step}s", tz="UTC")
            candles = [
                IngestCandle(
                    ts=ts,
                    symbol=symbol,
                    timeframe=timeframe,
                    open=1.0,
                    high=2.0,
                    low=0.5,
                    close=1.5,
                    volume=10.0,
                )
                for ts in idx
            ]
            return candles

    source = DummySource()
    ingestor = HistoricalOHLCVIngestor(source, writer, chunk_size_bars=10)
    start = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    end = start + pd.Timedelta(minutes=4)
    written = ingestor.backfill_range("TESTUSDT", "1m", start, end)

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*), MIN(symbol), MIN(timeframe) FROM {table_name};")
        count, sym, tf = cur.fetchone()

    writer.close()
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name};")
    conn.close()

    assert written == 5
    assert count == 5
    assert sym == "TESTUSDT"
    assert tf == "1m"
