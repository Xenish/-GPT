from __future__ import annotations

import os
from uuid import uuid4

import pandas as pd
import pytest

from finantradealgo.data_engine.data_backend import TimescaleBackend
from finantradealgo.data_engine.ingestion.models import IngestCandle
from finantradealgo.data_engine.ingestion.writer import TimescaleWarehouse
from finantradealgo.strategies.ema_cross import EMACrossStrategy
from finantradealgo.backtester.backtest_engine import Backtester, BacktestConfig


@pytest.mark.db
def test_db_lifecycle_migration_ingest_backtest(monkeypatch):
    dsn = os.getenv("FT_TIMESCALE_DSN") or os.getenv("FT_POSTGRES_DSN")
    if not dsn:
        pytest.skip("No Timescale/Postgres DSN set in FT_TIMESCALE_DSN/FT_POSTGRES_DSN")

    psycopg2 = pytest.importorskip("psycopg2")

    table = f"raw_ohlcv_{uuid4().hex[:8]}"
    with psycopg2.connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE {table} (
                    ts timestamptz NOT NULL,
                    symbol text NOT NULL,
                    timeframe text NOT NULL,
                    open double precision,
                    high double precision,
                    low double precision,
                    close double precision,
                    volume double precision,
                    vwap double precision,
                    PRIMARY KEY (symbol, timeframe, ts)
                );
                """
            )

    wh = TimescaleWarehouse(dsn, table_map={"ohlcv": table}, batch_size=500)

    idx = pd.date_range("2025-01-01 00:00:00", periods=30, freq="1min", tz="UTC")
    prices = pd.Series(range(1, 31), index=idx)
    candles = [
        IngestCandle(
            ts=ts,
            symbol="TESTUSDT",
            timeframe="1m",
            open=float(price),
            high=float(price + 0.5),
            low=float(price - 0.5),
            close=float(price + 0.2),
            volume=100.0,
            vwap=float(price),
        )
        for ts, price in prices.items()
    ]
    written = wh.upsert_ohlcv(candles)
    assert written == len(candles)

    backend = TimescaleBackend(dsn=dsn, tables={"ohlcv": table})
    df = backend.load_ohlcv("TESTUSDT", "1m")
    wh.close()

    with psycopg2.connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table};")

    assert not df.empty
    assert len(df) == len(candles)
    assert "close" in df.columns

    strat = EMACrossStrategy(fast=3, slow=5)
    bt = Backtester(strategy=strat, config=BacktestConfig(initial_cash=1000.0))
    result = bt.run(df)
    assert "equity_metrics" in result
    assert result["equity_metrics"]["final_equity"] > 0
