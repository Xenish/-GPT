import os

import pandas as pd
import pytest

from finantradealgo.storage.timeseries_db import TimeSeriesBackend, TimeSeriesDBClient, TimeSeriesDBConfig
from finantradealgo.system.config_loader import load_config
from finantradealgo.features.feature_pipeline import build_feature_pipeline_from_system_config


DSN = os.getenv("FT_TIMESCALE_DSN") or os.getenv("FT_POSTGRES_DSN")
psycopg2 = pytest.importorskip("psycopg2") if DSN else None

if not DSN:
    pytest.skip("DB DSN not set; skipping feature pipeline DB backend test", allow_module_level=True)


def _create_table(conn, table_name: str):
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


def _drop_table(conn, table_name: str):
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name};")
    conn.commit()


def test_feature_pipeline_reads_from_db_backend():
    table = "test_raw_ohlcv_feature"
    conn = psycopg2.connect(DSN)
    _create_table(conn, table)

    ts_cfg = TimeSeriesDBConfig(
        backend=TimeSeriesBackend.TIMESCALE,
        dsn=DSN,
        ohlcv_table=table,
    )
    client = TimeSeriesDBClient(ts_cfg)

    idx = pd.to_datetime(["2025-01-01 00:00:00Z", "2025-01-01 00:15:00Z"])
    df = pd.DataFrame(
        {"open": [1, 2], "high": [2, 3], "low": [0.5, 1.5], "close": [1.5, 2.5], "volume": [10, 20]},
        index=idx,
    )
    client.write_ohlcv("BTCUSDT", "15m", df)

    cfg = load_config("research")
    cfg["data_cfg"].backend = "timescale"
    cfg["data_cfg"].backend_params = {"dsn": DSN, "ohlcv_table": table}

    feat_df, meta = build_feature_pipeline_from_system_config(cfg, symbol="BTCUSDT", timeframe="15m")
    assert not feat_df.empty
    assert feat_df["timestamp"].is_monotonic_increasing

    _drop_table(conn, table)
    conn.close()
