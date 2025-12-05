import os
from pathlib import Path

import pandas as pd
import pytest

from finantradealgo.live_trading.factories import create_live_engine
from finantradealgo.storage.timeseries_db import TimeSeriesBackend, TimeSeriesDBClient, TimeSeriesDBConfig
from finantradealgo.system.config_loader import load_config

DSN = os.getenv("FT_TIMESCALE_DSN") or os.getenv("FT_POSTGRES_DSN")
psycopg2 = pytest.importorskip("psycopg2") if DSN else None

if not DSN:
    pytest.skip("DB DSN not set; skipping live replay_db integration test", allow_module_level=True)


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


def test_live_engine_replay_db(tmp_path: Path):
    table = "test_raw_ohlcv_live"
    conn = psycopg2.connect(DSN)
    _create_table(conn, table)

    ts_cfg = TimeSeriesDBConfig(
        backend=TimeSeriesBackend.TIMESCALE,
        dsn=DSN,
        ohlcv_table=table,
    )
    client = TimeSeriesDBClient(ts_cfg)

    idx = pd.to_datetime(["2025-01-01 00:00:00Z", "2025-01-01 00:15:00Z", "2025-01-01 00:30:00Z"])
    df = pd.DataFrame(
        {"open": [1, 2, 3], "high": [2, 3, 4], "low": [0.5, 1.5, 2.5], "close": [1.5, 2.5, 3.5], "volume": [10, 20, 30]},
        index=idx,
    )
    client.write_ohlcv("BTCUSDT", "15m", df)

    cfg = load_config("live")
    cfg["data_cfg"].backend = "timescale"
    cfg["data_cfg"].backend_params = {"dsn": DSN, "ohlcv_table": table}
    live_section = dict(cfg.get("live", {}) or {})
    live_section["mode"] = "paper"
    live_section["data_source"] = "replay_db"
    live_section["heartbeat_path"] = str(tmp_path / "hb_{run_id}.json")
    cfg["live"] = live_section
    cfg["live_cfg"] = cfg["live_cfg"].from_dict(
        live_section,
        default_symbol=cfg.get("symbol"),
        default_timeframe=cfg.get("timeframe"),
    )

    engine, strategy_name = create_live_engine(cfg, run_id="test_replay_db")

    # Run through the bars
    for _ in range(3):
        bar = engine.data_source.next_bar()
        if bar is None:
            break
        engine._on_bar(bar)

    assert engine.iteration > 0
    assert not engine.kill_switch_triggered_flag

    hb_path = Path(live_section["heartbeat_path"].format(run_id=engine.run_id))
    assert hb_path.exists()

    _drop_table(conn, table)
    conn.close()
