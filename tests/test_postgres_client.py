from __future__ import annotations

import os
from uuid import uuid4

import pytest

from finantradealgo.storage.postgres_client import DBConnectionConfig, PostgresClient


@pytest.mark.db
def test_postgres_client_smoke(monkeypatch):
    dsn = os.getenv("FT_TIMESCALE_DSN") or os.getenv("FT_POSTGRES_DSN")
    if not dsn:
        pytest.skip("No Postgres DSN set in FT_TIMESCALE_DSN/FT_POSTGRES_DSN")
    pytest.importorskip("psycopg2")

    cfg = DBConnectionConfig(dsn=dsn, minconn=1, maxconn=2)
    client = PostgresClient(cfg)

    table = f"tmp_smoke_{uuid4().hex[:8]}"
    client.execute(f"CREATE TABLE IF NOT EXISTS {table} (id serial primary key, val int);")
    client.execute(f"INSERT INTO {table} (val) VALUES (1);")
    row = client.fetchone(f"SELECT val FROM {table} LIMIT 1;")
    client.execute(f"DROP TABLE IF EXISTS {table};")
    client.close()

    assert row is not None
    assert row["val"] == 1
