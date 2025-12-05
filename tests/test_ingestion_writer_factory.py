from __future__ import annotations

import pytest

from finantradealgo.data_engine.ingestion.writer import (
    NullWarehouse,
    TimescaleWarehouse,
    build_warehouse_writer,
)
from finantradealgo.system.config_loader import WarehouseConfig


def test_build_warehouse_writer_none():
    cfg = WarehouseConfig(backend="none")
    writer = build_warehouse_writer(cfg)
    assert isinstance(writer, NullWarehouse)


def test_build_warehouse_writer_timescale_requires_dsn(monkeypatch):
    cfg = WarehouseConfig(backend="timescale", dsn_env="TEST_DSN_ENV")
    monkeypatch.delenv("TEST_DSN_ENV", raising=False)
    with pytest.raises(RuntimeError):
        build_warehouse_writer(cfg)

    pytest.importorskip("psycopg2")

    # Avoid real DB connection by stubbing TimescaleWarehouse
    created = {}

    class DummyTimescaleWarehouse:
        def __init__(self, dsn, table_map=None, batch_size=None):
            created["dsn"] = dsn
            created["table_map"] = table_map
            created["batch_size"] = batch_size

    monkeypatch.setattr("finantradealgo.data_engine.ingestion.writer.TimescaleWarehouse", DummyTimescaleWarehouse)
    monkeypatch.setenv("TEST_DSN_ENV", "postgres://user:pass@localhost:5432/db")
    writer = build_warehouse_writer(cfg)
    assert isinstance(writer, DummyTimescaleWarehouse)
    assert created["dsn"].startswith("postgres://")


def test_build_warehouse_writer_invalid_backend():
    cfg = WarehouseConfig(backend="invalid")
    with pytest.raises(ValueError):
        build_warehouse_writer(cfg)


@pytest.mark.parametrize("backend", ["csv", "duckdb"])
def test_build_warehouse_writer_unimplemented_backends(backend):
    cfg = WarehouseConfig(backend=backend)
    with pytest.raises(ValueError):
        build_warehouse_writer(cfg)
