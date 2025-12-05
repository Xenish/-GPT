import os

import pytest

from finantradealgo.system.config_loader import WarehouseConfig


def test_warehouse_none_allows_missing_env():
    cfg = WarehouseConfig(backend="none", dsn_env="FT_WAREHOUSE_DSN")
    assert cfg.get_dsn(allow_missing=True) is None
    assert cfg.get_dsn(allow_missing=False) is None


def test_warehouse_timescale_requires_env(monkeypatch):
    monkeypatch.delenv("FT_TIMESCALE_DSN", raising=False)
    cfg = WarehouseConfig(backend="timescale", dsn_env="FT_TIMESCALE_DSN")
    with pytest.raises(RuntimeError):
        cfg.get_dsn(allow_missing=False)
    # allow_missing should not raise
    assert cfg.get_dsn(allow_missing=True) is None


def test_warehouse_duckdb_allow_missing(monkeypatch):
    monkeypatch.delenv("FT_DUCKDB_PATH", raising=False)
    cfg = WarehouseConfig(backend="duckdb", dsn_env="FT_DUCKDB_PATH")
    assert cfg.get_dsn(allow_missing=True) is None
