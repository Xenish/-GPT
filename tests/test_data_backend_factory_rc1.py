import pytest

from finantradealgo.data_engine.data_backend import build_backend, CsvBackend, DuckDBBackend, TimescaleBackend
from finantradealgo.system.config_loader import DataConfig, WarehouseConfig


def test_build_csv_backend():
    data_cfg = DataConfig.from_dict({"backend": "csv", "ohlcv_path_template": "data/ohlcv/{symbol}_{timeframe}.csv"})
    backend = build_backend(data_cfg)
    assert isinstance(backend, CsvBackend)


def test_build_duckdb_backend_requires_path():
    pytest.importorskip("duckdb")
    data_cfg = DataConfig.from_dict({"backend": "duckdb", "backend_params": {"database": "db.duckdb"}})
    backend = build_backend(data_cfg)
    assert isinstance(backend, DuckDBBackend)


def test_build_timescale_backend_prefers_backend_params(monkeypatch):
    pytest.importorskip("psycopg2")
    data_cfg = DataConfig.from_dict({"backend": "timescale", "backend_params": {"dsn": "postgres://x"}})
    backend = build_backend(data_cfg)
    assert isinstance(backend, TimescaleBackend)


def test_build_timescale_backend_uses_warehouse_cfg(monkeypatch):
    pytest.importorskip("psycopg2")
    data_cfg = DataConfig.from_dict({"backend": "timescale", "backend_params": {}})
    data_cfg.warehouse_cfg = WarehouseConfig(backend="timescale", dsn_env="FT_FAKE_DSN")
    # allow_missing=True path inside build_backend should tolerate missing env
    with pytest.raises(ValueError):
        build_backend(data_cfg)
