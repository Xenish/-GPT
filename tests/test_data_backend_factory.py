import pytest

from finantradealgo.data_engine.data_backend import build_backend, CsvBackend
from finantradealgo.system.config_loader import DataConfig


def test_default_backend_is_csv():
    cfg = DataConfig.from_dict({})
    backend = build_backend(cfg)
    assert isinstance(backend, CsvBackend)
    assert backend.ohlcv_path_template == cfg.ohlcv_path_template


def test_timescale_backend_requires_dsn():
    cfg = DataConfig.from_dict({"backend": "timescale", "backend_params": {}})
    with pytest.raises(ValueError):
        build_backend(cfg)


def test_duckdb_backend_requires_path():
    cfg = DataConfig.from_dict({"backend": "duckdb", "backend_params": {}})
    with pytest.raises(ValueError):
        build_backend(cfg)
