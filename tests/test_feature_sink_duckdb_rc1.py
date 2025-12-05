import os
from pathlib import Path

import pandas as pd
import pytest

from finantradealgo.features.feature_builder import FeatureSink, FeatureSinkConfig


def test_duckdb_sink_write_and_read(tmp_path: Path):
    pytest.importorskip("duckdb")
    db_path = tmp_path / "features.duckdb"
    cfg = FeatureSinkConfig(kind="duckdb", duckdb_path=db_path, duckdb_table="features")
    sink = FeatureSink(cfg)

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:15:00"]),
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "timeframe": ["15m", "15m"],
            "feat1": [1.0, 2.0],
        }
    )

    ref = sink.write(df, symbol="BTCUSDT", timeframe="15m", mode="overwrite")
    assert db_path.exists()

    import duckdb  # type: ignore

    con = duckdb.connect(str(db_path), read_only=True)
    table = "features_15m"
    count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    assert count == 2
