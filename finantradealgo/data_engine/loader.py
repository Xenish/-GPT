from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


REQUIRED_COLUMNS: List[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


@dataclass
class DataConfig:
    timestamp_col: str = "timestamp"
    tz: str | None = None


def load_ohlcv_csv(path: str, config: DataConfig | None = None) -> pd.DataFrame:
    if config is None:
        config = DataConfig()

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], utc=True)
    df = df.sort_values(config.timestamp_col).reset_index(drop=True)

    return df


def _load_timeseries_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Timeseries file not found at {path}")
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"CSV at {path} missing timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_flow_features(
    symbol: str,
    timeframe: str,
    *,
    flow_dir: str | Path | None = None,
    base_dir: str | Path = "data",
) -> pd.DataFrame:
    if flow_dir:
        path = Path(flow_dir) / f"{symbol}_{timeframe}_flow.csv"
    else:
        path = Path(base_dir) / "flow" / f"{symbol}_{timeframe}_flow.csv"
    return _load_timeseries_csv(path)


def load_sentiment_features(
    symbol: str,
    timeframe: str,
    *,
    sentiment_dir: str | Path | None = None,
    base_dir: str | Path = "data",
) -> pd.DataFrame:
    if sentiment_dir:
        path = Path(sentiment_dir) / f"{symbol}_{timeframe}_sentiment.csv"
    else:
        path = Path(base_dir) / "sentiment" / f"{symbol}_{timeframe}_sentiment.csv"
    return _load_timeseries_csv(path)
